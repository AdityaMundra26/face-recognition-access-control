"""Face enrollment.

Loads an image (from disk or a webcam capture), detects a face with InsightFace's
``buffalo_l`` model pack, and produces a 512-d embedding ready to be stored in the
vector store.

The public entry point is :func:`enroll_face`. Detection failures (no face, or
more than one face) raise :class:`FaceEnrollmentError` with a clear message.
"""

from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

MODEL_PACK = "buffalo_l"
EMBEDDING_DIM = 512  # InsightFace's default ArcFace embedding size for buffalo_l

# (0, 0) selects InsightFace 1.0.x "auto" detection sizing, which tries both
# (128, 128) and (640, 640). This detects small / pre-cropped face images as well
# as full photos; a single fixed (640, 640) misses tightly-cropped faces.
DEFAULT_DET_SIZE = (0, 0)


class FaceEnrollmentError(Exception):
    """Raised when an image cannot be enrolled (no face or multiple faces)."""


# The FaceAnalysis app is expensive to construct (loads several ONNX models), so
# it is built once and reused.
_app: FaceAnalysis | None = None


def get_face_app(det_size: tuple[int, int] = DEFAULT_DET_SIZE) -> FaceAnalysis:
    """Return a lazily-initialised, cached InsightFace ``FaceAnalysis`` app.

    On first call this downloads the ``buffalo_l`` model pack (~280 MB) to
    ``~/.insightface`` if it is not already present, so it needs network access
    the first time it runs.

    ``det_size`` defaults to ``(0, 0)`` (auto). Pass a fixed size like
    ``(640, 640)`` for slightly faster, single-scale detection on normal photos.
    """
    global _app
    if _app is None:
        import onnxruntime

        providers = onnxruntime.get_available_providers()
        ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
        # InsightFace prints model-loading chatter straight to stdout; hush it
        # unless FACE_INGESTION_VERBOSE is set.
        quiet = not os.environ.get("FACE_INGESTION_VERBOSE")
        sink = io.StringIO() if quiet else None
        with contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext():
            app = FaceAnalysis(name=MODEL_PACK, providers=providers)
            app.prepare(ctx_id=ctx_id, det_size=det_size)
        _app = app
    return _app


def load_image(source: str | Path | int) -> np.ndarray:
    """Load a BGR image.

    ``source`` is either a path to an image file, or an integer webcam index
    (``0`` for the default camera), in which case a single frame is captured.
    """
    if isinstance(source, int):
        return _capture_from_webcam(source)

    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    # cv2.imread silently returns None on unreadable/corrupt files.
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def _capture_from_webcam(camera_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(camera_index)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {camera_index}")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to capture a frame from webcam {camera_index}")
        return frame
    finally:
        cap.release()


def detect_faces(image: np.ndarray) -> list:
    """Detect all faces in a BGR image. Returns a list of InsightFace ``Face``."""
    return get_face_app().get(image)


def detect_single_face(image: np.ndarray):
    """Detect exactly one face, or raise :class:`FaceEnrollmentError`."""
    faces = detect_faces(image)
    if len(faces) == 0:
        raise FaceEnrollmentError(
            "No face detected in the image. Use a clear, front-facing photo with "
            "the face well lit and large enough in frame."
        )
    if len(faces) > 1:
        raise FaceEnrollmentError(
            f"Detected {len(faces)} faces in the image. Enrollment needs exactly "
            "one person per image; crop or retake the photo."
        )
    return faces[0]


def generate_embedding(face, *, normalize: bool = True) -> np.ndarray:
    """Return the 512-d embedding for a detected ``face``.

    By default the L2-normalized embedding is returned, which makes cosine
    similarity equivalent to a dot product in the vector store. Pass
    ``normalize=False`` for the raw ArcFace embedding.
    """
    embedding = face.normed_embedding if normalize else face.embedding
    if embedding is None:
        raise FaceEnrollmentError(
            "The detected face has no embedding — the recognition model may have "
            "failed to load. Check that the buffalo_l model pack downloaded fully."
        )
    embedding = np.asarray(embedding, dtype=np.float32)
    if embedding.shape != (EMBEDDING_DIM,):
        raise FaceEnrollmentError(
            f"Unexpected embedding shape {embedding.shape}; expected ({EMBEDDING_DIM},)."
        )
    return embedding


def enroll_face(
    name: str,
    image_path: str | Path | int,
    *,
    normalize: bool = True,
) -> tuple[str, np.ndarray]:
    """Enroll a single face.

    Detects the face in ``image_path`` (a file path, or an int webcam index),
    generates its embedding, and returns ``(name, embedding)`` ready to be stored.

    Raises :class:`FaceEnrollmentError` if the image does not contain exactly one
    detectable face.
    """
    name = name.strip()
    if not name:
        raise ValueError("name must be a non-empty string")

    image = load_image(image_path)
    face = detect_single_face(image)
    embedding = generate_embedding(face, normalize=normalize)
    return name, embedding
