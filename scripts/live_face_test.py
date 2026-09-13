"""Real-time visual test of the InsightFace detection + recognition pipeline.

Opens the webcam, runs InsightFace's FaceAnalysis on every frame, and draws a
box around each detected face labeled with its recognized name (looked up in
the persistent vector store) or "unknown".

Keys (focus the video window):
    s   enroll the largest face in the current frame: prompts for a name in
        the terminal, then adds its embedding to the vector store
        (data/enrolled_faces/vector_store/) and archives the frame to
        data/enrolled_faces/test_samples/.
    q   quit

Run from the repo root:  python scripts/live_face_test.py [--camera N]
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ingestion.enrollment import (  # noqa: E402  (needs sys.path set first)
    FaceEnrollmentError,
    generate_embedding,
    get_face_app,
)
from src.recognition import FaceRecognizer, RecognitionResult  # noqa: E402
from src.vector_store import FaceVectorStore  # noqa: E402

ARCHIVE_DIR = REPO_ROOT / "data" / "enrolled_faces" / "test_samples"
WINDOW = "live face test  (s = enroll, q = quit)"

GREEN = (0, 255, 0)
YELLOW = (0, 215, 255)
RED = (0, 0, 255)


def _face_area(face) -> float:
    x1, y1, x2, y2 = face.bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def draw_overlay(
    frame: np.ndarray, pairs: list[tuple[object, RecognitionResult]]
) -> np.ndarray:
    """Draw a labeled box per (face, RecognitionResult) plus the live count."""
    annotated = frame.copy()
    for face, result in pairs:
        x1, y1, x2, y2 = face.bbox.astype(int)
        color = GREEN if result.matched else YELLOW
        label = (
            f"{result.name} {result.similarity:.2f}"
            if result.matched
            else f"unknown {result.similarity:.2f}"
        )
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated, label, (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )

    count_color = GREEN if len(pairs) == 1 else YELLOW if pairs else RED
    cv2.putText(
        annotated, f"Faces: {len(pairs)}", (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, count_color, 2, cv2.LINE_AA,
    )
    cv2.putText(
        annotated, "s = enroll   q = quit", (12, annotated.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return annotated


def _slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_")
    return slug or "unnamed"


def enroll_from_frame(frame: np.ndarray, faces: list, store: FaceVectorStore) -> None:
    """Prompt for a name and add the largest face's embedding to ``store``."""
    if not faces:
        print("  no face in frame - nothing enrolled")
        return
    if len(faces) > 1:
        print(f"  {len(faces)} faces in frame - enrolling the largest one")
    face = max(faces, key=_face_area)

    try:
        embedding = generate_embedding(face)
    except FaceEnrollmentError as exc:
        print(f"  could not build embedding: {exc}")
        return

    name = input("  enrollment name: ").strip()
    if not name:
        print("  empty name - nothing enrolled")
        return

    store.add(name, embedding)

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = ARCHIVE_DIR / f"{_slugify(name)}_{stamp}.jpg"
    cv2.imwrite(str(image_path), frame)
    print(f"  enrolled '{name}' in the vector store; archived frame to {image_path.name}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    parser.add_argument(
        "--det-size", type=int, default=640,
        help="square detection size; smaller is faster (default 640)",
    )
    args = parser.parse_args(argv)

    print("loading InsightFace models (first run downloads the buffalo_l pack)...")
    app = get_face_app(det_size=(args.det_size, args.det_size))
    store = FaceVectorStore()
    recognizer = FaceRecognizer(store)
    print(f"vector store has {store.count()} enrolled embedding(s): {store.list_names()}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"error: could not open webcam at index {args.camera}")
        return 1

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    print("webcam open - focus the video window; 's' to enroll, 'q' to quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("error: failed to read a frame from the webcam")
                return 1

            faces = app.get(frame)
            pairs = recognizer.identify_faces(faces)
            cv2.imshow(WINDOW, draw_overlay(frame, pairs))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                enroll_from_frame(frame, faces, store)

            # window closed via the title-bar X
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
