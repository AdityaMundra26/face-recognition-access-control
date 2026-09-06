"""Real-time visual test of the InsightFace detection pipeline.

Opens the webcam, runs InsightFace's FaceAnalysis on every frame, draws a box
around each detected face, and overlays the live face count.

Keys (focus the video window):
    s   save the current frame + embedding as a named enrollment
        (you'll be prompted for a name in the terminal)
    q   quit

Saved enrollments go to data/enrolled_faces/test_samples/ as a matching pair:
    <name>_<timestamp>.jpg   the captured frame
    <name>_<timestamp>.npy   the 512-d float32 embedding

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

SAVE_DIR = REPO_ROOT / "data" / "enrolled_faces" / "test_samples"
WINDOW = "live face test  (s = save enrollment, q = quit)"

GREEN = (0, 255, 0)
YELLOW = (0, 215, 255)
RED = (0, 0, 255)


def _face_area(face) -> float:
    x1, y1, x2, y2 = face.bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def draw_overlay(frame: np.ndarray, faces: list) -> np.ndarray:
    """Draw a box per face plus the live count. Returns the annotated frame."""
    annotated = frame.copy()
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), GREEN, 2)
        score = float(getattr(face, "det_score", 0.0) or 0.0)
        cv2.putText(
            annotated, f"{score:.2f}", (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1, cv2.LINE_AA,
        )

    count_color = GREEN if len(faces) == 1 else YELLOW if faces else RED
    cv2.putText(
        annotated, f"Faces: {len(faces)}", (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, count_color, 2, cv2.LINE_AA,
    )
    cv2.putText(
        annotated, "s = save   q = quit", (12, annotated.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return annotated


def _slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_")
    return slug or "unnamed"


def save_enrollment(frame: np.ndarray, faces: list) -> None:
    """Prompt for a name and save the frame + one embedding to SAVE_DIR."""
    if not faces:
        print("  no face in frame - nothing saved")
        return
    if len(faces) > 1:
        print(f"  {len(faces)} faces in frame - saving the largest one")
    face = max(faces, key=_face_area)

    try:
        embedding = generate_embedding(face)
    except FaceEnrollmentError as exc:
        print(f"  could not build embedding: {exc}")
        return

    name = input("  enrollment name: ").strip()
    if not name:
        print("  empty name - nothing saved")
        return

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = SAVE_DIR / f"{_slugify(name)}_{stamp}"
    cv2.imwrite(str(base.with_suffix(".jpg")), frame)
    np.save(base.with_suffix(".npy"), embedding)
    print(f"  saved {base.name}.jpg + .npy  (embedding {embedding.shape})")


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

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"error: could not open webcam at index {args.camera}")
        return 1

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    print("webcam open - focus the video window; 's' to save, 'q' to quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("error: failed to read a frame from the webcam")
                return 1

            faces = app.get(frame)
            cv2.imshow(WINDOW, draw_overlay(frame, faces))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                save_enrollment(frame, faces)

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
