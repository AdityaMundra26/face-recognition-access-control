"""Manual smoke test for the face enrollment module.

Usage:

    # explicit name=path pairs
    python demo_enrollment.py "Alice=data/samples/alice.jpg" "Bob=data/samples/bob.png"

    # or just paths (name is taken from the filename)
    python demo_enrollment.py data/samples/alice.jpg

    # with no arguments: enrolls every image in data/samples/, or, if that
    # directory is empty, runs a self-contained demo on InsightFace's bundled
    # images (one single-face enrollment + one multiple-faces error case)
    python demo_enrollment.py

For each image it prints the embedding shape, dtype and L2 norm. Explicit or
data/samples/ targets are also persisted into the real vector store
(data/enrolled_faces/vector_store/); the bundled fallback demo is not, since
it's a self-contained smoke test rather than real enrollment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from src.ingestion import FaceEnrollmentError, enroll_face
from src.vector_store import FaceVectorStore

SAMPLE_DIR = Path("data/samples")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def user_targets(args: list[str]) -> list[tuple[str, Path]]:
    """Targets from CLI args, or from data/samples/. Empty if neither is present."""
    if args:
        targets = []
        for arg in args:
            name, sep, path = arg.partition("=")
            if not sep:
                name, path = Path(arg).stem, arg
            targets.append((name.strip(), Path(path.strip())))
        return targets

    if SAMPLE_DIR.is_dir():
        return [
            (p.stem, p)
            for p in sorted(SAMPLE_DIR.iterdir())
            if p.suffix.lower() in IMAGE_SUFFIXES
        ]
    return []


def enroll_and_report(name: str, path: Path, store: FaceVectorStore | None = None) -> bool:
    """Enroll one image and print the result.

    If ``store`` is given, the embedding is also persisted there. Returns
    True if the image produced an embedding.
    """
    try:
        enrolled_name, embedding = enroll_face(name, path)
    except (FaceEnrollmentError, FileNotFoundError, ValueError) as exc:
        print(f"[FAIL] {name} ({path.name}): {exc}")
        return False

    print(
        f"[ OK ] {enrolled_name} ({path.name})\n"
        f"       shape={embedding.shape} dtype={embedding.dtype} "
        f"L2-norm={np.linalg.norm(embedding):.4f}\n"
        f"       first 5 values: {np.round(embedding[:5], 4)}"
    )
    if store is not None:
        store.add(enrolled_name, embedding)
        print(f"       stored in vector store (now {store.count()} embedding(s))")
    return True


def run_bundled_demo() -> int:
    """Self-contained demo using images shipped inside the insightface package."""
    import insightface.data as insight_data

    images = Path(insight_data.__file__).parent / "images"
    single_face = images / "Tom_Hanks_54745.png"  # one (pre-cropped) face
    group_photo = images / "t1.jpg"               # six faces

    print("No images in data/samples/ - running the bundled demo.\n")

    print("1. Single face -> should enroll and produce a 512-d embedding:")
    ok = enroll_and_report("Tom Hanks", single_face)

    print("\n2. Group photo -> should be rejected (more than one face):")
    try:
        enroll_face("group_photo", group_photo)
    except FaceEnrollmentError as exc:
        print(f"[ OK ] rejected as expected: {exc}")
        rejected = True
    else:
        print("[FAIL] expected a FaceEnrollmentError but enrollment succeeded")
        rejected = False

    return 0 if (ok and rejected) else 1


def main(argv: list[str]) -> int:
    targets = user_targets(argv)
    if not targets:
        return run_bundled_demo()

    store = FaceVectorStore()
    failures = sum(not enroll_and_report(name, path, store) for name, path in targets)
    print(
        f"\nEnrolled {len(targets) - failures}/{len(targets)} image(s) "
        f"(vector store now has {store.count()} total)."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
