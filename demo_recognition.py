"""Manual smoke test for end-to-end recognition: enroll -> detect -> identify.

Uses InsightFace's bundled sample images against a throwaway vector store, so
it needs the buffalo_l model pack (downloaded on first use of src.ingestion)
but no webcam.

    python demo_recognition.py
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from src.ingestion import detect_faces, generate_embedding
from src.recognition import FaceRecognizer
from src.vector_store import FaceVectorStore


def main() -> int:
    import insightface.data as insight_data

    images = Path(insight_data.__file__).parent / "images"
    tom_hanks = images / "Tom_Hanks_54745.png"  # one (pre-cropped) face
    group_photo = images / "t1.jpg"             # six faces, none enrolled

    import cv2

    persist_dir = Path(tempfile.mkdtemp(prefix="face_recognition_demo_"))
    ok = True

    try:
        store = FaceVectorStore(persist_dir=persist_dir)
        recognizer = FaceRecognizer(store)

        print("enrolling Tom Hanks from the bundled sample image...")
        image = cv2.imread(str(tom_hanks), cv2.IMREAD_COLOR)
        face = detect_faces(image)[0]
        store.add("Tom Hanks", generate_embedding(face))
        print(f"store now has {store.count()} embedding(s): {store.list_names()}")

        print("\n1. Re-identify the same photo -> should match Tom Hanks:")
        result = recognizer.identify(generate_embedding(face))
        print(f"   matched={result.matched} name={result.name} similarity={result.similarity:.4f}")
        if not result.matched or result.name != "Tom Hanks":
            print("[FAIL] expected a confident match on Tom Hanks")
            ok = False
        else:
            print("[ OK ]")

        print("\n2. Identify faces in an unenrolled group photo -> none should match:")
        group_image = cv2.imread(str(group_photo), cv2.IMREAD_COLOR)
        faces = detect_faces(group_image)
        pairs = recognizer.identify_faces(faces)
        for i, (_, res) in enumerate(pairs):
            print(f"   face {i}: matched={res.matched} best_similarity={res.similarity:.4f}")
        if any(res.matched for _, res in pairs):
            print("[FAIL] expected no matches among unenrolled faces")
            ok = False
        else:
            print("[ OK ]")
    finally:
        shutil.rmtree(persist_dir, ignore_errors=True)

    print("\nAll checks passed." if ok else "\nSome checks FAILED.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
