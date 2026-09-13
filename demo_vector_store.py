"""Manual smoke test for the face vector store.

Uses synthetic embeddings (no webcam or InsightFace model download required)
to exercise add / query / list / delete against a throwaway Chroma database.

    python demo_vector_store.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from src.vector_store import EMBEDDING_DIM, FaceVectorStore


def random_embedding(rng: np.random.Generator) -> np.ndarray:
    """An L2-normalized random vector, standing in for a real face embedding."""
    vec = rng.normal(size=EMBEDDING_DIM).astype(np.float32)
    return vec / np.linalg.norm(vec)


def nudge(embedding: np.ndarray, rng: np.random.Generator, amount: float = 0.05) -> np.ndarray:
    """A slightly-perturbed copy, standing in for a second photo of the same person."""
    noisy = embedding + rng.normal(scale=amount, size=embedding.shape).astype(np.float32)
    return noisy / np.linalg.norm(noisy)


def main() -> int:
    rng = np.random.default_rng(0)
    persist_dir = Path(tempfile.mkdtemp(prefix="face_vector_store_demo_"))
    ok = True

    try:
        store = FaceVectorStore(persist_dir=persist_dir)
        print(f"opened store at {persist_dir} (starts empty: count={store.count()})")

        alice = random_embedding(rng)
        bob = random_embedding(rng)
        store.add("Alice", alice)
        store.add("Alice", nudge(alice, rng))  # second enrollment photo
        store.add("Bob", bob)
        print(f"added 3 embeddings -> count={store.count()}, names={store.list_names()}")

        print("\n1. Query with a probe close to Alice -> best match should be Alice:")
        probe = nudge(alice, rng, amount=0.02)
        matches = store.query(probe, n_results=2)
        for m in matches:
            print(f"   {m.name}  similarity={m.similarity:.4f}  id={m.id}")
        if not matches or matches[0].name != "Alice":
            print("[FAIL] expected Alice as the top match")
            ok = False
        else:
            print("[ OK ]")

        print("\n2. delete_by_name('Alice') -> both Alice entries removed:")
        removed = store.delete_by_name("Alice")
        print(f"   removed {removed} entr{'y' if removed == 1 else 'ies'}, "
              f"count={store.count()}, names={store.list_names()}")
        if removed != 2 or store.list_names() != ["Bob"]:
            print("[FAIL] expected 2 removed and only Bob left")
            ok = False
        else:
            print("[ OK ]")

        print("\n3. Query with wrong-shaped embedding -> should raise ValueError:")
        try:
            store.query(np.zeros(10, dtype=np.float32))
        except ValueError as exc:
            print(f"[ OK ] rejected as expected: {exc}")
        else:
            print("[FAIL] expected a ValueError")
            ok = False
    finally:
        shutil.rmtree(persist_dir, ignore_errors=True)

    print("\nAll checks passed." if ok else "\nSome checks FAILED.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
