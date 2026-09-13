"""Persistent storage and similarity search for face embeddings.

Wraps a Chroma collection so callers can add, query, and manage named face
embeddings without touching the Chroma API directly. Embeddings are expected
to be L2-normalized (as :func:`src.ingestion.generate_embedding` produces by
default), so the collection is configured for cosine distance and query
results report cosine similarity.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np

# Matches src.ingestion.enrollment.EMBEDDING_DIM (ArcFace embedding size for
# the buffalo_l model pack). Duplicated here rather than imported so that
# using the vector store doesn't pull in InsightFace/OpenCV.
EMBEDDING_DIM = 512

DEFAULT_PERSIST_DIR = Path("data/enrolled_faces/vector_store")
COLLECTION_NAME = "enrolled_faces"


@dataclass(frozen=True)
class Match:
    """One nearest-neighbor result from :meth:`FaceVectorStore.query`."""

    id: str
    name: str
    similarity: float  # cosine similarity in [-1, 1]; 1.0 is an exact match


class FaceVectorStore:
    """Persistent store of named face embeddings, backed by Chroma.

    A person may be enrolled multiple times (e.g. one embedding per photo);
    each :meth:`add` call creates a new entry rather than overwriting
    previous ones for the same name.
    """

    def __init__(self, persist_dir: str | Path = DEFAULT_PERSIST_DIR) -> None:
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    def add(self, name: str, embedding: np.ndarray) -> str:
        """Store one embedding under ``name``. Returns the generated entry id."""
        name = name.strip()
        if not name:
            raise ValueError("name must be a non-empty string")
        _validate_embedding(embedding)

        entry_id = str(uuid.uuid4())
        self._collection.add(
            ids=[entry_id],
            embeddings=[embedding.astype(np.float32).tolist()],
            metadatas=[{"name": name}],
        )
        return entry_id

    def query(self, embedding: np.ndarray, n_results: int = 1) -> list[Match]:
        """Return up to ``n_results`` nearest enrolled embeddings, best first."""
        _validate_embedding(embedding)
        count = self.count()
        if count == 0:
            return []

        result = self._collection.query(
            query_embeddings=[embedding.astype(np.float32).tolist()],
            n_results=min(n_results, count),
            include=["metadatas", "distances"],
        )
        ids = result["ids"][0]
        distances = result["distances"][0]
        metadatas = result["metadatas"][0]
        return [
            Match(id=entry_id, name=metadata["name"], similarity=1.0 - distance)
            for entry_id, metadata, distance in zip(ids, metadatas, distances)
        ]

    def delete_by_name(self, name: str) -> int:
        """Remove all entries enrolled under ``name``. Returns the count removed."""
        existing = self._collection.get(where={"name": name})
        ids = existing["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def list_names(self) -> list[str]:
        """Return the distinct enrolled names, sorted."""
        if self.count() == 0:
            return []
        metadatas = self._collection.get(include=["metadatas"])["metadatas"]
        return sorted({metadata["name"] for metadata in metadatas})

    def count(self) -> int:
        """Return the total number of stored embeddings (across all names)."""
        return self._collection.count()


def _validate_embedding(embedding: np.ndarray) -> None:
    embedding = np.asarray(embedding)
    if embedding.shape != (EMBEDDING_DIM,):
        raise ValueError(
            f"Unexpected embedding shape {embedding.shape}; expected ({EMBEDDING_DIM},)."
        )
