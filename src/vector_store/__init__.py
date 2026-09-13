"""Storage and similarity search for face embeddings."""

from .store import (
    COLLECTION_NAME,
    DEFAULT_PERSIST_DIR,
    EMBEDDING_DIM,
    FaceVectorStore,
    Match,
)

__all__ = [
    "COLLECTION_NAME",
    "DEFAULT_PERSIST_DIR",
    "EMBEDDING_DIM",
    "FaceVectorStore",
    "Match",
]
