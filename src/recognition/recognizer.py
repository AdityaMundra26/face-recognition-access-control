"""Face recognition: match a live face against enrolled ones.

Ties together face detection/embedding (:mod:`src.ingestion`) and persisted,
named embeddings (:mod:`src.vector_store`) to answer "who is this face, if
anyone enrolled?".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.vector_store import FaceVectorStore

# Cosine similarity threshold above which the best match counts as a
# recognized person. ArcFace (buffalo_l) embeddings: genuine (same-person)
# pairs typically score well above 0.5 and impostor pairs well below, so 0.5
# leaves a margin against false accepts. Tune per deployment/camera.
DEFAULT_THRESHOLD = 0.5


@dataclass(frozen=True)
class RecognitionResult:
    """Outcome of matching one probe embedding against the enrolled store."""

    matched: bool
    name: str | None
    similarity: float  # best similarity found; 0.0 if the store is empty


class FaceRecognizer:
    """Matches face embeddings against a :class:`~src.vector_store.FaceVectorStore`."""

    def __init__(self, store: FaceVectorStore, *, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.store = store
        self.threshold = threshold

    def identify(self, embedding: np.ndarray) -> RecognitionResult:
        """Return the best enrolled match for ``embedding``, if it clears the threshold."""
        matches = self.store.query(embedding, n_results=1)
        if not matches:
            return RecognitionResult(matched=False, name=None, similarity=0.0)

        best = matches[0]
        matched = best.similarity >= self.threshold
        return RecognitionResult(
            matched=matched,
            name=best.name if matched else None,
            similarity=best.similarity,
        )

    def identify_faces(self, faces: list) -> list[tuple[object, RecognitionResult]]:
        """Identify each detected InsightFace ``Face`` in ``faces``.

        Pairs each face with its :class:`RecognitionResult` so callers (e.g. a
        live overlay) can draw a name next to the right bounding box.
        """
        from src.ingestion.enrollment import generate_embedding

        return [(face, self.identify(generate_embedding(face))) for face in faces]
