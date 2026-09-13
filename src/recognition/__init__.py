"""Face recognition: match live faces against enrolled embeddings."""

from .recognizer import DEFAULT_THRESHOLD, FaceRecognizer, RecognitionResult

__all__ = [
    "DEFAULT_THRESHOLD",
    "FaceRecognizer",
    "RecognitionResult",
]
