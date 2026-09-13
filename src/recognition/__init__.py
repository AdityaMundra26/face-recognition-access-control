"""Face recognition: match live faces against enrolled embeddings."""

from .liveness import CLOSED_THRESHOLD, OPEN_THRESHOLD, eye_openness, is_blink
from .recognizer import DEFAULT_THRESHOLD, FaceRecognizer, RecognitionResult

__all__ = [
    "CLOSED_THRESHOLD",
    "DEFAULT_THRESHOLD",
    "FaceRecognizer",
    "OPEN_THRESHOLD",
    "RecognitionResult",
    "eye_openness",
    "is_blink",
]
