"""Face recognition: match live faces against enrolled embeddings."""

from .liveness import CLOSED_THRESHOLD, OPEN_THRESHOLD, eye_openness, is_blink
from .recognizer import DEFAULT_THRESHOLD, THRESHOLD_ENV_VAR, FaceRecognizer, RecognitionResult

__all__ = [
    "CLOSED_THRESHOLD",
    "DEFAULT_THRESHOLD",
    "FaceRecognizer",
    "OPEN_THRESHOLD",
    "RecognitionResult",
    "THRESHOLD_ENV_VAR",
    "eye_openness",
    "is_blink",
]
