"""Blink-based liveness check using InsightFace's 106-point landmarks.

A still photo (or a photo of a phone/monitor showing a face) can't blink on
command, so requiring a detected blink between an "eyes open" shot and a
"blink now" shot is a cheap, dependency-free liveness signal — buffalo_l
already produces the 106-point landmarks needed, no extra model or package.

This is a deterrent against the common case of holding up a single static
photo, not a defense against a determined attacker with a video of the
target blinking. It's intentionally simple, matching the rest of this
project's threat model.

Eye point indices (33-42 and 87-96, each a contiguous 10-point eye contour)
were identified empirically by rendering InsightFace's landmark_2d_106
output on a sample face and inspecting each cluster's coordinates -- the
106-point layout isn't documented in the installed insightface package.
"""

from __future__ import annotations

import numpy as np

LEFT_EYE_INDICES = list(range(33, 43))
RIGHT_EYE_INDICES = list(range(87, 97))

# Eye openness = landmark cluster height / width. Empirically, open eyes on a
# real face score ~0.35-0.45; a closed eye collapses to a thin line, well
# under 0.2. The gap between OPEN_THRESHOLD and CLOSED_THRESHOLD is a dead
# zone so a partial blink or a slightly squinted eye counts as neither.
OPEN_THRESHOLD = 0.28
CLOSED_THRESHOLD = 0.18


def eye_openness(landmarks_2d_106: np.ndarray) -> float:
    """Average left/right eye openness (contour height / width).

    ``landmarks_2d_106`` is a face's ``landmark_2d_106`` attribute, a
    ``(106, 2)`` array of (x, y) points.
    """

    def ratio(indices: list[int]) -> float:
        points = landmarks_2d_106[indices]
        width = points[:, 0].max() - points[:, 0].min()
        height = points[:, 1].max() - points[:, 1].min()
        return float(height / width) if width > 0 else 0.0

    return (ratio(LEFT_EYE_INDICES) + ratio(RIGHT_EYE_INDICES)) / 2


def is_blink(before_openness: float, after_openness: float) -> bool:
    """True if openness went from clearly open to clearly closed."""
    return before_openness >= OPEN_THRESHOLD and after_openness <= CLOSED_THRESHOLD
