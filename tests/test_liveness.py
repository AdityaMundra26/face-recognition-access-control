"""Tests for src.recognition.liveness (blink-based liveness check)."""

from __future__ import annotations

import numpy as np

from src.ingestion import detect_faces, load_image
from src.recognition import CLOSED_THRESHOLD, OPEN_THRESHOLD, eye_openness, is_blink
from src.recognition.liveness import LEFT_EYE_INDICES, RIGHT_EYE_INDICES


def _landmarks_with_eye_height(height: float, width: float = 15.0) -> np.ndarray:
    """A 106-point array with both eye clusters set to a given height/width.

    Every non-eye point is placed at the origin — only the eye clusters
    matter for :func:`eye_openness`.
    """
    points = np.zeros((106, 2), dtype=np.float32)
    for indices, cx in [(LEFT_EYE_INDICES, 30.0), (RIGHT_EYE_INDICES, 70.0)]:
        # spread points evenly across [cx, cx+width] x [0, height]
        xs = np.linspace(cx, cx + width, num=len(indices))
        ys = np.linspace(0, height, num=len(indices))
        points[indices, 0] = xs
        points[indices, 1] = ys
    return points


def test_eye_openness_high_for_wide_open_eyes():
    open_eyes = _landmarks_with_eye_height(height=7.0, width=15.0)
    assert eye_openness(open_eyes) >= OPEN_THRESHOLD


def test_eye_openness_low_for_closed_eyes():
    closed_eyes = _landmarks_with_eye_height(height=1.0, width=15.0)
    assert eye_openness(closed_eyes) <= CLOSED_THRESHOLD


def test_is_blink_true_for_open_then_closed():
    assert is_blink(before_openness=0.4, after_openness=0.05) is True


def test_is_blink_false_if_never_open():
    assert is_blink(before_openness=0.2, after_openness=0.05) is False


def test_is_blink_false_if_never_closes():
    assert is_blink(before_openness=0.4, after_openness=0.3) is False


def test_is_blink_false_in_the_ambiguous_dead_zone():
    # between CLOSED_THRESHOLD and OPEN_THRESHOLD counts as neither state
    assert is_blink(before_openness=0.4, after_openness=0.22) is False


def test_eye_openness_on_a_real_open_eyed_face(bundled_images):
    image = load_image(bundled_images["single_face"])
    face = detect_faces(image)[0]
    assert eye_openness(face.landmark_2d_106) >= OPEN_THRESHOLD
