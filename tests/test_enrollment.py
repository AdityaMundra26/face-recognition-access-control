"""Tests for src.ingestion.enrollment (real InsightFace detection, no mocking)."""

from __future__ import annotations

import numpy as np
import pytest

from src.ingestion import (
    EMBEDDING_DIM,
    FaceEnrollmentError,
    detect_single_face,
    enroll_face,
    generate_embedding,
    load_image,
)


def test_detect_single_face_succeeds_on_one_face(bundled_images):
    image = load_image(bundled_images["single_face"])
    face = detect_single_face(image)
    assert face is not None


def test_detect_single_face_raises_on_no_face():
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    with pytest.raises(FaceEnrollmentError, match="No face detected"):
        detect_single_face(blank)


def test_detect_single_face_raises_on_multiple_faces(bundled_images):
    image = load_image(bundled_images["group_photo"])
    with pytest.raises(FaceEnrollmentError, match="Detected 6 faces"):
        detect_single_face(image)


def test_generate_embedding_shape_dtype_and_norm(bundled_images):
    image = load_image(bundled_images["single_face"])
    face = detect_single_face(image)
    embedding = generate_embedding(face)

    assert embedding.shape == (EMBEDDING_DIM,)
    assert embedding.dtype == np.float32
    assert np.linalg.norm(embedding) == pytest.approx(1.0, abs=1e-4)


def test_load_image_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "does_not_exist.jpg")


def test_enroll_face_happy_path(bundled_images):
    name, embedding = enroll_face("Tom Hanks", bundled_images["single_face"])
    assert name == "Tom Hanks"
    assert embedding.shape == (EMBEDDING_DIM,)


def test_enroll_face_empty_name_raises(bundled_images):
    with pytest.raises(ValueError, match="non-empty"):
        enroll_face("   ", bundled_images["single_face"])


def test_enroll_face_rejects_multiple_faces(bundled_images):
    with pytest.raises(FaceEnrollmentError):
        enroll_face("group", bundled_images["group_photo"])
