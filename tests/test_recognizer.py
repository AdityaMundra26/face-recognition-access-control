"""Tests for src.recognition.FaceRecognizer."""

from __future__ import annotations

import pytest

from src.ingestion import detect_faces, generate_embedding, load_image
from src.recognition import DEFAULT_THRESHOLD, FaceRecognizer
from src.vector_store import FaceVectorStore

from conftest import nudge, random_embedding


@pytest.fixture
def store(tmp_path) -> FaceVectorStore:
    return FaceVectorStore(persist_dir=tmp_path / "vector_store")


def test_identify_against_empty_store_is_unmatched(store, rng):
    recognizer = FaceRecognizer(store)
    result = recognizer.identify(random_embedding(rng))
    assert result.matched is False
    assert result.name is None
    assert result.similarity == 0.0


def test_identify_above_threshold_matches(store, rng):
    alice = random_embedding(rng)
    store.add("Alice", alice)
    recognizer = FaceRecognizer(store)

    result = recognizer.identify(nudge(alice, rng, amount=0.01))

    assert result.matched is True
    assert result.name == "Alice"
    assert result.similarity >= DEFAULT_THRESHOLD


def test_identify_below_threshold_is_unmatched_but_reports_similarity(store, rng):
    alice = random_embedding(rng)
    store.add("Alice", alice)
    # A second independent random vector is, in 512-d, overwhelmingly likely
    # to sit well below any reasonable similarity threshold.
    unrelated = random_embedding(rng)
    recognizer = FaceRecognizer(store)

    result = recognizer.identify(unrelated)

    assert result.matched is False
    assert result.name is None
    assert result.similarity < DEFAULT_THRESHOLD


def test_custom_threshold_is_respected(store, rng):
    alice = random_embedding(rng)
    store.add("Alice", alice)
    probe = nudge(alice, rng, amount=0.01)
    lenient = FaceRecognizer(store, threshold=-1.0)
    strict = FaceRecognizer(store, threshold=1.01)

    assert lenient.identify(probe).matched is True
    assert strict.identify(probe).matched is False


def test_identify_faces_pairs_each_face_with_a_result(store, bundled_images):
    image = load_image(bundled_images["single_face"])
    face = detect_faces(image)[0]
    store.add("Tom Hanks", generate_embedding(face))
    recognizer = FaceRecognizer(store)

    pairs = recognizer.identify_faces([face])

    assert len(pairs) == 1
    result_face, result = pairs[0]
    assert result_face is face
    assert result.matched is True
    assert result.name == "Tom Hanks"
