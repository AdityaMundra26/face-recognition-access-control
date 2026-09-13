"""Tests for src.vector_store.FaceVectorStore (real chromadb, no mocking)."""

from __future__ import annotations

import numpy as np
import pytest

from src.vector_store import FaceVectorStore

from conftest import nudge, random_embedding


@pytest.fixture
def store(tmp_path) -> FaceVectorStore:
    return FaceVectorStore(persist_dir=tmp_path / "vector_store")


def test_new_store_is_empty(store):
    assert store.count() == 0
    assert store.list_names() == []


def test_add_increments_count(store, rng):
    store.add("Alice", random_embedding(rng))
    store.add("Bob", random_embedding(rng))
    assert store.count() == 2
    assert store.list_names() == ["Alice", "Bob"]


def test_add_same_name_twice_keeps_both_entries(store, rng):
    embedding = random_embedding(rng)
    store.add("Alice", embedding)
    store.add("Alice", nudge(embedding, rng))
    assert store.count() == 2
    assert store.list_names() == ["Alice"]


def test_add_strips_and_rejects_empty_name(store, rng):
    with pytest.raises(ValueError, match="non-empty"):
        store.add("   ", random_embedding(rng))


def test_add_rejects_wrong_shape(store):
    with pytest.raises(ValueError, match="Unexpected embedding shape"):
        store.add("Alice", np.zeros(10, dtype=np.float32))


def test_query_empty_store_returns_no_matches(store, rng):
    assert store.query(random_embedding(rng)) == []


def test_query_finds_closest_match(store, rng):
    alice = random_embedding(rng)
    bob = random_embedding(rng)
    store.add("Alice", alice)
    store.add("Bob", bob)

    probe = nudge(alice, rng, amount=0.02)
    matches = store.query(probe, n_results=2)

    assert len(matches) == 2
    assert matches[0].name == "Alice"
    assert matches[0].similarity > matches[1].similarity


def test_query_rejects_wrong_shape(store):
    with pytest.raises(ValueError, match="Unexpected embedding shape"):
        store.query(np.zeros(10, dtype=np.float32))


def test_delete_by_name_removes_all_that_persons_entries(store, rng):
    embedding = random_embedding(rng)
    store.add("Alice", embedding)
    store.add("Alice", nudge(embedding, rng))
    store.add("Bob", random_embedding(rng))

    removed = store.delete_by_name("Alice")

    assert removed == 2
    assert store.list_names() == ["Bob"]
    assert store.count() == 1


def test_delete_by_name_missing_name_removes_nothing(store, rng):
    store.add("Bob", random_embedding(rng))
    assert store.delete_by_name("Nobody") == 0
    assert store.count() == 1
