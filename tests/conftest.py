"""Shared pytest fixtures and helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.vector_store import EMBEDDING_DIM


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def random_embedding(rng: np.random.Generator) -> np.ndarray:
    """An L2-normalized random vector, standing in for a real face embedding."""
    vec = rng.normal(size=EMBEDDING_DIM).astype(np.float32)
    return vec / np.linalg.norm(vec)


def nudge(embedding: np.ndarray, rng: np.random.Generator, amount: float = 0.05) -> np.ndarray:
    """A slightly-perturbed copy, standing in for a second photo of the same person."""
    noisy = embedding + rng.normal(scale=amount, size=embedding.shape).astype(np.float32)
    return noisy / np.linalg.norm(noisy)


@pytest.fixture(scope="session")
def bundled_images() -> dict[str, Path]:
    """Paths to InsightFace's bundled sample images (buffalo_l is downloaded locally)."""
    import insightface.data as insight_data

    images = Path(insight_data.__file__).parent / "images"
    return {
        "single_face": images / "Tom_Hanks_54745.png",  # one, pre-cropped face
        "group_photo": images / "t1.jpg",  # six faces
    }
