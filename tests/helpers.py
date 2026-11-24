"""Synthetic hands, so the tests need no webcam and no recorded data."""

from pathlib import Path

import numpy as np

from yishou.dataset import Sample, SampleWriter
from yishou.gestures import NUM_CLASSES


def synthetic_samples(per_burst: int = 30, bursts: int = 3, seed: int = 0) -> list[Sample]:
    """One random prototype hand per gesture, with small per-frame jitter.

    The prototypes are the same whatever the seed; `seed` only changes the jitter,
    so two seeds give two sessions of the same hands.
    """
    prototypes = np.random.default_rng(0).uniform(100, 500, size=(NUM_CLASSES, 21, 3))
    rng = np.random.default_rng(seed)
    samples = []
    for label, prototype in enumerate(prototypes):
        for burst in range(bursts):
            for _ in range(per_burst):
                points = (prototype + rng.normal(0, 2, size=(21, 3))).astype(np.float32)
                samples.append(Sample(label, burst, "Right", points))
    return samples


def write(samples: list[Sample], path: Path) -> None:
    with SampleWriter(path) as writer:
        for sample in samples:
            writer.write(sample)
