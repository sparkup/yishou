"""The smoothing that keeps the number on screen from flickering."""

import numpy as np
import pytest

from yishou.demo import Smoother
from yishou.gestures import NONE_LABEL, NUM_CLASSES


def sure(label: int, confidence: float = 0.9) -> np.ndarray:
    """Probabilities that put `confidence` on one label and spread the rest."""
    probabilities = np.full(NUM_CLASSES, (1 - confidence) / (NUM_CLASSES - 1), dtype=np.float32)
    probabilities[label] = confidence
    return probabilities


def test_smoother_ignores_single_frame_flickers() -> None:
    smoother = Smoother(NUM_CLASSES, window=5)
    outputs = [smoother.update(sure(label)).argmax() for label in [3, 3, 3, 8, 3, 3]]
    assert outputs == [3] * 6


def test_smoother_follows_a_held_change() -> None:
    smoother = Smoother(NUM_CLASSES, window=5)
    for label in [3, 3, 8, 8, 8, 8]:
        averaged = smoother.update(sure(label))
    assert averaged.argmax() == 8


def test_confidence_matches_the_number_shown() -> None:
    smoother = Smoother(NUM_CLASSES, window=4)
    for probabilities in [sure(5, 0.9), sure(5, 0.7), sure(2, 0.6), sure(5, 0.8)]:
        averaged = smoother.update(probabilities)
    assert averaged.argmax() == 5
    assert averaged[5] == pytest.approx((0.9 + 0.7 + 0.8 + 0.04) / 4, abs=1e-6)


def test_frames_without_a_hand_count_as_none() -> None:
    smoother = Smoother(NUM_CLASSES, window=3)
    smoother.update(sure(4))
    smoother.update(None)
    assert smoother.update(None).argmax() == NONE_LABEL
