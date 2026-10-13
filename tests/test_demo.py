"""The smoothing that keeps the number on screen from flickering."""

from yishou.demo import Smoother


def test_smoother_ignores_single_frame_flickers() -> None:
    smoother = Smoother(window=5)
    outputs = [smoother.update(label) for label in [3, 3, 3, 8, 3, 3]]
    assert outputs == [3] * 6


def test_smoother_follows_a_held_change() -> None:
    smoother = Smoother(window=5)
    outputs = [smoother.update(label) for label in [3, 3, 8, 8, 8, 8]]
    assert outputs[-1] == 8
