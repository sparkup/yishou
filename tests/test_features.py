"""The landmark normalisation: invariant to position, scale, rotation and handedness."""

import numpy as np
import pytest

from yishou.features import FEATURE_DIM, MIDDLE_MCP, to_features


@pytest.fixture
def hand() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(100, 500, size=(21, 3)).astype(np.float32)


def rotate_in_plane(points: np.ndarray, degrees: float) -> np.ndarray:
    theta = np.radians(degrees)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated = points.copy()
    rotated[:, :2] = points[:, :2] @ rotation.T
    return rotated


def test_shape_and_canonical_pose(hand: np.ndarray) -> None:
    features = to_features(hand, "Right")
    assert features.shape == (FEATURE_DIM,)

    points = features.reshape(21, 3)
    np.testing.assert_allclose(points[0], 0, atol=1e-5)  # wrist at the origin
    np.testing.assert_allclose(points[MIDDLE_MCP, :2], [0, -1], atol=1e-4)  # palm up, length 1


def test_does_not_modify_input(hand: np.ndarray) -> None:
    original = hand.copy()
    to_features(hand, "Left")
    np.testing.assert_array_equal(hand, original)


def test_invariant_to_translation(hand: np.ndarray) -> None:
    moved = hand + np.array([40.0, -25.0, 3.0], dtype=np.float32)
    np.testing.assert_allclose(to_features(moved, "Right"), to_features(hand, "Right"), atol=1e-4)


def test_invariant_to_scale(hand: np.ndarray) -> None:
    np.testing.assert_allclose(
        to_features(hand * 2.5, "Right"), to_features(hand, "Right"), atol=1e-4
    )


@pytest.mark.parametrize("degrees", [30, 90, 180, -135])
def test_invariant_to_in_plane_rotation(hand: np.ndarray, degrees: float) -> None:
    np.testing.assert_allclose(
        to_features(rotate_in_plane(hand, degrees), "Right"),
        to_features(hand, "Right"),
        atol=1e-3,
    )


def test_left_hand_matches_mirrored_right_hand(hand: np.ndarray) -> None:
    mirrored = hand.copy()
    mirrored[:, 0] = 640 - mirrored[:, 0]
    np.testing.assert_allclose(to_features(mirrored, "Left"), to_features(hand, "Right"), atol=1e-4)


def test_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        to_features(np.zeros((20, 3)), "Right")
