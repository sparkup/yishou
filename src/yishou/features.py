"""Turn raw hand landmarks into features that only describe the hand's shape.

A gesture is the same gesture wherever the hand is in the frame, however far it
is from the camera, however it is tilted, and whichever hand makes it. Removing
those four things here means the classifier does not have to learn them from
data.
"""

import numpy as np

NUM_LANDMARKS = 21
FEATURE_DIM = NUM_LANDMARKS * 3

# MediaPipe landmark indices
WRIST = 0
MIDDLE_MCP = 9  # knuckle at the base of the middle finger


def to_features(points: np.ndarray, handedness: str) -> np.ndarray:
    """Normalise (21, 3) pixel-space landmarks into a flat (63,) feature vector.

    The result is invariant to translation, scale, in-plane rotation and
    left/right hand.
    """
    if points.shape != (NUM_LANDMARKS, 3):
        raise ValueError(f"expected landmarks of shape (21, 3), got {points.shape}")

    pts = points.astype(np.float32)  # astype copies, so the caller's array is untouched

    # Translation: the wrist becomes the origin.
    pts -= pts[WRIST]

    # Handedness: mirror left hands so every hand looks like a right hand.
    if handedness == "Left":
        pts[:, 0] *= -1

    # Rotation: turn the hand in the image plane until the palm points straight up
    # (image y grows downwards, so "up" is the direction (0, -1)).
    palm = pts[MIDDLE_MCP, :2]
    theta = -np.pi / 2 - np.arctan2(palm[1], palm[0])
    cos, sin = np.cos(theta), np.sin(theta)
    rotation = np.array([[cos, -sin], [sin, cos]], dtype=np.float32)
    pts[:, :2] = pts[:, :2] @ rotation.T

    # Scale: the palm length becomes 1.
    scale = float(np.linalg.norm(palm))
    if scale > 1e-6:
        pts /= scale

    return pts.reshape(-1)
