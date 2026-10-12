"""Drawing helpers for the preview window."""

import cv2
import numpy as np

from yishou.tracking import Hand

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (120, 220, 80)
RED = (60, 60, 230)

# Pairs of landmark indices joined by a bone
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),  # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # index
    (5, 9), (9, 10), (10, 11), (11, 12),  # middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # little
    (0, 17),  # palm edge
)  # fmt: skip


def open_window(name: str, width: int = 960) -> None:
    """A resizable preview window that starts at `width` pixels wide, 16:9."""
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, width, width * 9 // 16)


def draw_hand(frame: np.ndarray, hand: Hand) -> None:
    pixels = hand.points[:, :2].astype(int)
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, tuple(pixels[start]), tuple(pixels[end]), WHITE, 2, cv2.LINE_AA)
    for point in pixels:
        cv2.circle(frame, tuple(point), 4, GREEN, -1, cv2.LINE_AA)


def draw_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.7,
    color: tuple[int, int, int] = WHITE,
) -> None:
    """Text with a dark outline so it stays readable on any background."""
    thickness = max(1, round(scale * 2))
    cv2.putText(frame, text, origin, FONT, scale, BLACK, thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, origin, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_number(frame: np.ndarray, text: str, confidence: float) -> None:
    """The predicted number, large, in the top-left corner, with a confidence bar."""
    draw_text(frame, text, (30, 170), scale=6.0)
    bar_width = 220
    cv2.rectangle(frame, (30, 200), (30 + bar_width, 214), BLACK, -1)
    cv2.rectangle(frame, (30, 200), (30 + int(bar_width * confidence), 214), GREEN, -1)
