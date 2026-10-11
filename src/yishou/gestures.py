"""The gesture vocabulary: one class per number, plus a "none" class.

The shapes follow the most common mainland Chinese convention. Several numbers
have regional variants (10 is also shown as crossed index and middle fingers,
for instance), so record whichever variant you want the model to learn.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Gesture:
    label: int
    name: str
    description: str


NONE_LABEL = 0

GESTURES: tuple[Gesture, ...] = (
    Gesture(0, "none", "Any hand shape that is not a number"),
    Gesture(1, "1", "Index finger up"),
    Gesture(2, "2", "Index and middle fingers up"),
    Gesture(3, "3", "Index, middle and ring fingers up"),
    Gesture(4, "4", "Four fingers up, thumb folded"),
    Gesture(5, "5", "Open hand"),
    Gesture(6, "6", "Thumb and little finger out, others folded"),
    Gesture(7, "7", "Thumb, index and middle fingertips pinched together"),
    Gesture(8, "8", "Thumb and index finger out in an L shape"),
    Gesture(9, "9", "Index finger hooked, others folded"),
    Gesture(10, "10", "Closed fist"),
)

NUM_CLASSES = len(GESTURES)

# Keyboard keys used while collecting: 1-9 for themselves, 0 for ten, n for none.
KEY_TO_LABEL: dict[str, int] = {str(n): n for n in range(1, 10)} | {
    "0": 10,
    "n": NONE_LABEL,
}
