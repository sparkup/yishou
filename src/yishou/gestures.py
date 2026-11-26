"""The gesture vocabulary: one class per number, a "none" class and two Easter eggs.

The shapes follow the most common mainland Chinese convention. Several numbers
have regional variants (10 is also shown as crossed index and middle fingers,
for instance), so record whichever variant you want the model to learn.

Two extra gestures are not numbers: the rock sign and the OK sign, the Easter
eggs of the web demo. They have their own classes so the model does not mistake
them for 6 or 3.

A gesture can also name a `pointer` finger. The model then learns one class
whatever the hand's angle, and the demo adds the direction that finger points,
so a single "thumb" class reads as "thumb up" or "thumb down".
"""

from dataclasses import dataclass

from yishou.features import THUMB


@dataclass(frozen=True, slots=True)
class Gesture:
    label: int
    name: str
    description: str
    pointer: tuple[int, int] | None = None  # (base, tip) landmarks, see yishou.features.THUMB


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
    Gesture(10, "10", "Closed fist, or index and middle fingers crossed"),
    Gesture(11, "rock", "Index and little finger up, thumb folded over the others or held out"),
    Gesture(12, "ok", "Thumb and index fingertips touching in a circle, other three up"),
    Gesture(13, "thumb", "Thumb out, other fingers folded, up or down", pointer=THUMB),
)

NUM_CLASSES = len(GESTURES)

# Keyboard keys used while collecting: 1-9 for themselves, 0 for ten, n for none,
# r for rock, o for OK and t for the thumb.
KEY_TO_LABEL: dict[str, int] = {str(n): n for n in range(1, 10)} | {
    "0": 10,
    "n": NONE_LABEL,
    "r": 11,
    "o": 12,
    "t": 13,
}
