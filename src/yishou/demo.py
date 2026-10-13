"""Live webcam demo: show the number the hand is making."""

from collections import Counter, deque
from pathlib import Path

import cv2

from yishou.features import to_features
from yishou.gestures import GESTURES, NONE_LABEL
from yishou.model import GestureMLP
from yishou.overlay import draw_hand, draw_number, draw_text, open_window
from yishou.tracking import HandTracker, frames

WINDOW = "yishou"


class Smoother:
    """Majority vote over the last few frames, so the number does not flicker."""

    def __init__(self, window: int = 7) -> None:
        self._recent: deque[int] = deque(maxlen=window)

    def update(self, label: int) -> int:
        self._recent.append(label)
        return Counter(self._recent).most_common(1)[0][0]


def demo(
    model_id: str,
    camera_index: int = 0,
    threshold: float = 0.6,
    record_path: Path | None = None,
) -> None:
    """Run the live demo until q is pressed.

    `model_id` is a local directory or a Hugging Face repo id. Predictions below
    `threshold` confidence are shown as no number. `record_path` saves the
    preview, overlay included, as an mp4.
    """
    model = GestureMLP.from_pretrained(model_id)
    smoother = Smoother()
    video: cv2.VideoWriter | None = None

    open_window(WINDOW)
    try:
        with HandTracker() as tracker:
            for frame in frames(camera_index):
                hand = tracker.detect(frame)
                label, confidence = NONE_LABEL, 0.0
                if hand is not None:
                    draw_hand(frame, hand)
                    probabilities = model.predict_proba(to_features(hand.points, hand.handedness))
                    best = int(probabilities.argmax())
                    if probabilities[best] >= threshold:
                        label, confidence = best, float(probabilities[best])

                label = smoother.update(label)
                if label != NONE_LABEL:
                    draw_number(frame, GESTURES[label].name, confidence)
                draw_text(frame, "q = quit", (20, frame.shape[0] - 20), scale=0.55)

                if record_path is not None:
                    if video is None:
                        height, width = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video = cv2.VideoWriter(str(record_path), fourcc, 30.0, (width, height))
                    video.write(frame)

                cv2.imshow(WINDOW, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if video is not None:
            video.release()
            print(f"Recording saved to {record_path}")
        cv2.destroyAllWindows()
