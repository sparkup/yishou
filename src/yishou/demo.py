"""Live webcam demo: show the number the hand is making."""

from collections import deque
from pathlib import Path

import cv2
import numpy as np

from yishou.features import to_features
from yishou.gestures import GESTURES, NONE_LABEL
from yishou.model import GestureMLP
from yishou.overlay import draw_hand, draw_number, draw_text, open_window
from yishou.tracking import HandTracker, frames

WINDOW = "yishou"


class Smoother:
    """Average the class probabilities over the last few frames, so the number does not flicker.

    Averaging probabilities rather than voting on labels keeps the confidence bar
    in step with the number on screen, and a frame the model is unsure about
    weighs less than one it is sure about.
    """

    def __init__(self, num_classes: int, window: int = 7) -> None:
        self._no_hand = np.zeros(num_classes, dtype=np.float32)
        self._no_hand[NONE_LABEL] = 1.0
        self._recent: deque[np.ndarray] = deque(maxlen=window)

    def update(self, probabilities: np.ndarray | None) -> np.ndarray:
        """Add one frame, None when no hand was found, and return the averaged probabilities."""
        self._recent.append(self._no_hand if probabilities is None else probabilities)
        return np.mean(self._recent, axis=0)


def demo(
    model_id: str,
    camera_index: int = 0,
    threshold: float = 0.6,
    record_path: Path | None = None,
) -> None:
    """Run the live demo until q is pressed.

    `model_id` is a local directory or a Hugging Face repo id. Predictions whose
    averaged confidence is below `threshold` are shown as no number. `record_path` saves the
    preview, overlay included, as an mp4.
    """
    model = GestureMLP.from_pretrained(model_id)
    smoother = Smoother(model.num_classes)
    video: cv2.VideoWriter | None = None

    open_window(WINDOW)
    try:
        with HandTracker() as tracker:
            for frame in frames(camera_index):
                hand = tracker.detect(frame)
                probabilities = None
                if hand is not None:
                    draw_hand(frame, hand)
                    probabilities = model.predict_proba(to_features(hand.points, hand.handedness))

                averaged = smoother.update(probabilities)
                label = int(averaged.argmax())
                confidence = float(averaged[label])
                if label != NONE_LABEL and confidence >= threshold:
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
