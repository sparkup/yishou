"""Record labelled hand landmarks from the webcam."""

import time
from collections import Counter
from pathlib import Path

import cv2

from yishou.dataset import Sample, SampleWriter, load_samples
from yishou.gestures import GESTURES, KEY_TO_LABEL
from yishou.overlay import GREEN, RED, draw_hand, draw_text, open_window
from yishou.tracking import HandTracker, frames

WINDOW = "yishou: collect"


def collect(data_path: Path, camera_index: int = 0) -> None:
    """Open a preview window and append recorded samples to `data_path`.

    Keys: 1-9 select that number, 0 selects ten, n selects "none",
    space starts and stops recording, q quits.
    """
    counts = Counter(sample.label for sample in load_samples(data_path))
    label = 1
    recording = False
    burst = 0

    open_window(WINDOW)
    with HandTracker() as tracker, SampleWriter(data_path) as writer:
        for frame in frames(camera_index):
            hand = tracker.detect(frame)
            if hand is not None:
                draw_hand(frame, hand)
                if recording:
                    writer.write(Sample(label, burst, hand.handedness, hand.points))
                    counts[label] += 1

            gesture = GESTURES[label]
            draw_text(frame, f"{gesture.name}: {gesture.description}", (20, 40))
            draw_text(
                frame,
                f"REC  {counts[label]} samples" if recording else f"{counts[label]} samples",
                (20, 75),
                color=RED if recording else GREEN,
            )
            draw_text(
                frame,
                "1-9, 0 = ten, n = none, r = rock, o = OK | space = record | q = quit",
                (20, frame.shape[0] - 20),
                scale=0.55,
            )
            cv2.imshow(WINDOW, frame)

            key = chr(cv2.waitKey(1) & 0xFF)
            if key == "q":
                break
            if key == " ":
                recording = not recording
                burst = time.time_ns() // 1_000_000  # a fresh id for each recording
            elif key in KEY_TO_LABEL:
                label = KEY_TO_LABEL[key]
                recording = False

    cv2.destroyAllWindows()
    print(f"Samples per gesture in {data_path}:")
    for gesture in GESTURES:
        print(f"  {gesture.name:>4}  {counts[gesture.label]}")
