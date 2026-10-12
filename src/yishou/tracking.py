"""Webcam frames and MediaPipe hand tracking."""

import time
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path.home() / ".cache" / "yishou" / "hand_landmarker.task"


@dataclass(frozen=True, slots=True)
class Hand:
    points: np.ndarray  # (21, 3) landmarks in pixels; z is relative depth on the same scale as x
    handedness: str  # "Left" or "Right"


def frames(
    camera_index: int = 0,
    mirror: bool = True,
    size: tuple[int, int] = (1280, 720),
    patience_s: float = 5.0,
) -> Iterator[np.ndarray]:
    """Yield BGR frames from the webcam until the caller stops asking.

    Frames are mirrored by default so the preview behaves like a mirror, which is
    also what MediaPipe assumes when it labels a hand as left or right. `size` is
    a request: the camera may pick the nearest resolution it supports.

    Cameras can fail a few reads while starting up, and an iPhone used as a
    Continuity Camera drops frames now and then, so failed reads are retried for
    up to `patience_s` seconds before giving up with an error.
    """
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(
            f"could not open camera {camera_index}; on macOS, check that your terminal "
            "has camera access in System Settings > Privacy & Security > Camera"
        )
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    last_frame_at = time.monotonic()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                if time.monotonic() - last_frame_at > patience_s:
                    raise RuntimeError(
                        f"camera {camera_index} stopped sending frames for {patience_s:.0f} s; "
                        "if it is an iPhone, check it is still connected as a Continuity Camera"
                    )
                time.sleep(0.05)
                continue
            last_frame_at = time.monotonic()
            yield cv2.flip(frame, 1) if mirror else frame
    finally:
        capture.release()


def ensure_model(path: Path = MODEL_PATH) -> Path:
    """Download the MediaPipe hand landmark model on first use."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading hand landmark model to {path}")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


class HandTracker:
    """Find one hand per frame. Use as a context manager."""

    def __init__(self, min_confidence: float = 0.5) -> None:
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(ensure_model())),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._landmarker.close()

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int | None = None) -> Hand | None:
        """Find the hand in one frame.

        `timestamp_ms` is the frame's time in a recorded video; leave it out for a
        live camera and the clock is used instead.
        """
        height, width = frame_bgr.shape[:2]
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
        )

        # Video mode requires strictly increasing timestamps.
        if timestamp_ms is None:
            timestamp_ms = time.monotonic_ns() // 1_000_000
        timestamp_ms = max(self._last_timestamp_ms + 1, timestamp_ms)
        self._last_timestamp_ms = timestamp_ms

        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.hand_landmarks:
            return None

        # Landmarks come back normalised to [0, 1] per axis; convert to pixels so the
        # frame's aspect ratio does not distort the hand.
        points = np.array(
            [(lm.x * width, lm.y * height, lm.z * width) for lm in result.hand_landmarks[0]],
            dtype=np.float32,
        )
        return Hand(points=points, handedness=result.handedness[0][0].category_name)
