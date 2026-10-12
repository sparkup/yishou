"""Read and write recorded samples as CSV.

Samples store the raw landmarks, not the features, so the feature engineering in
`features.py` can change without recording everything again.
"""

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np

from yishou.features import NUM_LANDMARKS

HEADER = ["label", "burst", "handedness"] + [
    f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in "xyz"
]


@dataclass(frozen=True, slots=True)
class Sample:
    label: int
    burst: int  # id of the continuous recording this frame belongs to
    handedness: str
    points: np.ndarray  # (21, 3) landmarks in pixels


class SampleWriter:
    """Append samples to a CSV file. Use as a context manager."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not path.exists() or path.stat().st_size == 0
        self._file = path.open("a", newline="")
        self._writer = csv.writer(self._file)
        if is_new:
            self._writer.writerow(HEADER)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._file.close()

    def write(self, sample: Sample) -> None:
        coords = (f"{value:.2f}" for value in sample.points.reshape(-1))
        self._writer.writerow([sample.label, sample.burst, sample.handedness, *coords])


def load_samples(path: Path) -> list[Sample]:
    """Read samples back, skipping rows of the wrong length.

    A session that is killed mid-write leaves a half row, which the next session
    then appends to, so a damaged row is dropped with a warning rather than
    stopping training.
    """
    if not path.exists():
        return []
    samples: list[Sample] = []
    skipped: list[int] = []
    with path.open(newline="") as file:
        reader = csv.reader(file)
        next(reader, None)  # header
        for line, row in enumerate(reader, start=2):
            if len(row) != len(HEADER):
                skipped.append(line)
                continue
            samples.append(
                Sample(
                    label=int(row[0]),
                    burst=int(row[1]),
                    handedness=row[2],
                    points=np.array(row[3:], dtype=np.float32).reshape(NUM_LANDMARKS, 3),
                )
            )
    if skipped:
        print(f"Warning: skipped {len(skipped)} damaged row(s) in {path}, lines {skipped}")
    return samples


def split_by_burst(
    samples: Sequence[Sample], val_fraction: float = 0.2
) -> tuple[list[Sample], list[Sample]]:
    """Split into train and validation sets, holding out whole bursts per gesture.

    Consecutive webcam frames are nearly identical, so a random split would put
    near-copies of the training frames in the validation set and flatter the
    score. Holding out entire bursts keeps validation honest. A gesture recorded
    in a single burst falls back to holding out the end of that burst.
    """
    by_label: dict[int, dict[int, list[Sample]]] = {}
    for sample in samples:
        by_label.setdefault(sample.label, {}).setdefault(sample.burst, []).append(sample)

    train: list[Sample] = []
    val: list[Sample] = []
    for bursts in by_label.values():
        ordered = [bursts[burst_id] for burst_id in sorted(bursts)]
        if len(ordered) >= 2:
            n_val = max(1, round(len(ordered) * val_fraction))
            for burst in ordered[:-n_val]:
                train.extend(burst)
            for burst in ordered[-n_val:]:
                val.extend(burst)
        else:
            (burst,) = ordered
            cut = int(len(burst) * (1 - val_fraction))
            train.extend(burst[:cut])
            val.extend(burst[cut:])
    return train, val
