"""Sample storage and the burst-based train/validation split."""

from pathlib import Path

import numpy as np

from tests.helpers import synthetic_samples, write
from yishou.dataset import load_samples, split_by_burst
from yishou.gestures import NUM_CLASSES


def test_dataset_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "data" / "samples.csv"
    samples = synthetic_samples(per_burst=2, bursts=1)
    write(samples[:5], path)
    write(samples[5:], path)  # appending must not repeat the header

    loaded = load_samples(path)
    assert len(loaded) == len(samples)
    assert loaded[0].label == samples[0].label
    assert loaded[0].handedness == "Right"
    np.testing.assert_allclose(loaded[0].points, samples[0].points, atol=0.01)


def test_load_missing_file_is_empty(tmp_path: Path) -> None:
    assert load_samples(tmp_path / "nope.csv") == []


def test_load_skips_damaged_rows(tmp_path: Path) -> None:
    path = tmp_path / "samples.csv"
    write(synthetic_samples(per_burst=1, bursts=1)[:2], path)
    with path.open("a") as file:
        file.write("3,0,Right,1.0,2.0\n")  # a row cut short, as a killed session leaves it
    assert len(load_samples(path)) == 2


def test_split_holds_out_whole_bursts() -> None:
    train_set, val_set = split_by_burst(synthetic_samples(per_burst=10, bursts=5))
    assert {s.burst for s in val_set} == {4}
    assert {s.burst for s in train_set} == {0, 1, 2, 3}


def test_split_single_burst_holds_out_the_tail() -> None:
    train_set, val_set = split_by_burst(synthetic_samples(per_burst=10, bursts=1))
    assert len(train_set) == 8 * NUM_CLASSES
    assert len(val_set) == 2 * NUM_CLASSES
