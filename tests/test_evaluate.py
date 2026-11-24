"""Scoring a trained model on new recordings."""

from pathlib import Path

import pytest

from tests.helpers import synthetic_samples, write
from yishou.evaluate import evaluate
from yishou.gestures import GESTURES
from yishou.train import train


def test_evaluate_scores_new_recordings(tmp_path: Path) -> None:
    data, out = tmp_path / "samples.csv", tmp_path / "model"
    write(synthetic_samples(), data)
    train(data, out, epochs=30)

    # Fresh jitter around the same prototype hands stands in for a new session.
    new = tmp_path / "new.csv"
    write(synthetic_samples(per_burst=5, bursts=1, seed=1), new)

    metrics = evaluate(str(out), new)
    assert metrics["samples"] == 5 * len(GESTURES)
    assert metrics["accuracy"] > 0.9
    assert set(metrics["accuracy_per_gesture"]) == {gesture.name for gesture in GESTURES}


def test_evaluate_without_samples_stops_with_a_message(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="yishou collect --data"):
        evaluate(str(tmp_path / "model"), tmp_path / "missing.csv")
