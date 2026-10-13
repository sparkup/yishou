"""A full train, save and reload cycle on synthetic hands."""

from pathlib import Path

import pytest

from tests.helpers import synthetic_samples, write
from yishou.features import to_features
from yishou.gestures import NUM_CLASSES
from yishou.model import GestureMLP
from yishou.train import train


def test_train_save_and_reload(tmp_path: Path) -> None:
    data, out = tmp_path / "samples.csv", tmp_path / "model"
    samples = synthetic_samples()
    write(samples, data)

    metrics = train(data, out, epochs=30)
    assert metrics["val_accuracy"] > 0.9
    assert (out / "metrics.json").exists()

    model = GestureMLP.from_pretrained(out)
    sample = samples[-1]
    probabilities = model.predict_proba(to_features(sample.points, sample.handedness))
    assert probabilities.shape == (NUM_CLASSES,)
    assert int(probabilities.argmax()) == sample.label


def test_train_without_samples_stops_with_a_message(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="run `yishou collect` first"):
        train(tmp_path / "missing.csv", tmp_path / "model")
