"""The model card written next to the weights before publishing."""

import json
from pathlib import Path

from yishou.gestures import GESTURES
from yishou.hub import model_card


def test_card_lists_every_gesture_and_the_accuracy(tmp_path: Path) -> None:
    metrics = {
        "val_accuracy": 0.9567,
        "val_accuracy_per_gesture": {"10": 0.8627},
        "val_samples": 100,
        "train_samples": 400,
    }
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))

    card = model_card(tmp_path, "someone/yishou")
    assert "95.7%" in card
    assert "| 10 | Closed fist | 86.3% |" in card
    assert "| 1 | Index finger up | n/a |" in card
    assert "yishou demo --model someone/yishou" in card
    for gesture in GESTURES:
        assert f"| {gesture.name} | {gesture.description} |" in card


def test_card_without_metrics_says_so(tmp_path: Path) -> None:
    assert "No metrics were recorded" in model_card(tmp_path, "someone/yishou")
