"""Train the gesture classifier on recorded samples."""

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from yishou.dataset import Sample, load_samples, split_by_burst
from yishou.features import to_features
from yishou.gestures import GESTURES, NUM_CLASSES
from yishou.model import GestureMLP


def to_tensors(samples: Sequence[Sample]) -> tuple[torch.Tensor, torch.Tensor]:
    features = np.stack([to_features(s.points, s.handedness) for s in samples])
    labels = np.array([s.label for s in samples], dtype=np.int64)
    return torch.from_numpy(features), torch.from_numpy(labels)


def confusion_matrix(predicted: torch.Tensor, actual: torch.Tensor) -> np.ndarray:
    """Rows are the true gesture, columns the predicted one."""
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for truth, guess in zip(actual.tolist(), predicted.tolist(), strict=True):
        matrix[truth, guess] += 1
    return matrix


def print_confusion(matrix: np.ndarray) -> None:
    names = [gesture.name for gesture in GESTURES]
    print("\nConfusion matrix (rows = truth, columns = prediction)")
    print("      " + "".join(f"{name:>5}" for name in names))
    for name, row in zip(names, matrix, strict=True):
        print(f"{name:>5} " + "".join(f"{count:>5}" for count in row))


def train(
    data_path: Path,
    out_dir: Path,
    epochs: int = 80,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    noise: float = 0.02,
    seed: int = 0,
) -> dict[str, object]:
    """Train on the samples in `data_path`, save the best model to `out_dir`.

    Returns the metrics that are also written to `out_dir/metrics.json`.
    """
    samples = load_samples(data_path)
    if not samples:
        raise SystemExit(f"no samples in {data_path}; run `yishou collect` first")

    recorded = {sample.label for sample in samples}
    missing = [gesture.name for gesture in GESTURES if gesture.label not in recorded]
    if missing:
        print(f"Warning: no samples recorded for: {', '.join(missing)}")

    train_samples, val_samples = split_by_burst(samples)
    if not train_samples or not val_samples:
        raise SystemExit("not enough samples to split into train and validation sets")
    x_train, y_train = to_tensors(train_samples)
    x_val, y_val = to_tensors(val_samples)
    print(f"{len(train_samples)} training samples, {len(val_samples)} validation samples")

    torch.manual_seed(seed)
    model = GestureMLP()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    best_accuracy = -1.0
    best_state: dict[str, torch.Tensor] = {}
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x_train))
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            # Jitter the landmarks a little so the model does not memorise exact poses.
            inputs = x_train[batch] + noise * torch.randn_like(x_train[batch])
            loss = loss_fn(model(inputs), y_train[batch])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * len(batch)

        model.eval()
        with torch.inference_mode():
            accuracy = (model(x_val).argmax(dim=-1) == y_val).float().mean().item()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {name: value.clone() for name, value in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == epochs:
            print(
                f"epoch {epoch:>3}  loss {total_loss / len(order):.4f}  val accuracy {accuracy:.3f}"
            )

    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        predicted = model(x_val).argmax(dim=-1)
    matrix = confusion_matrix(predicted, y_val)
    print_confusion(matrix)

    per_gesture = {
        gesture.name: round(matrix[gesture.label, gesture.label] / matrix[gesture.label].sum(), 4)
        for gesture in GESTURES
        if matrix[gesture.label].sum() > 0
    }
    metrics: dict[str, object] = {
        "val_accuracy": round(best_accuracy, 4),
        "val_accuracy_per_gesture": per_gesture,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "epochs": epochs,
    }

    model.save_pretrained(out_dir)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nBest validation accuracy {best_accuracy:.3f}; model saved to {out_dir}")
    return metrics
