"""Score a trained model on recordings it has never seen, such as someone else's hand.

Validation during training only holds out bursts of the same person, so it says
little about other hands, cameras or rooms. Recording a few bursts from someone
new and scoring the model on them gives a more honest number.
"""

from pathlib import Path

import torch

from yishou.dataset import load_samples
from yishou.model import GestureMLP
from yishou.train import confusion_matrix, per_gesture_accuracy, print_confusion, to_tensors


def evaluate(model_id: str, data_path: Path, revision: str | None = None) -> dict[str, object]:
    """Print the accuracy and confusion matrix of `model_id` on every sample in `data_path`.

    `model_id` is a local directory or a Hugging Face repo id, and `revision` a branch
    of that repo. Returns the metrics.
    """
    samples = load_samples(data_path)
    if not samples:
        raise SystemExit(f"no samples in {data_path}; record some with `yishou collect --data`")

    model = GestureMLP.from_pretrained(model_id, revision=revision)
    model.eval()
    features, labels = to_tensors(samples)
    with torch.inference_mode():
        predicted = model(features).argmax(dim=-1)

    matrix = confusion_matrix(predicted, labels)
    print_confusion(matrix)
    accuracy = (predicted == labels).float().mean().item()
    per_gesture = per_gesture_accuracy(matrix)

    print(f"\nAccuracy {accuracy:.3f} on {len(samples)} frames from {data_path}")
    weakest = sorted(per_gesture.items(), key=lambda item: item[1])[:3]
    print("Weakest gestures: " + ", ".join(f"{name} {score:.0%}" for name, score in weakest))
    return {
        "accuracy": round(accuracy, 4),
        "accuracy_per_gesture": per_gesture,
        "samples": len(samples),
    }
