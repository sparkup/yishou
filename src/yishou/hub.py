"""Publish a trained model to the Hugging Face Hub."""

import json
from pathlib import Path

from huggingface_hub import HfApi

from yishou.gestures import GESTURES

GITHUB_URL = "https://github.com/sparkup/yishou"

CARD_TEMPLATE = """\
---
license: mit
library_name: pytorch
tags:
  - gesture-recognition
  - hand-landmarks
  - mediapipe
  - pytorch_model_hub_mixin
---

# yishou

A small MLP that recognises the Chinese one-hand number gestures, 1 to 10, from
the 21 hand landmarks produced by MediaPipe.

- **Input:** 63 features, the (x, y, z) of each landmark after normalising for
  position, scale, in-plane rotation and left/right hand
- **Output:** {num_classes} classes, the numbers 1 to 10 plus "none"
- **Code:** {github_url}

## Gestures

| Class | Hand shape |
| --- | --- |
{gesture_rows}

## Evaluation

{evaluation}

Validation frames come from recordings that were held out whole, so they are not
near-duplicates of training frames. All data was recorded by one person on one
webcam, so expect lower accuracy on other hands, cameras and lighting.

## Usage

```bash
pip install git+{github_url}
yishou demo --model {repo_id}
```
"""


def model_card(model_dir: Path, repo_id: str) -> str:
    """The README shown on the model's Hugging Face page, with the gestures and accuracy."""
    metrics_path = model_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        evaluation = (
            f"Validation accuracy: **{metrics['val_accuracy']:.1%}** "
            f"({metrics['val_samples']} validation frames, "
            f"{metrics['train_samples']} training frames)."
        )
    else:
        evaluation = "No metrics were recorded for this model."

    return CARD_TEMPLATE.format(
        num_classes=len(GESTURES),
        github_url=GITHUB_URL,
        gesture_rows="\n".join(f"| {g.name} | {g.description} |" for g in GESTURES),
        evaluation=evaluation,
        repo_id=repo_id,
    )


def push(model_dir: Path, repo_id: str, private: bool = False) -> str:
    """Upload `model_dir` to the Hub with a generated model card. Returns the repo URL.

    Requires a Hugging Face token: run `hf auth login` or set `HF_TOKEN`.
    """
    if not (model_dir / "model.safetensors").exists():
        raise SystemExit(f"no trained model in {model_dir}; run `yishou train` first")

    (model_dir / "README.md").write_text(model_card(model_dir, repo_id))

    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=model_dir, repo_id=repo_id)
    return f"https://huggingface.co/{repo_id}"
