"""The gesture classifier: a small MLP over normalised hand landmarks."""

from collections.abc import Sequence

import numpy as np
import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from yishou.features import FEATURE_DIM
from yishou.gestures import NUM_CLASSES


class GestureMLP(nn.Module, PyTorchModelHubMixin):
    """Maps a (63,) feature vector to logits over the gesture classes.

    `PyTorchModelHubMixin` adds `save_pretrained`, `from_pretrained` and
    `push_to_hub`, storing the constructor arguments in `config.json` next to
    the weights.
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden: Sequence[int] = (128, 64),
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = input_dim
        for size in hidden:
            layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(dropout)]
            width = size
        layers.append(nn.Linear(width, num_classes))
        self.net = nn.Sequential(*layers)

    @property
    def num_classes(self) -> int:
        """Outputs of this model, which may predate gestures added since it was trained."""
        return self.net[-1].out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.inference_mode()
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Class probabilities for a single (63,) feature vector."""
        self.eval()
        logits = self(torch.from_numpy(features).float().unsqueeze(0))
        return logits.softmax(dim=-1).squeeze(0).numpy()
