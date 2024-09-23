import torch
import torch.nn as nn
import numpy as np

from typing import Any, Dict


class Model(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError

    def get_flat_weights(self) -> np.ndarray:
        device = next(self.parameters()).device
        self.cpu()
        weights = torch.cat(
            [
                p.detach().clone().flatten()
                for p in self.parameters()
                if p.requires_grad
            ],
            dim=0,
        ).numpy()
        self.to(device)
        return weights

    def load_flat_weights(self, weights: np.ndarray) -> None:
        i = 0
        for parameter in filter(lambda p: p.requires_grad == True, self.parameters()):
            shape = parameter.shape
            j = i + parameter.numel()
            new_parameter = torch.from_numpy(weights[i:j].reshape(shape)).float()
            parameter.data = new_parameter
            i = j
