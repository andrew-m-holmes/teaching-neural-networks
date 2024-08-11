import torch
import torch.nn as nn
import numpy as np

from typing import Any, Dict


class Model(nn.Module):

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: Any) -> Dict[str, Any]:
        return (
            self.model(inputs)
            if isinstance(inputs, torch.Tensor)
            else self.model(**inputs)
        )

    def get_flat_weights(self) -> np.ndarray:
        return torch.cat(
            [
                p.cpu().detach().clone().flatten()
                for p in self.model.parameters()
                if p.requires_grad
            ],
            dim=0,
        ).numpy()

    def load_flat_weights(self, weights: np.ndarray) -> None:
        i = 0
        for parameter in filter(
            lambda p: p.requires_grad == True, self.model.parameters()
        ):
            shape = parameter.shape
            j = i + parameter.numel()
            new_parameter = torch.from_numpy(weights[i:j].reshape(shape)).float()
            parameter.data = new_parameter
            i = j
