import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

from typing import Any, Dict
from transformers import BertModel


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


class MLP(Model):

    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(28 * 28, 512)
        self.norm_1 = nn.LayerNorm(512)
        self.drop_1 = nn.Dropout(0.4)
        self.linear_2 = nn.Linear(512, 512)
        self.norm_2 = nn.LayerNorm(512)
        self.drop_2 = nn.Dropout(0.2)
        self.linear_3 = nn.Linear(512, 512)
        self.norm_3 = nn.LayerNorm(512)
        self.linear_4 = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.norm_1(self.linear_1(x))
        x = self.drop_1(f.relu(x))

        x = self.norm_2(self.linear_2(x))
        x = self.drop_2(f.relu(x))

        x = self.norm_3(self.linear_3(x))
        x = self.linear_4(f.relu(x))
        return {"logits": x}
