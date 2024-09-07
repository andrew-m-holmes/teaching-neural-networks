import torch
import torch.nn as nn
import torch.nn.functional as f
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


class MLP(nn.Module):

    def __init__(self):
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

    def forward(self, x):
        x = self.norm_1(self.linear_1(x))
        x = self.drop_1(f.relu(x))

        x = self.norm_2(self.linear_2(x))
        x = self.drop_2(f.relu(x))

        x = self.norm_3(self.linear_3(x))
        x = self.linear_4(f.relu(x))
        return {"logits": x}
