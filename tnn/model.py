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
        return torch.cat(
            [
                p.cpu().detach().clone().flatten()
                for p in self.parameters()
                if p.requires_grad
            ],
            dim=0,
        ).numpy()

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


class BertForClassification(Model):

    def __init__(
        self,
        classes: int,
        hidden_size: int,
        name: str = "google-bert/bert-base-uncased",
    ) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained(name)
        self.linear = nn.Linear(hidden_size, classes)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, **kwargs) -> Dict[str, Any]:
        outputs = self.bert(**kwargs)
        cls_hidden_state = self.norm(outputs.pooler_output)
        logits = self.linear(cls_hidden_state)
        return {"logits": logits, "outputs": outputs}
