import torch
import torch.nn as nn
import numpy as np

from abc import ABC, abstractmethod
from typing import Any


class ModelInterface(ABC):

    @abstractmethod
    def forward(self, inputs: Any) -> Any:
        pass

    @abstractmethod
    def get_weights(self) -> np.ndarray:
        pass

    @abstractmethod
    def load_weights(self, weights: np.ndarray) -> None:
        pass


class TorchModel(ModelInterface):

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def forward(self, inputs: Any) -> Any:
        return self.model(inputs)

    def get_weights(self) -> np.ndarray:
        return np.array(
            torch.cat(
                [p.cpu().detach() for p in self.model.parameters() if p.requires_grad]
            )
        )

    def load_weights(self, weights: np.ndarray) -> None:
        start = 0

        for p in filter(lambda p: p.requires_grad, self.model.parameters()):
            end = start + p.numel()
            size = p.size()
            p.data = torch.from_numpy(weights[start:end].reshape(size))
            start = end


class HuggingFaceModel(ModelInterface):

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def forward(self, inputs: Any) -> Any:
        return self.model(**inputs)

    def get_weights(self) -> np.ndarray:
        return np.array(
            torch.cat(
                [p.cpu().detach() for p in self.model.parameters() if p.requires_grad]
            )
        )

    def load_weights(self, weights: np.ndarray) -> None:
        start = 0

        for p in filter(lambda p: p.requires_grad, self.model.parameters()):
            end = start + p.numel()
            size = p.size()
            p.data = torch.from_numpy(weights[start:end].reshape(size))
            start = end


def auto_model(model: Any) -> ModelInterface:
    return (
        HuggingFaceModel(model)
        if model.__module__.startswith("transformers")
        else TorchModel(model)
    )
