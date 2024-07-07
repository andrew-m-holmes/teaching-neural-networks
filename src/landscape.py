import h5py
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from datasets import load_dataset
from typing import Callable, List, Tuple, Optional


class Landscape:

    def __init__(
        self,
        model: nn.Module,
        trajectory: np.ndarray,
        method: str = "filternorm",
    ) -> None:
        self.model = model.cpu()
        self.trajectory = [a for a in trajectory]
        self.method = method
        self.weights = [
            w.flatten().numpy() for w in model.parameters() if w.requires_grad
        ]

    def pca(self):
        raise NotImplementedError

    def randomvecs(self):
        dirx, diry = [], []
        for weight in self.weights:
            dirx.append(weight)

    def custom(self):
        raise NotImplementedError

    def filternorm(self, weight: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def fromfiles(
        model: nn.Module,
        filepath: str,
        modelpath: Optional[str] = None,
        method: str = "filter",
    ) -> "Landscape":
        if modelpath is not None:
            statedict = torch.load(modelpath, map_location="cpu")
            model.load_state_dict(statedict)
        with h5py.File(filepath, mode="r") as file:
            trajectory = file["metrics"]["trajectory"]
            return Landscape(model, trajectory, method)
