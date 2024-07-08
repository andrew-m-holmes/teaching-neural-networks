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
    ) -> None:
        self.model = model.cpu()
        self.trajectory = trajectory

    def filternorm(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        vecs = ([], [])
        with torch.no_grad():

            for param in map(lambda p: p.flatten(), self.model.parameters()):
                wnorm = param.norm()
                vecx, vecy = torch.randn_like(param), torch.randn_like(param)
                vecx *= wnorm / vecx.norm()
                vecy *= wnorm / vecy.norm()
                vecs[0].append(vecx.numpy())
                vecs[1].append(vecy.numpy())
            return vecs

    def pca(self):
        raise NotImplementedError

    @staticmethod
    def fromfiles(
        model: nn.Module,
        filepath: str,
        modelpath: Optional[str] = None,
    ) -> "Landscape":
        if modelpath is not None:
            statedict = torch.load(modelpath, map_location="cpu")
            model.load_state_dict(statedict)
        with h5py.File(filepath, mode="r") as file:
            trajectory = file["metrics"]["trajectory"]
            return Landscape(model, trajectory)
