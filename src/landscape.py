import h5py
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from typing import Callable, List, Tuple, Optional


class Landscape:

    def __init__(
        self,
        model: nn.Module,
        lossfn: Callable[..., torch.Tensor],
        dataloader: data.DataLoader,
        trajectory: np.ndarray,
        filepath: Optional[str] = None,
    ) -> None:
        self.model = model.cpu()
        self.lossfn = lossfn
        self.dataloader = dataloader
        self.trajectory = trajectory
        self.filepath = filepath if filepath is not None else "./landscape.h5"
        self.parameters = [
            p.cpu().flatten().clone().detach()
            for p in model.parameters()
            if p.requires_grad
        ]

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

    def create(
        self,
        resolution: int = 10,
        bounds: Tuple[float, float] = (-10.0, 10.0),
        device: Optional[str] = None,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        vecx, vecy = self.filternorm()
        linspacex, linspacey = torch.linspace(
            *bounds, steps=resolution
        ), torch.linspace(*bounds, steps=resolution)
        A, B = torch.meshgrid(linspacex, linspacey, indexing="ij")

        Z = torch.zeros(resolution, resolution)
        for i in range(resolution):
            for j in range(resolution):
                parameters = []

                for p, x, y in zip(self.parameters, vecx, vecy):
                    parameters.append(p + A[i][j] * x + B[i][j] * y)

                loss = self.computeloss(parameters, device=device)
                Z[i, j] = loss
                if verbose:
                    print(f"Iteration: {i * 10 + j + 1}Loss: {loss:.4f}")

        A, B, Z = A.numpy(), B.numpy(), Z.numpy()
        with h5py.File(self.filepath, mode="w") as file:
            axesgroup = file.create_group("axes")
            axesgroup.create_dataset("X", data=Z)
            axesgroup.create_dataset("Y", data=Z)
            axesgroup.create_dataset("Z", data=Z)
        return A, B, Z

    def computeloss(
        self,
        parameters: List[torch.Tensor],
        device: Optional[str] = None,
    ) -> float:
        self.model.eval()
        self.setparameters(parameters)
        self.model.to(device)

        with torch.no_grad():
            testloss = 0
            for inputs, labels in self.dataloader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                    device, non_blocking=True
                )
                outputs = self.model(inputs)
                loss = self.lossfn(outputs, labels)
                testloss += loss.item()

        self.model.cpu()
        self.setparameters(self.parameters)
        testloss /= len(self.dataloader)
        return testloss

    def setparameters(self, parameters: List[torch.Tensor]) -> None:
        modelparams = [p for p in self.model.parameters() if p.requires_grad]
        assert len(parameters) == len(modelparams)

        for currparam, newparam in zip(modelparams, parameters):
            currparam.data = newparam.reshape(currparam.size())

    def filternorm(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        vecx, vecy = [], []

        for parameter in self.parameters:
            pnorm = parameter.norm()
            x = torch.randn_like(parameter)
            vecx.append(x * pnorm / x.norm())
            y = torch.randn_like(parameter)
            vecy.append(y * pnorm / y.norm())

        return vecx, vecy

    def pca(self):
        raise NotImplementedError
