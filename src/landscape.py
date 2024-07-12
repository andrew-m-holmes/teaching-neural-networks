import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from sklearn.decomposition import PCA
from typing import Callable, List, Tuple, Optional

Trajectory = torch.Tensor


class Landscape:

    def __init__(
        self,
        model: nn.Module,
        trajectory: Optional[Trajectory],
    ) -> None:

        self.model = model.cpu()
        self.trajectory = trajectory
        self.parameters = [
            p.cpu().flatten().clone().detach()
            for p in model.parameters()
            if p.requires_grad
        ]

    @staticmethod
    def fromfiles(
        model: nn.Module,
        modelpath: Optional[str] = None,
        trajpath: Optional[str] = None,
    ) -> "Landscape":

        if modelpath is not None:
            statedict = torch.load(modelpath, map_location="cpu")
            model.load_state_dict(statedict)

        trajectory = torch.load(trajpath) if trajpath is not None else None
        return Landscape(model, trajectory)

    def create(
        self,
        lossfn: Callable[..., torch.Tensor],
        dataloader: data.DataLoader,
        resolution: int = 10,
        bounds: Tuple[float, float] = (-10.0, 10.0),
        device: Optional[str] = None,
        printevery: Optional[int] = None,
        filepath: Optional[str] = None,
        mode="filter",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        verbose = bool(printevery is not None and printevery)
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

                loss = self.computeloss(parameters, lossfn, dataloader, device=device)
                Z[i, j] = loss

                iter = i * resolution + j + 1
                if (
                    printevery and iter % printevery == 0 and iter != 1
                ) or iter == printevery:
                    print(f"Iteration: {i * resolution + j + 1}, loss: {loss:.4f}")

        A, B, Z = A.numpy(), B.numpy(), Z.numpy()
        if filepath is None:
            filepath = "./landscape.h5"
        self.writetofiles(A, B, Z, filepath=filepath, verbose=verbose)
        return A, B, Z

    def computeloss(
        self,
        parameters: List[torch.Tensor],
        lossfn: Callable[..., torch.Tensor],
        dataloader: data.DataLoader,
        device: Optional[str] = None,
    ) -> float:

        self.model.eval()
        self.setparameters(parameters)
        self.model.to(device)

        with torch.no_grad():
            testloss = 0
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(
                    device, non_blocking=True
                )
                outputs = self.model(inputs)
                loss = lossfn(outputs, labels)
                testloss += loss.item()

        self.model.cpu()
        self.setparameters(self.parameters)
        testloss /= len(dataloader)
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

    def pca(self) -> Tuple[torch.Tensor, torch.Tensor]:

        assert self.trajectory is not None
        flatparams = torch.cat(self.parameters)
        diffmat = (self.trajectory - flatparams).numpy()

        pca = PCA(n_components=2)
        pca.fit(diffmat)
        pc1 = torch.from_numpy(pca.components_[0])
        pc2 = torch.from_numpy(pca.components_[1])
        return pc1, pc2

    def writetofiles(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Z: np.ndarray,
        filepath: str,
        verbose: bool = True,
    ) -> None:

        if verbose:
            print(f"Writing landscape to: {filepath}")

        with h5py.File(filepath, mode="w") as file:
            axesgroup = file.create_group("axes")
            axesgroup.create_dataset("A", data=A)
            axesgroup.create_dataset("B", data=B)
            axesgroup.create_dataset("Z", data=Z)

        if verbose:
            print("Landscape written")
