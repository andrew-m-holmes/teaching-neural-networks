import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from sklearn.decomposition import PCA
from typing import Callable, Tuple, Optional


Mesh = Tuple[np.ndarray, np.ndarray, np.ndarray]


class Landscape:

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable[..., torch.Tensor],
        trajectory: Optional[torch.Tensor] = None,
        write: bool = True,
        file_path: Optional[str] = None,
    ) -> None:

        self.model = model.cpu()
        self.loss_fn = loss_fn
        self.trajectory = trajectory
        self.write = write
        self.file_path = file_path

    @staticmethod
    def from_files(
        model,
        loss_fn: Callable[..., torch.Tensor],
        param_path: str,
        traj_path: Optional[str] = None,
        write: bool = True,
        file_path: Optional[str] = None,
    ) -> "Landscape":

        state_dict = torch.load(param_path, map_location="cpu")
        model.load_state_dict(state_dict)
        trajectory = (
            torch.load(traj_path, map_location="cpu") if traj_path is not None else None
        )

        return Landscape(model, loss_fn, trajectory, write, file_path)

    def create_landscape(
        self,
        dataloder: data.DataLoader,
        mode: str = "filter",
        custom_dirs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        resolution: int = 25,
        bounds: Tuple[float, float] = (-10.0, 10.0),
        device: Optional[str] = None,
        print_every: Optional[int] = None,
    ) -> Tuple[Mesh, Optional[np.ndarray]]:

        mode = mode.lower().strip()
        dir1, dir2 = torch.empty(0), torch.empty(0)
        trajectory = None

        if mode == "filter":
            dir1, dir2 = self.filter_norm()
        elif mode == "pca":
            dir1, dir2, trajectory = self.pca()
        elif mode == "custom":
            assert custom_dirs is not None
            dir1, dir2 = custom_dirs
        else:
            raise ValueError(f"Invalid mode: {mode}")

        x_coord = torch.linspace(*bounds, steps=resolution)
        y_coord = torch.linspace(*bounds, steps=resolution)
        X, Y = torch.meshgrid(x_coord, y_coord, indexing="ij")
        Z = torch.zeros((resolution, resolution))

        trained_parameters = self.flat_parameters()
        for i in range(resolution):
            for j in range(resolution):

                new_parameters = trained_parameters + X[i][j] * dir1 + Y[i][j] * dir2
                self.assign_parameters(new_parameters)
                loss = self.compute_loss(dataloder, device)
                Z[i][j] = loss
                self.assign_parameters(trained_parameters)

                iters = i * resolution + j + 1
                if print_every and (
                    (iters % print_every == 0 and iters != 1) or iters == print_every
                ):
                    print(f"Iteration: {iters}, loss: {loss:.4f}")

        X, Y, Z = X.numpy(), Y.numpy(), Z.numpy()
        if self.write:
            self.write_file(
                (X, Y, Z),
                trajectory=trajectory,
                file_path=self.file_path,
                verbose=bool(print_every),
            )
        return (X, Y, Z), trajectory

    def compute_loss(
        self,
        dataloader: data.DataLoader,
        device: Optional[str] = None,
    ) -> float:
        self.model.eval()

        with torch.no_grad():

            total_loss = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()

            return total_loss / len(dataloader)

    def assign_parameters(self, new_parameters: torch.Tensor) -> None:

        start = 0
        for param in self.model.parameters():
            if not param.requires_grad:
                continue

            end = start + param.numel()
            param_slice = new_parameters[start:end].reshape(param.size())
            param.data = param_slice
            start = end

    def flat_parameters(self) -> torch.Tensor:
        return torch.cat(
            [
                p.flatten().detach().clone()
                for p in self.model.parameters()
                if p.requires_grad
            ]
        )

    def filter_norm(self) -> Tuple[torch.Tensor, torch.Tensor]:
        dir1, dir2 = [], []

        with torch.no_grad():

            for param in self.model.parameters():
                if not param.requires_grad:
                    continue

                vec1 = torch.randn_like(param).flatten()
                vec1 = vec1 * param.norm() / vec1.norm()
                vec2 = torch.randn_like(param).flatten()
                vec2 = vec2 * param.norm() / vec2.norm()

                dir1.append(vec1)
                dir2.append(vec2)

            return torch.cat(dir1), torch.cat(dir2)

    def pca(self) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        assert self.trajectory is not None
        diff_matrix = self.trajectory - self.flat_parameters()
        pca = PCA(n_components=2)
        trajectory = pca.fit_transform(diff_matrix.numpy())

        return (
            torch.from_numpy(pca.components_[0]),
            torch.from_numpy(pca.components_[1]),
            trajectory.T,
        )

    def write_file(
        self,
        mesh: Mesh,
        trajectory: Optional[np.ndarray] = None,
        file_path: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        if file_path is None:
            file_path = "./landscape.h5"

        if verbose:
            print(f"Writing to f{file_path}")

        X, Y, Z = mesh
        with h5py.File(file_path, mode="w") as file:
            mesh_group = file.create_group("mesh")
            mesh_group.create_dataset("X", data=X)
            mesh_group.create_dataset("Y", data=Y)
            mesh_group.create_dataset("Z", data=Z)

            if trajectory is not None:
                mesh_group.create_dataset("trajectory", data=trajectory)

        if verbose:
            print(f"{file_path} written")


if __name__ == "__main__":

    x = torch.randn(9)
    y = x.reshape(3, 3)
    assert x.norm() == y.norm()
