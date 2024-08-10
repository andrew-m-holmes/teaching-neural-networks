import h5py
import numpy as np
import torch
import torch.utils.data as data

from sklearn.decomposition import PCA
from typing import Callable, Tuple, Optional, Dict, Union
from .model import Model


class MeshGrid:

    def __init__(
        self,
        model: Model,
        trajectory: np.ndarray,
        loss_fn: Callable[..., torch.Tensor],
        eval_dataloader: data.DataLoader,
        device: Optional[str] = None,
        path: Optional[str] = None,
        verbose: Optional[Union[int, bool]] = None,
    ) -> None:
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        if not verbose or verbose < 0:
            verbose = False
        else:
            verbose = int(verbose)

        self.model = model
        self.trajectory = trajectory
        self.loss_fn = loss_fn
        self.eval_dataloader = eval_dataloader
        self.device = device
        self.path = path
        self.verbose = verbose

    def create_mesh_grid(
        self,
        resolution: int = 25,
        endpoints: Tuple[float, float] = (-10.0, 10.0),
        mode: str = "pca",
    ) -> Dict[
        str, Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Optional[np.ndarray]]
    ]:
        mode = mode.lower().strip()
        if mode not in ("pca", "random"):
            raise ValueError(
                f"Invalid usage of mode parameter, mode can be ('pca', 'random')"
            )

        if mode == "pca":
            data = self._compute_pca_directions()
        else:
            data = self._compute_random_directions()

        x_coordinates = np.linspace(*endpoints, num=resolution)
        y_coordinates = np.linspace(*endpoints, num=resolution)
        X, Y = np.meshgrid(x_coordinates, y_coordinates, indexing="ij")
        Z = np.zeros((resolution, resolution))

        x_direction, y_direction = data.get("directions", (None, None))
        trained_weights = self.model.get_flat_weights()

        for i in range(resolution):
            for j in range(resolution):

                perturbed_weights = (
                    trained_weights + X[i][j] * x_direction + Y[i][j] * y_direction
                )

                loss = self._sample_loss_from_perturbation(perturbed_weights)
                Z[i][j] = loss

        self.model.load_flat_weights(trained_weights)

        mesh_grid = (X, Y, Z)
        return {
            "mesh_grid": mesh_grid,
            "variances": data.get("variances"),
        }

    def _sample_loss_from_perturbation(self, peturbed_weights: np.ndarray) -> float:
        with torch.no_grad():
            self.model.eval()
            self.model.load_flat_weights(peturbed_weights)
            self.model.to(self.device)

            n_batches = len(self.eval_dataloader)
            net_loss = 0

            for inputs, labels in self.eval_dataloader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                logits = self.model(inputs).get("logits")
                loss = self.loss_fn(logits, labels)
                net_loss += loss.item()

            self.model.cpu()
            return net_loss / n_batches

    def _compute_pca_directions(
        self,
    ) -> Dict[str, Union[Tuple[np.ndarray, np.ndarray], np.ndarray]]:
        train_variance_matrix = self.trajectory - self.model.get_flat_weights()
        pca = PCA(n_components=2)

        optim_path = pca.fit_transform(train_variance_matrix)
        components = pca.components_
        variance = pca.explained_variance_ratio_
        return {
            "optim_path": optim_path,
            "directions": components,
            "variance": variance,
        }

    def _compute_random_directions(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        directions_x, directions_y = [], []
        for weight in self.model.parameters():
            w_norm = weight.flatten().norm().numpy()

            x = np.random.randn(weight.numel())
            x *= w_norm / np.linalg.norm(x)

            y = np.random.randn(weight.numel())
            y *= w_norm / np.linalg.norm(y)

            directions_x.append(x)
            directions_y.append(y)

        return {
            "directions": (
                np.concat(directions_x, axis=0),
                np.concat(directions_y, axis=0),
            )
        }

    def _write_meshgrid(
        self,
        meshgrid: Tuple[np.ndarray, np.ndarray, np.ndarray],
        variance: Optional[np.ndarray] = None,
    ) -> None:
        if self.path is None:
            raise ValueError("'path' is None")

        with h5py.File(self.path, mode="a") as file:
            meshgrid_group = file.get("meshgrid")
            if not isinstance(meshgrid_group, h5py.Group):
                meshgrid_group = file.create_group("meshgrid")

            X, Y, Z = meshgrid
            meshgrid_group.create_dataset(name="X", data=X, dtype=np.float32)
            meshgrid_group.create_dataset(name="Y", data=Y, dtype=np.float32)
            meshgrid_group.create_dataset(name="Z", data=Z, dtype=np.float32)

            if variance is not None:
                meshgrid_group.create_dataset(
                    name="variance", data=variance, dtype=np.float32
                )

    @classmethod
    def from_files(
        cls,
        trainer_path: str,
        model: Model,
        loss_fn: Callable[..., torch.Tensor],
        eval_dataloader: data.DataLoader,
        device: Optional[str] = None,
        path: Optional[str] = None,
        verbose: Optional[Union[bool, int]] = None,
    ) -> "MeshGrid":

        with h5py.File(trainer_path, mode="r") as file:
            trajectory_group = file.get("trajectory")
            assert isinstance(trajectory_group, h5py.Group)
            weight_states = np.concatenate(
                [w for w in trajectory_group.values()], axis=0
            )

            trajectory = weight_states[:-1]
            final_weights = trajectory[-1]
            model.load_flat_weights(final_weights)
            return cls(
                model,
                trajectory,
                loss_fn,
                eval_dataloader,
                device=device,
                path=path,
                verbose=verbose,
            )
