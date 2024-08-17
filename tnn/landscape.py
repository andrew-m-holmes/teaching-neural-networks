import tnn
import os
import h5py
import torch
import torch.utils.data as data
import numpy as np

from .model import Model
from sklearn.decomposition import PCA
from typing import Callable, Tuple, Optional, Dict, Union, Any


class Landscape:

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

    def create_meshgrid(
        self,
        resolution: int = 25,
        endpoints: Tuple[float, float] = (-10.0, 10.0),
        mode: str = "pca",
    ) -> Dict[
        str,
        Union[
            Tuple[np.ndarray, np.ndarray, np.ndarray],
            Tuple[np.ndarray, np.ndarray],
            Optional[np.ndarray],
        ],
    ]:
        mode = mode.lower().strip()
        if mode not in ("pca", "random"):
            raise ValueError(
                f"Invalid usage of 'mode' parameter, mode can be ('pca', 'random')"
            )

        if self.verbose:
            print(f"meshgrid creation using {mode}\nmodel using {self.device}")
        if mode == "pca":
            data = self._compute_pca_directions()
        else:
            data = self._compute_random_directions()

        x_coordinates = np.linspace(*endpoints, num=resolution)
        y_coordinates = np.linspace(*endpoints, num=resolution)
        X, Y = np.meshgrid(x_coordinates, y_coordinates, indexing="ij")
        Z = np.zeros((resolution, resolution))

        directions = data.get("directions")
        assert directions is not None
        x_direction, y_direction = directions
        trained_weights = self.model.get_flat_weights()

        if self.verbose:
            print(f"meshgrid creation started")
        for i in range(resolution):
            for j in range(resolution):

                perturbed_weights = (
                    trained_weights + X[i][j] * x_direction + Y[i][j] * y_direction
                )

                loss = self._sample_loss_from_perturbation(perturbed_weights)
                Z[i][j] = loss

                if self.verbose:
                    n_iter = i * resolution + j + 1
                    if (n_iter % self.verbose == 0 and n_iter > 1) or (
                        n_iter == self.verbose
                    ):
                        print(f"(iter: {n_iter}): iter loss: {loss:.4f}")

        self.model.load_flat_weights(trained_weights)
        if self.verbose:
            print("meshgrid creation complete")

        meshgrid = (X, Y, Z)

        if self.path is not None:
            dirname = os.path.dirname(__file__)
            os.makedirs(dirname, exist_ok=True)
            self._write_landscape(meshgrid, data, bool(self.verbose))
        return {
            "meshgrid": meshgrid,
            "optim_path": data.get("optim_path"),
            "variance": data.get("variance"),
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
        x_directions, y_directions = [], []
        for weight in self.model.parameters():
            w_norm = weight.detach().flatten().norm().numpy()

            x = np.random.randn(weight.numel())
            x *= w_norm / np.linalg.norm(x)

            y = np.random.randn(weight.numel())
            y *= w_norm / np.linalg.norm(y)

            x_directions.append(x)
            y_directions.append(y)

        return {
            "directions": (
                np.concat(x_directions, axis=0),
                np.concat(y_directions, axis=0),
            )
        }

    def _write_landscape(
        self,
        meshgrid: Tuple[np.ndarray, np.ndarray, np.ndarray],
        data: Dict[str, Any],
        verbose: bool = False,
    ) -> None:
        if self.path is None:
            raise ValueError("'path' is None")

        with h5py.File(self.path, mode="a") as file:
            landscape_group = tnn._get_group("landscape", file, clear=True)
            meshgrid_group = tnn._get_group("meshgrid", landscape_group, clear=True)
            X, Y, Z = meshgrid
            meshgrid_dict = {"X": X, "Y": Y, "Z": Z}

            for name, arr in meshgrid_dict.items():
                meshgrid_group.create_dataset(name=name, data=arr, dtype=np.float32)
                if verbose:
                    print(
                        f"meshgrid array axis {name} saved to {self.path}/landscape/meshgrid/{name}"
                    )

            metadata_group = tnn._get_group("metadata", landscape_group, clear=True)
            for name, data in data.items():
                if name == "directions":
                    x_direction, y_direction = data
                    metadata_group.create_dataset(
                        name="x_direction", data=x_direction, dtype=np.float32
                    )
                    metadata_group.create_dataset(
                        name="y_direction", data=y_direction, dtype=np.float32
                    )

                    if self.verbose:
                        print(
                            f"x_direction saved to {self.path}/landscape/metadata/x_direction"
                        )
                        print(
                            f"y_direction saved to {self.path}/landscape/metadata/y_direction"
                        )

                elif data is not None:
                    metadata_group.create_dataset(
                        name=name, data=data, dtype=np.float32
                    )
                    if self.verbose:
                        print(f"{name} saved to {self.path}/landscape/metadata/{name}")

    @classmethod
    def from_file(
        cls,
        trainer_path: str,
        model: Model,
        loss_fn: Callable[..., torch.Tensor],
        eval_dataloader: data.DataLoader,
        device: Optional[str] = None,
        path: Optional[str] = None,
        verbose: Optional[Union[bool, int]] = None,
    ) -> "Landscape":

        with h5py.File(trainer_path, mode="r") as file:
            trajectory_group = file.get("trajectory")
            if not isinstance(trajectory_group, h5py.Group):
                raise RuntimeError(
                    "Could not find trajectory group to create Landscape"
                )
            weight_states = np.stack([w for w in trajectory_group.values()], axis=0)

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
