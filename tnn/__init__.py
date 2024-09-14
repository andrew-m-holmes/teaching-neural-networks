import h5py
import numpy as np

from .trainer import Trainer
from .landscape import Landscape
from .model import Model, MLP, BertForClassification
from .plot import *
from typing import Union


def get_metrics_from_file(path: str) -> Dict[str, np.ndarray]:
    with h5py.File(path, mode="r") as file:
        metrics_group = file.get("metrics")
        if not isinstance(metrics_group, h5py.Group):
            raise RuntimeError(f"could not find metrics from {path}")
        return {n: np.array(d) for n, d in metrics_group.items()}


def get_trajectory_from_file(path: str) -> np.ndarray:
    with h5py.File(path, mode="r") as file:
        trajectory_group = file.get("trajectory")
        if not isinstance(trajectory_group, h5py.Group):
            raise RuntimeError(f"could not locate trajectory from {path}")

        trajectory = np.vstack([d for d in map(np.array, trajectory_group.values())])
        return trajectory


def get_meshgrid_from_file(path: str) -> np.ndarray:
    with h5py.File(path, mode="r") as file:
        landscape_group = file.get("landscape")
        if not isinstance(landscape_group, h5py.Group):
            raise RuntimeError(f"could not find landscape from {path}")

        meshgrid = landscape_group.get("meshgrid")
        if meshgrid is None:
            raise RuntimeError(
                f"could not locate meshgrid in {path}/landscape/meshgrid"
            )
        return np.array(meshgrid)


def get_landscape_from_file(path: str) -> Dict[str, np.ndarray]:
    with h5py.File(path, mode="r") as file:
        landscape_group = file.get("landscape")
        if not isinstance(landscape_group, h5py.Group):
            raise RuntimeError(f"could not find landscape from {path}")

        landscape = {}
        meshgrid = landscape_group.get("meshgrid")
        if meshgrid is None:
            raise RuntimeError(
                f"could not locate meshgrid in {path}/landscape/meshgrid"
            )
        landscape["meshgrid"] = np.array(meshgrid)

        optim_path = landscape_group.get("optim_path")
        if optim_path is not None:
            landscape["optim_path"] = np.array(optim_path)

        variance = landscape_group.get("variance")
        if variance is not None:
            landscape["variance"] = np.array(variance)

        return landscape


def _get_group(
    name: str, module: Union[h5py.File, h5py.Group], clear: bool = False
) -> h5py.Group:
    group = module.get(name)
    if isinstance(group, h5py.Group):
        if clear:
            group.clear()
        return group
    return module.create_group(name)
