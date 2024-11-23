import h5py
import numpy as np

from .trainer import Trainer
from .plot import *
from typing import Union


def get_metrics_from_file(path: str) -> Dict[str, np.ndarray]:
    with h5py.File(path, mode="r") as file:
        metrics_group = file.get("metrics")
        if not isinstance(metrics_group, h5py.Group):
            raise RuntimeError(f"could not find metrics from {path}")
        return {n: np.array(d) for n, d in metrics_group.items()}


def _get_group(
    name: str, module: Union[h5py.File, h5py.Group], clear: bool = False
) -> h5py.Group:
    group = module.get(name)
    if isinstance(group, h5py.Group):
        if clear:
            group.clear()
        return group
    return module.create_group(name)
