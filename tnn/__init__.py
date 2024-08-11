import h5py

from .trainer import Trainer
from .landscape import Landscape
from .model import Model
from .plot import *
from typing import Union


def _get_group(
    name: str, module: Union[h5py.File, h5py.Group], clear: bool = False
) -> h5py.Group:
    group = module.get(name)
    if isinstance(group, h5py.Group):
        if clear:
            group.clear()
        return group
    return module.create_group(name)
