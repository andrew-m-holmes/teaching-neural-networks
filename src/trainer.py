import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optimizers


from .interface import auto_model
from dataclasses import dataclass
from typing import Tuple, List, Optional, Callable, Dict, Any, Iterable


@dataclass
class Config:

    optim: Dict[str, Any]
    loss: Dict[str, Any]


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        optim: str,
        config: Config,
        dataset: Iterable,
        batch_size: int = 128,
        epochs: int = 100,
        collate_fn: Optional[Callable] = None,
        device: Optional[str] = None,
        path: Optional[str] = None,
        print_every: Optional[int] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.collate_fn = collate_fn
        self.device = device
        self.path = path
        self.write = path is not None
        self.print_every = print_every

        self.model = auto_model(model)
        self.optim = self._create_optim(optim, config)
        self.loss = self._create_loss(config)

    def _create_optim(self, optim: str, config: Config):
        optimizer_map = {
            "adam": optimizers.Adam,
            "sgd": optimizers.SGD,
            "rmsprop": optimizers.RMSprop,
        }
        return optimizer_map[optim](**config.optim)

    def _create_loss(self, config: Config):
        return nn.CrossEntropyLoss(**config.loss)

    def _get_batch(self, start: int) -> Tuple:
        pass

    def _write_file(
        self, metrics: Dict[str, List[float]], trajectory: List[np.ndarray]
    ) -> None:
        pass
