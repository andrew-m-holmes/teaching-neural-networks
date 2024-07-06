import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt

from src.utils import test
from datasets import load_dataset
from matplotlib.animation import FuncAnimation

from typing import List, Tuple, Optional
from torch import Tensor
from torch.nn.modules.loss import _Loss as Loss
from torch.utils.data import DataLoader


def genrandvecs(model: nn.Module) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
    thetas = [p.detach().clone().flatten() for p in model.parameters()]
    deltas, etas = [], []

    for theta in thetas:
        vecx, vecy = filternorm(theta), filternorm(theta)
        deltas.append(vecx)
        etas.append(vecy)

    assert len(thetas) == len(deltas) == len(etas)
    return thetas, deltas, etas


def filternorm(layerparams: Tensor) -> Tensor:
    vec = torch.randn_like(layerparams)
    vec *= layerparams.norm() / vec.norm()
    return vec


def genminimizer(
    thetas: List[Tensor],
    deltas: List[Tensor],
    etas: List[Tensor],
    alpha: float,
    beta: float,
) -> List[Tensor]:
    thetahats = []

    for theta, delta, eta in zip(thetas, deltas, etas):
        assert theta.numel() == delta.numel() == eta.numel()
        newflatparams = theta.detach().flatten() + alpha * delta + beta * eta
        thetahats.append(newflatparams)

    return thetahats


def evaluate(
    model: nn.Module,
    origin: List[Tensor],
    thetahats: List[Tensor],
    lossfn: Loss,
    testloader: DataLoader,
    device: Optional[str] = None,
):
    setparams(model, thetahats)
    loss = test(model, lossfn, testloader, device=device)
    setparams(model, origin)
    return loss


def setparams(model: nn.Module, thetas: List[Tensor]) -> None:
    assert len(list(model.parameters())) == len(thetas)
    for param, theta in zip(model.parameters(), thetas):
        assert param.numel() == theta.numel()
        param.data = theta.reshape(param.size())
