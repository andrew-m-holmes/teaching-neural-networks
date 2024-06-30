import torch
import torch.nn as nn
import numpy as np
from tnn.resnet import ResNet
from typing import List


def flatten_params(model: nn.Module) -> List[torch.Tensor]:
    with torch.no_grad():
        return [p.flatten() for p in model.parameters() if p.requires_grad]


def main():
    resnet = ResNet()
    params = flatten_params(resnet)
    print(len(params))


if __name__ == "__main__":
    main()
