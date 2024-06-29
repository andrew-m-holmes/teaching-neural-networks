import torch
import torch.nn as nn
from typing import List, Tuple
from collections import OrderedDict


class ResNet(nn.Module):

    def __init__(self, layer_config: List[Tuple[int, int, int]]) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=layer_config[0][0],
            kernel_size=7,
            stride=2,
            padding=3,
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.batchnorm = nn.BatchNorm2d(layer_config[0][0])
        self.relu = nn.ReLU()
        self.layers = nn.Sequential(
            OrderedDict(
                {
                    f"layer_{l}": (
                        Layer(
                            in_channels,
                            out_channels,
                            stride=1,
                            n_blocks=n_blocks,
                            projection=False,
                        )
                        if not l
                        else Layer(
                            in_channels,
                            out_channels,
                            stride=2,
                            n_blocks=n_blocks,
                            projection=True,
                        )
                    )
                    for l, (in_channels, out_channels, n_blocks) in enumerate(
                        layer_config
                    )
                }
            )
        )
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.linear = nn.Linear(512, 1000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.batchnorm(self.conv(x))
        x = self.maxpool(self.relu(x))
        x = self.layers(x)
        x = torch.flatten(self.avgpool(x), start_dim=1)
        output = self.linear(x)
        return output


class Layer(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        n_blocks: int,
        projection: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            OrderedDict(
                {
                    f"block_{b}": (
                        Block(in_channels, out_channels, stride, projection)
                        if not b
                        else Block(
                            out_channels, out_channels, stride=1, projection=False
                        )
                    )
                    for b in range(n_blocks)
                }
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class Block(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        projection: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.projection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
            if projection
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.relu1(self.batchnorm1(self.conv1(x)))
        residual = self.batchnorm2(self.conv2(x))
        output = self.relu2(self.projection(skip) + residual)
        return output


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = torch.randn(64, 3, 224, 224).to(device)
    layer_config = [(64, 64, 3), (64, 128, 4), (128, 256, 5), (256, 512, 3)]
    resnet34 = ResNet(layer_config).to(device)
    # print(resnet34)
    print(resnet34(inputs).size())


if __name__ == "__main__":
    main()
