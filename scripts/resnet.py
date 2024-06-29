import torch
import torch.nn as nn


class ResNet(nn.Module):

    def __init__(self, layer_config: list) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=layer_config[0],
            kernel_size=7,
            stride=2,
            padding=3,
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.layers = nn.Sequential(
            nn.ModuleDict(
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
        self.linear = nn.Linear(1000, 1000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.conv(x))
        x = self.layers(x)
        output = self.linear(self.avgpool(x))
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
        self.blocks = nn.Sequential(
            nn.ModuleDict(
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
            else lambda x: x
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.relu1(self.batchnorm1(self.conv1(x)))
        residual = self.batchnorm2(self.conv2(x))
        output = self.relu2(self.projection(skip) + residual)
        return output
