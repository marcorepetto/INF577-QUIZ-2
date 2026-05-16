import torch
import torch.nn as nn


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: nn.Module = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet_backbone(nn.Module):
    """Lightweight ResNet-18 variant for 64x64 images."""

    def __init__(self, cfg, input_channels: int = 3):
        super(ResNet_backbone, self).__init__()
        self.cfg = cfg
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = self.cfg.stem.channels

        # Stem: 3x3 conv, stride 1, padding 1
        self.conv1 = nn.Conv2d(
            input_channels, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # Layers based on config (typically 4 layers, each with 2 BasicBlocks)
        self.layer1 = self._make_layer(self.cfg.blocks[0].channels, 2, stride=self.cfg.blocks[0].stride)
        self.layer2 = self._make_layer(self.cfg.blocks[1].channels, 2, stride=self.cfg.blocks[1].stride)
        self.layer3 = self._make_layer(self.cfg.blocks[2].channels, 2, stride=self.cfg.blocks[2].stride)
        self.layer4 = self._make_layer(self.cfg.blocks[3].channels, 2, stride=self.cfg.blocks[3].stride)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * BasicBlock.expansion, stride),
                norm_layer(planes * BasicBlock.expansion),
            )

        layers = []
        layers.append(
            BasicBlock(self.inplanes, planes, stride, downsample, norm_layer=norm_layer)
        )
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(
                BasicBlock(
                    self.inplanes,
                    planes,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, 3, 64, 64)
        x = self.conv1(x)  # (B, 16, 64, 64)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x) # (B, 16, 64, 64)
        x = self.layer2(x) # (B, 32, 32, 32)
        x = self.layer3(x) # (B, 64, 16, 16)
        x = self.layer4(x) # (B, 128, 8, 8)

        x = self.avgpool(x) # (B, 128, 1, 1)
        x = torch.flatten(x, 1) # (B, 128)

        return x
