import torch
import torch.nn as nn


class CNN_backbone(nn.Module):
  """Simple CNN backbone constructed from a Hydra config-like object."""

  def __init__(self, cfg, input_channels: int = 3):
    super(CNN_backbone, self).__init__()
    self.cfg = cfg

    ks = self.cfg.conv.kernel_size
    stride = self.cfg.conv.stride
    pad = self.cfg.conv.padding

    # Stem
    self.stem = nn.Sequential(
      nn.Conv2d(input_channels, self.cfg.conv.sizes.stem, kernel_size=ks, stride=stride, padding=pad),
      nn.BatchNorm2d(self.cfg.conv.sizes.stem),
      nn.ReLU(inplace=True),
    )

    # Block 1
    self.block1 = nn.Sequential(
      nn.Conv2d(self.cfg.conv.sizes.stem, self.cfg.conv.sizes.block_1, kernel_size=ks, stride=stride, padding=pad),
      nn.BatchNorm2d(self.cfg.conv.sizes.block_1),
      nn.ReLU(inplace=True),
      nn.Conv2d(self.cfg.conv.sizes.block_1, self.cfg.conv.sizes.block_1, kernel_size=ks, stride=stride, padding=pad),
      nn.BatchNorm2d(self.cfg.conv.sizes.block_1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=self.cfg.max_pool.kernel_size, stride=self.cfg.max_pool.stride),
    )

    # Block 2
    self.block2 = nn.Sequential(
      nn.Conv2d(self.cfg.conv.sizes.block_1, self.cfg.conv.sizes.block_2, kernel_size=ks, stride=stride, padding=pad),
      nn.BatchNorm2d(self.cfg.conv.sizes.block_2),
      nn.ReLU(inplace=True),
      nn.Conv2d(self.cfg.conv.sizes.block_2, self.cfg.conv.sizes.block_2, kernel_size=ks, stride=stride, padding=pad),
      nn.BatchNorm2d(self.cfg.conv.sizes.block_2),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=self.cfg.max_pool.kernel_size, stride=self.cfg.max_pool.stride),
    )

    # Block 3
    self.block3 = nn.Sequential(
      nn.Conv2d(self.cfg.conv.sizes.block_2, self.cfg.conv.sizes.block_3, kernel_size=ks, stride=stride, padding=pad),
      nn.BatchNorm2d(self.cfg.conv.sizes.block_3),
      nn.ReLU(inplace=True),
      nn.Conv2d(self.cfg.conv.sizes.block_3, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
    )

    # Global average pooling -> feature vector
    self.global_pool = nn.AdaptiveAvgPool2d(1)

  def forward(self, x):
    x = self.stem(x)
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.global_pool(x)
    x = torch.flatten(x, 1)
    return x