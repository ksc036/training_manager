from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return self.conv(torch.cat([skip, x], dim=1))


class TinySegNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 16) -> None:
        super().__init__()
        c1 = base_channels
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2
        c5 = c4 * 2
        self.inc = DoubleConv(in_channels, c1)
        self.down1 = DownBlock(c1, c2)
        self.down2 = DownBlock(c2, c3)
        self.down3 = DownBlock(c3, c4)
        self.down4 = DownBlock(c4, c5)
        self.up1 = UpBlock(c5, c4, c4)
        self.up2 = UpBlock(c4, c3, c3)
        self.up3 = UpBlock(c3, c2, c2)
        self.up4 = UpBlock(c2, c1, c1)
        self.outc = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 16) -> None:
        super().__init__()
        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        ]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.x0_0 = DoubleConv(in_channels, channels[0])
        self.x1_0 = DoubleConv(channels[0], channels[1])
        self.x2_0 = DoubleConv(channels[1], channels[2])
        self.x3_0 = DoubleConv(channels[2], channels[3])
        self.x4_0 = DoubleConv(channels[3], channels[4])

        self.x0_1 = DoubleConv(channels[0] + channels[1], channels[0])
        self.x1_1 = DoubleConv(channels[1] + channels[2], channels[1])
        self.x2_1 = DoubleConv(channels[2] + channels[3], channels[2])
        self.x3_1 = DoubleConv(channels[3] + channels[4], channels[3])

        self.x0_2 = DoubleConv(channels[0] * 2 + channels[1], channels[0])
        self.x1_2 = DoubleConv(channels[1] * 2 + channels[2], channels[1])
        self.x2_2 = DoubleConv(channels[2] * 2 + channels[3], channels[2])

        self.x0_3 = DoubleConv(channels[0] * 3 + channels[1], channels[0])
        self.x1_3 = DoubleConv(channels[1] * 3 + channels[2], channels[1])

        self.x0_4 = DoubleConv(channels[0] * 4 + channels[1], channels[0])
        self.outc = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def _upsample_to(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != target.shape[-2:]:
            return nn.functional.interpolate(
                x,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0_0 = self.x0_0(x)
        x1_0 = self.x1_0(self.pool(x0_0))
        x0_1 = self.x0_1(torch.cat([x0_0, self._upsample_to(self.up(x1_0), x0_0)], dim=1))

        x2_0 = self.x2_0(self.pool(x1_0))
        x1_1 = self.x1_1(torch.cat([x1_0, self._upsample_to(self.up(x2_0), x1_0)], dim=1))
        x0_2 = self.x0_2(
            torch.cat([x0_0, x0_1, self._upsample_to(self.up(x1_1), x0_0)], dim=1)
        )

        x3_0 = self.x3_0(self.pool(x2_0))
        x2_1 = self.x2_1(torch.cat([x2_0, self._upsample_to(self.up(x3_0), x2_0)], dim=1))
        x1_2 = self.x1_2(
            torch.cat([x1_0, x1_1, self._upsample_to(self.up(x2_1), x1_0)], dim=1)
        )
        x0_3 = self.x0_3(
            torch.cat([x0_0, x0_1, x0_2, self._upsample_to(self.up(x1_2), x0_0)], dim=1)
        )

        x4_0 = self.x4_0(self.pool(x3_0))
        x3_1 = self.x3_1(torch.cat([x3_0, self._upsample_to(self.up(x4_0), x3_0)], dim=1))
        x2_2 = self.x2_2(
            torch.cat([x2_0, x2_1, self._upsample_to(self.up(x3_1), x2_0)], dim=1)
        )
        x1_3 = self.x1_3(
            torch.cat([x1_0, x1_1, x1_2, self._upsample_to(self.up(x2_2), x1_0)], dim=1)
        )
        x0_4 = self.x0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, self._upsample_to(self.up(x1_3), x0_0)], dim=1)
        )
        return self.outc(x0_4)


class DeepLabV3Wrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from torchvision.models.segmentation import deeplabv3_resnet50

        self.model = deeplabv3_resnet50(
            weights=None,
            weights_backbone=None,
            num_classes=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x)
        return outputs["out"]


def build_segmentation_model(model_name: str) -> nn.Module:
    normalized_name = model_name.strip().lower()
    if normalized_name == "unet":
        return UNet()
    if normalized_name == "unetplusplus":
        return UNetPlusPlus()
    if normalized_name == "deeplabv3":
        return DeepLabV3Wrapper()
    if normalized_name == "external-script-model":
        return TinySegNet()
    raise ValueError(f"Unsupported model name: {model_name}")
