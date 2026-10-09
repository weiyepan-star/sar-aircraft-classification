"""
小型 CNN 基线模型（CNNBaseline）—— 本项目的**基线对照模型**。

设计目标（对应 experiment-plan.md 第 4.2 节，按 2026-10-08 的方案变更调整）：
  · 结构与参数量都"适中"：既能跑出可信基线，又不至于碾压或碾压不过迁移学习
  · 面向 **96×96 单通道** SAR 灰度切片图
  · 输入通道处理：数据集输出的是**单通道灰度**（`(N,1,96,96)`），
    因此第一层卷积的 `in_channels=1`，**直接对应，不做三通道复制**。
    （三通道复制只用于 ResNet18 这类 ImageNet 预训练模型，见 resnet18.py）

网络结构（**4 个卷积层**，每个卷积后接 BatchNorm + ReLU）：

    输入 (1, 96, 96)
      ├─ Conv3×3( 1→ 32) + BN + ReLU          → (32, 96, 96)
      ├─ MaxPool2                              → (32, 48, 48)
      ├─ Conv3×3(32→ 64) + BN + ReLU          → (64, 48, 48)
      ├─ MaxPool2                              → (64, 24, 24)
      ├─ Conv3×3(64→128) + BN + ReLU          → (128, 24, 24)
      ├─ MaxPool2                              → (128, 12, 12)
      ├─ Conv3×3(128→128) + BN + ReLU         → (128, 12, 12)   ← Grad-CAM 目标层
      ├─ GlobalAvgPool                         → (128,)
      ├─ Dropout(0.5)
      └─ Linear(128 → num_classes)

参数量约 **24 万**（默认 base_channels=32），与数据集规模（约 2400 张训练图）相称。

不使用任何预训练权重 —— 这正是"从零训练"基线的定义。
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv_bn_relu(cin: int, cout: int) -> nn.Sequential:
    """Conv3×3（same padding）+ BatchNorm + ReLU。"""
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class CNNBaseline(nn.Module):
    """
    4 层卷积的小型 CNN 基线。

    参数
    ----
    num_classes   : 类别数（SAR-ACD 为 6）
    base_channels : 第一层通道数，后续层依次 ×1/×2/×4/×4
    dropout       : 分类头前的 Dropout 比例
    in_channels   : 输入通道数，灰度图 = 1
    """

    def __init__(
        self,
        num_classes: int = 6,
        base_channels: int = 32,
        dropout: float = 0.5,
        in_channels: int = 1,
    ):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 4

        self.in_channels = in_channels
        self.base_channels = base_channels

        # 4 个卷积层；最后一个卷积层是 Grad-CAM 的目标层（保留 12×12 空间分辨率）
        self.features = nn.Sequential(
            _conv_bn_relu(in_channels, c1), nn.MaxPool2d(2),   # 96 -> 48
            _conv_bn_relu(c1, c2), nn.MaxPool2d(2),            # 48 -> 24
            _conv_bn_relu(c2, c3), nn.MaxPool2d(2),            # 24 -> 12
            _conv_bn_relu(c3, c4),                             # 12 -> 12
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(c4, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
#  自检：确认输入输出维度与参数量
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    for size in (96, 64, 128):
        m = CNNBaseline(num_classes=6).eval()
        x = torch.randn(2, 1, size, size)
        with torch.no_grad():
            y = m(x)
            f = m.forward_features(x)
        total = sum(p.numel() for p in m.parameters())
        print(f"  输入 {tuple(x.shape)} → 特征 {tuple(f.shape)} → 输出 {tuple(y.shape)} | 参数量 {total:,}")
    print("\n✅ CNNBaseline 自检通过")
