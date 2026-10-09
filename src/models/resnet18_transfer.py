"""
ResNet18 迁移学习模型（ResNet18Transfer）。

================================================================================
  1. 预训练权重来源（可追溯）
================================================================================
  · 来源：**torchvision 官方权重**，`torchvision.models.ResNet18_Weights.IMAGENET1K_V1`
  · 原始训练集：**ImageNet-1k**（1,281,167 张训练图，1,000 类）
  · 权重文件：`resnet18-f37072fd.pth`（约 45 MB），首次使用时由 torchvision 自动下载
  · 官方文档：https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
  · 论文：He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016
  · 注意：本权重是**自然彩色图像**上训练的，与 SAR 灰度图存在域差异 ——
    这正是本项目要验证的问题（experiment-plan.md 的假设 H1）。

================================================================================
  2. 分类层替换
================================================================================
  原 ResNet18 的 `fc` 是 `Linear(512 → 1000)`（ImageNet 类别数）。
  这里替换为 `Linear(512 → num_classes)`（本项目 num_classes = 6），
  并用小标准差正态初始化（mean=0, std=0.01）——因为它是**新随机初始化的层**，
  训练初期不能让它的输出过大而冲垮预训练特征。

================================================================================
  3. 冻结 / 解冻策略与学习率（默认 partial）
================================================================================
  参数 `freeze_policy` 三选一：

  ┌──────────┬────────────────────────────┬──────────────────────────────┐
  │ 取值     │ 可训练的部分               │ 含义                         │
  ├──────────┼────────────────────────────┼──────────────────────────────┤
  │ `none`   │ 全部（约 11.18 M）         │ 整体微调 full fine-tune      │
  │ `partial`│ layer3 + layer4 + fc       │ **默认**：浅层冻结、深层微调 │
  │ `all`    │ 仅 fc（约 3 K）            │ 线性探针 linear probe        │
  └──────────┴────────────────────────────┴──────────────────────────────┘

  **本项目默认用 `partial`**，理由：
    · 数据集很小（训练集仅 2,098 张）—— 整体微调 1,118 万参数**极易过拟合**；
    · ResNet 的**浅层**（conv1 / layer1 / layer2）学的是边缘、角点等
      **通用低层特征**，对 SAR 同样适用，冻结它们既省算力又防过拟合；
    · **深层**（layer3 / layer4）学的是任务相关的语义特征，
      ImageNet 的语义与 SAR 差异最大，**最需要针对 SAR 重新学习**；
    · 全连接层是全新初始化的，必须训练。

  **学习率（分层设置，discriminative fine-tuning）**：

      分类头 fc                 lr = 1e-3      （新初始化的层，用较大的 lr）
      可训练的骨干 layer3/4     lr = 1e-4      （预训练权重，用 1/10 的 lr 防止被冲垮）

  这两组参数由 `parameter_groups()` 生成，交给构造优化器的 `build_optimizer()` 使用
  （见 configs 里的 `train.lr` 与 `train.lr_backbone`）。

  ⚠️ **冻结时必须同时把被冻结模块设为 `eval()`**：
     否则 BatchNorm 的 running mean/var 仍会在前向时被更新，
     那就不是真正的"冻结"了。本类的 `train()` 已覆写处理这一点。

================================================================================
  4. 灰度图怎么进网络（本项目选择：复制成 3 通道）
================================================================================
  数据集输出的是**单通道灰度** `(N, 1, 96, 96)`，而 ImageNet 预训练的第一层
  期望 3 通道。两种做法：

  ┌──────────────────────┬────────────────────────────────────────────────────┐
  │ 做法                 │ 说明                                               │
  ├──────────────────────┼────────────────────────────────────────────────────┤
  │ **复制成 3 通道** ✅ │ `x.repeat(1,3,1,1)` —— 三个通道完全相同。          │
  │ （本项目采用）       │ **完整保留预训练的第一层权重**，不做任何改动。     │
  │                      │ 代价：第一层看到的是"三份一样的灰度"，             │
  │                      │ 与它在 ImageNet 上见过的彩色输入分布不同。         │
  ├──────────────────────┼────────────────────────────────────────────────────┤
  │ 改第一层为单通道     │ 新建 `Conv2d(1, 64, ...)`，权重取原 `conv1` 在       │
  │ （备选，in_channels=1）│ 输入通道维的**平均值**。                          │
  │                      │ 好处：第一层输入分布与灰度图匹配；                 │
  │                      │ 代价：丢掉了预训练三通道权重的结构（做了一次平均）。│
  └──────────────────────┴────────────────────────────────────────────────────┘

  **选择"复制成 3 通道"作为主方案**，因为：
    ① 它**不改动任何预训练权重**，是"最纯粹的迁移学习"设定；
    ② 复制后的输入在数值上与"灰度值等强度的三通道彩色图"等价，
       低层卷积的响应仍然有意义；
    ③ 备选方案（改第一层）会引入一次人工平均，**多了一个需要额外论证的假设**。
  备选方案已实现（`in_channels=1`），可作为消融实验，但**不是本次主配置**。
"""

from __future__ import annotations

# ⚠️ 导入顺序（本机环境特有，实测结论）：
#    PIL 必须在 torchvision 之前导入，否则报
#        ImportError: DLL load failed while importing _imaging
#    这是固定 DLL 加载顺序用的，本文件并不直接使用 PIL —— **请勿删除或调整顺序**。
#    详见 README.md 第 5.4 节「坑 3」。
from PIL import Image  # noqa: F401

import torch
import torch.nn as nn

# 归一化统计量：ImageNet-1k 的 RGB 均值/标准差（torchvision 官方推荐值）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

FREEZE_POLICIES = ("none", "partial", "all")
# partial 策略下被冻结的模块名（浅层）
PARTIAL_FROZEN_PREFIXES = ("conv1", "bn1", "layer1", "layer2")


class ResNet18Transfer(nn.Module):
    """ImageNet 预训练 ResNet18 + 新分类头，支持三种冻结策略。"""

    def __init__(
        self,
        num_classes: int = 6,
        pretrained: bool = True,
        weights: str = "IMAGENET1K_V1",
        freeze_policy: str = "partial",
        in_channels: int = 3,
    ):
        super().__init__()
        if freeze_policy not in FREEZE_POLICIES:
            raise ValueError(f"freeze_policy 必须是 {FREEZE_POLICIES}，收到 {freeze_policy!r}")

        from torchvision import models  # 延迟导入（见 gradcam.py 的导入顺序说明）

        w = getattr(models.ResNet18_Weights, weights) if pretrained else None
        backbone = models.resnet18(weights=w)

        # ---- 灰度处理 ----
        if in_channels == 1:
            # 备选方案：第一层改单通道，预训练权重在输入通道维求平均
            old = backbone.conv1
            new = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size,
                            stride=old.stride, padding=old.padding, bias=False)
            with torch.no_grad():
                new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
            backbone.conv1 = new

        self.in_channels = in_channels
        self.repeat_gray_to_rgb = (in_channels == 3)   # 复制成三通道（主方案）

        # ---- 替换分类层 ----
        in_features = backbone.fc.in_features          # 512
        backbone.fc = nn.Linear(in_features, num_classes)
        nn.init.normal_(backbone.fc.weight, mean=0.0, std=0.01)
        nn.init.zeros_(backbone.fc.bias)

        self.backbone = backbone
        self.freeze_policy = freeze_policy
        self.num_classes = num_classes
        self._apply_freeze(freeze_policy)

    # ------------------------------------------------------------------
    def _apply_freeze(self, policy: str) -> None:
        if policy == "none":
            return
        for name, p in self.backbone.named_parameters():
            if policy == "all":
                p.requires_grad = name.startswith("fc.")
            else:  # partial
                frozen = any(name == pre or name.startswith(pre + ".") for pre in PARTIAL_FROZEN_PREFIXES)
                p.requires_grad = (not frozen)

    @property
    def frozen_module_names(self) -> list[str]:
        return [n for n, p in self.backbone.named_parameters() if not p.requires_grad]

    def train(self, mode: bool = True):
        """
        训练模式下，**被冻结的模块仍然保持 eval()**。

        原因：BatchNorm 在前向时会更新 running_mean/running_var，
        若冻结的骨干还处于 train 模式，那些统计量照样会被改动 —— 那就不是真冻结。
        """
        super().train(mode)
        if mode and self.freeze_policy != "none":
            for name, module in self.backbone.named_children():
                if name == "fc":
                    continue
                trainable = any(p.requires_grad for p in module.parameters())
                if not trainable:
                    module.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.repeat_gray_to_rgb and x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)     # (N,1,H,W) -> (N,3,H,W)，三通道内容相同
        return self.backbone(x)

    # ------------------------------------------------------------------
    def parameter_groups(self, lr_head: float, lr_backbone: float) -> list[dict]:
        """
        生成分层学习率的参数组（供优化器使用）。

        分类头 fc 用 lr_head；可训练的骨干（layer3/layer4）用 lr_backbone。
        """
        head, backbone = [], []
        for name, p in self.backbone.named_parameters():
            if not p.requires_grad:
                continue
            (head if name.startswith("fc.") else backbone).append(p)

        groups = [{"params": head, "lr": lr_head, "name": "head"}]
        if backbone:
            groups.append({"params": backbone, "lr": lr_backbone, "name": "backbone"})
        return groups


# ---------------------------------------------------------------------------
#  自检
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    print("ResNet18Transfer 自检（不下载预训练权重，pretrained=False）\n")
    for policy in FREEZE_POLICIES:
        m = ResNet18Transfer(num_classes=6, pretrained=False, freeze_policy=policy).eval()
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        x = torch.randn(2, 1, 96, 96)
        with torch.no_grad():
            y = m(x)
        groups = m.parameter_groups(1e-3, 1e-4)
        print(f"  policy={policy:<8} 输入 {tuple(x.shape)} → 输出 {tuple(y.shape)} | "
              f"总参数 {total:,} | 可训练 {trainable:,} ({100*trainable/total:.2f}%) | "
              f"参数组 {[g['name'] for g in groups]}")
    print("\n✅ ResNet18Transfer 自检通过")
