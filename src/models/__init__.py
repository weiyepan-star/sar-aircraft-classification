"""
模型包：基线模型、迁移学习模型、以及构建工厂。

    src/models/
    ├── __init__.py              本文件：模型工厂 + 旧版 SmallCNN（仅兼容早期 checkpoint）
    ├── cnn_baseline.py          CNNBaseline —— 基线（4 个卷积层，单通道输入）
    └── resnet18_transfer.py     ResNet18Transfer —— ImageNet 迁移学习（详见该文件顶部说明）

支持的模型类型（configs 里的 `type` 字段）：

    CNNBaseline        基线，从零训练，单通道输入
    ResNet18Transfer   迁移学习，三种冻结策略（none / partial / all）
    ResNet18           同上（旧类型名，保留兼容）
    SmallCNN           旧版 6 卷积结构，**已冻结**，只为兼容早期冒烟运行而保留
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .cnn_baseline import CNNBaseline
from .resnet18_transfer import IMAGENET_MEAN, IMAGENET_STD, FREEZE_POLICIES, ResNet18Transfer

# 旧类型名兼容：ResNet18Classifier 现等价于 ResNet18Transfer
ResNet18Classifier = ResNet18Transfer

__all__ = [
    "CNNBaseline",
    "ResNet18Transfer",
    "ResNet18Classifier",
    "SmallCNN",
    "build_model",
    "get_model_config",
    "build_optimizer",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "FREEZE_POLICIES",
]


# ============================================================================
#  旧版 SmallCNN（6 个卷积层的早期结构）
#  ⚠️ 已冻结：**不要在新实验中使用**。保留它只是为了让
#     results/raw/smoke/smoke_small_cnn_seed42_* 这批早期冒烟运行的 checkpoint
#     仍然可以被 evaluate.py / gradcam.py 加载，保证历史结果可追溯（G6）。
# ============================================================================
class SmallCNN(nn.Module):
    """早期版本的小型 CNN（3 个 block × 2 个卷积）。已被 CNNBaseline 取代。"""

    def __init__(self, num_classes: int = 6, channels=(32, 64, 128), dropout: float = 0.5,
                 in_channels: int = 1):
        super().__init__()
        c1, c2, c3 = channels
        self.in_channels = in_channels
        self.features = nn.Sequential(
            self._block(in_channels, c1),
            self._block(c1, c2),
            self._block(c2, c3),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(c3, num_classes))
        self._init_weights()

    @staticmethod
    def _block(cin: int, cout: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)).flatten(1))


# ============================================================================
#  工厂
# ============================================================================
def build_model(model_cfg: dict, num_classes: int, logger=None):
    """按配置字典构建模型。"""
    mtype = model_cfg["type"]
    name = model_cfg["name"]

    if mtype == "CNNBaseline":
        model = CNNBaseline(
            num_classes=num_classes,
            base_channels=int(model_cfg.get("base_channels", 32)),
            dropout=float(model_cfg.get("dropout", 0.5)),
            in_channels=int(model_cfg.get("input_channels", 1)),
        )
    elif mtype in ("ResNet18Transfer", "ResNet18"):
        model = ResNet18Transfer(
            num_classes=num_classes,
            pretrained=bool(model_cfg.get("pretrained", True)),
            weights=model_cfg.get("weights", "IMAGENET1K_V1"),
            freeze_policy=model_cfg.get("freeze_policy", "partial"),
            in_channels=int(model_cfg.get("input_channels", 3)),
        )
    elif mtype == "SmallCNN":  # 旧版结构，仅为兼容早期冒烟 checkpoint
        model = SmallCNN(
            num_classes=num_classes,
            channels=tuple(model_cfg.get("channels", [32, 64, 128])),
            dropout=float(model_cfg.get("dropout", 0.5)),
            in_channels=int(model_cfg.get("input_channels", 1)),
        )
    else:
        raise ValueError(f"未知模型类型: {mtype}")

    if logger:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("模型 %s (%s): 总参数 %s, 可训练参数 %s (%.2f%%)",
                    name, mtype, f"{total:,}", f"{trainable:,}", 100 * trainable / max(total, 1))
        if isinstance(model, ResNet18Transfer):
            logger.info("  冻结策略 = %s | 冻结参数张量 %d 个 | 灰度处理 = %s",
                        model.freeze_policy, len(model.frozen_module_names),
                        "复制成 3 通道" if model.repeat_gray_to_rgb else "改第一层为单通道")
    return model


def get_model_config(cfg: dict, model_name: str) -> dict:
    for m in cfg["models"]:
        if m["name"] == model_name:
            return m
    available = [m["name"] for m in cfg["models"]]
    raise KeyError(f"配置里没有名为 {model_name!r} 的模型。可选: {available}")


def build_optimizer(model, cfg: dict, model_cfg: dict):
    """
    构建优化器。

    分层学习率（discriminative fine-tuning）：
      · ResNet18Transfer 用模型自带的 `parameter_groups()`：
        分类头 lr = train.lr，可训练骨干 lr = train.lr_backbone
      · 其他模型（如 CNNBaseline）用统一的 train.lr
    """
    t = cfg["train"]
    lr = float(t["lr"])
    wd = float(t.get("weight_decay", 0.0))
    opt_name = t.get("optimizer", "AdamW")

    if isinstance(model, ResNet18Transfer) and model_cfg.get("pretrained"):
        groups = model.parameter_groups(lr_head=lr,
                                        lr_backbone=float(t.get("lr_backbone", lr * 0.1)))
    else:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}]

    if opt_name == "AdamW":
        return torch.optim.AdamW(groups, lr=lr, weight_decay=wd)
    if opt_name == "SGD":
        return torch.optim.SGD(groups, lr=lr, weight_decay=wd, momentum=0.9, nesterov=True)
    raise ValueError(f"未知优化器: {opt_name}")
