"""
训练主程序（单模型 × 单随机种子）。

对应约束：
  G2  —— **本脚本绝不使用测试集**（测试集只在 evaluate.py 中评估一次）
  G4  —— 单次运行只负责一个种子；多种子由 run_all 调用多次，汇总时报告均值与波动
  G5  —— 日志（train.log）、逐轮记录（history.csv）、指标（metrics.json）全部落盘
  G6  —— 结果按 smoke / formal 分离存放；失败运行的产物同样保留
  IC2 —— 增强只在训练集（由 dataset.py 的 build_transforms 保证）
  IC5 —— 最佳模型依据**验证集 Macro F1** 选择

用法：
    python src/train.py --config configs/smoke_test.yaml --model small_cnn --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import build_dataloaders                                    # noqa: E402
from models import build_model, build_optimizer, get_model_config        # noqa: E402
from utils import (env_snapshot, format_seconds, get_logger, load_config,  # noqa: E402
                   make_run_dir, save_json, save_yaml, set_seed)

SMOKE_BANNER = """
================================================================================
  ⚠️  冒烟测试运行（mode = smoke）
  目的：验证数据管线 / 模型维度 / 反向传播 / 产物落盘是否正常
  **本结果不得作为正式实验结论引用**，正式结果见 results/raw/formal/
================================================================================
"""


# ----------------------------------------------------------------------------
#  指标
# ----------------------------------------------------------------------------
def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    from sklearn.metrics import accuracy_score, f1_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


# ----------------------------------------------------------------------------
#  单轮训练 / 验证
# ----------------------------------------------------------------------------
def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=False,
              desc="train"):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, n_seen = 0.0, 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with autocast_ctx(use_amp):
                out = model(x)
                loss = criterion(out, y)

            if is_train:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        n_seen += bs
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(out.detach().argmax(1).cpu().tolist())

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(n_seen, 1)
    metrics["n"] = n_seen
    return metrics


def autocast_ctx(enabled: bool):
    if not enabled:
        from contextlib import nullcontext

        return nullcontext()
    try:
        return torch.amp.autocast("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=True)


def make_grad_scaler(enabled: bool):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


# ----------------------------------------------------------------------------
#  管线自检（step 4：确认数据/维度/反向传播）
# ----------------------------------------------------------------------------
def sanity_check(model, loader, criterion, device, logger, num_classes, use_amp=False) -> dict:
    """① 数据能否读取 ② 模型输入输出维度 ③ 能否反向传播。

    注意：期望的类别数必须由 `num_classes` 传入，
    **不能**用"当前 batch 里出现过的标签种类数"来推断
    （batch 里恰好只出现 3 个类别时会被误判为不通过）。
    """
    logger.info("-" * 72)
    logger.info("管线自检（取一个 batch）")
    x, y = next(iter(loader))
    logger.info("  ① 数据读取   : 成功，输入 batch 形状 = %s, 标签形状 = %s", tuple(x.shape), tuple(y.shape))
    logger.info("     输入 dtype=%s, 取值范围 [%.3f, %.3f]", x.dtype, float(x.min()), float(x.max()))
    logger.info("     标签取值 = %s", sorted(set(y.tolist())))

    model.train()
    x_d, y_d = x.to(device), y.to(device)
    with autocast_ctx(use_amp):
        out = model(x_d)
        loss = criterion(out, y_d)
    logger.info("  ② 模型输出   : 成功，输出形状 = %s（应为 [batch, num_classes=%d]）",
                tuple(out.shape), num_classes)

    loss.backward()
    grad_norms = [
        float(p.grad.detach().norm()) for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]
    total_grad = float(sum(grad_norms))
    logger.info("  ③ 反向传播   : 成功，loss = %.4f，%d 个参数张量获得梯度，梯度总范数 = %.6f",
                float(loss.item()), len(grad_norms), total_grad)
    model.zero_grad(set_to_none=True)

    ok = tuple(out.shape)[1] == num_classes and total_grad > 0
    logger.info("  ④ 自检结论   : %s", "✅ 通过" if ok else "❌ 未通过")
    logger.info("-" * 72)

    return {
        "input_shape": list(x.shape),
        "output_shape": list(out.shape),
        "loss_value": float(loss.item()),
        "n_grad_tensors": len(grad_norms),
        "total_grad_norm": total_grad,
        "passed": bool(ok),
    }


# ----------------------------------------------------------------------------
#  主流程
# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="训练单个模型（单随机种子）")
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default=None, help="模型名（默认取配置里第一个 enabled 的）")
    ap.add_argument("--seed", type=int, default=None, help="随机种子（默认取 train.seeds[0]）")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖配置里的轮数（调试用）")
    ap.add_argument("--tag", default=None, help="运行标签（默认取 run.tag）")
    ap.add_argument("--run-dir", default=None,
                    help="显式指定运行目录（相对项目根或绝对路径）；默认自动生成带时间戳的目录")
    args = ap.parse_args()

    # ---- 配置 ----
    cfg = load_config(args.config)
    project_root = Path(args.config).resolve().parent.parent
    cfg["_project_root"] = str(project_root)

    enabled = [m for m in cfg["models"] if m.get("enabled", True)]
    model_name = args.model or enabled[0]["name"]
    model_cfg = get_model_config(cfg, model_name)

    run_cfg = cfg.get("run", {}) or {}
    mode = run_cfg.get("mode", "formal")
    tag = args.tag or run_cfg.get("tag", "run")
    seed = args.seed if args.seed is not None else int(cfg["train"]["seeds"][0])
    epochs = args.epochs if args.epochs is not None else int(cfg["train"]["epochs"])

    # ---- 运行目录（G6：smoke 与 formal 分离）----
    if args.run_dir:
        rd = Path(args.run_dir)
        run_dir = rd if rd.is_absolute() else (project_root / rd)
        run_dir.mkdir(parents=True, exist_ok=True)
        existing = [p for p in run_dir.iterdir() if p.name != ".gitkeep"]
        if existing:
            print(f"[警告] 运行目录已存在且非空，将写入/覆盖其中的产物: {run_dir}")
    else:
        results_raw = project_root / "results" / "raw"
        run_dir = make_run_dir(results_raw, mode, tag, model_name, seed)
    logger = get_logger(f"train.{mode}.{model_name}.{seed}", run_dir / "train.log")

    logger.info("=" * 72)
    logger.info("运行目录 : %s", run_dir)
    logger.info("模式     : %s%s", mode, "  ← 冒烟测试" if mode == "smoke" else "")
    logger.info("模型     : %s (%s)", model_name, model_cfg["type"])
    logger.info("随机种子 : %d", seed)
    logger.info("轮数     : %d", epochs)
    if mode == "smoke":
        logger.warning(SMOKE_BANNER)

    # ---- 配置文件副本 ----
    # **剔除机器相关字段**：`_project_root` / `_config_path` 是本机绝对路径，
    # 写进仓库后，别人克隆到别的目录时会据此把产物写回**原机器**（实测踩过这个坑）。
    save_yaml({k: v for k, v in cfg.items() if not k.startswith("_")},
              run_dir / "config_resolved.yaml")

    # ---- 设备 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    logger.info("设备     : %s%s", device, f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    logger.info("混合精度 : %s", "开启" if use_amp else "关闭")

    seed_info = set_seed(seed)

    # ---- 数据 ----
    logger.info("-" * 72)
    logger.info("构建数据管线")
    data = build_dataloaders(cfg, run_dir, logger=logger)
    loaders = data["loaders"]
    class_names = data["class_names"]
    num_classes = len(class_names)
    logger.info("类别(%d): %s", num_classes, class_names)

    # ---- 模型 ----
    logger.info("-" * 72)
    model = build_model(model_cfg, num_classes, logger=logger).to(device)
    optimizer = build_optimizer(model, cfg, model_cfg)
    criterion = nn.CrossEntropyLoss()

    sched_name = cfg["train"].get("scheduler", "cosine")
    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif sched_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3)
    else:
        scheduler = None

    scaler = make_grad_scaler(use_amp)

    # ---- 管线自检 ----
    checks = sanity_check(model, loaders["train"], criterion, device, logger, num_classes, use_amp)

    # ---- 训练循环 ----
    es_cfg = cfg["train"].get("early_stopping", {}) or {}
    es_on = bool(es_cfg.get("enabled", True))
    patience = int(es_cfg.get("patience", 10))

    history: list[dict] = []
    best_f1, best_epoch, bad_epochs = -1.0, -1, 0
    t_start = time.time()
    #  G2/IC5：全程**不触碰**测试集
    test_loader_used = False

    logger.info("=" * 72)
    logger.info("开始训练（测试集本次**未被使用**，见 G2/IC5）")
    header = f"{'epoch':>5} {'lr':>9} {'tr_loss':>9} {'tr_acc':>8} {'va_loss':>9} {'va_acc':>8} {'va_f1':>8} {'time':>8}"
    logger.info(header)
    logger.info("-" * len(header))

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        lr_now = float(optimizer.param_groups[0]["lr"])

        tr = run_epoch(model, loaders["train"], criterion, device, optimizer, scaler, use_amp)
        va = run_epoch(model, loaders["val"], criterion, device)

        if scheduler is not None:
            if sched_name == "plateau":
                scheduler.step(va["macro_f1"])
            else:
                scheduler.step()

        row = {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": tr["loss"],
            "train_acc": tr["accuracy"],
            "train_macro_f1": tr["macro_f1"],
            "val_loss": va["loss"],
            "val_acc": va["accuracy"],
            "val_macro_f1": va["macro_f1"],
            "epoch_time_sec": time.time() - t0,
        }
        history.append(row)

        logger.info(
            "%5d %9.6f %9.4f %8.4f %9.4f %8.4f %8.4f %8s",
            epoch, row["lr"], tr["loss"], tr["accuracy"], va["loss"], va["accuracy"],
            va["macro_f1"], format_seconds(row["epoch_time_sec"]),
        )

        # 最佳模型：依据**验证集 Macro F1**（IC5）
        if va["macro_f1"] > best_f1:
            best_f1, best_epoch, bad_epochs = va["macro_f1"], epoch, 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": model_name,
                    "model_cfg": model_cfg,
                    "class_names": class_names,
                    "epoch": epoch,
                    "val_macro_f1": best_f1,
                    "image_size": int(cfg["data"]["image_size"]),
                    "normalize": {"mean": data["mean"], "std": data["std"]},
                    "seed": seed,
                    "mode": mode,
                },
                run_dir / "best_model.pt",
            )
            logger.info("        ↑ 新的最佳（验证集 macro_f1 = %.4f），已保存 best_model.pt", best_f1)
        else:
            bad_epochs += 1
            if es_on and bad_epochs >= patience:
                logger.info("早停：验证集 macro_f1 连续 %d 轮未提升（最佳为第 %d 轮 %.4f）",
                            patience, best_epoch, best_f1)
                break

    torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, run_dir / "last_model.pt")

    total_time = time.time() - t_start

    # ---- 落盘（G5）----
    with open(run_dir / "history.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    metrics = {
        "mode": mode,
        "is_smoke": mode == "smoke",
        "model": model_name,
        "model_type": model_cfg["type"],
        "seed": seed,
        "epochs_planned": epochs,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "best_val_acc": max(h["val_acc"] for h in history),
        "total_time_sec": total_time,
        "n_params_total": sum(p.numel() for p in model.parameters()),
        "n_params_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "class_names": class_names,
        "data": {
            "n_train": len(loaders["train"].dataset),
            "n_val": len(loaders["val"].dataset),
            "n_test": len(loaders["test"].dataset),
            "image_size": int(cfg["data"]["image_size"]),
            "normalize": {"mean": data["mean"], "std": data["std"],
                          "mode": cfg.get("normalize", {}).get("mode", "train_stats")},
        },
        "split_checks": data["checks"],
        "sanity_check": checks,
        "test_set_used_in_training": test_loader_used,
        "note": ("冒烟测试结果，不得作为正式结论" if mode == "smoke"
                 else "正式实验结果"),
    }
    save_json(metrics, run_dir / "metrics.json")

    run_meta = {
        "run_dir": str(run_dir),
        "mode": mode,
        "tag": tag,
        "model": model_name,
        "seed": seed,
        "seed_settings": seed_info,
        "env": env_snapshot(),
        "config_path": str(Path(args.config).resolve()),
        "command": " ".join(sys.argv),
    }
    save_json(run_meta, run_dir / "run_meta.json")

    # ---- 汇总输出 ----
    logger.info("=" * 72)
    logger.info("训练完成")
    logger.info("  实际轮数      : %d / %d", len(history), epochs)
    logger.info("  最佳轮次      : 第 %d 轮", best_epoch)
    logger.info("  最佳验证 macro_f1: %.4f", best_f1)
    logger.info("  最佳验证 acc  : %.4f", max(h["val_acc"] for h in history))
    logger.info("  总耗时        : %s", format_seconds(total_time))
    logger.info("  测试集是否被使用: %s", "是" if test_loader_used else "否（符合 G2/IC5）")
    logger.info("  产物目录      : %s", run_dir)
    for f in sorted(run_dir.iterdir()):
        logger.info("      %-24s %8.1f KB", f.name, f.stat().st_size / 1024)
    if mode == "smoke":
        logger.warning("这是冒烟测试结果，已存于 results/raw/smoke/，**不会进入正式汇总**")
    logger.info("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
