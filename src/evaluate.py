"""
测试集评估（**整个项目中唯一允许使用测试集的地方**）。

对应约束：
  G2  —— 测试集只在最终评估时使用一次；本脚本维护"测试集使用台账"，
         若同一运行被评估超过 1 次会**大声告警**并记入台账。
  IC5 —— 使用训练阶段由**验证集**选出的 best_model.pt，绝不重新选择模型。
  G5  —— 所有指标落盘（test_metrics.json / predictions.csv / errors.csv）
  IC3 —— 报告 Accuracy / Macro F1 / 每类 P-R-F1 / 混淆矩阵（原始 + 归一化）
  IC4 —— 导出**全部**错误样例清单与高不确定性样例清单（供可视化与 Grad-CAM 使用）
  G6  —— 错误样例完整保留，不做任何筛选

用法：
    python src/evaluate.py --run results/raw/formal/small_cnn_seed42_20261008-120000
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import build_dataloaders                                  # noqa: E402
from models import build_model                                        # noqa: E402
from utils import (get_logger, load_config, load_json, save_json,     # noqa: E402
                   detect_run_mode, set_seed)

LEDGER_NAME = "test_evaluation_ledger.json"


# ----------------------------------------------------------------------------
#  台账：记录测试集被使用了几次（G2）
# ----------------------------------------------------------------------------
def update_ledger(run_dir: Path, entry: dict, logger) -> int:
    path = run_dir / LEDGER_NAME
    ledger = load_json(path) if path.is_file() else {"entries": []}
    ledger["entries"].append(entry)
    n = len(ledger["entries"])
    ledger["n_evaluations"] = n
    ledger["policy"] = "测试集原则上只评估一次（G2 / IC5）"
    save_json(ledger, path)

    if n == 1:
        logger.info("测试集使用台账：这是第 1 次评估（符合 G2）")
    else:
        logger.warning("=" * 72)
        logger.warning("⚠️ 测试集使用台账：这是第 %d 次评估本运行！", n)
        logger.warning("   G2 要求测试集只在最终方案确定后评估一次。")
        logger.warning("   若本次是因为修复缺陷而重跑，请在报告中明确说明；")
        logger.warning("   若是因为「看结果后调参」而重跑，则**已经违反 G2**。")
        logger.warning("=" * 72)
    return n


# ----------------------------------------------------------------------------
#  推理
# ----------------------------------------------------------------------------
@torch.no_grad()
def predict_all(model, loader, device, logger=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_true = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits.float(), dim=1)
        all_probs.append(probs.cpu().numpy())
        all_true.append(y.numpy())
    probs = np.concatenate(all_probs, axis=0)
    true = np.concatenate(all_true, axis=0)
    pred = probs.argmax(axis=1)
    if logger:
        logger.info("推理完成：%d 个样本", len(true))
    return true, pred, probs


# ----------------------------------------------------------------------------
#  指标（IC3）
# ----------------------------------------------------------------------------
def compute_test_metrics(y_true, y_pred, probs, class_names) -> dict:
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score)

    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names,
        output_dict=True, zero_division=0,
    )
    per_class = {
        name: {
            "precision": float(report[name]["precision"]),
            "recall": float(report[name]["recall"]),
            "f1": float(report[name]["f1-score"]),
            "support": int(report[name]["support"]),
        }
        for name in class_names
    }
    conf = probs.max(axis=1)
    correct = (y_true == y_pred)

    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": per_class,
        "label_mapping": {str(i): name for i, name in enumerate(class_names)},
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
        "n_errors": int((~correct).sum()),
        "mean_confidence": float(conf.mean()),
        "mean_confidence_correct": float(conf[correct].mean()) if correct.any() else None,
        "mean_confidence_wrong": float(conf[~correct].mean()) if (~correct).any() else None,
        "macro_weighted_f1_gap": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
            - f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


# ----------------------------------------------------------------------------
#  逐样本预测表 / 错误清单 / 不确定性清单（IC4）
# ----------------------------------------------------------------------------
def dump_predictions(records, y_true, y_pred, probs, class_names, run_dir: Path,
                     topk_uncertain: int = 50) -> dict:
    conf = probs.max(axis=1)
    rows = []
    for i, rec in enumerate(records):
        rows.append(
            {
                "relpath": rec["relpath"],
                "class_name": rec["class_name"],
                "true_label": int(y_true[i]),
                "true_name": class_names[int(y_true[i])],
                "pred_label": int(y_pred[i]),
                "pred_name": class_names[int(y_pred[i])],
                "confidence": round(float(conf[i]), 6),
                "correct": bool(y_true[i] == y_pred[i]),
            }
        )
    rows.sort(key=lambda r: r["relpath"])

    fields = list(rows[0].keys())
    with open(run_dir / "predictions.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    errors = [r for r in rows if not r["correct"]]
    with open(run_dir / "errors.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(errors)

    uncertain = sorted(rows, key=lambda r: r["confidence"])[:topk_uncertain]
    with open(run_dir / "uncertain.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(uncertain)

    return {
        "n_predictions": len(rows),
        "n_errors": len(errors),
        "n_uncertain_exported": len(uncertain),
        "error_class_pairs": dict(
            Counter(f"{r['true_name']}->{r['pred_name']}" for r in errors).most_common()
        ),
    }


# ----------------------------------------------------------------------------
#  主流程
# ----------------------------------------------------------------------------
def evaluate_run(run_dir: str | Path, config_path: str | Path | None = None,
                 split: str = "test", logger=None) -> dict:
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"运行目录不存在: {run_dir}")

    ckpt_path = run_dir / "best_model.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"找不到最佳模型权重: {ckpt_path}")

    if config_path is None:
        config_path = run_dir / "config_resolved.yaml"
    cfg = load_config(config_path)

    # 项目根：从运行目录向上找到含 results/ 的那一层
    project_root = run_dir
    while project_root != project_root.parent and not (project_root / "results").is_dir():
        project_root = project_root.parent
    cfg["_project_root"] = str(project_root)

    if logger is None:
        logger = get_logger(f"evaluate.{run_dir.name}", run_dir / "evaluate.log")
    logger.info("=" * 72)
    logger.info("测试集评估: %s", run_dir.name)
    logger.info("配置文件  : %s", config_path)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    class_names = ckpt["class_names"]
    model_name = ckpt.get("model_name", "unknown")
    seed = ckpt.get("seed", -1)
    logger.info("模型      : %s | 训练时最佳轮次 %s | 验证集 macro_f1 %s",
                model_name, ckpt.get("epoch"), ckpt.get("val_macro_f1"))
    logger.info("该 checkpoint 是依据**验证集**选出的（IC5），本次不改动模型选择")

    set_seed(int(seed) if seed >= 0 else 42)

    # ---- 复用训练时的同一个划分（G3：不得重新划分）----
    cfg.setdefault("split", {})["reuse_index"] = str(run_dir / "split_index.csv")
    data = build_dataloaders(cfg, run_dir, logger=logger,
                             artifact_dir=run_dir / "eval_artifacts")
    loaders = data["loaders"]
    records = [r for r in data["records"] if r["split"] == split]
    if not records:
        raise RuntimeError(f"划分 {split!r} 为空")

    # ---- 构建模型并载入权重（pretrained=False：权重来自 checkpoint，不需要联网）----
    model_cfg = dict(ckpt["model_cfg"])
    model_cfg["pretrained"] = False
    model = build_model(model_cfg, len(class_names), logger=None)
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    logger.info("设备      : %s | 评估划分: %s（%d 个样本）", device, split, len(records))

    # ---- 推理与指标 ----
    y_true, y_pred, probs = predict_all(model, loaders[split], device, logger=logger)
    metrics = compute_test_metrics(y_true, y_pred, probs, class_names)
    metrics.update(
        {
            "run": run_dir.name,
            "model": model_name,
            "seed": seed,
            "split": split,
            "mode": detect_run_mode(run_dir),
            "checkpoint": str(ckpt_path.name),
            "checkpoint_selected_by": "val_macro_f1 (IC5)",
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
            "device": str(device),
        }
    )

    dump = dump_predictions(records, y_true, y_pred, probs, class_names, run_dir,
                            topk_uncertain=int(cfg.get("error_analysis", {}).get(
                                "export_uncertain_topk", 50)))
    metrics["error_analysis"] = dump

    # ---- 台账（G2）：**先更新台账**，再把评估次数一并写进指标文件 ----
    n_eval = update_ledger(
        run_dir,
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "split": split,
            "n_samples": len(records),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "command": " ".join(sys.argv),
        },
        logger,
    )
    metrics["n_test_evaluations"] = n_eval
    save_json(metrics, run_dir / "test_metrics.json")

    # ---- 报告 ----
    logger.info("-" * 72)
    logger.info("测试集结果（%d 个样本）", metrics["n_samples"])
    logger.info("  Accuracy      : %.4f", metrics["accuracy"])
    logger.info("  Macro F1      : %.4f   ← 主指标", metrics["macro_f1"])
    logger.info("  Weighted F1   : %.4f", metrics["weighted_f1"])
    logger.info("  错分数        : %d / %d", metrics["n_errors"], metrics["n_samples"])
    logger.info("  平均置信度    : 全部 %.4f | 正确 %.4f | 错误 %s",
                metrics["mean_confidence"],
                metrics["mean_confidence_correct"] or float("nan"),
                f"{metrics['mean_confidence_wrong']:.4f}"
                if metrics["mean_confidence_wrong"] is not None else "n/a")
    logger.info("-" * 72)
    logger.info("  每类指标（IC3）：")
    logger.info("    %-12s %10s %10s %10s %8s", "类别", "Precision", "Recall", "F1", "Support")
    for name in class_names:
        m = metrics["per_class"][name]
        logger.info("    %-12s %10.4f %10.4f %10.4f %8d",
                    name, m["precision"], m["recall"], m["f1"], m["support"])
    logger.info("-" * 72)
    logger.info("  混淆矩阵（行=真实，列=预测；标签顺序 %s）：", class_names)
    for i, row in enumerate(metrics["confusion_matrix"]):
        logger.info("    %-12s %s", class_names[i], " ".join(f"{v:5d}" for v in row))
    em = metrics["error_analysis"]["error_class_pairs"]
    if em:
        logger.info("  主要错分方向：%s", em)
    else:
        logger.info("  本次无错分样本")
    logger.info("-" * 72)
    logger.info("产物：test_metrics.json / predictions.csv / errors.csv / uncertain.csv / %s",
                LEDGER_NAME)
    logger.info("=" * 72)
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description="测试集评估（唯一允许使用测试集的地方）")
    ap.add_argument("--run", required=True, help="运行目录（含 best_model.pt）")
    ap.add_argument("--config", default=None, help="配置文件（默认用运行目录里的 config_resolved.yaml）")
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    args = ap.parse_args()
    evaluate_run(args.run, args.config, args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
