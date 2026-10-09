"""
图表生成：训练曲线 / 混淆矩阵 / 每类指标 / 错误样例 / 数据探索。

对应约束：
  G5  —— 所有图表**由脚本生成**，数值全部来自产物文件（history.csv / test_metrics.json /
         predictions.csv / errors.csv），不在代码里硬编码任何数字
  G7  —— 每张图对应正文一个具体问题；不做装饰图
  IC3 —— 混淆矩阵（原始 + 归一化）、每类 P/R/F1
  IC4 —— 正确 / 错误 / 高不确定性样例可视化；每张样例标注 真实/预测/置信度/文件名
  G6  —— 错误样例**全部**导出，不做筛选

图内文字为英文（避免缺字），但已配置中文字体以防意外。

用法：
    python src/visualize.py --run results/raw/formal/small_cnn_seed42_20261008-120000
    python src/visualize.py --run <RUN> --fig eda          # 只出数据探索图
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (detect_run_mode, get_logger, load_config, load_json,  # noqa: E402
                   save_figure, setup_matplotlib)

setup_matplotlib()
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402


# ----------------------------------------------------------------------------
#  读取产物
# ----------------------------------------------------------------------------
def read_csv_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def cast_prediction_rows(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        out.append(
            {
                "relpath": r["relpath"],
                "true_label": int(r["true_label"]),
                "true_name": r["true_name"],
                "pred_label": int(r["pred_label"]),
                "pred_name": r["pred_name"],
                "confidence": float(r["confidence"]),
                "correct": r["correct"].lower() in {"true", "1", "yes"},
            }
        )
    return out


# ----------------------------------------------------------------------------
#  fig1：训练曲线
# ----------------------------------------------------------------------------
def plot_training_curves(history: list[dict], outdir: Path, prefix: str, name_tag: str,
                         model: str, seed: int, logger) -> list[Path]:
    ep = [int(h["epoch"]) for h in history]

    # --- fig1a：loss ---
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(ep, [float(h["train_loss"]) for h in history], "o-", label="Train loss", lw=2, ms=5)
    ax.plot(ep, [float(h["val_loss"]) for h in history], "s--", label="Val loss", lw=2, ms=5)
    best_ep = min(ep, key=lambda e: float(history[e - 1]["val_loss"]))
    ax.axvline(best_ep, color="green", ls=":", lw=1.4, alpha=0.8)
    ax.annotate(f"min val loss @ epoch {best_ep}", xy=(best_ep, float(history[best_ep - 1]["val_loss"])),
                xytext=(6, 12), textcoords="offset points", color="green", fontsize=9)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(f"{model} (seed={seed}) — Loss: train vs val\n"
                 f"[which question] is the model overfitting, and from which epoch?")
    ax.legend(); ax.grid(alpha=0.3)
    files = save_figure(fig, outdir, f"{prefix}fig1a_train_curve_{name_tag}")
    plt.close(fig)

    # --- fig1b：accuracy / macro-F1 ---
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(ep, [float(h["train_acc"]) for h in history], "o-", label="Train acc", lw=2, ms=5)
    ax.plot(ep, [float(h["val_acc"]) for h in history], "s--", label="Val acc", lw=2, ms=5)
    ax.plot(ep, [float(h["val_macro_f1"]) for h in history], "^-.", label="Val macro-F1", lw=2, ms=5)
    best_f1_ep = max(ep, key=lambda e: float(history[e - 1]["val_macro_f1"]))
    ax.axvline(best_f1_ep, color="green", ls=":", lw=1.4, alpha=0.8)
    ax.annotate(f"best val macro-F1 @ epoch {best_f1_ep}", xy=(best_f1_ep, float(history[best_f1_ep - 1]["val_macro_f1"])),
                xytext=(6, -18), textcoords="offset points", color="green", fontsize=9)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{model} (seed={seed}) — Accuracy & Macro-F1\n"
                 f"[which question] does accuracy hide what macro-F1 reveals?")
    ax.legend(); ax.grid(alpha=0.3)
    files += save_figure(fig, outdir, f"{prefix}fig1b_metric_curve_{name_tag}")
    plt.close(fig)

    logger.info("  训练曲线: %d 轮，最佳 macro-F1 在第 %d 轮", len(history), best_f1_ep)
    return files


# ----------------------------------------------------------------------------
#  fig3：混淆矩阵与每类指标
# ----------------------------------------------------------------------------
def plot_confusion(cm: list[list[float]], class_names: list[str], outdir: Path, prefix: str,
                   normalized: bool, logger) -> list[Path]:
    mat = np.array(cm, dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max() if mat.max() > 0 else 1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True (ground truth)")
    fmt = "{:.2f}" if normalized else "{:d}"
    thresh = mat.max() * 0.55 if mat.max() > 0 else 0.5
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, fmt.format(v if normalized else int(v)), ha="center", va="center",
                    color="white" if v > thresh else "black", fontsize=9)
    tag = "row-normalised" if normalized else "raw counts"
    ax.set_title(f"Confusion matrix ({tag}) — test set, n={int(mat.sum())}\n"
                 f"[which question] which class pairs get confused, and how badly?")
    suffix = "_norm" if normalized else ""
    files = save_figure(fig, outdir, f"{prefix}fig3{'b' if normalized else 'a'}_confusion_matrix{suffix}")
    plt.close(fig)
    return files


def plot_per_class_metrics(per_class: dict, class_names: list[str], outdir: Path,
                           prefix: str, logger) -> list[Path]:
    x = np.arange(len(class_names))
    w = 0.26
    p = [per_class[c]["precision"] for c in class_names]
    r = [per_class[c]["recall"] for c in class_names]
    f = [per_class[c]["f1"] for c in class_names]

    # --- fig3c：分组柱状图 ---
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.bar(x - w, p, w, label="Precision")
    ax.bar(x, r, w, label="Recall")
    ax.bar(x + w, f, w, label="F1")
    for xi, vals in zip(x, zip(p, r, f)):
        for off, v in zip((-w, 0, w), vals):
            ax.text(xi + off, v + 0.02, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_ylim(0, 1.12); ax.set_ylabel("Score")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.set_title("[which question] which class is the weak spot, and is it precision or recall?")
    files = save_figure(fig, outdir, f"{prefix}fig3c_per_class_metrics")
    plt.close(fig)

    # --- fig3d：F1 排序 ---
    order = sorted(range(len(class_names)), key=lambda i: f[i])
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(class_names) + 2.0))
    colors = ["#d62728" if f[i] == min(f) else ("#2ca02c" if f[i] == max(f) else "#1f77b4") for i in order]
    ax.barh([class_names[i] for i in order], [f[i] for i in order], color=colors)
    for k, i in enumerate(order):
        ax.text(f[i] + 0.01, k, f"{f[i]:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.1); ax.set_xlabel("F1 (test set)")
    ax.set_title("[which question] which class has the lowest F1?\n(red = worst, green = best)")
    ax.grid(axis="x", alpha=0.3)
    files += save_figure(fig, outdir, f"{prefix}fig3d_per_class_f1_rank")
    plt.close(fig)

    logger.info("  每类 F1: %s", {c: round(per_class[c]["f1"], 4) for c in class_names})
    return files


# ----------------------------------------------------------------------------
#  fig4：样例网格
# ----------------------------------------------------------------------------
def plot_sample_grid(rows: list[dict], image_root: Path, outdir: Path, prefix: str, fig_name: str,
                     title: str, ncols: int = 6, max_items: int = 36, logger=None) -> list[Path]:
    rows = rows[:max_items]
    if not rows:
        if logger:
            logger.warning("  %s: 没有可展示的样例，跳过", fig_name)
        return []

    n = len(rows)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.05, nrows * 2.65))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.axis("off")

    for ax, r in zip(axes, rows):
        try:
            with Image.open(image_root / r["relpath"]) as im:
                ax.imshow(np.asarray(im.convert("L")), cmap="gray")
        except (OSError, FileNotFoundError):
            ax.text(0.5, 0.5, "load failed", ha="center", va="center")
        color = "#1a7f37" if r["correct"] else "#d1242f"
        ax.set_title(
            f"{r['true_name']} → {r['pred_name']}\np={r['confidence']:.2f}\n{Path(r['relpath']).name}",
            color=color, fontsize=6.6,
        )
        for sp in ax.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(1.6)

    fig.suptitle(title, fontsize=10, y=1.0)
    fig.tight_layout()
    files = save_figure(fig, outdir, f"{prefix}{fig_name}")
    plt.close(fig)
    return files


# ----------------------------------------------------------------------------
#  fig0：数据探索（读取审计结果，不重新扫描）
# ----------------------------------------------------------------------------
def plot_eda(audit: dict, image_root: Path, outdir: Path, prefix: str, logger,
             n_samples: int = 12) -> list[Path]:
    counts = audit.get("class_counts_after_dedup") or audit.get("class_counts_raw") or {}
    raw = audit.get("class_counts_raw") or {}
    names = sorted(counts)
    w = audit.get("width", {}); h = audit.get("height", {})

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2))

    ax = axes[0, 0]
    ax.bar(names, [counts[c] for c in names], color="#4c78a8")
    for i, c in enumerate(names):
        ax.text(i, counts[c] + 4, str(counts[c]), ha="center", fontsize=8)
    ax.axhline(np.mean(list(counts.values())), color="red", ls="--", lw=1.2, label="mean")
    ax.set_ylim(0, max(counts.values()) * 1.18)
    ax.set_title("[which question] are the classes balanced?", fontsize=9.5)
    ax.set_ylabel("Images (after dedup)"); ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=25)

    ax = axes[0, 1]
    txt = (f"scanned: {audit.get('n_images_scanned')}\n"
           f"after dedup: {audit.get('n_after_dedup')}\n"
           f"dropped: {audit.get('n_dropped')}\n"
           f"corrupted: {audit.get('n_corrupted')}\n"
           f"colour modes: {audit.get('modes')}\n"
           f"unique sizes: {audit.get('n_unique_sizes')}\n"
           f"width  {w.get('min')}~{w.get('max')} (mean {w.get('mean', 0):.1f})\n"
           f"height {h.get('min')}~{h.get('max')} (mean {h.get('mean', 0):.1f})")
    ax.axis("off")
    ax.text(0.02, 0.95, txt, va="top", family="monospace", fontsize=9.2)
    ax.set_title("[which question] what does the dataset actually look like?", fontsize=9.5)

    ax = axes[1, 0]
    if raw:
        ax.bar(names, [raw.get(c, 0) for c in names], alpha=0.45, color="#999999", label="raw")
        ax.bar(names, [counts[c] for c in names], color="#4c78a8", label="after dedup")
        ax.legend(fontsize=8)
    ax.set_title("[which question] how many duplicates were removed per class?", fontsize=9.5)
    ax.set_ylabel("Images"); ax.tick_params(axis="x", rotation=25)

    ax = axes[1, 1]
    tiles, labels = [], []
    for c in names:
        files = sorted((image_root / c).glob("*.jpg"))
        if not files:
            continue
        try:
            with Image.open(files[0]) as im:
                tiles.append(np.asarray(im.convert("L").resize((64, 64)), dtype=float))
            labels.append(c)
        except OSError:
            continue
    if tiles:
        gap = np.full((64, 4), 255.0)
        row = []
        for i, t in enumerate(tiles):
            row.append(t)
            if i < len(tiles) - 1:
                row.append(gap)
        ax.imshow(np.concatenate(row, axis=1), cmap="gray")
        ax.set_xticks([32 + i * 68 for i in range(len(tiles))])
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
        ax.set_yticks([])
    else:
        ax.axis("off")
    ax.set_title("[which question] what do the images look like? (one per class, resized 64×64)",
                 fontsize=9.5)

    fig.suptitle("SAR-ACD dataset — exploratory summary (all numbers from results/raw/audit/data_audit.json)",
                 fontsize=10.5)
    fig.tight_layout()
    files = save_figure(fig, outdir, f"{prefix}fig0_data_eda")
    plt.close(fig)
    logger.info("  数据探索图已生成")
    return files


# ----------------------------------------------------------------------------
#  fig5：Grad-CAM 渲染（数据来自 src/gradcam.py 的产物）
# ----------------------------------------------------------------------------
def _cam_grid(grays: np.ndarray, cams: np.ndarray, metas: list[dict], outdir: Path,
              fig_name: str, title: str, max_rows: int, logger) -> list[Path]:
    n = min(len(metas), max_rows)
    if n == 0:
        return []
    fig, axes = plt.subplots(n, 3, figsize=(7.4, 2.3 * n), squeeze=False)
    for row in range(n):
        ax0, ax1, ax2 = axes[row]
        for ax in (ax0, ax1, ax2):
            ax.axis("off")
        ax0.imshow(grays[row], cmap="gray"); ax0.set_title("input (grayscale)", fontsize=7.5)
        ax1.imshow(cams[row], cmap="jet"); ax1.set_title("Grad-CAM", fontsize=7.5)
        ax2.imshow(grays[row], cmap="gray"); ax2.imshow(cams[row], cmap="jet", alpha=0.45)
        ax2.set_title("overlay", fontsize=7.5)
        m = metas[row]
        color = "#1a7f37" if m["correct"] else "#d1242f"
        ax0.text(-0.06, 0.5,
                 f"{m['true_name']} → {m['pred_name']}\np={m['confidence']:.2f}\n"
                 f"border={m['border_mass']:.2f}\n{Path(m['relpath']).name}",
                 transform=ax0.transAxes, ha="right", va="center", fontsize=6.5, color=color)
    extra = "" if len(metas) <= max_rows else f"  (showing {n} of {len(metas)})"
    fig.suptitle(f"{title}{extra}", fontsize=9.5, y=1.0)
    fig.tight_layout()
    files = save_figure(fig, outdir, fig_name)
    plt.close(fig)
    return files


def _ablation_fig(ab: dict, outdir: Path, fig_name: str) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    labels = ["full image\n(baseline)",
              "border filled\n(target kept)  ← decisive",
              "centre cropped\n(target may be cut)"]
    vals = [ab["baseline_accuracy"], ab["border_filled_accuracy"], ab["center_cropped_accuracy"]]
    bars = ax.bar(labels, vals, color=["#4c78a8", "#54a24b", "#f58518"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.4f}", ha="center", fontsize=10)
    top = max(vals) if max(vals) > 0 else 1.0
    ax.set_ylim(0, min(1.0, top * 1.32))
    ax.set_ylabel("Test accuracy")
    ax.set_title("[which question] does the model look at the aircraft or at the background?\n"
                 f"Δ(fill) = {ab['delta_fill']:+.4f}   Δ(crop) = {ab['delta_crop']:+.4f}\n"
                 f"{ab['verdict']}", fontsize=8.5)
    ax.grid(axis="y", alpha=0.3)
    files = save_figure(fig, outdir, fig_name)
    plt.close(fig)
    return files


def render_cam(run_dir: Path, outdir: Path, prefix: str, logger, max_rows: int = 16) -> list[Path]:
    """渲染 fig5a–fig5d。数据来源：src/gradcam.py 产出的 cam_data.npz + cam_manifest.json。"""
    cam_npz = run_dir / "cam_data.npz"
    cam_manifest = run_dir / "cam_manifest.json"
    if not cam_npz.is_file() or not cam_manifest.is_file():
        logger.warning("未找到 cam_data.npz / cam_manifest.json，请先运行 src/gradcam.py；跳过 fig5")
        return []

    data = np.load(cam_npz)
    man = load_json(cam_manifest)
    groups = man.get("groups", {})
    written: list[Path] = []

    specs = [
        ("correct", "fig5a_gradcam_correct",
         "[which question] on correct samples, does the model look at the target?"),
        ("wrong", "fig5b_gradcam_wrong",
         "[which question] where does the model look when it is wrong?"),
        ("background", "fig5c_gradcam_background",
         "[which question] which CORRECT predictions have their CAM on the background? "
         "(evidence for shortcut learning / hypothesis H3)"),
    ]
    for gname, figname, title in specs:
        metas = groups.get(gname, [])
        key_g, key_c = f"{gname}_gray", f"{gname}_cam"
        if not metas or key_g not in data:
            logger.warning("  %s: 无样例，跳过", figname)
            continue
        written += _cam_grid(data[key_g], data[key_c], metas, outdir,
                             f"{prefix}{figname}", title, max_rows, logger)
        logger.info("  %-40s %d 张", figname, min(len(metas), max_rows))

    ab = man.get("background_ablation", {}) or {}
    if ab.get("enabled"):
        written += _ablation_fig(ab, outdir, f"{prefix}fig5d_background_ablation")
        logger.info("  %-40s Δfill=%+.4f  Δcrop=%+.4f",
                    "fig5d_background_ablation", ab["delta_fill"], ab["delta_crop"])
    return written


# ----------------------------------------------------------------------------
#  主流程
# ----------------------------------------------------------------------------
def resolve_outdir(cfg: dict, run_dir: Path, mode: str) -> tuple[Path, str]:
    """冒烟运行的图写回自己的目录；正式运行的图进入项目 figures/ 并加运行名前缀。"""
    if mode == "formal":
        out = Path(cfg["_project_root"]) / cfg["paths"]["figures"]
        return out, f"{run_dir.name}__"
    return run_dir / "figures", ""


def main() -> int:
    ap = argparse.ArgumentParser(description="生成图表（数值全部来自产物文件）")
    ap.add_argument("--run", required=True, help="运行目录")
    ap.add_argument("--outdir", default=None, help="输出目录（默认按 mode 自动决定）")
    ap.add_argument("--fig", default="all", choices=["all", "curves", "metrics", "samples", "eda", "cam"])
    args = ap.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"运行目录不存在: {run_dir}")

    logger = get_logger(f"visualize.{run_dir.name}", run_dir / "visualize.log")

    metrics_path = run_dir / "test_metrics.json"
    history_path = run_dir / "history.csv"
    if not history_path.is_file():
        raise FileNotFoundError(f"缺少 {history_path}，请先运行 train.py")

    # mode 判定（权威来源是 train.py 的 metrics.json，见 utils.detect_run_mode）
    mode = detect_run_mode(run_dir)

    cfg_path = run_dir / "config_resolved.yaml"
    cfg = load_config(cfg_path) if cfg_path.is_file() else {"paths": {"figures": "figures"},
                                                            "_project_root": "."}
    project_root = Path(cfg.get("_project_root", "."))
    if not (project_root / "data").is_dir():
        project_root = run_dir
        while project_root != project_root.parent and not (project_root / "results").is_dir():
            project_root = project_root.parent
    image_root = project_root / cfg["paths"]["data_root"]
    # 图文件格式统一为 pdf + png（见 configs/default.yaml 的 output.figures_format）

    outdir, prefix = resolve_outdir(cfg, run_dir, mode)
    if args.outdir:
        outdir, prefix = Path(args.outdir), ""

    logger.info("=" * 72)
    logger.info("生成图表: %s", run_dir.name)
    logger.info("模式      : %s | 输出目录: %s", mode, outdir)
    if mode == "smoke":
        logger.warning("冒烟运行的图写入自身目录，**不会进入项目的 figures/**")

    history = read_csv_rows(history_path)
    model = (load_json(metrics_path).get("model") if metrics_path.is_file()
             else cfg.get("models", [{}])[0].get("name", "model"))
    seed = (load_json(metrics_path).get("seed") if metrics_path.is_file() else 0)
    name_tag = f"{model}_seed{seed}"
    written: list[Path] = []

    # ---- fig1 曲线 ----
    if args.fig in ("all", "curves"):
        logger.info("-" * 72); logger.info("fig1 训练曲线")
        written += plot_training_curves(history, outdir, prefix, name_tag, model, seed, logger)

    # ---- fig3 指标 ----
    if args.fig in ("all", "metrics"):
        logger.info("-" * 72); logger.info("fig3 混淆矩阵与每类指标")
        if not metrics_path.is_file():
            logger.warning("  未找到 test_metrics.json，请先运行 evaluate.py；跳过 fig3/fig4")
        else:
            m = load_json(metrics_path)
            class_names = [m["label_mapping"][str(i)] for i in range(len(m["label_mapping"]))]
            written += plot_confusion(m["confusion_matrix"], class_names, outdir, prefix, False, logger)
            written += plot_confusion(m["confusion_matrix_normalized"], class_names, outdir, prefix, True, logger)
            written += plot_per_class_metrics(m["per_class"], class_names, outdir, prefix, logger)

    # ---- fig4 样例 ----
    if args.fig in ("all", "samples") and metrics_path.is_file():
        logger.info("-" * 72); logger.info("fig4 错误样例可视化")
        preds = cast_prediction_rows(read_csv_rows(run_dir / "predictions.csv"))
        errors = cast_prediction_rows(read_csv_rows(run_dir / "errors.csv"))
        uncertain = cast_prediction_rows(read_csv_rows(run_dir / "uncertain.csv"))
        if not preds:
            logger.warning("  未找到 predictions.csv，请先运行 evaluate.py；跳过 fig4")
        else:
            correct = [r for r in preds if r["correct"]]
            per_cls = defaultdict(list)
            for r in correct:
                per_cls[r["true_name"]].append(r)
            for v in per_cls.values():
                v.sort(key=lambda r: -r["confidence"])
            n_per = int((cfg.get("error_analysis", {}) or {}).get("export_correct_per_class", 6))
            correct_sel = [r for c in sorted(per_cls) for r in per_cls[c][:n_per]]

            written += plot_sample_grid(correct_sel, image_root, outdir, prefix, "fig4a_correct_samples",
                                        "[which question] what do confidently-correct samples look like?", logger=logger)
            written += plot_sample_grid(
                errors, image_root, outdir, prefix, "fig4b_wrong_samples",
                f"[which question] every misclassified test sample (total {len(errors)}) — all kept, none filtered",
                max_items=max(len(errors), 1), logger=logger)
            written += plot_sample_grid(
                uncertain[:24], image_root, outdir, prefix, "fig4c_uncertain_samples",
                "[which question] where is the model unsure? (lowest max-softmax)", logger=logger)

            pairs = defaultdict(list)
            for r in errors:
                pairs[(r["true_name"], r["pred_name"])].append(r)
            pair_rows = [r for k in sorted(pairs, key=lambda k: -len(pairs[k]))
                         for r in pairs[k][:6]]
            if pair_rows:
                written += plot_sample_grid(
                    pair_rows, image_root, outdir, prefix, "fig4d_confusion_pairs",
                    "[which question] what do the main confusion pairs look like?", logger=logger)

    # ---- fig0 数据探索 ----
    if args.fig in ("all", "eda"):
        audit_path = project_root / "results" / "raw" / "audit" / "data_audit.json"
        if audit_path.is_file():
            logger.info("-" * 72); logger.info("fig0 数据探索")
            written += plot_eda(load_json(audit_path), image_root, outdir, prefix, logger)
        else:
            logger.warning("未找到 %s，跳过 fig0（可先运行 src/dataset.py）", audit_path)

    # ---- fig5 Grad-CAM ----
    if args.fig in ("all", "cam"):
        logger.info("-" * 72); logger.info("fig5 Grad-CAM 可解释性（数据来自 gradcam.py）")
        written += render_cam(run_dir, outdir, prefix, logger)

    logger.info("=" * 72)
    logger.info("共生成 %d 个图文件：", len(written))
    for p in written:
        logger.info("  %-64s %7.1f KB", p.name, p.stat().st_size / 1024)
    logger.info("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
