"""
模型对比：把两个（或多个）模型的汇总结果放在一起比较。

对应约束：
  G3  —— **校验两个模型用的是同一份数据划分**（比较 summary.json 里的划分哈希），
         不一致直接报错 —— 否则"对比"没有意义
  G4  —— 按 experiment-plan.md 第 6.3 节的**稳健性判据**判断差异是否显著：
         均值差必须大于"两个模型中较大的那个标准差"，才可称"更好"；
         否则如实写"差异在随机波动范围内"
  G5  —— 所有数字来自各模型的 summary.json，表格与图由脚本生成
  G7  —— 每张图对应一个具体问题

产物（默认写到 results/summary/comparison_<A>_vs_<B>/）：
    comparison.md / comparison.json
    figures/fig6a_model_comparison.*        总体指标均值±标准差
    figures/fig6b_per_class_f1.*            逐类别 F1 对比
    figures/fig6c_per_class_delta.*         逐类别差值（B−A）
    figures/fig6d_confusion_side_by_side.*  两个模型的平均混淆矩阵（归一化）
    figures/fig6e_confusion_delta.*         混淆矩阵差值（B−A）
    figures/fig6f_per_seed.*                逐种子散点（让波动可见）

用法：
    python src/compare_models.py \
        --a results/summary/cnn_baseline --b results/summary/resnet18_transfer \
        --label-a "CNN 基线" --label-b "ResNet18 迁移学习"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger, load_json, save_figure, save_json, setup_matplotlib  # noqa: E402

setup_matplotlib()
import matplotlib.pyplot as plt  # noqa: E402


# ============================================================================
#  读取
# ============================================================================
def load_summary(path: Path) -> dict:
    p = Path(path)
    if p.is_dir():
        p = p / "summary.json"
    if not p.is_file():
        raise FileNotFoundError(f"找不到汇总文件: {p}（请先运行 src/summarize.py）")
    s = load_json(p)
    s["_path"] = str(p)
    return s


def verify_same_split(a: dict, b: dict, logger) -> dict:
    """G3：两个模型必须用同一份划分，否则不可比。"""
    ha = (a.get("split_consistency") or {}).get("split_hash")
    hb = (b.get("split_consistency") or {}).get("split_hash")
    logger.info("划分一致性校验（G3）：")
    logger.info("    A = %s", (ha or "缺失")[:16])
    logger.info("    B = %s", (hb or "缺失")[:16])
    if not ha or not hb:
        raise AssertionError("有汇总文件缺少 split_consistency.split_hash，无法确认可比性")
    if ha != hb:
        raise AssertionError(
            "❌ 两个模型使用了**不同的数据划分**，不可比较！\n"
            f"    A={ha}\n    B={hb}\n"
            "    请确认两者的 split.seed 相同。"
        )
    if a["class_names"] != b["class_names"]:
        raise AssertionError(f"类别不一致: {a['class_names']} vs {b['class_names']}")
    logger.info("  ✅ 同一份划分、同一套类别 —— 对比成立")
    return {"passed": True, "split_hash": ha}


# ============================================================================
#  统计与判定
# ============================================================================
def compare_overall(a: dict, b: dict, key: str) -> dict:
    ma, sa = a[key]["mean"], a[key]["std"]
    mb, sb = b[key]["mean"], b[key]["std"]
    delta = mb - ma
    # 稳健性判据（experiment-plan.md 第 6.3 节）：差值需大于两者中较大的标准差
    threshold = max(sa, sb)
    significant = abs(delta) > threshold
    return {
        "a_mean": ma, "a_std": sa, "b_mean": mb, "b_std": sb,
        "delta": delta, "threshold": threshold, "significant": bool(significant),
        "verdict": ("B 显著更好" if (significant and delta > 0) else
                    "A 显著更好" if (significant and delta < 0) else
                    "差异在随机波动范围内（不显著）"),
    }


def compare_per_class(a: dict, b: dict, metric: str = "f1") -> dict:
    out = {}
    for c in a["class_names"]:
        da, db = a["per_class"][c][metric], b["per_class"][c][metric]
        out[c] = {
            "a_mean": da["mean"], "a_std": da["std"],
            "b_mean": db["mean"], "b_std": db["std"],
            "delta": db["mean"] - da["mean"],
            "threshold": max(da["std"], db["std"]),
            # 注意：support 在「类别」这一层，不在「指标」层
            "support": a["per_class"][c].get("support"),
        }
    return out


# ============================================================================
#  表格
# ============================================================================
def build_markdown(a: dict, b: dict, la: str, lb: str, overall: dict, per_class: dict,
                   cm_a: np.ndarray, cm_b: np.ndarray) -> str:
    md: list[str] = []
    md.append(f"# 模型对比：{la} vs {lb}\n")
    md.append(f"- **{la}**：`{a['model']}`，{a['n_runs']} 个种子 {a['seeds']}，"
              f"测试集 {a.get('n_test', 446)} 个样本")
    md.append(f"- **{lb}**：`{b['model']}`，{b['n_runs']} 个种子 {b['seeds']}")
    md.append("- 数据划分：**同一份**（已校验 `split_hash` 一致）")
    md.append("- 该对比为**受控对比**：除模型本身外，去重规则/划分/输入尺寸/归一化/增强/轮数/batch/种子全部相同\n")

    md.append("## 一、总体指标对比（测试集）\n")
    md.append(f"| 指标 | {la} | {lb} | 差值（{lb} − {la}） | 波动阈值 | 判定 |")
    md.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for key, label in (("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("weighted_f1", "Weighted F1")):
        d = overall[key]
        mark = "**" if key == "macro_f1" else ""
        md.append(f"| {mark}{label}{mark} | {d['a_mean']:.4f} ± {d['a_std']:.4f} "
                  f"| {d['b_mean']:.4f} ± {d['b_std']:.4f} "
                  f"| **{d['delta']:+.4f}** | {d['threshold']:.4f} | {d['verdict']} |")
    md.append("")
    md.append("> **判定依据**（experiment-plan.md 第 6.3 节）：均值差需大于"
              "「两个模型中较大的标准差」才可视为超出随机波动。\n")

    md.append("## 二、逐类别 F1 对比\n")
    md.append(f"| 类别 | Support | {la} F1 | {lb} F1 | 差值 | 更好的 |")
    md.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for c in sorted(per_class, key=lambda k: -per_class[k]["delta"]):
        d = per_class[c]
        better = lb if d["delta"] > 0 else la
        md.append(f"| {c} | {d['support']} | {d['a_mean']:.4f} ± {d['a_std']:.4f} "
                  f"| {d['b_mean']:.4f} ± {d['b_std']:.4f} | {d['delta']:+.4f} | {better} |")
    md.append("")

    md.append("## 三、平均混淆矩阵对比（按行归一化，行=真实，列=预测）\n")
    names = a["class_names"]
    for label, cm in ((la, cm_a), (lb, cm_b)):
        md.append(f"**{label}**\n")
        md.append("| 真实\\预测 | " + " | ".join(names) + " |")
        md.append("| --- | " + " | ".join(["---:"] * len(names)) + " |")
        for i, c in enumerate(names):
            md.append(f"| **{c}** | " + " | ".join(f"{v:.2f}" for v in cm[i]) + " |")
        md.append("")

    md.append(f"**差值矩阵（{lb} − {la}）**：正值表示 {lb} 在该格更「重」，"
              f"对角线上为正即 {lb} 对该类识别更准\n")
    delta = cm_b - cm_a
    md.append("| 真实\\预测 | " + " | ".join(names) + " |")
    md.append("| --- | " + " | ".join(["---:"] * len(names)) + " |")
    for i, c in enumerate(names):
        md.append(f"| **{c}** | " + " | ".join(f"{v:+.3f}" for v in delta[i]) + " |")
    md.append("")

    md.append("## 四、结论\n")
    d = overall["macro_f1"]
    md.append(f"- **主指标 Macro F1**：{la} = {d['a_mean']:.4f}，{lb} = {d['b_mean']:.4f}，"
              f"差值 **{d['delta']:+.4f}**")
    md.append(f"- 波动阈值（两者较大标准差）= {d['threshold']:.4f}")
    md.append(f"- **判定：{d['verdict']}**")
    if d["significant"] and d["delta"] > 0:
        md.append(f"\n> {lb} 在 Macro F1 上比 {la} 高 **{d['delta']*100:.2f} 个百分点**"
                  f"（相对提升 {d['delta']/d['a_mean']*100:.1f}%），且超出种子波动范围。")
    elif not d["significant"]:
        md.append(f"\n> 两者差异（{d['delta']:+.4f}）**未超过种子波动**（{d['threshold']:.4f}），"
                  "不能断言谁更好。")
    md.append("")
    return "\n".join(md)


# ============================================================================
#  图
# ============================================================================
def plot_figures(a: dict, b: dict, la: str, lb: str, overall: dict, per_class: dict,
                 cm_a: np.ndarray, cm_b: np.ndarray, figdir: Path, prefix: str, logger) -> list[Path]:
    names = a["class_names"]
    written: list[Path] = []

    # ---- fig6a：总体指标（问答：哪个模型好、好多少、稳不稳）----
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    metrics = [("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("weighted_f1", "Weighted F1")]
    x = np.arange(len(metrics)); w = 0.34
    ma = [overall[k]["a_mean"] for k, _ in metrics]
    sa = [overall[k]["a_std"] for k, _ in metrics]
    mb = [overall[k]["b_mean"] for k, _ in metrics]
    sb = [overall[k]["b_std"] for k, _ in metrics]
    b1 = ax.bar(x - w / 2, ma, w, yerr=sa, capsize=6, label=la, color="#4c78a8",
                error_kw={"elinewidth": 1.5, "ecolor": "#333"})
    b2 = ax.bar(x + w / 2, mb, w, yerr=sb, capsize=6, label=lb, color="#f58518",
                error_kw={"elinewidth": 1.5, "ecolor": "#333"})
    for bars, vals, stds in ((b1, ma, sa), (b2, mb, sb)):
        for bar, v, s in zip(bars, vals, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, v + s + 0.012, f"{v:.4f}",
                    ha="center", fontsize=8.5)
    for i, (k, _) in enumerate(metrics):
        d = overall[k]
        ax.annotate(f"{d['delta']:+.4f}\n({'显著' if d['significant'] else '不显著'})",
                    (i, max(ma[i], mb[i]) + max(sa[i], sb[i]) + 0.075), ha="center", fontsize=8,
                    color="#1a7f37" if d["significant"] else "#8a6d3b")
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in metrics])
    top = max(max(np.array(ma) + np.array(sa)), max(np.array(mb) + np.array(sb)))
    ax.set_ylim(0, min(1.0, top * 1.42))
    ax.set_ylabel(f"Test score (mean ± std over {a['n_runs']} seeds)")
    ax.legend()
    ax.set_title("[which question] which model is better, by how much, and is it beyond seed noise?",
                 fontsize=9.5)
    ax.grid(axis="y", alpha=0.3)
    written += save_figure(fig, figdir, f"{prefix}fig6a_model_comparison")
    plt.close(fig)

    # ---- fig6b：逐类别 F1 对比 ----
    order = sorted(names, key=lambda c: -per_class[c]["delta"])
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    fa = [per_class[c]["a_mean"] for c in order]
    sa_ = [per_class[c]["a_std"] for c in order]
    fb = [per_class[c]["b_mean"] for c in order]
    sb_ = [per_class[c]["b_std"] for c in order]
    ax.bar(x - w / 2, fa, w, yerr=sa_, capsize=5, label=la, color="#4c78a8",
           error_kw={"elinewidth": 1.2, "ecolor": "#333"})
    ax.bar(x + w / 2, fb, w, yerr=sb_, capsize=5, label=lb, color="#f58518",
           error_kw={"elinewidth": 1.2, "ecolor": "#333"})
    for i, c in enumerate(order):
        ax.text(i, max(fa[i], fb[i]) + 0.045, f"{per_class[c]['delta']:+.3f}",
                ha="center", fontsize=8,
                color="#1a7f37" if per_class[c]["delta"] > 0 else "#d1242f")
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylim(0, min(1.0, max(max(fa), max(fb)) * 1.35))
    ax.set_ylabel("F1 (test set)")
    ax.legend()
    ax.set_title("[which question] which classes does each model handle better?\n"
                 "(number above each pair = delta, green = B better)", fontsize=9.5)
    ax.grid(axis="y", alpha=0.3)
    written += save_figure(fig, figdir, f"{prefix}fig6b_per_class_f1")
    plt.close(fig)

    # ---- fig6c：逐类别差值条形图 ----
    fig, ax = plt.subplots(figsize=(7.6, 0.5 * len(order) + 2.4))
    deltas = [per_class[c]["delta"] for c in order]
    colors = ["#1a7f37" if d > 0 else "#d1242f" for d in deltas]
    ax.barh(order[::-1], deltas[::-1], color=colors[::-1])
    for i, c in enumerate(order[::-1]):
        d = per_class[c]["delta"]
        ax.text(d + (0.004 if d >= 0 else -0.004), i, f"{d:+.4f}", va="center",
                ha="left" if d >= 0 else "right", fontsize=9)
    ax.axvline(0, color="#333", lw=1)
    ax.set_xlabel(f"F1 delta  ({lb} − {la})")
    ax.set_title("[which question] where exactly does the difference come from?", fontsize=9.5)
    ax.grid(axis="x", alpha=0.3)
    written += save_figure(fig, figdir, f"{prefix}fig6c_per_class_delta")
    plt.close(fig)

    # ---- fig6d：两个模型的平均混淆矩阵并排（归一化）----
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6))
    for ax, label, cm in ((axes[0], la, cm_a), (axes[1], lb, cm_b)):
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if cm[i, j] > 0.55 else "black")
        ax.set_title(label, fontsize=10)
    fig.suptitle("[which question] are the error patterns different, or the same mistakes?\n"
                 "(row-normalised mean confusion matrix)", fontsize=10)
    fig.tight_layout()
    written += save_figure(fig, figdir, f"{prefix}fig6d_confusion_side_by_side")
    plt.close(fig)

    # ---- fig6e：混淆矩阵差值（对角线为正 = B 更准）----
    delta_cm = cm_b - cm_a
    lim = max(abs(delta_cm).max(), 1e-6)
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    im = ax.imshow(delta_cm, cmap="RdYlGn", vmin=-lim, vmax=lim)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right"); ax.set_yticklabels(names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{delta_cm[i, j]:+.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(f"[which question] which specific confusions did {lb} fix or worsen?\n"
                 f"green = better for {lb}, red = worse (diagonal = correctly classified)",
                 fontsize=9.5)
    written += save_figure(fig, figdir, f"{prefix}fig6e_confusion_delta")
    plt.close(fig)

    # ---- fig6f：逐种子散点 ----
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4))
    for ax, key, label in ((axes[0], "accuracy", "Accuracy"), (axes[1], "macro_f1", "Macro F1")):
        va, vb = a[key]["values"], b[key]["values"]
        ax.scatter([0] * len(va), va, s=75, color="#4c78a8", zorder=3, label=la)
        ax.scatter([1] * len(vb), vb, s=75, color="#f58518", zorder=3, label=lb)
        ax.hlines(np.mean(va), -0.2, 0.2, color="#4c78a8", lw=2)
        ax.hlines(np.mean(vb), 0.8, 1.2, color="#f58518", lw=2)
        for xv, vals, col in ((0, va, "#4c78a8"), (1, vb, "#f58518")):
            for v, sd in zip(vals, a["seeds"] if xv == 0 else b["seeds"]):
                ax.annotate(f"{v:.4f}", (xv, v), textcoords="offset points",
                            xytext=(9, -3), fontsize=7.5, color=col)
        ax.set_xticks([0, 1]); ax.set_xticklabels([la, lb])
        lo = min(min(va), min(vb)); hi = max(max(va), max(vb)); pad = max(0.02, (hi - lo) * 0.35)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel(label)
        ax.set_title(f"[which question] {label}: are the seed spreads overlapping?", fontsize=9.5)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    written += save_figure(fig, figdir, f"{prefix}fig6f_per_seed")
    plt.close(fig)

    logger.info("  生成 %d 个对比图到 %s", len(written), figdir)
    return written


# ============================================================================
#  主流程
# ============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="两个模型的对比汇总")
    ap.add_argument("--a", required=True, help="模型 A 的汇总目录或 summary.json")
    ap.add_argument("--b", required=True, help="模型 B 的汇总目录或 summary.json")
    ap.add_argument("--label-a", default=None, help="模型 A 的显示名")
    ap.add_argument("--label-b", default=None, help="模型 B 的显示名")
    ap.add_argument("--out", default=None, help="输出目录")
    ap.add_argument("--figure-dir", default=None, help="图输出目录（默认 <project>/figures）")
    args = ap.parse_args()

    A = load_summary(args.a)
    B = load_summary(args.b)
    la = args.label_a or A["model"]
    lb = args.label_b or B["model"]

    project_root = Path(A["_path"]).resolve()
    while project_root != project_root.parent and not (project_root / "results").is_dir():
        project_root = project_root.parent

    outdir = Path(args.out) if args.out else (
        project_root / "results" / "summary" / f"comparison_{A['model']}_vs_{B['model']}")
    outdir.mkdir(parents=True, exist_ok=True)
    figdir = Path(args.figure_dir) if args.figure_dir else (project_root / "figures")

    logger = get_logger("compare_models", outdir / "compare.log")
    logger.info("=" * 72)
    logger.info("模型对比：%s  vs  %s", la, lb)

    split_info = verify_same_split(A, B, logger)

    overall = {k: compare_overall(A, B, k)
               for k in ("accuracy", "macro_f1", "weighted_f1")}
    per_class = compare_per_class(A, B, "f1")
    cm_a = np.array(A["mean_confusion_matrix_normalized"], dtype=float)
    cm_b = np.array(B["mean_confusion_matrix_normalized"], dtype=float)

    md = build_markdown(A, B, la, lb, overall, per_class, cm_a, cm_b)
    (outdir / "comparison.md").write_text(md, encoding="utf-8")

    result = {
        "label_a": la, "label_b": lb,
        "model_a": A["model"], "model_b": B["model"],
        "seeds_a": A["seeds"], "seeds_b": B["seeds"],
        "split_consistency": split_info,
        "overall": overall,
        "per_class_f1": per_class,
        "confusion_matrix_normalized_a": cm_a.tolist(),
        "confusion_matrix_normalized_b": cm_b.tolist(),
        "confusion_matrix_delta": (cm_b - cm_a).tolist(),
        "class_names": A["class_names"],
        "verdict_macro_f1": overall["macro_f1"]["verdict"],
    }
    save_json(result, outdir / "comparison.json")
    logger.info("  写入 %s（comparison.md / comparison.json）", outdir)

    prefix = f"comparison_{A['model']}_vs_{B['model']}__"
    plot_figures(A, B, la, lb, overall, per_class, cm_a, cm_b, figdir, prefix, logger)

    logger.info("=" * 72)
    for k, label in (("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("weighted_f1", "Weighted F1")):
        d = overall[k]
        logger.info("  %-12s %s=%.4f±%.4f  %s=%.4f±%.4f  Δ=%+.4f (阈值 %.4f) → %s",
                    label, la, d["a_mean"], d["a_std"], lb, d["b_mean"], d["b_std"],
                    d["delta"], d["threshold"], d["verdict"])
    logger.info("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
