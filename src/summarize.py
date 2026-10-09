"""
多运行汇总：跨随机种子的均值与波动（G4）。

对应约束：
  G4  —— 报告**均值与波动**（标准差 + 逐种子明细），不得只保留最好的一次
  G3  —— **校验所有运行使用的是同一份数据划分**（比较 split_index.csv 的哈希），
         否则模型之间不可比，汇总无意义
  G5  —— 汇总表与图全部由脚本生成，数字来自各运行的 test_metrics.json / history.csv
  G7  —— 每张图对应一个具体问题

产物：
  results/summary/<name>/summary.json        完整汇总（含逐种子明细）
  results/summary/<name>/summary.md          人读版表格
  results/summary/<name>/per_seed.csv        逐种子一行
  results/summary/<name>/per_class.csv       逐类别均值/标准差
  results/summary/<name>/mean_confusion_matrix.csv
  figures/fig1c_train_curves_all_seeds.*     三种子的训练曲线叠加
  figures/fig2a_summary_metrics.*            Accuracy / Macro F1 均值±标准差
  figures/fig2b_per_seed_scatter.*           逐种子散点（让波动可见）
  figures/fig3a_mean_confusion_matrix.*      平均混淆矩阵（+ 归一化版）

用法：
    python src/summarize.py --runs results/raw/cnn_seed42 results/raw/cnn_seed123 \
        results/raw/cnn_seed2024 --name cnn_baseline
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (get_logger, load_json, save_figure, save_json,   # noqa: E402
                   setup_matplotlib)

setup_matplotlib()
import matplotlib.pyplot as plt  # noqa: E402


# ============================================================================
#  读取单个运行
# ============================================================================
def load_run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    tm_path = run_dir / "test_metrics.json"
    mt_path = run_dir / "metrics.json"
    if not tm_path.is_file():
        raise FileNotFoundError(
            f"{run_dir} 缺少 test_metrics.json —— 请先运行 src/evaluate.py --run {run_dir}"
        )
    if not mt_path.is_file():
        raise FileNotFoundError(f"{run_dir} 缺少 metrics.json —— 请确认这是 train.py 产生的运行目录")

    tm = load_json(tm_path)
    mt = load_json(mt_path)

    history = []
    hp = run_dir / "history.csv"
    if hp.is_file():
        with open(hp, encoding="utf-8", newline="") as f:
            history = [{k: (int(v) if k == "epoch" else float(v)) for k, v in row.items()}
                       for row in csv.DictReader(f)]

    split_hash = None
    sp = run_dir / "split_index.csv"
    if sp.is_file():
        split_hash = hashlib.md5(sp.read_bytes()).hexdigest()

    # 测试集评估次数：优先取指标文件；早期运行没写这个字段时，回退到台账（G2）
    n_eval = tm.get("n_test_evaluations")
    if n_eval is None:
        ledger = run_dir / "test_evaluation_ledger.json"
        if ledger.is_file():
            n_eval = load_json(ledger).get("n_evaluations")

    return {
        "run_dir": str(run_dir),
        "name": run_dir.name,
        "seed": tm.get("seed", mt.get("seed")),
        "model": tm.get("model", mt.get("model")),
        "mode": tm.get("mode", mt.get("mode")),
        "is_smoke": tm.get("is_smoke", mt.get("is_smoke")),
        "n_test_evaluations": n_eval,
        "accuracy": tm["accuracy"],
        "macro_f1": tm["macro_f1"],
        "weighted_f1": tm["weighted_f1"],
        "per_class": tm["per_class"],
        "label_mapping": tm["label_mapping"],
        "confusion_matrix": np.array(tm["confusion_matrix"], dtype=float),
        "confusion_matrix_normalized": np.array(tm["confusion_matrix_normalized"], dtype=float),
        "n_errors": tm["n_errors"],
        "n_samples": tm["n_samples"],
        "best_epoch": mt.get("best_epoch"),
        "epochs_run": mt.get("epochs_run"),
        "total_time_sec": mt.get("total_time_sec"),
        "n_params_total": mt.get("n_params_total"),
        "history": history,
        "split_hash": split_hash,
    }


# ============================================================================
#  G3：校验所有运行共用同一份划分
# ============================================================================
def verify_same_split(runs: list[dict], logger) -> dict:
    hashes = {r["name"]: r["split_hash"] for r in runs}
    missing = [k for k, v in hashes.items() if not v]
    if missing:
        raise AssertionError(f"以下运行缺少 split_index.csv，无法确认划分一致性: {missing}")
    uniq = set(hashes.values())
    logger.info("划分一致性校验（G3）：%d 个运行的 split_index.csv 哈希", len(runs))
    for name, h in hashes.items():
        logger.info("    %-24s %s", name, h[:16])
    if len(uniq) != 1:
        raise AssertionError(
            "❌ 汇总的各个运行使用了**不同的数据划分**，它们之间不可比！\n"
            f"   不同的划分哈希: {sorted(uniq)}\n"
            "   请确认所有运行共用同一个 split.seed（configs 里为 42）。"
        )
    logger.info("  ✅ 全部一致 —— 所有运行共用同一份划分，可以汇总比较")
    return {"passed": True, "split_hash": list(uniq)[0], "per_run": hashes}


def verify_same_dataset(runs: list[dict], logger) -> None:
    """同一个模型的多个种子应该来自同一次数据管线（类别数与测试集样本数一致）。"""
    n_samples = {r["n_samples"] for r in runs}
    classes = {tuple(sorted(r["label_mapping"].values())) for r in runs}
    if len(n_samples) != 1:
        raise AssertionError(f"各运行的测试集样本数不一致: {n_samples}")
    if len(classes) != 1:
        raise AssertionError(f"各运行的类别集合不一致: {classes}")
    logger.info("  ✅ 类别与测试集样本数一致（%d 类，%d 个测试样本）",
                len(list(classes)[0]), list(n_samples)[0])


# ============================================================================
#  汇总计算
# ============================================================================
def aggregate(runs: list[dict]) -> dict:
    acc = np.array([r["accuracy"] for r in runs], dtype=float)
    mf1 = np.array([r["macro_f1"] for r in runs], dtype=float)
    wf1 = np.array([r["weighted_f1"] for r in runs], dtype=float)

    class_names = [runs[0]["label_mapping"][str(i)] for i in range(len(runs[0]["label_mapping"]))]
    per_class = {}
    for c in class_names:
        for metric in ("precision", "recall", "f1"):
            vals = np.array([r["per_class"][c][metric] for r in runs], dtype=float)
            per_class.setdefault(c, {})[metric] = {
                "mean": float(vals.mean()), "std": float(vals.std(ddof=0)),
                "values": vals.tolist(),
            }
        per_class[c]["support"] = int(runs[0]["per_class"][c]["support"])

    cms = np.stack([r["confusion_matrix"] for r in runs])
    cms_n = np.stack([r["confusion_matrix_normalized"] for r in runs])

    def stat(a):
        return {"mean": float(a.mean()), "std": float(a.std(ddof=0)), "values": a.tolist()}

    return {
        "n_runs": len(runs),
        "seeds": [r["seed"] for r in runs],
        "model": runs[0]["model"],
        "class_names": class_names,
        "accuracy": stat(acc),
        "macro_f1": stat(mf1),
        "weighted_f1": stat(wf1),
        "n_errors": {"values": [r["n_errors"] for r in runs],
                     "mean": float(np.mean([r["n_errors"] for r in runs]))},
        "per_class": per_class,
        "mean_confusion_matrix": cms.mean(axis=0).tolist(),
        "std_confusion_matrix": cms.std(axis=0, ddof=0).tolist(),
        "mean_confusion_matrix_normalized": cms_n.mean(axis=0).tolist(),
        "per_run": [
            {
                "name": r["name"], "seed": r["seed"], "accuracy": r["accuracy"],
                "macro_f1": r["macro_f1"], "weighted_f1": r["weighted_f1"],
                "n_errors": r["n_errors"], "best_epoch": r["best_epoch"],
                "epochs_run": r["epochs_run"], "total_time_sec": r["total_time_sec"],
                "n_params_total": r["n_params_total"],
                "n_test_evaluations": r["n_test_evaluations"],
            }
            for r in runs
        ],
    }


# ============================================================================
#  落盘：表格
# ============================================================================
def write_tables(agg: dict, runs: list[dict], outdir: Path, logger) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    written = []

    # ---- per_seed.csv ----
    p = outdir / "per_seed.csv"
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "seed", "accuracy", "macro_f1", "weighted_f1",
                                          "n_errors", "best_epoch", "epochs_run",
                                          "total_time_sec", "n_params_total",
                                          "n_test_evaluations"])
        w.writeheader()
        w.writerows(agg["per_run"])
    written.append(p)

    # ---- per_class.csv ----
    p = outdir / "per_class.csv"
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "support",
                    "precision_mean", "precision_std",
                    "recall_mean", "recall_std",
                    "f1_mean", "f1_std"])
        for c in agg["class_names"]:
            d = agg["per_class"][c]
            w.writerow([c, d["support"],
                        f"{d['precision']['mean']:.6f}", f"{d['precision']['std']:.6f}",
                        f"{d['recall']['mean']:.6f}", f"{d['recall']['std']:.6f}",
                        f"{d['f1']['mean']:.6f}", f"{d['f1']['std']:.6f}"])
    written.append(p)

    # ---- mean_confusion_matrix.csv ----
    for tag, key in (("mean_confusion_matrix", "mean_confusion_matrix"),
                     ("mean_confusion_matrix_normalized", "mean_confusion_matrix_normalized")):
        p = outdir / f"{tag}.csv"
        with open(p, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["true\\pred"] + agg["class_names"])
            for i, c in enumerate(agg["class_names"]):
                w.writerow([c] + [f"{v:.4f}" for v in agg[key][i]])
        written.append(p)

    # ---- summary.json ----
    written.append(save_json(agg, outdir / "summary.json"))

    # ---- summary.md ----
    md = []
    md.append(f"# 汇总报告：{agg['model']}\n")
    md.append(f"- 运行数：**{agg['n_runs']}**　随机种子：**{agg['seeds']}**")
    md.append("- 数据划分：**同一份**（已校验 split_index.csv 哈希一致）")
    md.append("- 测试集样本数：%d　类别数：%d\n" % (runs[0]["n_samples"], len(agg["class_names"])))

    md.append("## 一、总体指标（均值 ± 标准差）\n")
    md.append("| 指标 | 均值 | 标准差 | 逐种子 |")
    md.append("| --- | ---: | ---: | --- |")
    for label, key in (("Accuracy", "accuracy"), ("Macro F1", "macro_f1"), ("Weighted F1", "weighted_f1")):
        d = agg[key]
        vals = ", ".join(f"{v:.4f}" for v in d["values"])
        md.append(f"| **{label}** | **{d['mean']:.4f}** | {d['std']:.4f} | {vals} |")
    md.append("")

    md.append("## 二、逐类别指标（均值 ± 标准差）\n")
    md.append("| 类别 | Support | Precision | Recall | F1 |")
    md.append("| --- | ---: | ---: | ---: | ---: |")
    for c in agg["class_names"]:
        d = agg["per_class"][c]
        md.append(f"| {c} | {d['support']} | {d['precision']['mean']:.4f} ± {d['precision']['std']:.4f} "
                  f"| {d['recall']['mean']:.4f} ± {d['recall']['std']:.4f} "
                  f"| {d['f1']['mean']:.4f} ± {d['f1']['std']:.4f} |")
    md.append("")

    md.append("## 三、平均混淆矩阵（行=真实，列=预测）\n")
    md.append("| 真实\\预测 | " + " | ".join(agg["class_names"]) + " |")
    md.append("| --- | " + " | ".join(["---:"] * len(agg["class_names"])) + " |")
    for i, c in enumerate(agg["class_names"]):
        md.append(f"| **{c}** | " + " | ".join(f"{v:.2f}" for v in agg["mean_confusion_matrix"][i]) + " |")
    md.append("")

    md.append("## 四、逐种子明细\n")
    md.append("| 运行 | seed | Accuracy | Macro F1 | 错分 | 最佳轮次 | 实际轮数 | 耗时(s) | 测试集评估次数 |")
    md.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in agg["per_run"]:
        md.append(f"| {r['name']} | {r['seed']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} "
                  f"| {r['n_errors']} | {r['best_epoch']} | {r['epochs_run']} "
                  f"| {r['total_time_sec']:.0f} | {r['n_test_evaluations']} |")
    md.append("")

    md.append("## 五、波动解读（G4）\n")
    md.append(f"- Accuracy 标准差 **{agg['accuracy']['std']:.4f}**，"
              f"极差 {max(agg['accuracy']['values']) - min(agg['accuracy']['values']):.4f}")
    md.append(f"- Macro F1 标准差 **{agg['macro_f1']['std']:.4f}**，"
              f"极差 {max(agg['macro_f1']['values']) - min(agg['macro_f1']['values']):.4f}")
    md.append(f"- 结论：后续比较其他模型时，**均值差需大于约 "
              f"{max(agg['accuracy']['std'], agg['macro_f1']['std']):.4f} 才可视为超出随机波动**"
              f"（experiment-plan.md 第 6.3 节的稳健性判据）。")

    p = outdir / "summary.md"
    p.write_text("\n".join(md), encoding="utf-8")
    written.append(p)
    return written


# ============================================================================
#  图
# ============================================================================
def plot_summary_figures(agg: dict, runs: list[dict], outdir: Path, prefix: str, logger) -> list[Path]:
    written: list[Path] = []
    names = agg["class_names"]

    # ---- fig2a：均值 ± 标准差（问答：基线的基础分是多少、稳不稳）----
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    labels = ["Accuracy", "Macro F1", "Weighted F1"]
    means = [agg["accuracy"]["mean"], agg["macro_f1"]["mean"], agg["weighted_f1"]["mean"]]
    stds = [agg["accuracy"]["std"], agg["macro_f1"]["std"], agg["weighted_f1"]["std"]]
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    bars = ax.bar(labels, means, yerr=stds, capsize=8, color=colors,
                  error_kw={"elinewidth": 1.6, "ecolor": "#333333"})
    for b, m, s in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width() / 2, m + s + 0.015, f"{m:.4f}\n±{s:.4f}",
                ha="center", fontsize=9.5)
    ax.set_ylim(0, min(1.0, max(means) * 1.35) if max(means) > 0 else 1.0)
    ax.set_ylabel(f"Test score  (n={agg['n_runs']} seeds)")
    ax.set_title(f"[which question] what is the baseline's floor, and how stable is it?\n"
                 f"{agg['model']} — mean ± std over seeds {agg['seeds']}", fontsize=9.5)
    ax.grid(axis="y", alpha=0.3)
    written += save_figure(fig, outdir, f"{prefix}fig2a_summary_metrics")
    plt.close(fig)

    # ---- fig2b：逐种子散点（问答：均值是不是被某一次好运拉高的）----
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    metric_names = ["Accuracy", "Macro F1"]
    data = [agg["accuracy"]["values"], agg["macro_f1"]["values"]]
    for i, (label, vals, color) in enumerate(zip(metric_names, data, ["#4c78a8", "#f58518"])):
        ax.scatter([i] * len(vals), vals, s=70, color=color, zorder=3, label=f"{label} per seed")
        ax.hlines(np.mean(vals), i - 0.22, i + 0.22, color="#333333", lw=2, zorder=2)
        for v in vals:
            ax.annotate(f"{v:.4f}", (i, v), textcoords="offset points", xytext=(8, -3), fontsize=8)
    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_xticklabels(metric_names)
    ax.set_ylim(min(min(v) for v in data) - 0.05, max(max(v) for v in data) + 0.05)
    ax.set_ylabel("Test score")
    ax.set_title("[which question] is the mean driven by one lucky seed?\n"
                 "(black bar = mean, dots = individual seeds)", fontsize=9.5)
    ax.grid(axis="y", alpha=0.3)
    written += save_figure(fig, outdir, f"{prefix}fig2b_per_seed_scatter")
    plt.close(fig)

    # ---- fig1c：三种子训练曲线叠加（问答：收敛是否稳定、是否过拟合）----
    have = [r for r in runs if r["history"]]
    if have:
        fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))
        cmap = plt.get_cmap("tab10")
        for i, r in enumerate(have):
            ep = [h["epoch"] for h in r["history"]]
            c = cmap(i % 10)
            axes[0].plot(ep, [h["train_loss"] for h in r["history"]], "-", color=c, alpha=0.45, lw=1.3)
            axes[0].plot(ep, [h["val_loss"] for h in r["history"]], "--", color=c, lw=2,
                         label=f"seed {r['seed']}")
            axes[1].plot(ep, [h["train_acc"] for h in r["history"]], "-", color=c, alpha=0.45, lw=1.3,
                         label=f"seed {r['seed']} train")
            axes[1].plot(ep, [h["val_acc"] for h in r["history"]], "--", color=c, lw=2,
                         label=f"seed {r['seed']} val")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].set_title("[which question] do all seeds converge the same way?\n"
                          "solid = train, dashed = val", fontsize=9.5)
        axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].set_ylim(0, 1.02)
        axes[1].set_title("[which question] where does each seed stop improving?", fontsize=9.5)
        axes[1].legend(fontsize=7.5, ncol=2); axes[1].grid(alpha=0.3)
        fig.tight_layout()
        written += save_figure(fig, outdir, f"{prefix}fig1c_train_curves_all_seeds")
        plt.close(fig)

    # ---- fig3a：平均混淆矩阵 ----
    for tag, mat, norm in (("fig3a_mean_confusion_matrix", agg["mean_confusion_matrix"], False),
                           ("fig3b_mean_confusion_matrix_norm",
                            agg["mean_confusion_matrix_normalized"], True)):
        m = np.array(mat, dtype=float)
        fig, ax = plt.subplots(figsize=(7.0, 6.0))
        im = ax.imshow(m, cmap="Blues", vmin=0, vmax=m.max() if m.max() > 0 else 1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right"); ax.set_yticklabels(names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True (ground truth)")
        fmt = "{:.2f}" if norm else "{:.1f}"
        thr = m.max() * 0.55 if m.max() > 0 else 0.5
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, fmt.format(m[i, j]), ha="center", va="center", fontsize=9,
                        color="white" if m[i, j] > thr else "black")
        ax.set_title(f"[which question] which class pairs are confused, averaged over "
                     f"{agg['n_runs']} seeds?\n(n={int(np.array(agg['mean_confusion_matrix']).sum())} "
                     f"test samples × {agg['n_runs']} seeds)", fontsize=9.5)
        written += save_figure(fig, outdir, f"{prefix}{tag}")
        plt.close(fig)

    logger.info("  生成 %d 个图文件到 %s", len(written), outdir)
    return written


# ============================================================================
#  主流程
# ============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="多运行汇总（跨随机种子的均值与波动）")
    ap.add_argument("--runs", nargs="+", required=True, help="各运行的目录")
    ap.add_argument("--name", default=None, help="汇总名称（默认取模型名）")
    ap.add_argument("--figure-dir", default=None, help="图输出目录（默认 <project>/figures）")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    run_dirs = [Path(r) for r in args.runs]
    project_root = run_dirs[0].resolve()
    while project_root != project_root.parent and not (project_root / "results").is_dir():
        project_root = project_root.parent
    logger = get_logger("summarize", project_root / "results" / "summary" / "_logs" / "summarize.log")

    logger.info("=" * 72)
    logger.info("汇总 %d 个运行（项目根: %s）", len(run_dirs), project_root)
    runs = [load_run(d) for d in run_dirs]
    runs.sort(key=lambda r: (r["seed"] is None, r["seed"]))
    for r in runs:
        logger.info("  %-24s seed=%-6s acc=%.4f macroF1=%.4f (%s)",
                    r["name"], r["seed"], r["accuracy"], r["macro_f1"],
                    "冒烟" if r["is_smoke"] else "正式")

    if any(r["is_smoke"] for r in runs):
        raise AssertionError(
            "❌ 汇总中混入了**冒烟测试**运行！冒烟结果不得作为正式结论（G6）。"
            f" 冒烟运行: {[r['name'] for r in runs if r['is_smoke']]}"
        )

    logger.info("-" * 72)
    logger.info("一致性校验")
    split_info = verify_same_split(runs, logger)
    verify_same_dataset(runs, logger)

    agg = aggregate(runs)
    agg["split_consistency"] = split_info
    agg["project_root"] = str(project_root)

    name = args.name or agg["model"] or "summary"
    outdir = project_root / "results" / "summary" / name
    logger.info("-" * 72)
    logger.info("写入汇总表 → %s", outdir)
    for p in write_tables(agg, runs, outdir, logger):
        logger.info("    %s", p.name)

    if not args.no_figures:
        figdir = Path(args.figure_dir) if args.figure_dir else (project_root / "figures")
        logger.info("-" * 72)
        logger.info("生成汇总图 → %s", figdir)
        plot_summary_figures(agg, runs, figdir, f"{name}__", logger)

    logger.info("=" * 72)
    logger.info("汇总完成：%s", name)
    logger.info("  Accuracy   : %.4f ± %.4f   (逐种子 %s)",
                agg["accuracy"]["mean"], agg["accuracy"]["std"],
                [round(v, 4) for v in agg["accuracy"]["values"]])
    logger.info("  Macro F1   : %.4f ± %.4f   (逐种子 %s)",
                agg["macro_f1"]["mean"], agg["macro_f1"]["std"],
                [round(v, 4) for v in agg["macro_f1"]["values"]])
    logger.info("  Weighted F1: %.4f ± %.4f", agg["weighted_f1"]["mean"], agg["weighted_f1"]["std"])
    logger.info("  平均每类 F1: %s",
                {c: round(agg["per_class"][c]["f1"]["mean"], 4) for c in agg["class_names"]})
    logger.info("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
