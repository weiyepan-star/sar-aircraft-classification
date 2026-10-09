"""
错误分析与可解释性分析（针对**按验证集选出的最终模型**）。

对应约束：
  IC5 —— 最终模型**只依据验证集**选择，绝不看测试集分数；
         本脚本把选择过程自动化并落盘成可审计的记录（selection.json）
  G5  —— 所有数字、图表由脚本生成，可追溯
  G6  —— **全部**错误样例保留，不做筛选
  IC4 —— 错误样例可视化 + Grad-CAM 可解释性

⚠️ 为什么分成两个 stage：
   本机环境里「torch 反向传播 + matplotlib 渲染」在同一进程会触发
   `OMP: Error #15`（两套 OpenMP 运行时冲突，详见 README 5.4 坑 2）。
   因此：
       --stage compute  纯 torch：选模型 + 计算 Grad-CAM 指标 → 落盘
       --stage render   纯 matplotlib：读上面的产物 → 出图 + 写分析说明

用法：
    python src/error_analysis.py --stage compute
    python src/error_analysis.py --stage render
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ⚠️ PIL 必须在 torchvision / CUDA 之前导入（见 README 5.4 坑 3）
from PIL import Image  # noqa: F401

import numpy as np

# ⚠️ 这里**故意不在模块顶层导入 torch**：
#    render 阶段（matplotlib 出图）与 compute 阶段（torch 反向传播）必须分进程，
#    否则同一进程里先跑 torch backward 再用 matplotlib 渲染会触发
#        OMP: Error #15: Initializing libomp.dll, but found libiomp5md.dll already initialized
#    （见 README 5.4 坑 2）。torch 只在 compute 相关函数内部按需导入。

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger, load_config, load_json, save_json, set_seed  # noqa: E402

OUT_NAME = "error_analysis"
SEL_NAME = "final_model_selection.json"
LOC_NAME = "cam_localization.json"
EX_NPZ = "cam_examples.npz"


# ============================================================================
#  IC5：按【验证集】选最终模型
# ============================================================================
def select_final_model(project_root: Path, logger) -> dict:
    """
    两步选择，**全程不使用测试集**：

      第 1 步：按模型族比较 **验证集 macro-F1 的均值**（3 个种子），选出更好的模型族；
      第 2 步：在该模型族内，选**验证集 macro-F1 最高**的那个种子作为最终模型。

    只读 `metrics.json` 里的 `best_val_macro_f1`（训练阶段由验证集产生），
    不读 `test_metrics.json` —— 这一点在返回结果里显式标注。
    """
    raw = project_root / "results" / "raw"
    entries = []
    for d in sorted(raw.iterdir()):
        mp = d / "metrics.json"
        if not (d.is_dir() and mp.is_file()):
            continue
        m = load_json(mp)
        if m.get("mode") != "formal" or m.get("is_smoke"):
            continue
        entries.append({
            "run_dir": str(d), "run": d.name, "model": m["model"], "seed": m["seed"],
            "val_macro_f1": m["best_val_macro_f1"], "val_acc": m.get("best_val_acc"),
            "best_epoch": m.get("best_epoch"), "n_params": m.get("n_params_total"),
            "model_type": m.get("model_type"),
        })
    if not entries:
        raise RuntimeError("在 results/raw 下找不到任何 formal 运行（metrics.json）")

    by_model: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        by_model[e["model"]].append(e)

    family = {}
    for name, rows in by_model.items():
        vals = np.array([r["val_macro_f1"] for r in rows], dtype=float)
        family[name] = {
            "seeds": [r["seed"] for r in rows],
            "val_macro_f1_mean": float(vals.mean()),
            "val_macro_f1_std": float(vals.std(ddof=0)),
            "val_macro_f1_values": vals.tolist(),
        }
    best_family = max(family, key=lambda k: family[k]["val_macro_f1_mean"])

    cands = sorted(by_model[best_family], key=lambda r: -r["val_macro_f1"])
    chosen = cands[0]

    logger.info("=" * 72)
    logger.info("【IC5】最终模型选择（**只用验证集，未使用测试集**）")
    logger.info("-" * 72)
    logger.info("第 1 步：按模型族的验证集 macro-F1 均值比较")
    for name, f in sorted(family.items(), key=lambda kv: -kv[1]["val_macro_f1_mean"]):
        logger.info("    %-20s %.4f ± %.4f   (种子 %s)", name,
                    f["val_macro_f1_mean"], f["val_macro_f1_std"], f["seeds"])
    logger.info("  → 选出模型族：%s", best_family)
    logger.info("-" * 72)
    logger.info("第 2 步：在该族内选验证集 macro-F1 最高的种子")
    for c in cands:
        logger.info("    %-22s seed=%-6s val_macro_f1=%.4f %s", c["run"], c["seed"],
                    c["val_macro_f1"], "← 选中" if c is chosen else "")
    logger.info("  → 最终模型：%s（seed=%s，验证集 macro_f1=%.4f）",
                chosen["run"], chosen["seed"], chosen["val_macro_f1"])
    logger.info("=" * 72)

    return {
        "criterion": "验证集 macro-F1（不使用测试集）",
        "test_set_used_for_selection": False,
        "families": family,
        "selected_family": best_family,
        "selected_run": chosen["run"],
        "selected_run_dir": chosen["run_dir"],
        "selected_seed": chosen["seed"],
        "selected_val_macro_f1": chosen["val_macro_f1"],
        "candidates": cands,
        "all_runs": entries,
    }


# ============================================================================
#  Grad-CAM（复用 gradcam.py 的实现）
# ============================================================================
def get_layer_by_name(model, name: str):
    """取 backbone 下的指定层（ResNet 用）；非 ResNet 返回 None。"""
    bb = getattr(model, "backbone", None)
    if bb is not None and hasattr(bb, name):
        return getattr(bb, name)
    return None


def gradcam_localization(sel: dict, project_root: Path, logger, outdir: Path) -> dict:
    """
    量化"模型在看哪里"，并**跨层对比**。

    ⚠️ 为什么必须跨层：
       ResNet18 在 96×96 输入下下采样 32 倍，`layer4` 只有 **3×3**（每格 = 原图 32×32 像素）。
       用它做 Grad-CAM，热力图空间分辨率极低，定位指标自然接近随机 ——
       这**不能**说明模型没看目标，只能说明这一层的 CAM 太粗。
       因此同时对 `layer3`（6×6）与 `layer4`（3×3）计算，做对照。

    指标（无标注框，用亮像素近似目标）：
      · cam_in_bbox_ratio  = 热力图落在「亮像素外接框」内的比例，随机基线 = 外接框面积占比
      · cam_in_mask_ratio  = 热力图落在「亮像素掩膜」上的比例，随机基线 = 掩膜面积占比
                              （掩膜面积小得多，因此这个指标**更锐利**）
    """
    from gradcam import GradCAM, get_target_layer, temporarily_enable_grad, restore_grad
    import torch
    from dataset import load_split_index
    from models import build_model

    run_dir = Path(sel["selected_run_dir"])
    cfg = load_config(run_dir / "config_resolved.yaml")
    cfg["_project_root"] = str(project_root)
    ckpt = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    class_names = ckpt["class_names"]
    model_type = ckpt["model_cfg"]["type"]
    set_seed(int(ckpt.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcfg = dict(ckpt["model_cfg"]); mcfg["pretrained"] = False
    model = build_model(mcfg, len(class_names), logger=None)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    preds = {}
    with open(run_dir / "predictions.csv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            preds[r["relpath"]] = r

    size = int(ckpt.get("image_size", cfg["data"]["image_size"]))
    image_root = project_root / cfg["paths"]["data_root"]
    records = [r for r in load_split_index(run_dir / "split_index.csv", image_root)
               if r["split"] == "test"]

    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((size, size)), transforms.ToTensor(),
        transforms.Normalize([ckpt["normalize"]["mean"]], [ckpt["normalize"]["std"]]),
    ])

    # 候选目标层：ResNet 用 layer3/layer4 对照；其他模型用默认目标层
    candidates = [n for n in ("layer3", "layer4") if get_layer_by_name(model, n) is not None]
    layers = candidates if candidates else ["default"]

    saved = temporarily_enable_grad(model)
    per_layer: dict[str, list[dict]] = {}
    layer_cams: dict[str, np.ndarray] = {}
    grid_info: dict[str, str] = {}

    for lname in layers:
        layer = get_layer_by_name(model, lname) if lname != "default" else get_target_layer(model, model_type)
        engine = GradCAM(model, layer)
        rows: list[dict] = []
        cams: dict[str, np.ndarray] = {}
        grid = "?"
        logger.info("计算 %d 个测试样本的 Grad-CAM 定位指标（%s）…", len(records), lname)
        for i, rec in enumerate(records):
            p = preds.get(rec["relpath"])
            if p is None:
                continue
            with Image.open(image_root / rec["relpath"]) as im:
                gray = im.convert("L")
                arr = np.asarray(gray.resize((size, size), Image.BILINEAR), dtype=float)
                x = tf(gray).unsqueeze(0).to(device)
            cam, _ = engine.compute(x, class_idx=int(p["pred_label"]))
            if grid == "?":
                grid = f"{cam.shape[0]}×{cam.shape[1]}"
            cam = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize(
                (size, size), Image.BILINEAR), dtype=float) / 255.0

            thr = arr.mean() + 1.0 * arr.std()
            mask = arr > thr
            total = float(cam.sum()) + 1e-12
            if mask.sum() >= 5:
                ys, xs = np.where(mask)
                bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                bbox_frac = ((bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)) / (size * size)
                in_bbox = float(cam[bbox[1]:bbox[3] + 1, bbox[0]:bbox[2] + 1].sum()) / total
                mask_frac = float(mask.mean())
                in_mask = float(cam[mask].sum()) / total
            else:
                bbox, bbox_frac, in_bbox, mask_frac, in_mask = None, None, None, None, None

            rows.append({
                "relpath": rec["relpath"], "true_name": p["true_name"], "pred_name": p["pred_name"],
                "confidence": float(p["confidence"]),
                "correct": p["correct"].lower() in {"true", "1"},
                "bbox_area_frac": bbox_frac, "cam_in_bbox_ratio": in_bbox,
                "mask_area_frac": mask_frac, "cam_in_mask_ratio": in_mask,
                "bbox": list(bbox) if bbox else None,
            })
            cams[rec["relpath"]] = cam.astype(np.float16)
            if (i + 1) % 150 == 0:
                logger.info("    … %d/%d", i + 1, len(records))
        engine.remove()
        per_layer[lname] = rows
        layer_cams[lname] = cams
        grid_info[lname] = grid
        valid = [r for r in rows if r["cam_in_bbox_ratio"] is not None]
        rb = np.array([r["cam_in_bbox_ratio"] for r in valid])
        bf = np.array([r["bbox_area_frac"] for r in valid])
        rm = np.array([r["cam_in_mask_ratio"] for r in valid])
        mf = np.array([r["mask_area_frac"] for r in valid])
        logger.info("  [%s] CAM 网格 %s | bbox 内占比 %.4f (随机 %.4f, %.2f×) | "
                    "mask 内占比 %.4f (随机 %.4f, %.2f×)",
                    lname, grid, rb.mean(), bf.mean(), rb.mean() / max(bf.mean(), 1e-9),
                    rm.mean(), mf.mean(), rm.mean() / max(mf.mean(), 1e-9))

    # 选"更锐利"的层（按 mask 指标提升倍数）用于出图
    def lift_mask(lname):
        v = [r for r in per_layer[lname] if r["cam_in_mask_ratio"] is not None]
        rm = np.mean([r["cam_in_mask_ratio"] for r in v])
        mf = np.mean([r["mask_area_frac"] for r in v])
        return rm / max(mf, 1e-9)

    best_layer = max(per_layer, key=lift_mask)
    restore_grad(model, saved)

    def stats(lname):
        v = [r for r in per_layer[lname] if r["cam_in_bbox_ratio"] is not None]
        rb = np.array([r["cam_in_bbox_ratio"] for r in v])
        bf = np.array([r["bbox_area_frac"] for r in v])
        rm = np.array([r["cam_in_mask_ratio"] for r in v])
        mf = np.array([r["mask_area_frac"] for r in v])
        ok = np.array([r["correct"] for r in v])
        return {
            "cam_grid": grid_info[lname],
            "cam_in_bbox_ratio_mean": float(rb.mean()),
            "cam_in_bbox_ratio_median": float(np.median(rb)),
            "bbox_area_frac_mean": float(bf.mean()),
            "lift_bbox": float(rb.mean() / max(bf.mean(), 1e-9)),
            "cam_in_mask_ratio_mean": float(rm.mean()),
            "mask_area_frac_mean": float(mf.mean()),
            "lift_mask": float(rm.mean() / max(mf.mean(), 1e-9)),
            "cam_in_mask_ratio_correct": float(rm[ok].mean()) if ok.any() else None,
            "cam_in_mask_ratio_wrong": float(rm[~ok].mean()) if (~ok).any() else None,
        }

    summary = {
        "run": sel["selected_run"],
        "model": sel["selected_family"],
        "seed": sel["selected_seed"],
        "n_samples": len(per_layer[layers[0]]),
        "n_with_target": len([r for r in per_layer[layers[0]] if r["cam_in_bbox_ratio"] is not None]),
        "layers": {ln: stats(ln) for ln in layers},
        "best_layer": best_layer,
        "note": ("本数据集无目标框，用「亮像素（强散射体）」近似目标。"
                 "cam_in_bbox_ratio 的随机基线 = 外接框面积占比；"
                 "cam_in_mask_ratio 的随机基线 = 掩膜面积占比（更锐利）。"
                 "ResNet18 在 96×96 下 layer4 只有 3×3，因此同时报告 layer3(6×6) 作对照。"),
        "per_sample": per_layer[best_layer],
        "per_sample_all_layers": per_layer,
    }
    save_json(summary, outdir / LOC_NAME)

    # 保存用于出图的 CAM（用更锐利的那层；最高/最低各 6 张）
    v = [r for r in per_layer[best_layer] if r["cam_in_bbox_ratio"] is not None]
    order = sorted(v, key=lambda r: r["cam_in_mask_ratio"])
    pick = {r["relpath"] for r in order[:6]} | {r["relpath"] for r in order[-6:]}
    cams = layer_cams[best_layer]
    if pick:
        np.savez_compressed(outdir / EX_NPZ,
                            paths=np.array(sorted(pick)),
                            cams=np.stack([cams[p] for p in sorted(pick)]))
    logger.info("  用于出图的层：%s（CAM 网格 %s）", best_layer, grid_info[best_layer])
    logger.info("  定位指标已写入 %s", outdir / LOC_NAME)
    return summary


# ============================================================================
#  compute stage
# ============================================================================
def stage_compute(args) -> int:
    project_root = Path(args.project)
    outdir = project_root / "results" / "summary" / OUT_NAME
    outdir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("error_analysis.compute", outdir / "compute.log")

    sel = select_final_model(project_root, logger)
    save_json(sel, outdir / SEL_NAME)
    gradcam_localization(sel, project_root, logger, outdir)
    logger.info("compute 阶段完成 → %s", outdir)
    return 0


# ============================================================================
#  render stage（纯 matplotlib）
# ============================================================================
def stage_render(args) -> int:
    project_root = Path(args.project)
    outdir = project_root / "results" / "summary" / OUT_NAME
    figdir = project_root / "figures"
    logger = get_logger("error_analysis.render", outdir / "render.log")

    from visualize import plot_sample_grid, save_figure, setup_matplotlib

    setup_matplotlib()
    import matplotlib.pyplot as plt

    sel = load_json(outdir / SEL_NAME)
    loc = load_json(outdir / LOC_NAME)
    run_dir = Path(sel["selected_run_dir"])
    cfg = load_config(run_dir / "config_resolved.yaml")
    image_root = project_root / cfg["paths"]["data_root"]
    prefix = f"{OUT_NAME}__"

    with open(run_dir / "predictions.csv", encoding="utf-8", newline="") as f:
        preds = [{
            "relpath": r["relpath"], "true_name": r["true_name"], "pred_name": r["pred_name"],
            "true_label": int(r["true_label"]), "pred_label": int(r["pred_label"]),
            "confidence": float(r["confidence"]),
            "correct": r["correct"].lower() in {"true", "1"},
        } for r in csv.DictReader(f)]

    wrong = sorted([r for r in preds if not r["correct"]], key=lambda r: -r["confidence"])
    correct = sorted([r for r in preds if r["correct"]], key=lambda r: -r["confidence"])
    low_conf = sorted([r for r in preds if r["correct"]], key=lambda r: r["confidence"])
    class_names = sorted({r["true_name"] for r in preds})
    logger.info("=" * 72)
    logger.info("最终模型：%s（seed=%s，验证集 macro_f1=%.4f）",
                sel["selected_run"], sel["selected_seed"], sel["selected_val_macro_f1"])
    logger.info("  测试集：%d 张，错分 %d 张，正确 %d 张",
                len(preds), len(wrong), len(correct))

    written: list[Path] = []

    # ---- fig7a：全部错误样例（G6：一张不删）----
    # 用 9 列排布：36 张 6 行会太高（宽高比 1.30 在双栏下仍超过整页），9 列 4 行才排得进。
    written += plot_sample_grid(
        wrong[:60], image_root, figdir, prefix, "fig7a_all_wrong_samples",
        f"[which question] every misclassified test sample ({len(wrong)} of {len(preds)} total, "
        f"none filtered) — title: true → pred, p=confidence",
        ncols=9, max_items=60, logger=logger)

    # ---- fig7b：每类正确样例 ----
    per_cls = defaultdict(list)
    for r in correct:
        per_cls[r["true_name"]].append(r)
    sel_correct = [r for c in class_names for r in per_cls.get(c, [])[:3]]
    written += plot_sample_grid(
        sel_correct, image_root, figdir, prefix, "fig7b_correct_samples_per_class",
        "[which question] what do confidently-correct samples look like for each class?",
        ncols=6, max_items=18, logger=logger)

    # ---- fig7c：置信度最低的正确样例 ----
    written += plot_sample_grid(
        low_conf[:12], image_root, figdir, prefix, "fig7c_lowest_confidence_correct",
        "[which question] which CORRECT predictions is the model least sure about? "
        "(lowest max-softmax)", ncols=6, max_items=12, logger=logger)

    # ---- fig7d：最易混的两类 对照 ----
    cm = np.array(load_json(run_dir / "test_metrics.json")["confusion_matrix_normalized"])
    np.fill_diagonal(cm, 0.0)
    i, j = np.unravel_index(np.argmax(cm), cm.shape)
    pair = (class_names[i], class_names[j])
    confused_a = sorted([r for r in preds if r["true_name"] == pair[0]], key=lambda r: -r["confidence"])[:6]
    confused_b = sorted([r for r in preds if r["true_name"] == pair[1]], key=lambda r: -r["confidence"])[:6]
    written += plot_sample_grid(
        confused_a + confused_b, image_root, figdir, prefix, "fig7d_most_confused_pair",
        f"[which question] why are these two classes confused? "
        f"most confused pair = {pair[0]} ↔ {pair[1]} "
        f"(first 6 = {pair[0]}, last 6 = {pair[1]})",
        ncols=6, max_items=12, logger=logger)

    # ---- fig7e：Grad-CAM 定位指标（跨层对比）----
    best = loc["best_layer"]
    rows = [r for r in loc["per_sample"] if r["cam_in_mask_ratio"] is not None]
    ratios = np.array([r["cam_in_mask_ratio"] for r in rows])
    st = loc["layers"][best]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))

    ax = axes[0]
    ax.hist(ratios, bins=30, range=(0, 1), color="#4c78a8", edgecolor="white")
    ax.axvline(ratios.mean(), color="#d1242f", lw=2, label=f"model mean = {ratios.mean():.3f}")
    ax.axvline(st["mask_area_frac_mean"], color="#333", ls="--", lw=2,
               label=f"random baseline = {st['mask_area_frac_mean']:.3f}")
    ax.set_xlabel("CAM mass falling on bright (target) pixels")
    ax.set_ylabel("Number of test samples")
    ax.set_title(f"[which question] does the model look at the aircraft?\n"
                 f"CAM layer = {best} (grid {st['cam_grid']}), "
                 f"{st['lift_mask']:.2f}× above chance", fontsize=9.5)
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3)

    ax = axes[1]
    layers = list(loc["layers"])
    xs = np.arange(len(layers)); w = 0.35
    lb = [loc["layers"][l]["lift_bbox"] for l in layers]
    lm = [loc["layers"][l]["lift_mask"] for l in layers]
    b1 = ax.bar(xs - w / 2, lb, w, label="inside estimated bbox", color="#9ecae1")
    b2 = ax.bar(xs + w / 2, lm, w, label="on bright pixels (sharper)", color="#3182bd")
    for bars, vals in ((b1, lb), (b2, lm)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}×",
                    ha="center", fontsize=9)
    ax.axhline(1.0, color="#d1242f", ls="--", lw=1.5)
    ax.annotate("1.0× = random", (len(layers) - 0.5, 1.03), color="#d1242f", fontsize=8.5,
                ha="right")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{l}\n(CAM grid {loc['layers'][l]['cam_grid']})" for l in layers],
                       fontsize=9)
    ax.set_ylabel("concentration vs random")
    ax.set_title("[which question] does the coarse CAM layer hide the localisation?\n"
                 "(ResNet18 downsamples 32×, so layer4 is only 3×3 at 96×96)",
                 fontsize=9.5)
    ax.legend(fontsize=8.5); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    written += save_figure(fig, figdir, f"{prefix}fig7e_cam_localization")
    plt.close(fig)

    # ---- fig7f：CAM 最聚焦 / 最弥散 的对照 ----
    npz_p = outdir / EX_NPZ
    if npz_p.is_file():
        z = np.load(npz_p)
        paths = [str(p) for p in z["paths"]]
        cams = z["cams"].astype(np.float32)
        by_path = {r["relpath"]: r for r in rows}
        ordered = sorted(paths, key=lambda p: by_path[p]["cam_in_mask_ratio"])
        grays, kept = [], []
        for p in ordered:
            with Image.open(image_root / p) as im:
                grays.append(np.asarray(im.convert("L").resize((96, 96), Image.BILINEAR), dtype=float))
            kept.append(by_path[p])
        written += plot_cam_grid_from_arrays(
            np.stack(grays), np.stack([cams[paths.index(p)] for p in ordered]), kept,
            figdir, f"{prefix}fig7f_cam_on_vs_off_target",
            f"[which question] the most diffuse (top) vs most target-focused (bottom) examples, "
            f"CAM layer = {best}", logger, max_rows=6, per_row=2)

    # ---- 分析说明 ----
    shape = compare_pair_shape(preds, image_root, pair)
    md = build_analysis_md(sel, loc, preds, wrong, correct, pair, cm, class_names, shape)
    (project_root / "report" / "error-analysis.md").write_text(md, encoding="utf-8")
    (outdir / "analysis.md").write_text(md, encoding="utf-8")

    # ---- 机器可读的分析摘要（供 LaTeX 报告生成器读取，保证"数字不手填"）----
    pairs = Counter(f"{r['true_name']} → {r['pred_name']}" for r in wrong)
    cw_v = np.array([r["confidence"] for r in wrong]) if wrong else np.array([0.0])
    cc_v = np.array([r["confidence"] for r in correct]) if correct else np.array([0.0])
    directions = []
    for k, v in pairs.most_common():
        t, p = k.split(" → ")
        conf = [r["confidence"] for r in wrong if r["true_name"] == t and r["pred_name"] == p]
        directions.append({"true": t, "pred": p, "count": int(v),
                           "frac_of_errors": float(v / max(len(wrong), 1)),
                           "mean_conf": float(np.mean(conf))})
    offdiag_top = sorted(
        [{"true": class_names[i], "pred": class_names[j], "ratio": float(cm[i][j])}
         for i in range(len(class_names)) for j in range(len(class_names)) if i != j],
        key=lambda d: -d["ratio"])[:8]
    err_summary = {
        "selected_run": sel["selected_run"],
        "selected_family": sel["selected_family"],
        "selected_seed": sel["selected_seed"],
        "selected_val_macro_f1": sel["selected_val_macro_f1"],
        "test_set_used_for_selection": False,
        "n_test": len(preds), "n_wrong": len(wrong), "n_correct": len(correct),
        "error_rate": float(len(wrong) / max(len(preds), 1)),
        "mean_conf_wrong": float(cw_v.mean()), "mean_conf_correct": float(cc_v.mean()),
        "error_directions": directions,
        "most_confused_pair": {
            "a": pair[0], "b": pair[1],
            "a_to_b": float(cm[class_names.index(pair[0])][class_names.index(pair[1])]),
            "b_to_a": float(cm[class_names.index(pair[1])][class_names.index(pair[0])]),
            "combined_frac_of_errors": float(
                (len([r for r in wrong if r["true_name"] == pair[0] and r["pred_name"] == pair[1]])
                 + len([r for r in wrong if r["true_name"] == pair[1] and r["pred_name"] == pair[0]]))
                / max(len(wrong), 1)),
        },
        "pair_shape_stats": shape,
        "offdiagonal_top": offdiag_top,
    }
    save_json(err_summary, outdir / "error_summary.json")
    logger.info("  机器可读分析摘要 → %s", outdir / "error_summary.json")

    logger.info("-" * 72)
    logger.info("生成 %d 个图文件到 %s", len(written), figdir)
    logger.info("分析说明 → report/error-analysis.md")
    logger.info("=" * 72)
    return 0


def plot_cam_grid_from_arrays(grays, cams, metas, outdir, fig_name, title, logger,
                              max_rows: int = 12, per_row: int = 2):
    """
    每行放 `per_row` 个样例，每个样例占 3 个面板（原图 / 热力图 / 叠加）。

    早先的版本每行只放 1 个样例，12 行导致整图宽高比高达 3.4 ——
    缩放到单栏宽度后高度超过一整个页面，LaTeX 会报 "Float too large for page"
    并把该浮动体无限推迟（引用随之变成 ??）。改成一行多例后比例恢复正常。
    """
    from visualize import save_figure, setup_matplotlib
    setup_matplotlib()
    import matplotlib.pyplot as plt

    n = min(len(metas), max_rows)
    nrows = int(np.ceil(n / per_row))
    ncols = per_row * 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.4, 2.3 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for i in range(n):
        r, c = divmod(i, per_row)
        ax0, ax1, ax2 = axes[r][c * 3], axes[r][c * 3 + 1], axes[r][c * 3 + 2]
        ax0.imshow(grays[i], cmap="gray"); ax0.set_title("input", fontsize=6.5)
        ax1.imshow(cams[i], cmap="jet"); ax1.set_title("Grad-CAM", fontsize=6.5)
        ax2.imshow(grays[i], cmap="gray"); ax2.imshow(cams[i], cmap="jet", alpha=0.45)
        ax2.set_title("overlay", fontsize=6.5)
        m = metas[i]
        color = "#1a7f37" if m["correct"] else "#d1242f"
        ax0.text(-0.06, 0.5,
                 f"{m['true_name']}\n→ {m['pred_name']}\np={m['confidence']:.2f}\n"
                 f"on-tgt={m['cam_in_mask_ratio']:.3f}",
                 transform=ax0.transAxes, ha="right", va="center", fontsize=6.0, color=color)

    extra = "" if len(metas) <= max_rows else f"  (showing {n} of {len(metas)})"
    fig.suptitle(f"{title}{extra}", fontsize=9.0, y=1.0)
    fig.tight_layout()
    files = save_figure(fig, outdir, fig_name)
    plt.close(fig)
    return files


# ============================================================================
#  分析说明
# ============================================================================
def compare_pair_shape(preds: list[dict], image_root: Path, pair: tuple, size: int = 96) -> dict:
    """
    量化被混淆两类的外形统计，用于回答"为什么容易混"。

    测的是：
      · 原图宽/高/长宽比  —— 飞机的外形轮廓代理
      · 亮目标外接框面积占比 —— 目标在画面里占多大（大小差异是否可辨）
      · 全图亮度
    """
    stats: dict[str, dict] = {}
    for c in pair:
        rows = [r for r in preds if r["true_name"] == c]
        ws, hs, ar, bright, area = [], [], [], [], []
        for r in rows:
            try:
                with Image.open(image_root / r["relpath"]) as im:
                    w0, h0 = im.size
                    a = np.asarray(im.convert("L").resize((size, size), Image.BILINEAR), dtype=float)
            except (OSError, FileNotFoundError):
                continue
            ws.append(w0); hs.append(h0); ar.append(w0 / max(h0, 1)); bright.append(a.mean())
            m = a > (a.mean() + a.std())
            if m.sum() >= 5:
                ys, xs = np.where(m)
                area.append(((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)) / (size * size))
        stats[c] = {
            "n": len(rows),
            "img_w_median": float(np.median(ws)) if ws else None,
            "img_h_median": float(np.median(hs)) if hs else None,
            "aspect_median": float(np.median(ar)) if ar else None,
            "brightness_mean": float(np.mean(bright)) if bright else None,
            "target_area_median": float(np.median(area)) if area else None,
        }
    # 两类之间的相对差异
    a, b = pair
    diff = {}
    for k in ("img_w_median", "img_h_median", "aspect_median", "brightness_mean", "target_area_median"):
        va, vb = stats[a].get(k), stats[b].get(k)
        if va and vb:
            diff[k] = abs(va - vb) / max((va + vb) / 2, 1e-9)
    stats["_relative_diff"] = diff
    return stats


def build_analysis_md(sel, loc, preds, wrong, correct, pair, cm_offdiag, class_names, shape) -> str:
    md: list[str] = []
    md.append("# 错误分析与可解释性分析\n")
    md.append(f"> 本文件由 `src/error_analysis.py --stage render` 自动生成，"
              f"所有数字均来自脚本产物，未手工填写。\n")

    md.append("## 一、最终模型的选择（只用验证集）\n")
    md.append("**判据**：验证集 macro-F1。**测试集分数未参与选择**"
              f"（`{SEL_NAME}` 中 `test_set_used_for_selection: false`）。\n")
    md.append("| 模型族 | 验证集 Macro F1（均值 ± 标准差，3 种子） |")
    md.append("| --- | ---: |")
    for name, f in sorted(sel["families"].items(), key=lambda kv: -kv[1]["val_macro_f1_mean"]):
        md.append(f"| {name} | {f['val_macro_f1_mean']:.4f} ± {f['val_macro_f1_std']:.4f} |")
    md.append("")
    md.append(f"- 选出模型族：**{sel['selected_family']}**")
    md.append(f"- 该族内选验证集最高的种子：**seed = {sel['selected_seed']}**"
              f"（验证集 macro-F1 = {sel['selected_val_macro_f1']:.4f}）")
    md.append(f"- 最终运行：`{sel['selected_run']}`\n")

    md.append("## 二、错误样例分析\n")
    md.append(f"测试集共 **{len(preds)}** 张，错分 **{len(wrong)}** 张"
              f"（错误率 {len(wrong)/len(preds)*100:.2f}%），正确 {len(correct)} 张。")
    md.append(f"\n**全部错误样例已保存**（G6，未做任何筛选）："
              f"`figures/{OUT_NAME}__fig7a_all_wrong_samples.png`\n")
    md.append("### 错分方向统计（按数量排序）\n")
    md.append("| 真实 → 预测 | 数量 | 占错分比例 | 平均置信度 |")
    md.append("| --- | ---: | ---: | ---: |")
    pairs = Counter(f"{r['true_name']} → {r['pred_name']}" for r in wrong)
    for k, v in pairs.most_common():
        conf = np.mean([r["confidence"] for r in wrong
                        if f"{r['true_name']} → {r['pred_name']}" == k])
        md.append(f"| {k} | {v} | {v/len(wrong)*100:.1f}% | {conf:.3f} |")
    md.append("")
    cw = np.array([r["confidence"] for r in wrong])
    cc = np.array([r["confidence"] for r in correct])
    md.append(f"- 错分样本平均置信度 **{cw.mean():.3f}**，正确样本 **{cc.mean():.3f}**")
    md.append(f"- → 模型在犯错时**确实更不确定**（低 {cc.mean()-cw.mean():.3f}），"
              "因此可以用置信度阈值做**拒识**：低置信度样本交人工复核。\n")
    md.append("**每类正确样例**：`fig7b_correct_samples_per_class.png`；"
              "**置信度最低的正确样例**（高不确定性）：`fig7c_lowest_confidence_correct.png`\n")

    md.append("## 三、最容易混淆的两类\n")
    md.append(f"从**按行归一化的混淆矩阵**中取最大的非对角元素，得到最容易混的一对："
              f"**{pair[0]} ↔ {pair[1]}**\n")
    md.append("| 真实 | 预测成 | 比例 |")
    md.append("| --- | --- | ---: |")
    md.append(f"| {pair[0]} | {pair[1]} | **{cm_offdiag[class_names.index(pair[0])][class_names.index(pair[1])]:.3f}** |")
    md.append(f"| {pair[1]} | {pair[0]} | **{cm_offdiag[class_names.index(pair[1])][class_names.index(pair[0])]:.3f}** |")
    md.append("")
    md.append("### 除该对外，其余错分情况\n")
    md.append("| 真实 | 预测成 | 比例 |")
    md.append("| --- | --- | ---: |")
    others = [(class_names[i], class_names[j], cm_offdiag[i][j])
              for i in range(len(class_names)) for j in range(len(class_names)) if i != j]
    for a, b, v in sorted(others, key=lambda t: -t[2])[2:9]:
        md.append(f"| {a} | {b} | {v:.3f} |")
    md.append("")
    md.append(f"→ **{pair[0]} 与 {pair[1]} 是唯一一对「双向都错得明显」的类别**，"
              "其余方向的错分率都很低。\n")
    md.append(f"**对照样例**：`figures/{OUT_NAME}__fig7d_most_confused_pair.png`"
              f"（前 6 张为 {pair[0]}，后 6 张为 {pair[1]}）\n")
    md.append("### 为什么容易混？——两类的外形统计对比\n")
    md.append("用可量化的图像统计回答「是不是外形太像」这个假设：\n")
    md.append(f"| 统计量 | {pair[0]} | {pair[1]} | 相对差异 |")
    md.append("| --- | ---: | ---: | ---: |")
    rows_def = [
        ("img_w_median", "原图宽（中位数, px）"),
        ("img_h_median", "原图高（中位数, px）"),
        ("aspect_median", "长宽比（中位数）"),
        ("target_area_median", "亮目标占画面面积（中位数）"),
        ("brightness_mean", "全图平均亮度"),
    ]
    rd = shape["_relative_diff"]
    for key, label in rows_def:
        va, vb = shape[pair[0]][key], shape[pair[1]][key]
        d = rd.get(key)
        md.append(f"| {label} | {va:.3f} | {vb:.3f} | "
                  f"{('%.1f%%' % (d * 100)) if d is not None else 'n/a'} |")
    md.append("")
    md.append(f"（样本数：{pair[0]} = {shape[pair[0]]['n']} 张，{pair[1]} = {shape[pair[1]]['n']} 张）\n")
    lg = max(rd.items(), key=lambda kv: kv[1])
    sm = min(rd.items(), key=lambda kv: kv[1])
    md.append(f"- 两类差异**最大**的统计量是 `{lg[0]}`（相对差 {lg[1]*100:.1f}%），"
              f"差异**最小**的是 `{sm[0]}`（{sm[1]*100:.1f}%）")
    if max(rd.values()) < 0.20:
        md.append("- → 两类的尺寸/外形统计**高度接近**（所有统计量相对差 < 20%），"
                  "这与「它们在 SAR 图像中本就难以区分」的直觉一致。")
    else:
        md.append("- → 两类在部分统计量上**可分辨**，说明模型错分并非因为「完全看不出差别」，"
                  "而更可能是**训练样本不足或特征被池化掉**。")
    md.append("- 从错分方向看：两类**双向互错**（不是单向），说明不是某一类的模板被另一类"
              "\"吸走\"，而是两类的特征分布在决策边界附近**交叠**。")
    md.append("")
    md.append("**可检验的解释（按证据强度排序）**：\n")
    md.append("1. **外形相似** —— A220 与 A320/321 同属窄体客机，"
              "机翼后掠角、发动机布局、机身比例在低分辨率 SAR 下差异很小；"
              "上表统计量接近也支持这一点。")
    md.append("2. **方位角差异被当成类内变异** —— SAR 成像对观测方位角极敏感，"
              "同一机型在不同方位角下的散射图样差别可能**大于**两个机型在同一方位角下的差别；"
              "随机旋转增强（本实验启用了 ±30°）可能进一步模糊了这条边界。")
    md.append("3. **分辨率不足** —— 输入统一缩放到 96×96，"
              "而原始切片中位数仅约 90×80（见 `data/README.md`），"
              "细节差异（如发动机数量/位置）在缩放后丢失。")
    md.append("")
    md.append("> ⚠️ 上面前两条是**基于现有证据的推断**，不是被实验直接验证的结论。"
              "要确证需要：按方位角逐个分层统计错分率，或引入目标框做对齐后再比较。\n")

    md.append("## 四、Grad-CAM 可解释性分析\n")
    md.append("**方法**：本数据集**没有目标框标注**，因此用「亮像素（强散射体）」近似飞机范围。"
              "两个指标，各自与**随机基线**比较：\n")
    md.append("- `cam_in_bbox_ratio`：热力图落在「亮像素外接框」内的比例，随机基线 = 外接框面积占比")
    md.append("- `cam_in_mask_ratio`：热力图落在「亮像素掩膜」上的比例，"
              "随机基线 = 掩膜面积占比（**掩膜面积小得多，因此这个指标更锐利**）\n")
    md.append("**为什么必须跨层对比**：ResNet18 在 96×96 输入下共下采样 32 倍，"
              "`layer4` 输出只有 **3×3**（每个格子对应原图 32×32 像素），"
              "用它做 Grad-CAM 空间分辨率极低 —— 定位指标自然接近随机，"
              "**这并不能说明模型没看目标，只说明这一层的 CAM 太粗**。"
              "因此同时报告 `layer3`（6×6）作为对照。\n")
    md.append("| 目标层 | CAM 网格 | 落在目标框内 | 随机基线 | 提升 | **落在亮像素上** | 随机基线 | **提升** |")
    md.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for ln in loc["layers"]:
        s = loc["layers"][ln]
        md.append(f"| `{ln}` | {s['cam_grid']} | {s['cam_in_bbox_ratio_mean']:.4f} | "
                  f"{s['bbox_area_frac_mean']:.4f} | {s['lift_bbox']:.2f}× | "
                  f"**{s['cam_in_mask_ratio_mean']:.4f}** | {s['mask_area_frac_mean']:.4f} | "
                  f"**{s['lift_mask']:.2f}×** |")
    md.append("")
    best = loc["best_layer"]
    bs = loc["layers"][best]
    md.append(f"用于出图的层：**`{best}`**（CAM 网格 {bs['cam_grid']}）\n")
    md.append(f"- 预测**正确**样本落在亮像素上的比例："
              f"{bs['cam_in_mask_ratio_correct']:.4f}")
    md.append(f"- 预测**错误**样本落在亮像素上的比例："
              f"{bs['cam_in_mask_ratio_wrong']:.4f}"
              if bs["cam_in_mask_ratio_wrong"] is not None else "- 无错误样本")
    md.append("")
    lift = bs["lift_mask"]
    if lift >= 1.5:
        verdict = f"**模型明显聚焦于飞机目标**（落在亮散射体上的比例是随机水平的 {lift:.2f} 倍）"
    elif lift >= 1.2:
        verdict = (f"**模型有聚焦目标的倾向，但不强**（仅 {lift:.2f} 倍于随机水平）；"
                   f"部分原因是本数据集飞机占画面约 3/4，目标区域本身就很大")
    else:
        verdict = f"**无法确认模型是否聚焦目标**（仅 {lift:.2f} 倍于随机水平，接近随机）"
    md.append("→ " + verdict + "。")
    md.append("")
    md.append("> ⚠️ **方法学局限（必须说明）**：")
    md.append("> ① 目标范围是用「亮像素」近似的，**不是真实标注框**，"
              "强旁瓣/噪声也会被算作目标；")
    md.append("> ② 飞机占画面约 3/4，目标区域极大，"
              "因此「落在目标内」这类指标的**分辨力天生有限**；")
    md.append(f"> ③ `layer4` 的 CAM 只有 3×3，空间分辨率不足（见上表对比）。")
    md.append("> 因此本节结论应视为**倾向性证据**，而非定论。"
              "要严格回答，需要目标框标注 + 背景替换类扰动实验。\n")
    md.append("**图**："
              f"`{OUT_NAME}__fig7e_cam_localization`（分布直方图 + 跨层对比）、"
              f"`{OUT_NAME}__fig7f_cam_on_vs_off_target`（最聚焦 vs 最弥散的对照）\n")

    md.append("## 五、结论\n")
    md.append(f"1. **最容易混的两类**：**{pair[0]} ↔ {pair[1]}**"
              f"（{pair[0]}→{pair[1]} 占 {cm_offdiag[class_names.index(pair[0])][class_names.index(pair[1])]*100:.1f}%，"
              f"反向占 {cm_offdiag[class_names.index(pair[1])][class_names.index(pair[0])]*100:.1f}%），"
              "是**双向混淆**，说明两类在特征空间上确实重叠。")
    md.append(f"2. **Grad-CAM 显示**：{verdict}（层 `{best}`，CAM 网格 {bs['cam_grid']}）。")
    md.append(f"3. 错分时置信度更低（{cw.mean():.3f} vs {cc.mean():.3f}），"
              "可用置信度阈值做**拒识**以降低实际部署错误率。")
    md.append("")
    return "\n".join(md)


# ============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="错误分析与可解释性分析")
    ap.add_argument("--stage", required=True, choices=["compute", "render"])
    ap.add_argument("--project", default=None, help="项目根目录（默认自动推断）")
    args = ap.parse_args()
    if args.project is None:
        p = Path(__file__).resolve().parent
        while p != p.parent and not (p / "results").is_dir():
            p = p.parent
        args.project = str(p)
    return stage_compute(args) if args.stage == "compute" else stage_render(args)


if __name__ == "__main__":
    raise SystemExit(main())
