"""
Grad-CAM 可解释性观察 + 背景消融（**纯 torch 计算，不导入 matplotlib**）。

⚠️ 为什么不在这里画图：
   本机 conda 环境同时存在 MKL 的 `libiomp5md.dll` 与 llvm-openmp 的 `libomp.dll`。
   在同一进程里"先跑 torch 反向传播、再用 matplotlib 渲染"会触发
        OMP: Error #15: Initializing libomp.dll, but found libiomp5md.dll already initialized.
   官方给出的 `KMP_DUPLICATE_LIB_OK=TRUE` 是"不安全、不推荐"的绕过；而删除 llvm-openmp
   又无法由 environment.yml 复现。因此采用**结构性方案**：
       本脚本只负责计算并落盘 → `src/visualize.py --fig cam` 负责渲染。
   两个进程各自只加载一套运行时，冲突自然消失。

对应约束：
  IC4 —— 观察四类对象：分对且高置信 / 分错 / 高不确定性 / **分对但热力图落在背景**
  G5  —— 热力图数据与量化指标全部落盘，可追溯、可复算
  G6  —— 分错的样例全部保留，不做筛选

**量化设计**：本数据集无目标框标注，无法直接判断"热力图是否落在目标上"。
这里用可复现的代理指标
    border_mass_ratio = 热力图在「中心 50% 区域」之外的响应占比
越接近 1 越关注背景。并用**背景消融**（裁掉外围后重新评估）交叉验证。

用法：
    python src/gradcam.py --run results/raw/formal/resnet18_finetune_seed42_20261008-120000
    python src/visualize.py --run <同一运行目录> --fig cam     # 渲染 fig5*
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# ⚠️ 导入顺序在本机环境下**至关重要**（实测结论）：
#     PIL 必须在 torchvision 与 CUDA 初始化**之前**导入，否则报
#         ImportError: DLL load failed while importing _imaging
#     实测： PIL → torch → CUDA → torchvision   ✅
#            torchvision → PIL                 ❌
#            torch → CUDA → torchvision        ❌
#     原因：torchvision 的 C++ 扩展会抢先加载一组 DLL，之后再导入 PIL 的
#           C 扩展 _imaging 就会失败。先导入 PIL 可避免这个次序问题。
from PIL import Image  # noqa: E402  —— 必须放在最前面，勿调整顺序

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import build_model                                        # noqa: E402
from utils import (detect_run_mode, get_logger, load_config, load_json,   # noqa: E402
                   save_json, set_seed)

CAM_DATA = "cam_data.npz"
CAM_MANIFEST = "cam_manifest.json"


# ============================================================================
#  Grad-CAM（手写 hook 实现，不依赖第三方库）
# ============================================================================
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.acts = None
        self.grads = None
        self._h1 = target_layer.register_forward_hook(self._forward_hook)
        self._h2 = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.acts = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.grads = grad_output[0].detach()

    def remove(self):
        self._h1.remove()
        self._h2.remove()

    def compute(self, x: torch.Tensor, class_idx: int | None = None):
        """返回 (cam[H,W] ∈ [0,1], 使用的类别索引)"""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1)[0].item())
        logits[0, class_idx].backward()
        if self.grads is None or self.acts is None:
            raise RuntimeError(
                "Grad-CAM 拿不到梯度。请确认目标层参数 requires_grad=True"
                "（本脚本已用 temporarily_enable_grad 处理冻结骨干的情况）"
            )
        weights = self.grads.mean(dim=(2, 3), keepdim=True)          # 空间维全局平均
        cam = torch.relu((weights * self.acts).sum(dim=1))[0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy(), class_idx


def get_target_layer(model: nn.Module, model_type: str) -> nn.Module:
    """
    Grad-CAM 的目标层：
      · ResNet 系列 → `backbone.layer4` 的最后一个残差块（保留 12×12 空间分辨率）
      · 自定义 CNN → `features` 里最后一个卷积层

    ⚠️ 用**属性判断**而不是类型字符串判断 ——
    早期版本只认字符串 "ResNet18"，模型类型改成 "ResNet18Transfer" 后
    就落到了 CNN 分支并报 `AttributeError: 'ResNet18Transfer' object has no attribute 'features'`。
    """
    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "layer4"):
        return backbone.layer4[-1]

    features = getattr(model, "features", None)
    if features is not None:
        convs = [m for m in features.modules() if isinstance(m, nn.Conv2d)]
        if convs:
            return convs[-1]

    raise RuntimeError(f"无法为模型类型 {model_type!r} 找到 Grad-CAM 目标层")


def temporarily_enable_grad(model: nn.Module) -> dict:
    """冻结骨干时梯度不会流到目标层，这里临时打开（不更新参数），用完恢复。"""
    saved = {}
    for name, p in model.named_parameters():
        saved[name] = p.requires_grad
        p.requires_grad_(True)
    return saved


def restore_grad(model: nn.Module, saved: dict) -> None:
    for name, p in model.named_parameters():
        if name in saved:
            p.requires_grad_(saved[name])


# ============================================================================
#  量化：热力图落在背景的占比
# ============================================================================
def border_mass_ratio(cam: np.ndarray, center_frac: float = 0.5) -> float:
    """响应落在「中心 center_frac 区域」之外的比例。越接近 1 越关注背景。"""
    h, w = cam.shape
    ch, cw = max(1, int(h * center_frac)), max(1, int(w * center_frac))
    r0, c0 = (h - ch) // 2, (w - cw) // 2
    total = float(cam.sum()) + 1e-12
    inner = float(cam[r0 : r0 + ch, c0 : c0 + cw].sum())
    return (total - inner) / total


# ============================================================================
#  背景消融
# ============================================================================
class BorderFill:
    """
    把图像外围 `border_ratio` 的区域替换为常数值（归一化后 0 = 训练集均值）。

    与「裁剪」相比，这个变体**完整保留飞机本体**，只移除背景纹理，
    因此能更干净地回答"模型是靠飞机还是靠背景"。

    （裁剪的缺陷：本数据集是飞机切片图，裁掉外围很可能直接把机翼/机头切掉，
      于是准确率下降的原因无法区分是"背景被移除"还是"目标被破坏"。）
    """

    def __init__(self, border_ratio: float, fill: float = 0.0):
        self.border_ratio = border_ratio
        self.fill = fill

    def __call__(self, t: torch.Tensor) -> torch.Tensor:  # 输入为已归一化的 CHW
        _, h, w = t.shape
        k = int(round(min(h, w) * self.border_ratio))
        if k <= 0:
            return t
        out = t.clone()
        out[:, :k, :] = self.fill
        out[:, h - k :, :] = self.fill
        out[:, :, :k] = self.fill
        out[:, :, w - k :] = self.fill
        return out


def _accuracy_with(model, records, tf, device, batch_size: int = 32) -> float:
    from torch.utils.data import DataLoader

    from dataset import SarAircraftDataset

    loader = DataLoader(SarAircraftDataset(records, transform=tf),
                        batch_size=batch_size, shuffle=False)
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            correct += int((model(x.to(device)).argmax(1).cpu() == y).sum())
    return correct / max(len(records), 1)


def _target_extent_ratio(records, image_root: Path, size: int, border_ratio: float,
                         max_n: int = 200) -> float:
    """
    估计"填充外围会切掉多少机体"。

    用亮像素（强散射体）的外接框近似飞机范围，计算它落在「保留的中心区」之外的比例。
    该值越大，说明消融越可能**同时切除目标本体**，结论就越不可信。

    本数据集实测：中位数约 **0.80**，且 0% 的样本飞机完全落在中心区内
    —— 说明这个消融在本数据集上**是被混淆的**（飞机占画面约 3/4）。
    """
    k = int(round(size * border_ratio))
    inner = size - 2 * k
    outs = []
    for r in records[:max_n]:
        try:
            with Image.open(image_root / r["relpath"]) as im:
                a = np.asarray(im.convert("L").resize((size, size), Image.BILINEAR), dtype=float)
        except (OSError, FileNotFoundError):
            continue
        m = a > (a.mean() + a.std())
        if m.sum() < 5:
            continue
        ys, xs = np.where(m)
        iw = max(0, min(xs.max(), k + inner - 1) - max(xs.min(), k) + 1)
        ih = max(0, min(ys.max(), k + inner - 1) - max(ys.min(), k) + 1)
        box = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
        outs.append(1 - (iw * ih) / max(box, 1))
    return float(np.median(outs)) if outs else float("nan")


def background_ablation(run_dir: Path, cfg: dict, ckpt: dict, border_ratio: float, device, logger) -> dict:
    """
    三个变体在**同一进程、同一批测试样本**上对比：

      full   : Resize + Normalize                      （基线）
      fill   : 再把外围 border 替换为训练集均值         ← **关键诊断**
      crop   : 中心裁剪掉 border 后放大回原尺寸         （有混淆，仅作参考）

    判据：
      · fill 后准确率**基本不变** → 模型主要看目标，背景贡献有限
      · fill 后准确率**明显下降** → 模型确实依赖背景纹理（假设 H3 成立）
    """
    from torchvision import transforms

    from dataset import load_split_index

    project_root = Path(cfg["_project_root"])
    image_root = project_root / cfg["paths"]["data_root"]
    size = int(ckpt.get("image_size", cfg["data"]["image_size"]))
    mean = ckpt["normalize"]["mean"]; std = ckpt["normalize"]["std"]

    records = [r for r in load_split_index(run_dir / "split_index.csv", image_root)
               if r["split"] == "test"]

    model_cfg = dict(ckpt["model_cfg"]); model_cfg["pretrained"] = False
    model = build_model(model_cfg, len(ckpt["class_names"]), logger=None)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    base = [transforms.Resize((size, size)), transforms.ToTensor(),
            transforms.Normalize([mean], [std])]
    tf_full = transforms.Compose(base)
    tf_fill = transforms.Compose(base + [BorderFill(border_ratio)])
    keep = 1.0 - border_ratio
    side = max(8, int(size * keep))
    tf_crop = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop((side, side)),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([mean], [std]),
    ])

    acc_full = _accuracy_with(model, records, tf_full, device)
    acc_fill = _accuracy_with(model, records, tf_fill, device)
    acc_crop = _accuracy_with(model, records, tf_crop, device)

    d_fill = acc_fill - acc_full
    d_crop = acc_crop - acc_full

    # 关键：先判断这个消融本身是不是被混淆了
    target_out = _target_extent_ratio(records, image_root, size, border_ratio)
    confounded = bool(target_out > 0.4)

    if confounded:
        verdict = (f"无法判定 —— 填充外围会同时切除大部分机体"
                   f"（目标外接框有 {target_out:.0%} 落在保留区之外），"
                   f"该消融不能区分「背景」与「目标」，H3 不能据此确认")
    elif abs(d_fill) <= 0.02:
        verdict = "背景贡献有限；模型主要看目标区域（假设 H3 不成立）"
    elif d_fill < 0:
        verdict = "模型依赖背景纹理（假设 H3 成立）"
    else:
        verdict = "背景被替换后反而变好，需进一步分析"

    logger.info("  背景消融（%d 个测试样本，外围 %.0f%%，保留中心 %d×%d）:",
                len(records), border_ratio * 100, size - 2 * int(round(size * border_ratio)),
                size - 2 * int(round(size * border_ratio)))
    logger.info("    ① 原图（基线）              acc = %.4f", acc_full)
    logger.info("    ② 外围替换为训练集均值      acc = %.4f  (Δ %+.4f)", acc_fill, d_fill)
    logger.info("    ③ 中心裁剪后放大            acc = %.4f  (Δ %+.4f)", acc_crop, d_crop)
    logger.info("    混淆检查: 目标外接框有 %.0f%% 落在保留区之外 → %s",
                target_out * 100, "❌ 消融被混淆" if confounded else "✅ 消融可解释")
    logger.info("    结论: %s", verdict)
    if confounded:
        logger.warning("    注：本数据集的飞机约占画面 3/4，填充/裁剪外围都会切掉机体，")
        logger.warning("        因此 ②③ 的准确率下降**不能**解释为「模型依赖背景」。")
        logger.warning("        要真正回答 H3，需要目标框标注，或用只扰动背景、不动目标的方案。")

    return {
        "border_ratio": border_ratio,
        "n_test": len(records),
        "baseline_accuracy": acc_full,
        "border_filled_accuracy": acc_fill,
        "delta_fill": d_fill,
        "center_cropped_accuracy": acc_crop,
        "delta_crop": d_crop,
        "target_bbox_outside_center_ratio": target_out,
        "confounded": confounded,
        "verdict": verdict,
        "enabled": True,
        "note": ("fill = 外围替换为训练集均值；crop = 中心裁剪后放大。"
                 "本数据集目标占画面约 3/4，两种扰动都会切除机体，结论仅供参考。"),
    }


# ============================================================================
#  辅助
# ============================================================================
def read_predictions(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "relpath": r["relpath"], "true_name": r["true_name"], "pred_name": r["pred_name"],
                "true_label": int(r["true_label"]), "pred_label": int(r["pred_label"]),
                "confidence": float(r["confidence"]),
                "correct": r["correct"].lower() in {"true", "1", "yes"},
            })
    return rows


# ============================================================================
#  主流程
# ============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="Grad-CAM 计算 + 背景消融（不画图，画图见 visualize.py --fig cam）")
    ap.add_argument("--run", required=True)
    ap.add_argument("--bg-threshold", type=float, default=0.5,
                    help="border_mass_ratio 超过该值即判为「看背景」（默认 0.5）")
    ap.add_argument("--max-wrong", type=int, default=None, help="分错样例最多算几张热力图")
    ap.add_argument("--no-ablation", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"运行目录不存在: {run_dir}")
    ckpt_path = run_dir / "best_model.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"缺少 {ckpt_path}")

    logger = get_logger(f"gradcam.{run_dir.name}", run_dir / "gradcam.log")
    cfg = load_config(run_dir / "config_resolved.yaml")
    project_root = run_dir
    while project_root != project_root.parent and not (project_root / "results").is_dir():
        project_root = project_root.parent
    cfg["_project_root"] = str(project_root)

    metrics_path = run_dir / "test_metrics.json"
    metrics = load_json(metrics_path) if metrics_path.is_file() else {}
    mode = detect_run_mode(run_dir)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    class_names = ckpt["class_names"]
    model_name = ckpt.get("model_name", "model")
    model_type = ckpt["model_cfg"]["type"]
    seed = ckpt.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = dict(ckpt["model_cfg"]); model_cfg["pretrained"] = False
    model = build_model(model_cfg, len(class_names), logger=None)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    logger.info("=" * 72)
    logger.info("Grad-CAM 计算: %s | 模型 %s (%s) | 设备 %s", run_dir.name, model_name, model_type, device)
    if not metrics:
        logger.warning("未找到 test_metrics.json，建议先运行 evaluate.py")

    preds = read_predictions(run_dir / "predictions.csv")
    if not preds:
        raise RuntimeError("缺少 predictions.csv，请先运行 src/evaluate.py")

    from torchvision import transforms
    size = int(ckpt.get("image_size", cfg["data"]["image_size"]))
    mean = ckpt["normalize"]["mean"]; std = ckpt["normalize"]["std"]
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                             transforms.Normalize([mean], [std])])
    tf_disp = transforms.Resize((size, size))
    image_root = project_root / cfg["paths"]["data_root"]

    saved_grad = temporarily_enable_grad(model)
    engine = GradCAM(model, get_target_layer(model, model_type))

    def make_sample(rec: dict) -> dict | None:
        path = image_root / rec["relpath"]
        try:
            with Image.open(path) as im:
                gray = im.convert("L")
                disp = np.asarray(tf_disp(gray), dtype=np.float32)
                x = tf(gray).unsqueeze(0).to(device)
        except (OSError, FileNotFoundError) as exc:
            logger.warning("    读取失败 %s: %s", rec["relpath"], exc)
            return None
        cam, _ = engine.compute(x, class_idx=rec["pred_label"])
        # 把 CAM 放大到展示尺寸
        cam = np.asarray(
            Image.fromarray((cam * 255).astype(np.uint8)).resize(
                (disp.shape[1], disp.shape[0]), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        return {**rec, "gray": disp, "cam": cam, "border_mass": border_mass_ratio(cam)}

    # ---- 选取四类观察对象 ----
    correct = sorted([r for r in preds if r["correct"]], key=lambda r: -r["confidence"])
    per_cls_n = int((cfg.get("gradcam", {}) or {}).get("n_correct_per_class", 3))
    by_cls: dict[str, list[dict]] = {}
    for r in correct:
        by_cls.setdefault(r["true_name"], []).append(r)
    sel_correct = [r for c in sorted(by_cls) for r in by_cls[c][:per_cls_n]]

    wrong = [r for r in preds if not r["correct"]]
    limit = args.max_wrong if args.max_wrong is not None else int(
        (cfg.get("gradcam", {}) or {}).get("n_wrong_max", 20) or 20)
    sel_wrong = wrong[:limit]

    logger.info("-" * 72)
    logger.info("选取样例: 分对 %d 张（每类最多 %d）, 分错 %d/%d 张",
                len(sel_correct), per_cls_n, len(sel_wrong), len(wrong))

    groups: dict[str, list[dict]] = {"correct": [], "wrong": [], "background": []}
    arrays: dict[str, np.ndarray] = {}

    for gname, sel in (("correct", sel_correct), ("wrong", sel_wrong)):
        samples = [s for s in (make_sample(r) for r in sel) if s]
        groups[gname] = [
            {k: s[k] for k in ("relpath", "true_name", "pred_name", "confidence", "correct", "border_mass")}
            for s in samples
        ]
        if samples:
            arrays[f"{gname}_gray"] = np.stack([s["gray"] for s in samples])
            arrays[f"{gname}_cam"] = np.stack([s["cam"] for s in samples])
        logger.info("  %-10s 完成 %d 张", gname, len(samples))

    # 分对了但热力图偏背景（假设 H3 的证据）
    bg_rel = {s["relpath"] for s in groups["correct"] if s["border_mass"] > args.bg_threshold}
    sel_bg = [r for r in correct if r["relpath"] in bg_rel][:12]
    samples_bg = [s for s in (make_sample(r) for r in sel_bg) if s]
    groups["background"] = [
        {k: s[k] for k in ("relpath", "true_name", "pred_name", "confidence", "correct", "border_mass")}
        for s in samples_bg
    ]
    if samples_bg:
        arrays["background_gray"] = np.stack([s["gray"] for s in samples_bg])
        arrays["background_cam"] = np.stack([s["cam"] for s in samples_bg])
    logger.info("  分对但 CAM 偏背景（border > %.2f）: %d 张", args.bg_threshold, len(samples_bg))

    # ---- 背景消融 ----
    ablation = {"enabled": False}
    if not args.no_ablation and metrics:
        logger.info("-" * 72); logger.info("背景消融")
        res = background_ablation(
            run_dir, cfg, ckpt,
            float((cfg.get("background_ablation", {}) or {}).get("crop_ratio", 0.3)), device, logger)
        res["evaluate_metrics_accuracy"] = metrics["accuracy"]
        ablation = res

    engine.remove()
    restore_grad(model, saved_grad)

    # ---- 落盘（G5）----
    np.savez_compressed(run_dir / CAM_DATA, **arrays)
    manifest = {
        "run": run_dir.name,
        "model": model_name,
        "model_type": model_type,
        "seed": seed,
        "mode": mode,
        "image_size": size,
        "groups": groups,
        "border_mass_ratio": {
            "threshold": args.bg_threshold,
            "mean": float(np.mean([s["border_mass"] for g in groups.values() for s in g]))
            if any(groups.values()) else None,
            "n_above_threshold": len(samples_bg),
        },
        "background_ablation": ablation,
        "note": ("本数据集无目标框标注；border_mass_ratio = 热力图在中心 50% 区域之外的响应占比，"
                 "是可复现的代理指标。图由 src/visualize.py --fig cam 渲染（避免 torch 与 "
                 "matplotlib 同进程触发 OMP 冲突）"),
    }
    save_json(manifest, run_dir / CAM_MANIFEST)

    logger.info("=" * 72)
    logger.info("Grad-CAM 计算完成")
    for g, v in groups.items():
        logger.info("  %-12s %d 张", g, len(v))
    logger.info("  产物: %s + %s", CAM_DATA, CAM_MANIFEST)
    if ablation.get("enabled"):
        logger.info("  背景消融: Δfill = %+.4f  → %s", ablation["delta_fill"], ablation["verdict"])
    logger.info("  下一步: python src/visualize.py --run %s --fig cam", run_dir)
    logger.info("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
