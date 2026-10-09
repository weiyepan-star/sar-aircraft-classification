"""
SAR-ACD 数据加载：扫描 → 去重 → 分层划分 → 自检 → 归一化统计 → DataLoader。

对应约束：
  G2  —— 三集严格隔离；划分后强制断言（路径交集 + MD5 交集 + 分层偏差）
  G3  —— 固定随机种子分层抽样；划分可落盘复用（所有模型共用同一份划分）
  G5  —— 所有统计量落盘（split_index.csv / dedup_report.csv / train_stats.json）
  IC0 —— 尺寸、通道、损坏、重复检查
  IC2 —— 增强只在训练集；验证/测试只做 resize + 归一化
  IC0.3 —— 归一化统计量只在训练集上计算

用法：
    从命令行单独运行可只做数据审计：
        python src/dataset.py --config configs/smoke_test.yaml
"""

from __future__ import annotations

import csv
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger, load_config, load_json, save_json  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ============================================================================
#  1. 扫描
# ============================================================================
def scan_dataset(image_root: str | Path) -> list[dict]:
    """
    按文件夹名作为类别扫描数据集。

    返回：[{"path", "relpath", "class_name", "label", "md5"}, ...]
    顺序：先按类别名排序，再按文件名排序（保证跨机器一致）。
    """
    image_root = Path(image_root)
    if not image_root.is_dir():
        raise FileNotFoundError(f"图像根目录不存在: {image_root}")

    class_dirs = sorted([d for d in image_root.iterdir() if d.is_dir()], key=lambda p: p.name)
    if not class_dirs:
        raise RuntimeError(f"在 {image_root} 下没有找到任何类别子目录")

    records: list[dict] = []
    for label, cdir in enumerate(class_dirs):
        files = sorted(
            [f for f in cdir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS],
            key=lambda p: p.name,
        )
        for f in files:
            records.append(
                {
                    "path": str(f),
                    "relpath": f"{cdir.name}/{f.name}",
                    "class_name": cdir.name,
                    "label": label,
                    "md5": md5_of(f),
                }
            )
    return records


def md5_of(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# ============================================================================
#  2. 图像质量检查（IC0）
# ============================================================================
def inspect_images(records: list[dict], logger=None, sample: int | None = None) -> dict:
    """
    逐张检查：能否解码、通道数、尺寸。返回统计字典（可落盘）。

    sample=None 表示全量检查（正式审计用）；冒烟测试可传小数以加速。
    """
    subset = records if sample is None else records[:sample]
    n_bad = 0
    modes = Counter()
    sizes: list[tuple[int, int]] = []
    bad_files: list[str] = []

    for rec in subset:
        try:
            with Image.open(rec["path"]) as im:
                im.verify()  # 校验文件完整性
            with Image.open(rec["path"]) as im:
                modes[im.mode] += 1
                sizes.append(im.size)
        except Exception as exc:  # noqa: BLE001
            n_bad += 1
            bad_files.append(f"{rec['relpath']}: {exc}")
            if logger:
                logger.warning("损坏文件: %s (%s)", rec["relpath"], exc)

    w = np.array([s[0] for s in sizes]) if sizes else np.array([0])
    h = np.array([s[1] for s in sizes]) if sizes else np.array([0])
    info = {
        "checked": len(subset),
        "n_corrupted": n_bad,
        "corrupted_files": bad_files[:50],
        "modes": dict(modes),
        "n_unique_sizes": int(len(set(map(tuple, sizes)))),
        "width": {"min": int(w.min()), "max": int(w.max()), "mean": float(w.mean())},
        "height": {"min": int(h.min()), "max": int(h.max()), "mean": float(h.mean())},
    }
    if logger:
        logger.info(
            "图像检查: %d 张, 损坏 %d 张, 模式 %s, 唯一尺寸 %d 种, 尺寸 %d×%d ~ %d×%d",
            info["checked"], n_bad, info["modes"], info["n_unique_sizes"],
            int(w.min()), int(h.min()), int(w.max()), int(h.max()),
        )
    return info


# ============================================================================
#  3. 去重（IC0.1 / 必须在划分之前执行）
# ============================================================================
def deduplicate(
    records: list[dict],
    cross_class_policy: str = "drop_both",
    same_class_policy: str = "keep_first",
    logger=None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    按 MD5 去重。

    - 跨类别重复（同一张图出现在两个类别下）：默认 `drop_both` —— **两侧全部剔除**
      （无法判断哪个标签正确，保留任一侧都是注入错误标签）
    - 类内重复：默认 `keep_first` —— 保留第一张

    返回 (kept, dropped, report_rows)
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        groups[rec["md5"]].append(rec)

    kept: list[dict] = []
    dropped: list[dict] = []
    report: list[dict] = []

    n_cross = n_same = 0
    for md5, group in groups.items():
        if len(group) == 1:
            kept.append(group[0])
            continue

        classes = {r["class_name"] for r in group}
        group_sorted = sorted(group, key=lambda r: r["relpath"])

        if len(classes) > 1:  # 跨类别重复
            n_cross += 1
            for r in group_sorted:
                dropped.append(dict(r, drop_reason="cross_class_duplicate"))
            report.append(
                {
                    "md5": md5,
                    "kind": "cross_class",
                    "n_files": len(group),
                    "classes": "|".join(sorted(classes)),
                    "kept": "",
                    "dropped": "|".join(x["relpath"] for x in group_sorted),
                }
            )
        else:  # 类内重复
            n_same += 1
            keep = group_sorted[0]
            kept.append(keep)
            for r in group_sorted[1:]:
                r = dict(r, drop_reason="same_class_duplicate")
                dropped.append(r)
            report.append(
                {
                    "md5": md5,
                    "kind": "same_class",
                    "n_files": len(group),
                    "classes": sorted(classes)[0],
                    "kept": keep["relpath"],
                    "dropped": "|".join(x["relpath"] for x in group_sorted[1:]),
                }
            )

    kept.sort(key=lambda r: (r["class_name"], r["relpath"]))
    dropped.sort(key=lambda r: r["relpath"])

    if logger:
        logger.info(
            "去重: %d → %d 张（删除 %d）| 跨类别重复 %d 组（两侧全删），类内重复 %d 组（保留 1 张）",
            len(records), len(kept), len(dropped), n_cross, n_same,
        )
        if n_cross:
            logger.warning(
                "发现 %d 组**跨类别重复**：同一张图被标成两种类别，已两侧剔除（见 dedup_report.csv）",
                n_cross,
            )
    return kept, dropped, report


# ============================================================================
#  4. 限制每类张数（仅冒烟测试用）
# ============================================================================
def limit_per_class(records: list[dict], max_per_class: int | None, logger=None) -> list[dict]:
    if not max_per_class:
        return records
    by_cls: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_cls[rec["class_name"]].append(rec)
    out: list[dict] = []
    for cname in sorted(by_cls):
        picked = sorted(by_cls[cname], key=lambda r: r["relpath"])[:max_per_class]
        out.extend(picked)
        if logger:
            logger.info("  [限制] %s: %d → %d 张", cname, len(by_cls[cname]), len(picked))
    out.sort(key=lambda r: (r["class_name"], r["relpath"]))
    if logger:
        logger.info("限制每类最多 %d 张 → 共 %d 张（**仅用于冒烟测试**）", max_per_class, len(out))
    return out


# ============================================================================
#  5. 分层划分（G3 / IC1）
# ============================================================================
def stratified_split(
    records: list[dict], ratios: dict, seed: int, logger=None
) -> list[dict]:
    """
    分层划分：先在**每个类别内部**按比例切分，再合并。

    每类分配规则：
        n_val  = max(1, floor(n * val_ratio))
        n_test = max(1, floor(n * test_ratio))
        n_train = n - n_val - n_test     （余数全部归训练集）
    这样可保证**每个类别在三个集合中的比例一致**。
    """
    val_r = float(ratios["val"])
    test_r = float(ratios["test"])

    by_cls: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_cls[rec["class_name"]].append(rec)

    rng = np.random.default_rng(seed)
    out: list[dict] = []

    for cname in sorted(by_cls):
        group = sorted(by_cls[cname], key=lambda r: r["relpath"])
        n = len(group)
        if n < 3:
            raise RuntimeError(
                f"类别 {cname} 只有 {n} 张，无法三分（每条至少需要 3 张）。"
                f"请检查 data.max_per_class 设置。"
            )
        n_val = max(1, int(np.floor(n * val_r)))
        n_test = max(1, int(np.floor(n * test_r)))
        if n - n_val - n_test < 1:
            n_val = 1
            n_test = 1
        n_train = n - n_val - n_test

        perm = rng.permutation(n)  # 固定种子 → 每次结果相同
        idx_train = set(perm[:n_train].tolist())
        idx_val = set(perm[n_train : n_train + n_val].tolist())
        idx_test = set(perm[n_train + n_val :].tolist())

        for i, rec in enumerate(group):
            split = "train" if i in idx_train else "val" if i in idx_val else "test"
            out.append(dict(rec, split=split))

    merge_order = {"train": 0, "val": 1, "test": 2}
    out.sort(key=lambda r: (r["class_name"], r["relpath"]))
    if logger:
        c = Counter(r["split"] for r in out)
        logger.info(
            "分层划分(seed=%d): train=%d, val=%d, test=%d, 合计=%d",
            seed, c["train"], c["val"], c["test"], len(out),
        )
    return out


def assert_split_valid(records: list[dict], stratify_tol: float = 0.01, logger=None) -> dict:
    """
    划分后的**强制自检**（G2 / IC1）。任一项不通过 → 抛异常，阻断训练。

    1. 三个集合的路径交集为空
    2. 三个集合的 MD5 交集为空（防止漏网重复图跨集）
    3. 各集合类别比例与总体一致（偏差 <= stratify_tol）
    4. 三集合计 == 总数
    """
    sets = {s: [r for r in records if r["split"] == s] for s in ("train", "val", "test")}
    paths = {s: {r["relpath"] for r in v} for s, v in sets.items()}
    md5s = {s: {r["md5"] for r in v} for s, v in sets.items()}

    # 1) 路径交集
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = paths[a] & paths[b]
        if inter:
            raise AssertionError(f"[自检失败] {a} 与 {b} 的样本路径交集非空: {list(inter)[:5]}")

    # 2) MD5 交集
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = md5s[a] & md5s[b]
        if inter:
            raise AssertionError(
                f"[自检失败] {a} 与 {b} 的内容(MD5)交集非空 —— 存在数据泄漏: {list(inter)[:5]}"
            )

    # 3) 分层偏差
    total_cls = Counter(r["class_name"] for r in records)
    overall = {c: n / len(records) for c, n in total_cls.items()}
    max_dev = 0.0
    for s, v in sets.items():
        if not v:
            raise AssertionError(f"[自检失败] 集合 {s} 为空")
        cnt = Counter(r["class_name"] for r in v)
        for cname, prop in overall.items():
            dev = abs(cnt.get(cname, 0) / len(v) - prop)
            max_dev = max(max_dev, dev)

    # 4) 合计
    if sum(len(v) for v in sets.values()) != len(records):
        raise AssertionError("[自检失败] 三集合计 != 总数")

    result = {
        "n": {s: len(v) for s, v in sets.items()},
        "path_overlap": 0,
        "md5_overlap": 0,
        "max_class_proportion_deviation": round(max_dev, 4),
        "stratify_tol": stratify_tol,
        "passed": max_dev <= stratify_tol,
    }
    if logger:
        logger.info(
            "划分自检: 路径交集=空 ✅ | MD5 交集=空 ✅ | 类别比例最大偏差=%.4f (容差 %.2f) %s",
            max_dev, stratify_tol, "✅" if result["passed"] else "❌",
        )
    if not result["passed"]:
        raise AssertionError(
            f"[自检失败] 分层偏差 {max_dev:.4f} 超过容差 {stratify_tol}，分层未生效"
        )
    return result


# ============================================================================
#  6. 落盘 / 读取划分索引
# ============================================================================
def save_split_index(records: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["relpath", "label", "class_name", "split", "md5"])
        w.writeheader()
        for r in sorted(records, key=lambda x: x["relpath"]):
            w.writerow({k: r[k] for k in w.fieldnames})
    return path


def load_split_index(path: str | Path, image_root: str | Path) -> list[dict]:
    """读取冻结的划分索引，恢复成 records（用于正式实验：所有模型共用同一份划分）。"""
    image_root = Path(image_root)
    out: list[dict] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out.append(
                {
                    "path": str(image_root / row["relpath"]),
                    "relpath": row["relpath"],
                    "class_name": row["class_name"],
                    "label": int(row["label"]),
                    "split": row["split"],
                    "md5": row["md5"],
                }
            )
    return out


def save_dedup_report(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["md5", "kind", "n_files", "classes", "kept", "dropped"])
        w.writeheader()
        w.writerows(rows)
    return path


# ============================================================================
#  7. 归一化统计量（IC0.3：只能在训练集上算）
# ============================================================================
def compute_mean_std(records: list[dict], image_size: int, logger=None) -> tuple[float, float]:
    """只用**训练集**计算灰度均值/标准差（在 [0,1] 尺度上）。"""
    train = [r for r in records if r["split"] == "train"]
    if not train:
        raise RuntimeError("训练集为空，无法计算归一化统计量")

    total = 0.0
    total_sq = 0.0
    count = 0
    for rec in train:
        with Image.open(rec["path"]) as im:
            arr = np.asarray(
                im.convert("L").resize((image_size, image_size), Image.BILINEAR), dtype=np.float64
            ) / 255.0
        total += arr.sum()
        total_sq += (arr**2).sum()
        count += arr.size

    mean = total / count
    var = max(total_sq / count - mean**2, 1e-12)
    std = float(np.sqrt(var))
    if logger:
        logger.info("归一化统计量(仅训练集 %d 张): mean=%.4f, std=%.4f", len(train), mean, std)
    return float(mean), float(std)


# ============================================================================
#  8. Dataset 与 Transform（IC2）
# ============================================================================
def build_transforms(cfg: dict, split: str, mean: float, std: float):
    """
    训练集：Resize + 随机翻转 + 随机旋转 + 随机仿射 + ToTensor + Normalize
    验证/测试：**只做 Resize + ToTensor + Normalize**（IC2：不允许任何随机变换）
    """
    from torchvision import transforms

    size = int(cfg["data"]["image_size"])
    common = [transforms.Resize((size, size)), transforms.ToTensor()]

    if split != "train":
        return transforms.Compose(common + [transforms.Normalize([mean], [std])])

    aug = (cfg.get("augment", {}) or {}).get("train", {}) or {}
    ops = [transforms.Resize((size, size))]

    if aug.get("horizontal_flip"):
        ops.append(transforms.RandomHorizontalFlip(0.5))
    if aug.get("vertical_flip"):
        ops.append(transforms.RandomVerticalFlip(0.5))
    rot = float(aug.get("rotation_degrees") or 0)
    if rot:
        ops.append(transforms.RandomRotation(rot))

    translate = float(aug.get("translate") or 0)
    scale = aug.get("scale")
    if translate or scale:
        sc = tuple(scale) if scale else None
        ops.append(
            transforms.RandomAffine(
                degrees=0, translate=(translate, translate) if translate else None, scale=sc
            )
        )

    return transforms.Compose(ops + common[1:] + [transforms.Normalize([mean], [std])])


class SarAircraftDataset:
    """返回 (image_tensor[C,H,W], label)。灰度图为 1 通道。"""

    def __init__(self, records: list[dict], transform=None):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        from PIL import Image as _Image

        rec = self.records[idx]
        with _Image.open(rec["path"]) as im:
            img = im.convert("L")
            if self.transform is not None:
                img = self.transform(img)
        return img, rec["label"]


# ============================================================================
#  9. 组装 DataLoader
# ============================================================================
def build_dataloaders(cfg: dict, run_dir: str | Path, logger=None,
                      artifact_dir: str | Path | None = None) -> dict:
    """
    完成「扫描 → 检查 → 去重 → 限制 → 划分 → 自检 → 统计 → DataLoader」全流程，
    并把中间产物写入 artifact_dir（默认 = run_dir）（G5 可追溯）。

    参数 `artifact_dir` 用于 **评估阶段**：评估会复用训练时的划分索引，
    此时中间产物应写到别处，避免覆写训练阶段留下的 `split_index.csv`。

    返回 dict：loaders / records / class_names / stats / audit
    """
    from torch.utils.data import DataLoader

    from utils import make_generator, seed_worker

    if logger is None:
        logger = get_logger("dataset")

    root = Path(cfg["_project_root"])
    run_dir = Path(run_dir)
    art_dir = Path(artifact_dir) if artifact_dir else run_dir
    art_dir.mkdir(parents=True, exist_ok=True)
    image_root = root / cfg["paths"]["data_root"]
    image_size = int(cfg["data"]["image_size"])
    seed = int(cfg["split"]["seed"])

    # ---- 1) 扫描 ----
    records = scan_dataset(image_root)
    logger.info("扫描完成: %d 张图像, %d 个类别", len(records), len({r["class_name"] for r in records}))

    # ---- 2) 质量检查（IC0）----
    #  全量审计 3032 张图较慢；若缓存存在且与当前扫描结果一致则复用（G5：结果仍可追溯）
    raw_counts = dict(sorted(Counter(r["class_name"] for r in records).items()))
    canonical_audit = root / "results" / "raw" / "audit" / "data_audit.json"
    audit = None
    if canonical_audit.is_file():
        try:
            cached = load_json(canonical_audit)
            if (cached.get("n_images_scanned") == len(records)
                    and cached.get("class_counts_raw") == raw_counts):
                audit = cached
                logger.info("复用已缓存的全量图像审计结果（数据集未变化）: %s", canonical_audit)
        except (OSError, ValueError) as exc:  # noqa: BLE001
            logger.warning("审计缓存不可用，将重新检查: %s", exc)

    if audit is None:
        audit = inspect_images(records, logger=logger, sample=cfg["data"].get("audit_sample"))
        audit["n_images_scanned"] = len(records)
        audit["class_counts_raw"] = raw_counts
        save_json(audit, canonical_audit)
        logger.info("图像审计结果已写入: %s", canonical_audit)

    # ---- 3) 去重（必须在划分之前）----
    dedup_cfg = cfg.get("dedup", {}) or {}
    if dedup_cfg.get("enabled", True):
        records, dropped, report = deduplicate(
            records,
            cross_class_policy=dedup_cfg.get("cross_class_policy", "drop_both"),
            same_class_policy=dedup_cfg.get("same_class_policy", "keep_first"),
            logger=logger,
        )
        save_dedup_report(report, art_dir / "dedup_report.csv")
    else:
        dropped, report = [], []
    audit["n_after_dedup"] = len(records)
    audit["n_dropped"] = len(dropped)
    audit["class_counts_after_dedup"] = dict(
        sorted(Counter(r["class_name"] for r in records).items())
    )
    # 注意：去重统计是在**限制每类张数之前**记录的，所以这里反映的始终是全量结果。
    # 把含去重后统计的完整审计写回规范文件（下游报告需要这些字段）。
    save_json(audit, canonical_audit)

    # ---- 4) 限制每类张数（仅冒烟测试）----
    records = limit_per_class(records, cfg["data"].get("max_per_class"), logger=logger)

    # ---- 5) 划分 ----
    reuse = (cfg.get("split", {}) or {}).get("reuse_index")
    reuse_path = None
    if reuse:
        p = Path(reuse)
        reuse_path = p if p.is_absolute() else (root / p)
    if reuse_path and reuse_path.is_file():
        # 复用冻结划分：所有模型共用同一份划分（G3）。
        # 注意：此处**不再施加 max_per_class**，否则会破坏已冻结的三集比例。
        if cfg["data"].get("max_per_class"):
            logger.warning("同时设置了 reuse_index 与 max_per_class，为保证划分一致，max_per_class 已被忽略")
        records = load_split_index(reuse_path, image_root)
        logger.info("复用已冻结的划分索引: %s (%d 张)", reuse_path, len(records))
    else:
        records = stratified_split(records, cfg["split"]["ratios"], seed, logger=logger)

    # ---- 6) 强制自检（G2 / IC1）----
    checks = assert_split_valid(records, cfg["split"].get("assert_stratify_tol", 0.01), logger=logger)
    save_split_index(records, art_dir / "split_index.csv")

    # ---- 7) 归一化统计量（IC0.3）----
    norm_cfg = cfg.get("normalize", {}) or {}
    if norm_cfg.get("mode", "train_stats") == "div255":
        mean, std = 0.0, 1.0
        logger.info("归一化方式: x/255（不做 z-score）")
    else:
        mean, std = compute_mean_std(records, image_size, logger=logger)
    save_json({"mode": norm_cfg.get("mode", "train_stats"), "mean": mean, "std": std,
               "source": "train_split_only"}, art_dir / "train_stats.json")

    # ---- 8) 数据集与 DataLoader ----
    class_names = sorted({r["class_name"] for r in records})
    loaders = {}
    for split in ("train", "val", "test"):
        subset = [r for r in records if r["split"] == split]
        tf = build_transforms(cfg, split, mean, std)
        ds = SarAircraftDataset(subset, transform=tf)
        loaders[split] = DataLoader(
            ds,
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=(split == "train"),
            num_workers=int(cfg["train"].get("num_workers", 0)),
            pin_memory=True,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=make_generator(seed),
        )
        logger.info(
            "  %-5s: %4d 张 | 增强=%s | %d 个 batch",
            split, len(subset), "是" if split == "train" else "否", len(loaders[split]),
        )

    return {
        "loaders": loaders,
        "records": records,
        "class_names": class_names,
        "mean": mean,
        "std": std,
        "checks": checks,
        "audit": audit,
        "dropped": dropped,
    }


# ============================================================================
#  命令行：只做数据审计
# ============================================================================
def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="SAR-ACD 数据扫描 / 去重 / 划分（不含训练）")
    ap.add_argument("--config", required=True, help="配置文件路径")
    ap.add_argument("--output", default=None, help="中间产物输出目录（默认 results/raw/audit）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["_project_root"] = str(Path(args.config).resolve().parent.parent)

    out = Path(args.output) if args.output else Path(cfg["_project_root"]) / "results" / "raw" / "audit"
    logger = get_logger("dataset-main", out / "dataset_audit.log")

    data = build_dataloaders(cfg, out, logger=logger)
    save_json(data["audit"], out / "data_audit.json")
    save_json(data["checks"], out / "split_checks.json")
    logger.info("审计产物已写入: %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
