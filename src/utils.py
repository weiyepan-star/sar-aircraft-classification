"""
通用工具：配置加载、随机种子、JSON 读写、日志、运行目录管理。

对应约束：
  G3 —— set_seed() 固定全部随机源
  G5 —— save_json() 保证所有指标落盘可追溯
  G6 —— 运行目录区分 smoke / formal，失败运行也保留
"""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# ============================================================================
#  配置加载（支持 _base_ 继承，便于写"极小"的覆盖配置）
# ============================================================================
def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并。注意：**列表按整体替换**，不做元素级合并（语义可预测）。"""
    out = dict(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_config(path: str | Path) -> dict:
    """
    读取 YAML 配置。若含 `_base_` 键，则先加载基配置再深度合并本文件的覆盖项。

    `_base_` 路径相对于**当前配置文件所在目录**。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base_ref = cfg.pop("_base_", None)
    if base_ref:
        base = load_config(path.parent / base_ref)
        cfg = _deep_merge(base, cfg)
    cfg["_config_path"] = str(path)
    return cfg


# ============================================================================
#  随机种子（G3）
# ============================================================================
def set_seed(seed: int) -> dict:
    """
    固定全部随机源，返回本次的确定性设置快照（用于写入 run_meta.json）。
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    info: dict[str, Any] = {"seed": seed, "cudnn_benchmark": None, "cudnn_deterministic": None}
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        info["torch_initial_seed"] = torch.initial_seed()
        info["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
        info["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
    except ImportError:
        info["torch"] = "未安装"
    return info


def seed_worker(worker_id: int) -> None:
    """
    DataLoader 的 worker_init_fn：让每个 worker 的随机性也可复现。

    ⚠️ 注意：**Windows 上 C 的 long 是 32 位**，`np.random.seed()` 的取值必须 < 2**31，
    否则会抛 `OverflowError: Python int too large to convert to C long`。
    PyTorch 官方示例里用的是 `% 2**32`，在 Windows 上会踩这个坑；
    这里改用 `% 2**31`。（该问题只在 `num_workers > 0` 时才会暴露。）
    """
    worker_seed = (int(np.random.get_state()[1][0]) + worker_id) % (2**31)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int):
    """为 DataLoader 提供一个固定种子的 torch.Generator。"""
    import torch

    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ============================================================================
#  JSON / 路径
# ============================================================================
def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"无法序列化的类型: {type(obj)}")


def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)
    return path


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_yaml(obj: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    return path


def resolve_project_root(start: str | Path) -> Path:
    """
    从 `start`（文件或目录）向上查找项目根 —— 即同时含
    `src/`、`configs/`、`environment.yml` 的那一层目录。

    ⚠️ **为什么不能采信配置里的 `_project_root`**：
       那一项是**训练当时那台机器**的绝对路径，会随 `config_resolved.yaml`
       一起进入版本库。克隆到别的目录后若直接采信它，产物会被写回**原机器的路径**
       —— 本项目在干净目录复现验证时**实测踩到过这个坑**（图表被写回了原始项目）。
       因此所有脚本都必须从"自己在哪里"推导项目根，而不是读配置。
    """
    p = Path(start).resolve()
    if p.is_file():
        p = p.parent
    for cand in (p, *p.parents):
        if ((cand / "src").is_dir() and (cand / "configs").is_dir()
                and (cand / "environment.yml").is_file()):
            return cand
    raise FileNotFoundError(
        f"无法从 {start} 向上找到项目根（应含有 src/、configs/、environment.yml）")


def detect_run_mode(run_dir: str | Path) -> str:
    """
    判定一次运行是 `smoke` 还是 `formal`。

    判定顺序（**权威优先**）：
      1. `metrics.json`（train.py 产出，含 mode 字段）
      2. `test_metrics.json`（evaluate.py 产出）
      3. 按路径推断（含 `/smoke/` 视为 smoke）
      4. 兜底 `formal`

    ⚠️ 这个函数的存在是有教训的：早期版本让 `visualize.py` 只看
    `test_metrics.json` 的 `mode` 字段，而该字段当时并未写入，
    于是**正式运行被判成 smoke，图表被写进了运行目录而不是项目 figures/**。
    """
    run_dir = Path(run_dir)
    for fname in ("metrics.json", "test_metrics.json"):
        p = run_dir / fname
        if p.is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    m = json.load(f)
                if m.get("mode"):
                    return str(m["mode"])
            except (OSError, ValueError):
                pass
    s = str(run_dir).replace("\\", "/")
    if "/smoke/" in s or s.rstrip("/").endswith("/smoke"):
        return "smoke"
    return "formal"


def make_run_dir(
    root: str | Path,
    mode: str,
    tag: str,
    model: str,
    seed: int,
    timestamp: str | None = None,
) -> Path:
    """
    建立本次运行的目录。

    结构： `<root>/<mode>/<tag>_<model>_seed<seed>_<时间戳>/`

    mode = `smoke` → 冒烟测试，**不得当作正式结果**（G6 / 用户要求）
    mode = `formal` → 正式实验
    """
    if mode not in {"smoke", "formal"}:
        raise ValueError(f"mode 必须是 'smoke' 或 'formal'，收到: {mode!r}")
    ts = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(root) / mode / f"{tag}_{model}_seed{seed}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ============================================================================
#  日志
# ============================================================================
def force_utf8_console() -> None:
    """
    Windows 控制台默认是 GBK，遇到 ✅ / ⚠️ 等字符会抛 UnicodeEncodeError。

    日志文件本身是 UTF-8（不受影响），但控制台 handler 会因此报错并打断输出。
    这里把标准流重设为 UTF-8 且 errors='replace'，保证控制台永不因编码中断。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass  # 非标准流（如被重定向）时忽略


def get_logger(name: str, log_file: str | Path | None = None, level: int = logging.INFO):
    """同时输出到控制台与文件（UTF-8）。"""
    force_utf8_console()

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ============================================================================
#  环境快照
# ============================================================================
def env_snapshot() -> dict:
    """记录运行环境，便于事后追溯（G8）。"""
    snap: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "numpy": np.__version__,
    }
    try:
        import torch

        snap["torch"] = torch.__version__
        snap["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            snap["cuda_device"] = torch.cuda.get_device_name(0)
            snap["cuda_version"] = torch.version.cuda
    except ImportError:
        snap["torch"] = "未安装"
    try:
        import sklearn

        snap["sklearn"] = sklearn.__version__
    except ImportError:
        pass
    return snap


# ============================================================================
#  小工具
# ============================================================================
def setup_matplotlib():
    """
    统一 matplotlib 配置：Agg 后端（不弹窗）+ 尽量使用中文字体。
    图内文字默认用英文（见 experiment-plan.md 第 8 节），
    但仍然配置中文字体，避免任何中文标签出现缺字方框。
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager, rcParams

    available = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                 "Source Han Sans SC", "PingFang SC", "DejaVu Sans"):
        if cand in available:
            rcParams["font.sans-serif"] = [cand] + rcParams["font.sans-serif"]
            break
    rcParams["axes.unicode_minus"] = False
    rcParams["figure.dpi"] = 120
    rcParams["savefig.bbox"] = "tight"
    return matplotlib


def save_figure(fig, outdir, name: str, formats=("pdf", "png"), dpi: int = 200):
    """同时保存矢量与位图，返回已写入的路径列表。"""
    from pathlib import Path as _Path

    outdir = _Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        p = outdir / f"{name}.{ext}"
        fig.savefig(p, format=ext, dpi=dpi)
        written.append(p)
    return written


def count_parameters(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def format_seconds(sec: float) -> str:
    if sec < 60:
        return f"{sec:.1f}s"
    m, s = divmod(int(sec), 60)
    return f"{m}m{s:02d}s"
