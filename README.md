# SAR 民航飞机图像分类（六分类）

> **任务**：SAR（合成孔径雷达）图像中 6 类民航飞机的分类
> **数据**：SAR-ACD 公开数据集（本地实测 3032 张灰度切片图，去重后 2990 张）
> **核心对比**：紧凑 CNN（从零训练） vs ResNet18（ImageNet 迁移学习）
> **主结果**：Macro F1 **0.6233 → 0.9147**（+0.2913），且种子波动缩小约 6.3 倍
> **状态**：实验、分析、报告全部完成，并通过干净目录复现验证

---

## 目录

- [一、项目简介](#一项目简介)
- [二、环境安装](#二环境安装)
- [三、数据准备](#三数据准备)
- [四、快速验证（不重新训练）](#四快速验证不重新训练)
- [五、完整训练命令](#五完整训练命令)
- [六、测试与评价命令](#六测试与评价命令)
- [七、图表生成命令](#七图表生成命令)
- [八、报告编译命令](#八报告编译命令)
- [九、项目目录结构](#九项目目录结构)
- [十、仓库内容说明（什么入库、什么不入库）](#十仓库内容说明什么入库什么不入库)
- [十一、结果摘要](#十一结果摘要)
- [十二、注意事项与本机环境坑](#十二注意事项与本机环境坑)
- [十三、许可证与引用](#十三许可证与引用)

---

## 一、项目简介

本项目在 SAR-ACD 六类民用飞机数据集上，用**完全相同的划分与评估流程**对比
**紧凑 CNN（从零训练）** 与 **ResNet18（ImageNet 迁移学习）**，
并进一步做错误分析与 Grad-CAM 可解释性分析。

**研究问题**：在样本量小、灰度、目标占画面比例大的 SAR 数据上，
迁移学习到底有没有优势、优势有多大、代价是什么？

**受控对比**：除模型本身外，去重规则 / 数据划分 / 输入尺寸 / 归一化 /
数据增强 / 轮数 / batch size / 优化器 / 随机种子**全部相同**
（脚本逐项校验过，见 `src/` 与实验记录）。

| 项目 | 内容 |
| --- | --- |
| 类别数 | 6（A220 / A320321 / A330 / ARJ21 / Boeing737 / Boeing787） |
| 图像 | 单通道灰度 JPEG，uint8，尺寸不固定（35×32 ~ 200×150），635 种尺寸 |
| 去重后样本 | **2990** 张（原始 3032，剔除 42） |
| 划分 | train 2098 / val 446 / test 446，分层抽样，固定种子 42 |
| 输入 | 96 × 96 |
| 主指标 | **Macro F1**（另报 Accuracy、每类 P/R/F1、混淆矩阵） |
| 稳定性 | 3 个随机种子（42 / 123 / 2024），报告均值 ± 标准差 |
| 硬件 | NVIDIA RTX 4060 Laptop 8 GB（CUDA 12.4） |

---

## 二、环境安装

### 2.1 标准安装（推荐）

```bash
conda env create -f environment.yml
conda activate sar-aircraft
```

### 2.2 本机（Windows + conda 26）需要绕开 ToS 限制

conda 26 对 `defaults` 频道有服务条款限制，而本机 `~/.condarc` 只读，
因此 `conda env create -f environment.yml` 会报 `CondaToSNonInteractiveError`。
**实测可用**的写法：

```powershell
& "C:\ProgramData\miniconda3\shell\condabin\conda-hook.ps1"
conda create -y -n sar-aircraft --override-channels `
    -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge `
    --file environment.yml --env-spec environment.yml
```

> `environment.yml` 里已写 `- nodefaults`，并显式指定 PyTorch 的 CUDA 源
> （`--extra-index-url https://download.pytorch.org/whl/cu124`）。

### 2.3 安装后自检

```bash
python -c "import torch, torchvision, numpy, pandas, sklearn, matplotlib, seaborn, PIL, yaml; print('deps ok')"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

期望输出形如：`2.6.0+cu124 True`。
若 `cuda_available` 为 `False`，说明装成了 CPU 版：

```bash
pip uninstall -y torch torchvision
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

⏱ **预计耗时**：conda 依赖约 2 分钟 + PyTorch CUDA wheel 约 2.4 GB（实测 2.8 MB/s，约 15 分钟）。

---

## 三、数据准备

### 3.1 下载

```bash
git clone https://github.com/AICyberTeam/SAR-ACD.git data/SAR-ACD
```

**放在哪**：项目根目录下的 `data/SAR-ACD/`（即上面命令的目标路径），
最终结构应为 `data/SAR-ACD/images/<类别名>/*.jpg`。

**预期结果**：检出 3033 个文件（3032 张 `.jpg` + 1 个 `README.md`），约 5 MB。

### 3.2 核对数据（可选但建议）

```bash
python src/dataset.py --config configs/cnn_baseline.yaml
```

产物写入 `results/raw/audit/`：

- `data_audit.json` —— 逐类张数、尺寸分布、亮度统计、损坏文件数、去重前后对比
- `split_index.csv` —— 划分索引
- `dataset_audit.log`

**实测结果**（与官方描述有出入，已在报告中说明）：

| 项目 | 官方 | 本地实测 |
| --- | --- | --- |
| 总数 | 3032 | **3032** ✅ |
| 每类数量 | "约 500" | **464 ~ 528**（比例 1.138:1） |
| 尺寸 | 未说明 | 635 种（35×32 ~ 200×150） |
| 重复图 | 未说明 | **22 组，其中 20 组跨类别** ⚠️ |

> ⚠️ **跨类别重复**指同一张图出现在两个类别目录下。它既是标签矛盾，
> 又会在随机划分时造成**数据泄漏**（同一张图同时进训练集和测试集）。
> 本项目在**划分之前**用 MD5 逐字节去重，跨类别重复**两侧全删**，类内重复保留 1 张。

### 3.3 网络不畅时的备选

```powershell
git clone --depth 1 https://github.com/AICyberTeam/SAR-ACD.git data\SAR-ACD
# 或下载 ZIP 后解压：
#   https://github.com/AICyberTeam/SAR-ACD/archive/refs/heads/main.zip
```

---

## 四、快速验证（不重新训练）

**用途**：在**不训练**的前提下，验证「评估 → Grad-CAM → 图表 → 汇总 → 报告」
整条链路能否跑通。适合刚克隆仓库、想确认环境是否装对时使用。

> 前置条件：仓库中已包含 **最终模型**（`results/raw/resnet18_seed123/best_model.pt`）
> 与各次运行的指标产物。**完整重训见第五节。**

```bash
# ---- 每次运行前都要先激活环境（见第十二节坑 1）----
conda activate sar-aircraft
```

```bash
# ① 复现最终模型的评估（测试集只评估一次，会写入台账）
python src/evaluate.py --run results/raw/resnet18_seed123

# ② 复现 Grad-CAM + 背景消融（纯 torch 计算，不画图）
python src/gradcam.py --run results/raw/resnet18_seed123

# ③ 出图（训练曲线/混淆矩阵/错误样例/Grad-CAM），写入 figures/
python src/visualize.py --run results/raw/resnet18_seed123

# ④ 错误分析与可解释性分析（两阶段）
python src/error_analysis.py --stage compute    # 选最终模型 + 算 CAM 定位指标
python src/error_analysis.py --stage render     # 出图 + 写 report/error-analysis.md

# ⑤ 汇总（校验划分一致性 + 均值/波动）
python src/summarize.py --runs results/raw/cnn_seed42 results/raw/cnn_seed123 `
    results/raw/cnn_seed2024 --name cnn_baseline
python src/summarize.py --runs results/raw/resnet18_seed42 results/raw/resnet18_seed123 `
    results/raw/resnet18_seed2024 --name resnet18_transfer

# ⑥ 两模型对比
python src/compare_models.py --a results/summary/cnn_baseline `
    --b results/summary/resnet18_transfer `
    --label-a "CNN 基线" --label-b "ResNet18 迁移学习"

# ⑦ 生成报告 LaTeX（数字从 results/ 注入，无手填）
python src/make_latex_report.py
```

> PowerShell 换行用反引号 `` ` ``；bash 用 `\`。

**验证成功的判据**：

- 每个命令退出码 `0`
- `results/raw/resnet18_seed123/` 下出现 `test_metrics.json`、`predictions.csv`、`errors.csv`
- `results/summary/error_analysis/` 下出现 `final_model_selection.json`、`cam_localization.json`
- `figures/` 中相应图文件被刷新
- `report/acm/main.tex` 被重新生成

---

## 五、完整训练命令

> ⏱ **完整重训耗时**：CNN 约 8 分钟/种子，ResNet18 约 8.5 分钟/种子；
> 6 次运行（2 模型 × 3 种子）合计约 **50 分钟**（RTX 4060 Laptop）。
> 若只想验证链路，用第四节的快速验证路径。

```bash
conda activate sar-aircraft
```

### 5.1 CNN 基线（3 个种子）

```bash
python src/train.py --config configs/cnn_baseline.yaml --model cnn_baseline --seed 42   --run-dir results/raw/cnn_seed42
python src/train.py --config configs/cnn_baseline.yaml --model cnn_baseline --seed 123  --run-dir results/raw/cnn_seed123
python src/train.py --config configs/cnn_baseline.yaml --model cnn_baseline --seed 2024 --run-dir results/raw/cnn_seed2024
```

### 5.2 ResNet18 迁移学习（3 个种子）

```bash
python src/train.py --config configs/resnet18_transfer.yaml --model resnet18_transfer --seed 42   --run-dir results/raw/resnet18_seed42
python src/train.py --config configs/resnet18_transfer.yaml --model resnet18_transfer --seed 123  --run-dir results/raw/resnet18_seed123
python src/train.py --config configs/resnet18_transfer.yaml --model resnet18_transfer --seed 2024 --run-dir results/raw/resnet18_seed2024
```

**两个模型的关键差异**（其余超参完全一致，见 `configs/`）：

| 项目 | CNN 基线 | ResNet18 |
| --- | --- | --- |
| 权重 | 从零训练 | ImageNet-1k 预训练（torchvision 自动下载约 45 MB） |
| 输入通道 | 1（灰度直接进网络） | 3（灰度复制成 3 通道，保留全部预训练权重） |
| 冻结策略 | — | 冻结 `conv1/bn1/layer1/layer2`，微调 `layer3/layer4/fc` |
| 学习率 | 1e-3（统一） | 分类头 1e-3 / 骨干 1e-4（分层） |
| 参数量 | 241,382 | 11,179,590（可训练 10,496,518，93.89%） |

### 5.3 训练产物

每次运行写入 `results/raw/<run-dir>/`：
`config_resolved.yaml`、`train.log`、`history.csv`（逐轮指标）、`best_model.pt`、
`last_model.pt`、`metrics.json`、`run_meta.json`、`split_index.csv`、
`dedup_report.csv`、`train_stats.json`。

**模型选择依据**：验证集 Macro F1（**不使用测试集**）。

### 5.4 小样本冒烟测试（约 13 秒）

```bash
python src/train.py --config configs/smoke_test.yaml --model cnn_baseline --seed 42
```

用每类 10 张、2 轮，验证数据管线 / 模型维度 / 反向传播 / 产物落盘。
产物写入 `results/raw/smoke/`，**不得作为正式结论引用**。

---

## 六、测试与评价命令

```bash
# 测试集评估：用验证集选出的 checkpoint，测试集只评估一次
python src/evaluate.py --run results/raw/<run-dir>
```

产物：`test_metrics.json`（Accuracy / Macro F1 / Weighted F1 / 每类 P-R-F1 /
混淆矩阵原始与归一化 / 标签映射）、`predictions.csv`（逐样本）、
`errors.csv`（**全部**错误样例）、`uncertain.csv`（低置信度）、
`test_evaluation_ledger.json`（**测试集使用台账**）。

> **G2 守护**：`test_evaluation_ledger.json` 记录每次测试集评估的时间与命令。
> 同一运行被评估第 2 次时会**大声告警**——防止"看结果后调参"。

**汇总（3 个种子）**：

```bash
python src/summarize.py --runs <run1> <run2> <run3> --name <名称>
```

汇总前会**校验三个运行的划分哈希一致**（不一致直接报错），
并**拒绝混入冒烟运行**。

---

## 七、图表生成命令

```bash
# 单次运行的全部图（曲线/混淆矩阵/每类指标/错误样例/EDA/Grad-CAM 渲染）
python src/visualize.py --run results/raw/<run-dir>

# 只出某一类图
python src/visualize.py --run results/raw/<run-dir> --fig curves|metrics|samples|eda|cam
```

```bash
# 两模型对比图（6 张）
python src/compare_models.py --a results/summary/cnn_baseline --b results/summary/resnet18_transfer
```

```bash
# 错误分析与可解释性分析（两阶段；compute 用 torch，render 用 matplotlib）
python src/error_analysis.py --stage compute
python src/error_analysis.py --stage render
```

**输出位置**：正式运行（`run.mode: formal`）的图写入项目根 **`figures/`**；
冒烟运行的图写入其自身目录。每张图同时输出 `.pdf`（矢量）与 `.png`（位图）。

**图表清单**（前缀为运行名或 `error_analysis__` / `comparison_…__`）：

| 前缀 | 内容 |
| --- | --- |
| `fig0_data_eda` | 数据探索（类别分布 / 尺寸 / 亮度 / 样例） |
| `fig1a/1b/1c` | 训练曲线（loss、accuracy+Macro F1、三种子叠加） |
| `fig2a/2b` | 汇总指标均值±标准差、逐种子散点 |
| `fig3a/3b/3c/3d` | 混淆矩阵（原始/归一化）、每类 P-R-F1、F1 排序 |
| `fig4a/4b/4c/4d` | 正确样例、**全部错误样例**、高不确定性样例、混淆对 |
| `fig5a/5b/5c/5d` | Grad-CAM（分对/分错/分对但偏背景）、背景消融 |
| `fig6a~6f` | 两模型对比（总体/每类/差值/混淆矩阵并排/差值矩阵/逐种子） |
| `fig7a~7f` | 错误分析（全部错分、每类正确、低置信度、最易混两类、CAM 定位、CAM 对照） |

---

## 八、报告编译命令

ACM 双栏英文技术报告，源码在 `report/acm/`。

```bash
cd report/acm
python ../../src/make_latex_report.py   # 先在项目根生成 main.tex（数字从 results/ 注入）
bash build.sh                            # 4 遍编译 + 质量报告
```

或（从项目根）：

```bash
python src/make_latex_report.py
cd report/acm && bash build.sh
```

**依赖**：TeX Live 需包含 `texlive-publishers`（提供 `acmart.cls`）与
`texlive-fonts-extra`（提供 acmart 默认的 Libertine 字体）。
模板依赖已随仓库保存于 `report/acm/tex/`（`acmart.cls` + `ACM-Reference-Format.bst`）。

**产物**：`report/acm/main.pdf`（6 页）+ `preview/page-*.png`（每页预览）。

**质量报告**（`build.sh` 末尾输出）：Overfull hbox / vbox、缺字、
未定义引用、BibTeX 警告、字体嵌入统计。当前实测全部为 0（Underfull vbox 1 处轻微）。

> **报告数字不手填**：`make_latex_report.py` 从 `results/summary/` 读取 83 个值，
> 用 `@PLACEHOLDER@` 注入模板；实际用到的值落盘为 `report/acm/report_numbers.json` 供审计。

---

## 九、项目目录结构

```text
sar-aircraft-classification/
├── README.md                      # 本文件
├── experiment-plan.md             # 实验方案（含假设 H1–H3、变更记录）
├── learning-record.md             # 学习记录（过程、理解、遗留疑问）
├── environment.yml                # conda 环境定义
├── .gitignore
│
├── configs/
│   ├── default.yaml               # 默认配置（其他配置继承它）
│   ├── cnn_baseline.yaml          # CNN 基线正式实验（全量数据 / 40 轮 / 3 种子）
│   ├── resnet18_transfer.yaml     # ResNet18 正式实验（与基线控制变量一致）
│   └── smoke_test.yaml            # 冒烟测试（每类 10 张 / 2 轮）
│
├── data/
│   ├── README.md                  # 数据来源、引用、本地实测统计
│   └── SAR-ACD/                   # 【不入库】git clone 得到，见第三节
│       └── images/<类别>/*.jpg
│
├── src/
│   ├── utils.py                   # 配置加载/种子/日志/运行目录/JSON
│   ├── dataset.py                 # 扫描→质量检查→去重→分层划分→强制自检→DataLoader
│   ├── models/
│   │   ├── __init__.py            # 模型工厂 + 冻结的旧版 SmallCNN
│   │   ├── cnn_baseline.py        # CNNBaseline（4 卷积层，241,382 参数）
│   │   └── resnet18_transfer.py   # ResNet18Transfer（含冻结策略/学习率说明）
│   ├── train.py                   # 训练（单模型 × 单种子）
│   ├── evaluate.py                # 测试集评估（唯一允许用测试集的地方）
│   ├── gradcam.py                 # Grad-CAM 计算 + 背景消融（纯 torch）
│   ├── visualize.py               # 全部图表渲染（纯 matplotlib）
│   ├── summarize.py               # 多运行汇总（划分一致性 + 均值/波动）
│   ├── compare_models.py          # 两模型对比
│   ├── error_analysis.py          # 错误分析 + 可解释性（compute/render 双阶段）
│   └── make_latex_report.py       # 生成 ACM 报告 LaTeX
│
├── results/
│   ├── raw/
│   │   ├── audit/                 # 数据审计产物
│   │   ├── smoke/                 # 冒烟测试运行
│   │   ├── cnn_seed{42,123,2024}/         # CNN 基线 3 种子
│   │   └── resnet18_seed{42,123,2024}/    # ResNet18 3 种子
│   └── summary/
│       ├── cnn_baseline/                  # 基线汇总（表 + 图）
│       ├── resnet18_transfer/             # 迁移学习汇总
│       ├── comparison_cnn_baseline_vs_resnet18_transfer/   # 两模型对比
│       └── error_analysis/                # 最终模型选择 + CAM 定位指标
│
├── figures/                       # 全部图表（.pdf + .png，共 224 个文件）
│
├── report/
│   ├── error-analysis.md          # 错误分析与可解释性分析（脚本生成）
│   └── acm/                       # ACM 双栏技术报告
│       ├── main.tex / refs.bib / main.pdf / build.sh
│       ├── report_numbers.json    # 报告中实际用到的每个数值（审计用）
│       ├── tex/                   # acmart.cls + ACM-Reference-Format.bst
│       └── preview/               # 每页 PNG 预览
│
└── skills/
    └── research-project-delivery/ # 本项目使用的科研交付 Skill
        ├── SKILL.md               # 通用约束 G1–G8
        ├── references/image-classification.md   # 图像分类专属规则 IC0–IC5
        └── skill-usage-record.md  # 使用与修改记录（含全部缺陷修复）
```

---

## 十、仓库内容说明（什么入库、什么不入库）

| 内容 | 入库 | 原因 |
| --- | --- | --- |
| `src/` `configs/` `skills/` `report/` | ✅ | 源码与交付物 |
| `results/raw/*/`（指标、日志、划分索引、预测、错误清单） | ✅ | 可复现性证据 |
| `results/summary/` | ✅ | 汇总表与图 |
| `figures/`（PDF + PNG） | ✅ | 交付图表 |
| **最终模型** `resnet18_seed123/best_model.pt`（约 43 MB） | ✅ | 支持"不重训即可验证链路" |
| 基线 3 个 `best_model.pt`（各约 1 MB） | ✅ | 体积小 |
| `data/SAR-ACD/`（约 5 MB） | ❌ | 可 `git clone` 重建；且无 LICENSE，不宜再分发 |
| conda 环境 | ❌ | 可由 `environment.yml` 重建 |
| `__pycache__` 等缓存 | ❌ | 自动生成 |
| LaTeX 中间文件（`.aux/.log/.bbl/...`） | ❌ | `build.sh` 每次重跑 |
| `last_model.pt`（每个约 43 MB） | ❌ | 与 best 重复 |
| 另外 2 个 ResNet18 种子的 `best_model.pt`（各约 43 MB） | ❌ | 冗余；指标已入库，重训命令见第五节 |
| `results/raw/*/figures/`（约 45 MB） | ❌ | 早期缺陷留下的**重复副本**，内容与根 `figures/` 完全一致 |

**仓库体积**：约 140 MB（其中 figures 约 78 MB、最终模型约 43 MB）。

---

## 十一、结果摘要

### 总体（测试集，3 种子均值 ± 标准差）

| 指标 | CNN 基线 | **ResNet18 迁移学习** | 差值 | 判定 |
| --- | ---: | ---: | ---: | --- |
| Accuracy | 0.6293 ± 0.0274 | **0.9163 ± 0.0042** | +0.2870 | 显著更好 |
| **Macro F1** | 0.6233 ± 0.0291 | **0.9147 ± 0.0046** | **+0.2913** | **显著更好** |
| Weighted F1 | 0.6250 ± 0.0295 | 0.9165 ± 0.0045 | +0.2915 | 显著更好 |

> 判定依据（实验前固定）：均值差需大于两者中较大的标准差（阈值 0.0291）。

**两个关键结论**：

1. 迁移学习不仅更准（+29 个百分点），**种子波动还小了约 6.3 倍** —— 更可信、更可复现。
2. 收益集中在基线最弱的类（Boeing787 +0.3926、Boeing737 +0.3440），
   **最小的提升在 A320321（+0.1624）**。

### 错误分析（最终模型 = `resnet18_seed123`，按**验证集**选出）

- 测试集 446 张，错分 **36 张（8.07%）**
- **A220 ↔ A320321 一对占全部错误的 52.8%**，且是**双向**混淆
- 两类的外形统计（宽/高/长宽比/目标面积/亮度）**相对差全部 < 20%** —— 高度相似
- 错分平均置信度 0.760 vs 正确 0.958 → 可用置信度阈值做**拒识**

### Grad-CAM（结论保持克制）

| 目标层 | CAM 网格 | 落在亮散射体上 | 随机基线 | 提升 |
| --- | --- | ---: | ---: | ---: |
| `layer3` | 6×6 | 0.0878 | 0.0653 | **1.34×** |
| `layer4` | 3×3 | 0.0774 | 0.0653 | 1.18× |

**模型有聚焦目标的倾向，但不强（1.34×）**。
⚠️ 方法学限制：数据集**无目标框**，目标用"亮像素"近似；飞机占画面约 3/4，
定位类指标分辨力天生有限；`layer4` 在 96×96 下只有 3×3。
因此该结论定位为**倾向性证据，而非定论**。

### 假设检验（方案里预先写下的假设）

| 假设 | 内容 | 结果 |
| --- | --- | --- |
| H1 | 迁移学习更好，但**不一定大幅领先** | ❌ **被推翻**：实测领先 29 个百分点 |
| H2 | 冻结骨干会比微调更稳 | 未单独验证（本次采用 partial 冻结） |
| H3 | 模型会部分依赖背景 | ⚠️ **无法判定**：背景消融被证明是混淆的（见下） |

> **一次重要的自我纠错**：最初的"背景消融"（填充/裁剪外围）显示准确率暴跌，
> 看似支持 H3。但进一步量化发现**飞机外接框有 81% 落在被扰动区域** ——
> 该消融切掉的主要是**机体**而非背景，**不能用它判断模型是否依赖背景**。
> 已在 `src/gradcam.py` 中加入自动混淆自检，并把结论改为"无法判定"。

---

## 十二、注意事项与本机环境坑

### 坑 1：必须**激活环境**再运行

直接用 `...\envs\sar-aircraft\python.exe` 跑含 matplotlib 的脚本会
**以 `0xC06D007F` 静默崩溃（没有任何 Python 报错）**。
原因：本环境用 Intel MKL，其 DLL 在 `<env>\Library\bin`，只有激活时才进 PATH。

```powershell
& "C:\ProgramData\miniconda3\shell\condabin\conda-hook.ps1"
conda activate sar-aircraft
```

### 坑 2：torch 与 matplotlib **不能同进程**

同一进程里"先跑 torch 反向传播、再用 matplotlib 渲染"会触发：

```
OMP: Error #15: Initializing libomp.dll, but found libiomp5md.dll already initialized.
```

因此代码**刻意拆成两个进程**：`gradcam.py` / `error_analysis.py --stage compute`
只做 torch 计算并落盘；`visualize.py` / `error_analysis.py --stage render`
只做 matplotlib 渲染。**不要"顺手"把 matplotlib 加回 compute 阶段。**
（官方给的 `KMP_DUPLICATE_LIB_OK=TRUE` 属"不安全"绕过，本项目不采用。）

### 坑 3：PIL 必须在 torchvision / CUDA **之前**导入

否则报 `ImportError: DLL load failed while importing _imaging`。
涉及文件已在最顶部固定 `from PIL import Image` 并加注释，**请勿调整导入顺序**。

### 其他

- **数据泄漏**：本数据集有 20 组跨类别重复，**去重必须在划分之前**，
  划分后脚本会自动断言"三个子集的内容哈希两两不相交"。
- **测试集只用一次**：`evaluate.py` 会写入台账；重复评估会告警。
- **文档与代码的一致性**：`src/make_latex_report.py` 只从 `results/` 读数字，
  报告中不存在手填值。

---

## 十三、许可证与引用

### 数据集

- **SAR-ACD 仓库没有 LICENSE 文件**，按默认著作权规则属"保留所有权利"；
- 本项目为**课程/研究目的的本地实验**；
- **原始图像不入版本库**，如需公开再分发数据或模型，请先联系数据集作者；
- 版权归属：国家高分辨率对地观测系统重大专项 / 中国科学院空天信息创新研究院。

```bibtex
@ARTICLE{9754573,
  author  = {Sun, Xian and Lv, Yixuan and Wang, Zhirui and Fu, Kun},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  title   = {SCAN: Scattering Characteristics Analysis Network for Few-Shot
             Aircraft Classification in High-Resolution SAR Images},
  year    = {2022},
  volume  = {60},
  pages   = {1--17},
  doi     = {10.1109/TGRS.2022.3166174}
}
```

### 本项目代码

本项目代码为课程作业产物，可自由参考；
其中 ResNet18 使用 `torchvision` 的 ImageNet 预训练权重（BSD-3-Clause）。
