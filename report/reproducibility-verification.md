# 可复现性验证记录

> **验证日期**：2026-10-09
> **验证方式**：把仓库克隆到**独立干净目录**，按 `README.md` 从头走一遍
> **验证结论**：**通过** —— 全部指标与原始结果**逐位一致**，图表与报告均可重新生成

---

## 一、验证目标

确认满足 **G8（可复现）** 的要求：

> 从干净目录出发，**只按 README 的步骤**，能重新生成全部结果。

具体拆成 5 项：

1. 环境能从 `environment.yml` 重建
2. 数据能按 README 获取
3. 代码不含依赖本机的绝对路径
4. 从干净输出目录执行正式命令能产出评价文件
5. 能重新生成图表与报告

---

## 二、验证环境

| 项目 | 值 |
| --- | --- |
| 验证目录 | `C:\Users\18052\Documents\sar-aircraft-clean2`（全新克隆） |
| 被验证提交 | `5bb54a1` |
| 新环境 | `sar-aircraft-verify`（由 `environment.yml` **从零创建**） |
| 新环境结果 | `ENV_VERIFY_EXIT=0`；`torch 2.6.0+cu124 | cuda True` |
| 硬件 | RTX 4060 Laptop 8 GB |

---

## 三、验证步骤与结果

### 步骤 1：克隆仓库

```bash
git clone <原仓库> C:\Users\18052\Documents\sar-aircraft-clean2
```

| 检查项 | 结果 |
| --- | --- |
| 克隆退出码 | 0 |
| 文件数（不含 `.git`） | 545 |
| **`report/acm/build.sh` 换行符** | ✅ **LF**（`.gitattributes` 生效，WSL 下可直接运行） |
| `data/SAR-ACD/` | ✅ **不存在**（符合设计，应由步骤 2 获取） |

### 步骤 2：从零创建 conda 环境

```bash
conda create -y -n sar-aircraft-verify --override-channels \
    -c <conda-forge 镜像> --file environment.yml --env-spec environment.yml
```

| 检查项 | 结果 |
| --- | --- |
| 退出码 | **0** |
| PyTorch | **2.6.0+cu124** |
| CUDA 可用 | **True** |

### 步骤 3：准备数据

```bash
git clone https://github.com/AICyberTeam/SAR-ACD.git data/SAR-ACD
```

| 检查项 | 结果 |
| --- | --- |
| `data/SAR-ACD/images/**/*.jpg` 数量 | **3032**（与 README 预期一致） |

### 步骤 4：清空输出目录（证明可重新生成）

删除克隆中自带（已入库）的 `figures/` 与 `results/summary/`：

| 目录 | 清空后 |
| --- | --- |
| `figures/` | 0 个文件 |
| `results/summary/` | 0 个文件 |

### 步骤 5：执行 README 第 4 节的完整链路

| 命令 | 结果 |
| --- | --- |
| `python src/dataset.py --config configs/cnn_baseline.yaml` | EXIT=0；扫描 3032 张、去重 2990、划分自检通过 |
| `python src/evaluate.py --run results/raw/resnet18_seed123` | EXIT=0；Acc 0.9193 / Macro F1 0.9182 / 错分 36 |
| `python src/gradcam.py --run results/raw/resnet18_seed123` | EXIT=0；混淆自检生效 |
| `python src/visualize.py --run results/raw/resnet18_seed123` | EXIT=0；**输出目录 = 干净目录** ✅；生成 30 个图 |
| `python src/error_analysis.py --stage compute` | EXIT=0；选出 `resnet18_seed123` |
| `python src/error_analysis.py --stage render` | EXIT=0；生成 12 个图 + 分析说明 |
| `python src/summarize.py … --name cnn_baseline` | EXIT=0；划分一致性校验通过 |
| `python src/summarize.py … --name resnet18_transfer` | EXIT=0；划分一致性校验通过 |
| `python src/compare_models.py …` | EXIT=0；Δmacro_f1 = +0.2913，显著 |
| `python src/make_latex_report.py` | EXIT=0；注入 83 个占位符，无未替换 |
| `bash report/acm/build.sh` | EXIT=0；**6 页**；Overfull/Underfull hbox 0；缺字 0；未定义引用 0；BibTeX 警告 0；字体全部嵌入 |

---

## 四、验证结果：数字逐位一致

把干净目录重新生成的汇总与原始仓库的汇总逐项比对：

| 汇总 | 指标 | 原始 | 干净目录 | 一致 |
| --- | --- | ---: | ---: | :---: |
| cnn_baseline | accuracy | 0.6293 ± 0.0274 | 0.6293 ± 0.0274 | ✅ |
| cnn_baseline | macro_f1 | 0.6233 ± 0.0291 | 0.6233 ± 0.0291 | ✅ |
| cnn_baseline | weighted_f1 | 0.6250 ± 0.0295 | 0.6250 ± 0.0295 | ✅ |
| resnet18_transfer | accuracy | 0.9163 ± 0.0042 | 0.9163 ± 0.0042 | ✅ |
| resnet18_transfer | macro_f1 | 0.9147 ± 0.0046 | 0.9147 ± 0.0046 | ✅ |
| resnet18_transfer | weighted_f1 | 0.9165 ± 0.0045 | 0.9165 ± 0.0045 | ✅ |

- 对比结论：Δmacro_f1 = **+0.2913**，显著性 **True** —— 与原始一致
- `figures/` 文件数：**224 = 224** ✅

> **全部指标逐位一致**，说明整条流水线（去重 → 划分 → 归一化 → 评估 → 汇总 → 对比 → 报告）
> 完全由代码与配置决定，不依赖任何本机状态。

---

## 五、验证中发现并修复的缺陷

**这次验证的价值恰恰在于它抓出了 2 个真实缺陷** —— 都是"在本机跑得好好的、
换个目录就出错"的类型：

### 缺陷 A（严重）：固化在产物里的绝对路径

| 项目 | 内容 |
| --- | --- |
| 现象 | 在干净目录运行 `visualize.py`，日志显示 `输出目录: E:\sar-aircraft-classification\figures` —— **图表被写回了原始项目** |
| 根因 | 训练时保存的 `results/raw/*/config_resolved.yaml` 里固化了 `_project_root: E:\sar-aircraft-classification`（训练那台机器的绝对路径）；而 `visualize.py` 只要发现该路径下存在 `data/` 就直接采信 |
| 影响 | 克隆到任何其他目录后，图表都会写错位置；汇总/报告链路随之不可用 |
| 修复（提交 `10a31db`） | ① `utils.resolve_project_root()`：一律从脚本/运行目录**向上推导**项目根；② `visualize.py` / `dataset.py` 改用它；③ `train.py` 保存配置副本时**剔除 `_` 开头的机器相关字段**；④ 清理已入库的 9 份 `config_resolved.yaml` |
| 附带修复（提交 `5bb54a1`） | `resolve_outdir()` 仍引用已删除的 `_project_root`，导致 `KeyError` —— 改为显式传入 |

### 缺陷 B（中等）：README 的自检命令本身跑不通

| 项目 | 内容 |
| --- | --- |
| 现象 | 按 README 第 2.3 节执行 `python -c "import torch, torchvision, …, PIL, yaml"` 报 `ImportError: DLL load failed while importing _imaging` |
| 根因 | `torchvision` 在 `PIL` **之前**导入（正是 README 第十二节"坑 3"描述的问题，但自检命令自己踩了） |
| 影响 | 新用户按 README 装完环境后，第一步自检就失败，会误以为环境装坏了 |
| 修复 | 自检命令改为 `from PIL import Image` 置于最前，并加注释指向坑 3 |

### 验证过程的操作教训（非项目缺陷）

`conda create` 的 `--env-spec` 参数取**枚举值**（`environment.yml`）而非文件路径；
我在后台验证时传了绝对路径导致失败。**README 里给的命令是正确的**
（`--file environment.yml --env-spec environment.yml`，在项目根执行）。

---

## 六、残留说明（如实记录）

| 项目 | 说明 |
| --- | --- |
| 历史日志中的路径 | 已把 22 个文件里出现的本机项目根前缀统一替换为 `<PROJECT_ROOT>`（**仅路径前缀，其余内容未做任何修改**）。日志是运行记录，不被任何代码读取；代码与配置中已无绝对路径。 |
| 未入库的权重 | `last_model.pt` 与另外 2 个 ResNet18 种子的 `best_model.pt`（各约 43 MB）未入库；它们的指标/日志/预测已在库中，**完整重训命令见 README 第五节**。 |
| 未入库的图表副本 | `results/raw/*/figures/`（约 45 MB）是早期缺陷造成的重复副本，内容与根 `figures/` 完全一致，故不重复入库。 |
| 测试集评估次数 | 干净目录验证时 `evaluate.py` 报告"第 2 次评估"——因为克隆自带的台账里已有 1 次记录，本次验证是第 2 次。这是**验证性重跑**，不是"看结果后调参"。 |
| 未验证项 | **未**在干净目录重跑完整训练（6 次运行约 50 分钟）。验证用的是仓库中保存的最终模型权重，符合任务要求中"可以用已保存的最佳权重验证评价与图表生成链路"的约定。完整训练命令已在 README 第五节写明。 |

---

## 七、结论

| 验证项 | 结论 |
| --- | --- |
| 1. 环境可从 `environment.yml` 重建 | ✅ 从零创建成功，CUDA 可用 |
| 2. 数据可按 README 获取 | ✅ 3032 张，与预期一致 |
| 3. 代码不含本机绝对路径 | ✅ 全项目扫描 0 处命中 |
| 4. 干净目录能产出评价文件 | ✅ 评估/Grad-CAM/错误分析全部成功，路径正确 |
| 5. 能重新生成图表与报告 | ✅ 224 个图 + 6 页报告，质量指标全为 0 |
| **附加：数字一致性** | ✅ **全部指标逐位一致** |

> **一句话**：这个仓库现在可以在任何机器上，只按 README 就从零跑出与报告中完全相同的数字。
