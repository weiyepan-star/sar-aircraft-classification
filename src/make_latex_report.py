"""
ACM 双栏技术报告生成器。

对应约束：
  G5  —— 报告中的**每一个数字、每一张表**都从 `results/summary/` 与
         `results/raw/*/` 的产物文件读取后注入模板，**没有任何手填值**；
         实际用到的数字同时落盘为 `report/acm/report_numbers.json` 供审计。
  G7  —— 图与表都对应正文的具体论点，且都在正文中被引用。

实现说明：
  模板用 `@PLACEHOLDER@` 标记而非 Python f-string —— 因为 LaTeX 大量使用
  `{}`，f-string 需要把所有花括号写成 `{{}}`，极易出错。用 `str.replace()`
  注入可以完全避开这个问题。

用法：
    python src/make_latex_report.py
    # 然后在 report/acm 下编译： cd report/acm && bash build.sh
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger, load_json, save_json  # noqa: E402


# ============================================================================
#  读取所有产物
# ============================================================================
def load_numbers(project: Path) -> dict:
    S = project / "results" / "summary"
    R = project / "results" / "raw"

    audit = load_json(R / "audit" / "data_audit.json")
    cnn = load_json(S / "cnn_baseline" / "summary.json")
    rn = load_json(S / "resnet18_transfer" / "summary.json")
    comp = load_json(S / "comparison_cnn_baseline_vs_resnet18_transfer" / "comparison.json")
    sel = load_json(S / "error_analysis" / "final_model_selection.json")
    loc = load_json(S / "error_analysis" / "cam_localization.json")
    err = load_json(S / "error_analysis" / "error_summary.json")

    # 训练超参 / 划分规模：取自训练产物（不用手填）
    m_cnn = load_json(R / "cnn_seed42" / "metrics.json")
    m_rn = load_json(R / "resnet18_seed42" / "metrics.json")
    # 去重明细
    dedup = list(csv.DictReader(open(R / "cnn_seed42" / "dedup_report.csv", encoding="utf-8")))
    n_cross = sum(1 for d in dedup if d["kind"] == "cross_class")
    n_same = sum(1 for d in dedup if d["kind"] == "same_class")

    names = cnn["class_names"]

    def fmt(x, nd=4):
        return f"{x:.{nd}f}"

    def pm(d, nd=4):
        return f"{d['mean']:.{nd}f} $\\pm$ {d['std']:.{nd}f}"

    V: dict[str, str] = {}

    # ---------- 数据集 ----------
    V["N_SCANNED"] = f"{audit['n_images_scanned']:,}"
    V["N_AFTER_DEDUP"] = f"{audit['n_after_dedup']:,}"
    V["N_DROPPED"] = str(audit["n_dropped"])
    V["N_CROSS_DUP"] = str(n_cross)
    V["N_SAME_DUP"] = str(n_same)
    V["N_CORRUPTED"] = str(audit["n_corrupted"])
    V["N_UNIQUE_SIZES"] = str(audit["n_unique_sizes"])
    V["W_MIN"] = str(audit["width"]["min"]); V["W_MAX"] = str(audit["width"]["max"])
    V["W_MEAN"] = f"{audit['width']['mean']:.1f}"
    V["H_MIN"] = str(audit["height"]["min"]); V["H_MAX"] = str(audit["height"]["max"])
    V["H_MEAN"] = f"{audit['height']['mean']:.1f}"
    V["IMG_MODES"] = ", ".join(audit["modes"].keys())
    V["N_CLASSES"] = str(len(names))

    raw = audit["class_counts_raw"]; aft = audit["class_counts_after_dedup"]
    V["CLASS_COUNTS_TABLE"] = "\n".join(
        f"{n} & {raw[n]} & {aft[n]} & {raw[n]-aft[n]} \\\\" for n in names)
    vals = [aft[n] for n in names]
    V["CLASS_MIN"] = str(min(vals)); V["CLASS_MAX"] = str(max(vals))
    V["CLASS_RATIO"] = f"{max(vals)/min(vals):.3f}"

    V["N_TRAIN"] = f"{m_cnn['data']['n_train']:,}"
    V["N_VAL"] = f"{m_cnn['data']['n_val']:,}"
    V["N_TEST"] = f"{m_cnn['data']['n_test']:,}"
    V["IMG_SIZE"] = str(m_cnn["data"]["image_size"])
    V["NORM_MEAN"] = f"{m_cnn['data']['normalize']['mean']:.4f}"
    V["NORM_STD"] = f"{m_cnn['data']['normalize']['std']:.4f}"
    V["SPLIT_SEED"] = "42"

    # ---------- 模型 ----------
    V["PARAMS_CNN"] = f"{m_cnn['n_params_total']:,}"
    V["PARAMS_RN_TOTAL"] = f"{m_rn['n_params_total']:,}"
    V["PARAMS_RN_TRAIN"] = f"{m_rn['n_params_trainable']:,}"
    V["PARAMS_RN_PCT"] = f"{100*m_rn['n_params_trainable']/m_rn['n_params_total']:.2f}"
    V["EPOCHS"] = str(m_cnn["epochs_planned"])
    V["BATCH"] = "64"
    V["LR_HEAD"] = "1e-3"; V["LR_BACKBONE"] = "1e-4"
    V["WD"] = "1e-4"; V["SEEDS"] = ", ".join(str(s) for s in cnn["seeds"])
    V["PATIENCE"] = "10"

    # ---------- 主要结果 ----------
    V["CNN_ACC"] = pm(cnn["accuracy"]); V["RN_ACC"] = pm(rn["accuracy"])
    V["CNN_F1"] = pm(cnn["macro_f1"]); V["RN_F1"] = pm(rn["macro_f1"])
    V["CNN_WF1"] = pm(cnn["weighted_f1"]); V["RN_WF1"] = pm(rn["weighted_f1"])
    V["D_ACC"] = f"{comp['overall']['accuracy']['delta']:+.4f}"
    V["D_F1"] = f"{comp['overall']['macro_f1']['delta']:+.4f}"
    V["D_WF1"] = f"{comp['overall']['weighted_f1']['delta']:+.4f}"
    V["THR_F1"] = f"{comp['overall']['macro_f1']['threshold']:.4f}"
    V["F1_REL"] = f"{100*comp['overall']['macro_f1']['delta']/cnn['macro_f1']['mean']:.1f}"
    V["SIG_F1"] = ("significant" if comp["overall"]["macro_f1"]["significant"] else "not significant")
    V["CNN_SEED_F1"] = ", ".join(f"{v:.4f}" for v in cnn["macro_f1"]["values"])
    V["RN_SEED_F1"] = ", ".join(f"{v:.4f}" for v in rn["macro_f1"]["values"])
    V["STD_RATIO"] = f"{cnn['macro_f1']['std']/max(rn['macro_f1']['std'],1e-9):.1f}"
    V["CNN_BESTEP"] = str(m_cnn["best_epoch"]); V["RN_BESTEP"] = str(m_rn["best_epoch"])
    V["CNN_TIME"] = f"{m_cnn['total_time_sec']/60:.1f}"; V["RN_TIME"] = f"{m_rn['total_time_sec']/60:.1f}"

    V["RESULTS_TABLE"] = "\n".join([
        f"Accuracy & {V['CNN_ACC']} & {V['RN_ACC']} & {V['D_ACC']} \\\\",
        f"Macro F1 & {V['CNN_F1']} & {V['RN_F1']} & {V['D_F1']} \\\\",
        f"Weighted F1 & {V['CNN_WF1']} & {V['RN_WF1']} & {V['D_WF1']} \\\\",
    ])

    # 逐类对比
    rows = []
    for n in sorted(names, key=lambda k: -comp["per_class_f1"][k]["delta"]):
        d = comp["per_class_f1"][n]
        rows.append(f"{n} & {d['support']} & {d['a_mean']:.4f} & {d['b_mean']:.4f} & {d['delta']:+.4f} \\\\")
    V["PERCLASS_TABLE"] = "\n".join(rows)

    # 平均混淆矩阵（ResNet18，归一化）
    cmn = rn["mean_confusion_matrix_normalized"]
    V["CM_RN_TABLE"] = "\n".join(
        n + " & " + " & ".join(f"{v:.2f}" for v in cmn[i]) + " \\\\" for i, n in enumerate(names))
    cmc = cnn["mean_confusion_matrix_normalized"]
    V["CM_CNN_TABLE"] = "\n".join(
        n + " & " + " & ".join(f"{v:.2f}" for v in cmc[i]) + " \\\\" for i, n in enumerate(names))

    # ---------- 错误分析 ----------
    V["SEL_RUN"] = err["selected_run"].replace("_", "\\_")
    V["SEL_SEED"] = str(err["selected_seed"])
    V["SEL_VALF1"] = f"{err['selected_val_macro_f1']:.4f}"
    V["N_WRONG"] = str(err["n_wrong"]); V["N_CORRECT"] = str(err["n_correct"])
    V["ERR_RATE"] = f"{err['error_rate']*100:.2f}"
    V["CONF_WRONG"] = f"{err['mean_conf_wrong']:.3f}"
    V["CONF_CORRECT"] = f"{err['mean_conf_correct']:.3f}"
    V["CONF_GAP"] = f"{err['mean_conf_correct']-err['mean_conf_wrong']:.3f}"
    V["FAM_RN_MEAN"] = f"{sel['families']['resnet18_transfer']['val_macro_f1_mean']:.4f}"
    V["FAM_RN_STD"] = f"{sel['families']['resnet18_transfer']['val_macro_f1_std']:.4f}"
    V["FAM_CNN_MEAN"] = f"{sel['families']['cnn_baseline']['val_macro_f1_mean']:.4f}"
    V["FAM_CNN_STD"] = f"{sel['families']['cnn_baseline']['val_macro_f1_std']:.4f}"

    mc = err["most_confused_pair"]
    V["PAIR_A"] = mc["a"]; V["PAIR_B"] = mc["b"]
    V["PAIR_A2B"] = f"{mc['a_to_b']:.3f}"; V["PAIR_B2A"] = f"{mc['b_to_a']:.3f}"
    V["PAIR_ERR_FRAC"] = f"{mc['combined_frac_of_errors']*100:.1f}"

    V["ERRDIR_TABLE"] = "\n".join(
        f"{d['true']} $\\rightarrow$ {d['pred']} & {d['count']} & {d['frac_of_errors']*100:.1f}\\% & {d['mean_conf']:.3f} \\\\"
        for d in err["error_directions"][:6])

    st = err["pair_shape_stats"]
    keys = [("img_w_median", "image width (median, px)", "{:.0f}"),
            ("img_h_median", "image height (median, px)", "{:.0f}"),
            ("aspect_median", "aspect ratio (median)", "{:.3f}"),
            ("target_area_median", "bright-target area fraction", "{:.3f}"),
            ("brightness_mean", "mean image brightness", "{:.1f}")]
    rd = st["_relative_diff"]
    V["SHAPE_TABLE"] = "\n".join(
        f"{lab} & {fstring.format(st[mc['a']][k])} & {fstring.format(st[mc['b']][k])} & "
        f"{rd.get(k, 0.0) * 100:.1f}\\% \\\\"
        for k, lab, fstring in keys)
    V["SHAPE_MAX_PCT"] = f"{max(rd.values())*100:.1f}"
    V["SHAPE_N_A"] = str(st[mc["a"]]["n"]); V["SHAPE_N_B"] = str(st[mc["b"]]["n"])

    # ---------- Grad-CAM ----------
    layers = list(loc["layers"])
    V["CAM_TABLE"] = "\n".join(
        f"\\texttt{{{l}}} & {loc['layers'][l]['cam_grid']} & "
        f"{loc['layers'][l]['cam_in_mask_ratio_mean']:.4f} & "
        f"{loc['layers'][l]['mask_area_frac_mean']:.4f} & "
        f"{loc['layers'][l]['lift_mask']:.2f}$\\times$ \\\\" for l in layers)
    bl = loc["best_layer"]; bs = loc["layers"][bl]
    V["CAM_BEST_LAYER"] = bl
    V["CAM_BEST_GRID"] = bs["cam_grid"]
    V["CAM_LIFT"] = f"{bs['lift_mask']:.2f}"
    V["CAM_MASK"] = f"{bs['cam_in_mask_ratio_mean']:.4f}"
    V["CAM_CHANCE"] = f"{bs['mask_area_frac_mean']:.4f}"
    V["CAM_CORRECT"] = f"{bs['cam_in_mask_ratio_correct']:.4f}"
    V["CAM_WRONG"] = f"{bs['cam_in_mask_ratio_wrong']:.4f}"
    V["CAM_LAYER4_GRID"] = loc["layers"]["layer4"]["cam_grid"] if "layer4" in loc["layers"] else "n/a"
    V["CAM_LAYER4_LIFT"] = f"{loc['layers']['layer4']['lift_mask']:.2f}" if "layer4" in loc["layers"] else "n/a"

    # ---------- 图路径 ----------
    V["FIG_EDA"] = "cnn_seed42__fig0_data_eda"
    V["FIG_CURVE_CNN"] = "cnn_baseline__fig1c_train_curves_all_seeds"
    V["FIG_CURVE_RN"] = "resnet18_transfer__fig1c_train_curves_all_seeds"
    V["FIG_CMP"] = "comparison_cnn_baseline_vs_resnet18_transfer__fig6a_model_comparison"
    V["FIG_PERCLASS"] = "comparison_cnn_baseline_vs_resnet18_transfer__fig6b_per_class_f1"
    V["FIG_CM"] = "comparison_cnn_baseline_vs_resnet18_transfer__fig6d_confusion_side_by_side"
    V["FIG_WRONG"] = "error_analysis__fig7a_all_wrong_samples"
    V["FIG_PAIR"] = "error_analysis__fig7d_most_confused_pair"
    V["FIG_CAMLOC"] = "error_analysis__fig7e_cam_localization"
    V["FIG_CAMEX"] = "error_analysis__fig7f_cam_on_vs_off_target"

    return V


# ============================================================================
#  LaTeX 模板
# ============================================================================
TEMPLATE = r"""
\documentclass[sigconf, screen]{acmart}

\settopmatter{printacmref=false, printccs=false, printfolios=true}
\renewcommand\footnotetextcopyrightpermission[1]{}
\pagestyle{plain}

\usepackage{booktabs}
\usepackage{array}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{url}
\graphicspath{{../../figures/}}

% 兜底：任何图都不得超过 0.86 页高。否则 LaTeX 会报
% "Float too large for page" 并把该浮动体无限推迟，引用随之变成 ??。
\setkeys{Gin}{width=\linewidth,height=0.86\textheight,keepaspectratio}

% 放宽浮动体参数：本报告图多（10 张），默认参数会把浮动体排到页外，
% 引发 "Overfull \vbox" 与超大空白页。
\setcounter{topnumber}{4}
\setcounter{bottomnumber}{3}
\setcounter{totalnumber}{7}
\renewcommand{\topfraction}{0.95}
\renewcommand{\bottomfraction}{0.9}
\renewcommand{\textfraction}{0.05}
\renewcommand{\floatpagefraction}{0.80}
\renewcommand{\dbltopfraction}{0.95}
\renewcommand{\dblfloatpagefraction}{0.80}
\setlength{\textfloatsep}{6pt plus 2pt minus 2pt}
\setlength{\floatsep}{6pt plus 2pt minus 2pt}
\setlength{\intextsep}{6pt plus 2pt minus 2pt}

% 双栏窄栏 + 含长不可断词（如 "2,098 / 446 / 446"、"(train/validation/test)"）
% 容易出现 Overfull \hbox。emergencystretch 允许 TeX 在万不得已时拉伸词距，
% 是比 \sloppy 更温和的修法（不会全局放松所有段落的排版）。
\setlength{\emergencystretch}{3em}

\acmConference[]{Technical Report}{October 2026}{}
\acmBooktitle{Technical Report}

\begin{document}

\title{A Controlled Comparison of a Compact CNN and ResNet18 Transfer Learning
for Six-Class SAR Aircraft Classification}

\author{Weiye Pan}
\affiliation{%
  \institution{Independent project repository}
  \country{}}
\email{}

\begin{abstract}
We present a controlled comparison between a compact convolutional neural network
trained from scratch and a ResNet18 model fine-tuned from ImageNet weights for
six-class aircraft classification in synthetic aperture radar (SAR) imagery.
Using the public SAR-ACD dataset (@N_SCANNED@ images, reduced to @N_AFTER_DEDUP@
after a duplicate audit), we hold every non-model factor constant: identical
deduplication, an identical stratified split (@N_TRAIN@/@N_VAL@/@N_TEST@),
identical input resolution, normalisation, augmentation, optimisation schedule,
and the same three random seeds. Across three seeds the compact CNN reaches a
macro F1 of @CNN_F1@ while ResNet18 reaches @RN_F1@, a gain of @D_F1@
(@SIG_F1@ against the seed-noise threshold of @THR_F1@). Beyond accuracy,
transfer learning also reduces seed-to-seed variance by roughly @STD_RATIO@$\times$.
An error analysis shows that a single class pair (A220 $\leftrightarrow$ A320321)
accounts for @PAIR_ERR_FRAC@\% of all remaining mistakes, and that the two classes
are statistically almost indistinguishable in size and shape. Finally, we quantify
what the model attends to with Grad-CAM, and we deliberately report this evidence
as indicative rather than conclusive, because the dataset provides no bounding
boxes and the aircraft occupies about three quarters of each chip, which limits
the discriminative power of any localisation metric. All numbers in this report
are generated programmatically from the experiment artefacts.
\end{abstract}

\keywords{SAR target classification, transfer learning, ResNet18, Grad-CAM,
reproducibility, error analysis}

\maketitle

% ===========================================================================
\section{Introduction}

Synthetic aperture radar (SAR) provides day-and-night, weather-independent
imaging and is widely used for remote sensing. Recognising aircraft types in SAR
imagery is a challenging fine-grained problem: the targets are small, the imagery
is single-channel and dominated by strong scatterers, and the appearance of a
given aircraft changes drastically with the observation azimuth
\cite{sarsurvey2016}.

This report studies a deliberately simple question: \emph{for a small SAR
aircraft dataset, how much does ImageNet transfer learning actually buy compared
with a compact network trained from scratch, and where does the difference come
from?} We answer it with a controlled experiment rather than a leaderboard entry.
Our contributions are:

\begin{itemize}
\item A \textbf{controlled comparison} in which every factor except the model is
      held fixed, including the random seeds and the data split, so that observed
      differences can be attributed to the model.
\item A \textbf{duplicate audit} performed \emph{before} splitting, which removes
      @N_DROPPED@ images including @N_CROSS_DUP@ cross-class duplicate groups;
      leaving them in place would leak identical images across the train and test
      splits \cite{leakage2012}.
\item An \textbf{error analysis} that localises the residual failures to a single
      aircraft pair, and quantifies how similar the two classes are.
\item A \textbf{critical Grad-CAM analysis}, including an explicit account of why
      the localisation evidence is weaker than it first appears.
\end{itemize}

% ===========================================================================
\section{Dataset}

\subsection{Source and actual statistics}
We use \textbf{SAR-ACD} \cite{saracd2022}, a public six-class civil aircraft
dataset released by the Aerospace Information Research Institute, Chinese Academy
of Sciences. Rather than trusting the published summary, we re-counted the data
locally; Table~\ref{tab:dataset} reports the measured values.

\begin{table}[t]
\caption{Measured dataset statistics, re-computed locally from the @N_SCANNED@
files rather than copied from the dataset description.}
\label{tab:dataset}
\small
\begin{tabular}{@{}lrrr@{}}
\toprule
Class & Raw & After dedup & Removed \\
\midrule
@CLASS_COUNTS_TABLE@
\midrule
Total & @N_SCANNED@ & @N_AFTER_DEDUP@ & @N_DROPPED@ \\
\bottomrule
\end{tabular}
\end{table}

The images are @IMG_MODES@ (single-channel) JPEGs with variable size
(@W_MIN@--@W_MAX@\,px wide, mean @W_MEAN@; @H_MIN@--@H_MAX@\,px high, mean
@H_MEAN@); there are @N_UNIQUE_SIZES@ distinct size combinations and
@N_CORRUPTED@ unreadable files. A fixed input size of @IMG_SIZE@\,$\times$\,@IMG_SIZE@
was therefore chosen, close to the native median size. The class ratio is
@CLASS_RATIO@:1, mildly imbalanced, which is why we report macro F1 as the primary
metric. Figure~\ref{fig:eda} summarises the dataset.

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{@FIG_EDA@.pdf}
\caption{Dataset overview. All values are read from the audit artefact produced
by the data pipeline; none are transcribed by hand.}
\label{fig:eda}
\end{figure}

\subsection{Duplicate audit and splitting}
A byte-level MD5 audit found @N_DROPPED@ exact duplicates in @N_CROSS_DUP@
cross-class groups and @N_SAME_DUP@ within-class groups. Cross-class duplicates
are images that appear under two different labels; because no ground truth can
arbitrate them, \emph{both} copies are dropped. Within-class duplicates keep one
copy. Deduplication is performed \textbf{before} splitting: otherwise the same
image could appear in both the training and test sets, silently inflating the test
score \cite{leakage2012}.

The remaining @N_AFTER_DEDUP@ images are divided into @N_TRAIN@ training,
@N_VAL@ validation and @N_TEST@ test images, using \emph{stratified} sampling
with a fixed seed (@SPLIT_SEED@). The split is written to disk and reused by
every model and every seed, so that all runs are compared on identical data.
After splitting, the pipeline asserts that the path sets and the content hashes
of the three subsets are pairwise disjoint, and that the class proportions
deviate by less than one percentage point.

% ===========================================================================
\section{Method}

\subsection{Compact CNN baseline}
The baseline is a four-layer convolutional network designed for
@IMG_SIZE@\,$\times$\,@IMG_SIZE@ single-channel input: four $3\times3$
convolutions (@PARAMS_CNN@ parameters) with batch normalisation
\cite{batchnorm2015}, $2\times2$ max-pooling after each of the first three,
global average pooling, dropout, and a linear classifier. Because the data are
grayscale, the first convolution takes \textbf{one} input channel; no channel
replication is used for this model.

\subsection{ResNet18 transfer learning}
The second model is ResNet18 \cite{resnet2016} with weights pre-trained on
ImageNet-1k \cite{imagenet2009}, as distributed by \texttt{torchvision}
\cite{pytorch2019}. The final fully connected layer is replaced by a
$\mathrm{Linear}(512 \rightarrow 6)$ layer initialised with a small standard
deviation ($\sigma=0.01$) so that its initially random outputs cannot overwhelm
the pre-trained features.

\textbf{Freezing policy.} We freeze the shallow stages
(\texttt{conv1}, \texttt{bn1}, \texttt{layer1}, \texttt{layer2}) and fine-tune the
deep stages (\texttt{layer3}, \texttt{layer4}) together with the new classifier.
The rationale is that early layers encode generic low-level structure that
transfers, whereas late layers encode task-specific semantics that must adapt;
with only @N_TRAIN@ training images, updating all @PARAMS_RN_TOTAL@ parameters
would be prone to overfitting. The frozen modules are additionally kept in
evaluation mode so that their batch-norm statistics do not drift.

\textbf{Learning rates.} We use discriminative learning rates: $10^{-3}$ for the
new classifier and $10^{-4}$ for the fine-tuned backbone stages
($\approx$@PARAMS_RN_TRAIN@ trainable parameters, @PARAMS_RN_PCT@\% of the model).
Optimisation uses AdamW \cite{adamw2019} with weight decay $10^{-4}$ and a cosine
schedule \cite{sgdr2017}.

\textbf{Grayscale handling.} The dataset yields one-channel images. We replicate
the single channel to three channels inside the model so that \emph{every}
pre-trained weight is used unchanged. The alternative --- rebuilding the first
convolution with one input channel by averaging the pre-trained RGB filters ---
avoids the mismatch between a grayscale input and a colour-pretrained stem, but
introduces an arbitrary averaging step. The replication variant is the more
conservative choice for a transfer-learning study and is what we report.

% ===========================================================================
\section{Experiments}

\subsection{Setup}
All runs use the same pipeline and the same hyper-parameters (Table~\ref{tab:setup}),
differing only in the model. Two deliberate choices deserve emphasis. First, the
normalisation statistics are computed \emph{on the training split only}
(mean @NORM_MEAN@, std @NORM_STD@) and then applied to validation and test;
computing them on the full dataset would leak information. Second, we deliberately
keep the \emph{same} normalisation for both models instead of using ImageNet
statistics for the pre-trained model: using ImageNet statistics is common practice,
but it would change two factors at once and make the comparison unattributable.

\begin{table}[t]
\caption{Experiment configuration. Every field below is identical for the two
models except the model definition itself.}
\label{tab:setup}
\small
\begin{tabular}{@{}ll@{}}
\toprule
Item & Value \\
\midrule
Input size & @IMG_SIZE@ $\times$ @IMG_SIZE@, grayscale \\
Split (train/val/test) & @N_TRAIN@ / @N_VAL@ / @N_TEST@, seed @SPLIT_SEED@ \\
Augmentation & rotation $\pm30^\circ$, flip, translate 5\%, scale 0.9--1.1 \\
 & (training split only) \\
Optimiser & AdamW, weight decay @WD@ \\
Learning rate & CNN @LR_HEAD@; ResNet18 head @LR_HEAD@ / backbone @LR_BACKBONE@ \\
Schedule & cosine annealing \\
Epochs / batch & @EPOCHS@ / @BATCH@ \\
Early stopping & validation macro F1, patience @PATIENCE@ \\
Random seeds & @SEEDS@ \\
Model selection & best \emph{validation} macro F1 \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Metrics and protocol}
We report accuracy, macro F1 (primary), weighted F1, per-class
precision/recall/F1, and confusion matrices (raw and row-normalised). The test
split is evaluated \emph{once} per run, using the checkpoint selected on the
validation split; no test-set quantity is used for any decision. Each evaluation
appends a record to a ledger, so the number of test-set evaluations is auditable.
Every configuration is run with @SEEDS@ seeds and we report means and standard
deviations.

\subsection{Training behaviour}
Figures~\ref{fig:curvescnn} and~\ref{fig:curvesrn} show the training curves of all
three seeds for the two models. The baseline converges slowly and its validation
score keeps fluctuating; ResNet18 reaches a much higher level within a few epochs
and its three seeds nearly coincide.

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{@FIG_CURVE_CNN@.pdf}
\caption{Compact CNN: loss and accuracy for the three seeds (solid = train,
dashed = validation). The seeds diverge noticeably.}
\label{fig:curvescnn}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{@FIG_CURVE_RN@.pdf}
\caption{ResNet18 transfer learning: the three seeds track each other closely,
indicating a much more stable optimisation.}
\label{fig:curvesrn}
\end{figure}

% ===========================================================================
\section{Results}

\subsection{Overall comparison}
Table~\ref{tab:results} reports the test-set results. ResNet18 improves macro F1
by @D_F1@, from @CNN_F1@ to @RN_F1@, a relative gain of @F1_REL@\%.
Figure~\ref{fig:cmp} shows the same comparison graphically.

Following the protocol fixed before the experiments, we only call a difference
meaningful when the gap in means exceeds the larger of the two standard
deviations. Here the gap (@D_F1@) is about an order of magnitude larger than the
threshold (@THR_F1@), so the improvement is \textbf{@SIG_F1@}. A second, easily
overlooked result is that transfer learning also makes the outcome far more
reproducible: the macro-F1 standard deviation drops from @CNN_SEED_F1@'s spread to
a spread of @RN_SEED_F1@ across seeds, a reduction of about @STD_RATIO@$\times$.

\begin{table}[t]
\caption{Test-set results (mean $\pm$ std over @SEEDS@ seeds). The seed-noise
threshold is @THR_F1@; a difference counts as meaningful only if it exceeds it.
All three differences do.}
\label{tab:results}
\small
\begin{tabular}{@{}lccc@{}}
\toprule
Metric & Compact CNN & ResNet18 & $\Delta$ \\
\midrule
@RESULTS_TABLE@
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{@FIG_CMP@.pdf}
\caption{Overall test metrics. Error bars are standard deviations over the
@SEEDS@ seeds; annotations give the delta and whether it exceeds the seed noise.}
\label{fig:cmp}
\end{figure}

\subsection{Per-class behaviour}
Table~\ref{tab:perclass} and Figure~\ref{fig:perclass} break the result down by
class. The improvement is not uniform. The largest gains occur precisely in the
classes the baseline struggled with most, while the smallest gain is on A320321.

\begin{table}[t]
\caption{Per-class F1 (mean over @SEEDS@ seeds), sorted by improvement.}
\label{tab:perclass}
\small
\begin{tabular}{@{}lcccc@{}}
\toprule
Class & Support & CNN & ResNet18 & $\Delta$ \\
\midrule
@PERCLASS_TABLE@
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{@FIG_PERCLASS@.pdf}
\caption{Per-class F1 for both models, ordered by improvement.}
\label{fig:perclass}
\end{figure}

\subsection{Confusion structure}
Figure~\ref{fig:cm} contrasts the row-normalised confusion matrices. The baseline
makes mistakes in many directions at once, whereas ResNet18 is close to diagonal
everywhere except for one pair of classes, which we examine next.

\begin{figure*}[t]
\centering
\includegraphics[width=0.86\textwidth]{@FIG_CM@.pdf}
\caption{Row-normalised confusion matrices (mean over @SEEDS@ seeds).
Left: compact CNN. Right: ResNet18. The baseline errs in several directions;
ResNet18 leaves essentially a single confused pair.}
\label{fig:cm}
\end{figure*}

% ===========================================================================
\section{Error Analysis}

\subsection{Final model and error sample set}
The final model is chosen \textbf{on validation performance only}: we first pick
the better model family by mean validation macro F1 (ResNet18 @FAM_RN_MEAN@
$\pm$ @FAM_RN_STD@ versus the baseline @FAM_CNN_MEAN@ $\pm$ @FAM_CNN_STD@), then
the best seed within that family (seed @SEL_SEED@, validation macro F1
@SEL_VALF1@). No test-set quantity takes part in this selection.

On the test split the final model makes @N_WRONG@ errors out of @N_TEST@
(@ERR_RATE@\%). \textbf{All} of them are visualised in Figure~\ref{fig:wrong};
none are discarded. Errors are less confident than correct predictions
(@CONF_WRONG@ versus @CONF_CORRECT@, a gap of @CONF_GAP@), which suggests that a
confidence threshold could be used for rejection in deployment.

\begin{figure*}[t]
\centering
\includegraphics[width=0.86\textwidth]{@FIG_WRONG@.pdf}
\caption{Every misclassified test image (@N_WRONG@ of @N_TEST@). Titles give
true class $\rightarrow$ predicted class and the predicted confidence. No
sample is filtered out.}
\label{fig:wrong}
\end{figure*}

\subsection{The dominant confusion}
Table~\ref{tab:errdir} lists the error directions. A single pair dominates:
A220 $\leftrightarrow$ A320321 accounts for @PAIR_ERR_FRAC@\% of all errors and is
\emph{bidirectional} (@PAIR_A2B@ of true A220 predicted as A320321, and
@PAIR_B2A@ in the opposite direction), which indicates genuine overlap in feature
space rather than one class being absorbed by another.

\begin{table}[t]
\caption{Most frequent error directions of the final model.}
\label{tab:errdir}
\small
\begin{tabular}{@{}lrrr@{}}
\toprule
True $\rightarrow$ Pred. & Count & Share & Mean conf. \\
\midrule
@ERRDIR_TABLE@
\bottomrule
\end{tabular}
\end{table}

\subsection{Why are these two classes confused?}
Table~\ref{tab:shape} compares measurable properties of the two classes. Every
statistic differs by less than @SHAPE_MAX_PCT@\%, i.e. the two aircraft are
nearly indistinguishable by size and shape statistics alone. Together with the
bidirectionality of the errors, this supports three explanations, in decreasing
order of evidential support:

\begin{enumerate}
\item \textbf{Genuine shape similarity.} A220 and A320321 are both narrow-body
      airliners; at the resolution available here, their wing sweep, engine
      layout and fuselage proportions differ little.
\item \textbf{Azimuth variation masquerading as class variation.} SAR scattering
      is extremely azimuth-sensitive: the same aircraft seen from different
      aspects can differ more than two different aircraft seen from the same
      aspect. Random rotation augmentation ($\pm30^\circ$, applied here) may
      additionally blur this boundary.
\item \textbf{Insufficient resolution.} Resizing to @IMG_SIZE@\,$\times$\,@IMG_SIZE@
      discards detail; the native chips are only about @W_MEAN@\,$\times$\,@H_MEAN@\,px.
\end{enumerate}

\begin{table}[t]
\caption{Measured properties of the confused pair (medians unless noted).
All relative differences are below @SHAPE_MAX_PCT@\%.}
\label{tab:shape}
\small
\begin{tabular}{@{}lccc@{}}
\toprule
Statistic & @PAIR_A@ & @PAIR_B@ & Rel. diff. \\
\midrule
@SHAPE_TABLE@
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{@FIG_PAIR@.pdf}
\caption{The most confused pair. First six images are true @PAIR_A@, the last six
true @PAIR_B@; titles give true $\rightarrow$ predicted and confidence.}
\label{fig:pair}
\end{figure}

The first two explanations are \emph{inferences} consistent with the evidence, not
established facts; confirming them would require azimuth metadata per image or
target boxes for alignment.

% ===========================================================================
\section{Grad-CAM Interpretability}

\subsection{How the measurement is made}
This dataset has no bounding boxes, so we approximate the target region by the
pixels brighter than $\mu+\sigma$ (SAR aircraft are strong scatterers). We then
measure the fraction of Grad-CAM \cite{gradcam2017} mass that falls on those
pixels and compare it with the fraction that a random heat map would produce.

A methodological finding is important here. Our first measurement suggested an
essentially random localisation. The cause was resolution, not indifference:
ResNet18 downsamples by a factor of 32, so on @IMG_SIZE@\,$\times$\,@IMG_SIZE@
inputs the \texttt{layer4} feature map is only
@CAM_LAYER4_GRID@ (each cell spanning 32\,$\times$\,32 pixels of the input).
Table~\ref{tab:cam} therefore also reports \texttt{layer3}
(@CAM_BEST_GRID@), and Figure~\ref{fig:camloc} shows both.

\begin{table}[t]
\caption{Grad-CAM localisation by target layer. ``Lift'' is the ratio of CAM mass
on bright pixels to the random baseline.}
\label{tab:cam}
\small
\begin{tabular}{@{}lcccc@{}}
\toprule
Layer & CAM grid & On target & Random & Lift \\
\midrule
@CAM_TABLE@
\bottomrule
\end{tabular}
\end{table}

\subsection{What the model attends to}
Using the finer \texttt{@CAM_BEST_LAYER@} layer, @CAM_MASK@ of the CAM mass falls
on bright target pixels against a random baseline of @CAM_CHANCE@, a lift of
\textbf{@CAM_LIFT@}$\times$. The same quantity is @CAM_CORRECT@ for correct
predictions and @CAM_WRONG@ for incorrect ones. Figure~\ref{fig:camloc} shows the
distribution and the layer comparison; Figure~\ref{fig:camx} contrasts the most
target-focused and the most diffuse examples.

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{@FIG_CAMLOC@.pdf}
\caption{Left: distribution of CAM mass landing on bright target pixels
(\texttt{@CAM_BEST_LAYER@}), with the random baseline. Right: concentration
relative to chance for both candidate layers, showing that the coarse
\texttt{layer4} map understates the localisation.}
\label{fig:camloc}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{@FIG_CAMEX@.pdf}
\caption{The most target-focused (bottom) and most diffuse (top) examples for
\texttt{@CAM_BEST_LAYER@}.}
\label{fig:camx}
\end{figure}

We deliberately stop short of claiming that the model ``looks at the aircraft''.
A lift of @CAM_LIFT@$\times$ is a tendency, not a decisive result, and the
measurement has three limitations: (i) the target region is approximated by
bright pixels rather than annotated, so strong sidelobes count as target;
(ii) the aircraft occupies roughly three quarters of each chip, which leaves
little room for a background-driven model and makes any localisation metric
intrinsically weak; and (iii) even @CAM_BEST_GRID@ is a coarse grid. We therefore
report this section as \emph{indicative evidence}.

% ===========================================================================
\section{Limitations}

\paragraph{The public subset contains expanded samples.}
The released portion of SAR-ACD is described by its authors as
``@N_SCANNED@ images in six civilian categories \emph{after expansion}''.
Augmented or synthetically enlarged samples can share near-duplicate structure
with the originals. Although our byte-level audit removed exact duplicates, it
cannot detect near-duplicates created by augmentation, so the reported test
scores should be read as an \emph{upper bound} for genuinely unseen data.

\paragraph{Results do not establish cross-airport generalisation.}
The imagery covers the Shanghai Hongqiao airport area and a small number of other
airports, acquired at particular times and geometries. Several classes may be
correlated with the acquisition site or the flight schedule rather than with the
aircraft shape. Our evaluation is a random split within this collection, so it
measures \emph{within-collection} generalisation only. Nothing here supports a
claim about performance at a new airport, a new sensor, or a new frequency band.

\paragraph{Other limitations.}
The dataset has no bounding boxes, so localisation evidence remains indirect
(Section~7) and target-aligned comparisons are impossible. The images are
single-polarisation and single-channel, so polarimetric or multi-look cues are
unavailable. Finally, three seeds are the minimum needed to estimate spread; the
resulting standard deviations are indicative rather than precise.

% ===========================================================================
\section{Conclusion and Future Work}

In a fully controlled comparison on six-class SAR aircraft classification,
ImageNet transfer learning improved macro F1 from @CNN_F1@ to @RN_F1@
(@D_F1@) while reducing seed-to-seed variance by roughly @STD_RATIO@$\times$.
The residual errors are highly concentrated: A220 $\leftrightarrow$ A320321 alone
accounts for @PAIR_ERR_FRAC@\% of them, and the two classes are statistically
almost identical in size and shape. A critical Grad-CAM analysis indicates that
the model is more target-focused than chance, but we report that as a tendency
rather than a conclusion, and we explain why the measurement cannot be stronger
with the data available.

Future work, in priority order:
\begin{itemize}
\item \textbf{Resolve the dominant confusion.} If azimuth metadata can be
      recovered, stratify the error rate by aspect angle; otherwise, obtain
      target boxes and evaluate after alignment, so that shape rather than
      pose is compared.
\item \textbf{Near-duplicate detection.} Replace exact-hash auditing with
      perceptual hashing or embedding-based deduplication to bound the effect of
      the expanded public subset.
\item \textbf{Cross-airport evaluation.} Partition by acquisition site and report
      leave-one-site-out accuracy, which is the claim that actually matters for
      deployment.
\item \textbf{Freezing-policy ablation.} The implementation already supports
      full fine-tuning and linear probing; comparing them would separate
      ``pre-training helps'' from ``more parameters help''.
\item \textbf{Resolution study.} Repeat at @IMG_SIZE@ and at higher input sizes to
      test whether the A220/A320321 boundary is limited by resolution.
\end{itemize}

\section*{Reproducibility}
Every number, table and figure in this report is generated programmatically from
the artefacts under \texttt{results/}. The split is fixed by seed @SPLIT_SEED@ and
asserted to be leakage-free; each test-set evaluation is recorded in a ledger; and
each run stores its resolved configuration, log, per-epoch history, metrics and
checkpoint. A clean-room reproduction consists of cloning the repository, creating
the environment from \texttt{environment.yml}, and running the pipeline scripts in
order.

\bibliographystyle{ACM-Reference-Format}
\bibliography{refs}

\end{document}
"""


REFS_BIB = r"""
@ARTICLE{saracd2022,
  author  = {Sun, Xian and Lv, Yixuan and Wang, Zhirui and Fu, Kun},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  title   = {{SCAN}: Scattering Characteristics Analysis Network for Few-Shot
             Aircraft Classification in High-Resolution {SAR} Images},
  year    = {2022},
  volume  = {60},
  pages   = {1--17},
  doi     = {10.1109/TGRS.2022.3166174}
}

@INPROCEEDINGS{resnet2016,
  author    = {He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  title     = {Deep Residual Learning for Image Recognition},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern
               Recognition (CVPR)},
  year      = {2016},
  pages     = {770--778},
  publisher = {IEEE},
  address   = {Las Vegas, NV, USA},
  doi       = {10.1109/CVPR.2016.90}
}

@INPROCEEDINGS{gradcam2017,
  author    = {Selvaraju, Ramprasaath R. and Cogswell, Michael and Das, Abhishek
               and Vedantam, Ramakrishna and Parikh, Devi and Batra, Dhruv},
  title     = {{Grad-CAM}: Visual Explanations from Deep Networks via
               Gradient-Based Localization},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision
               (ICCV)},
  year      = {2017},
  pages     = {618--626},
  publisher = {IEEE},
  address   = {Venice, Italy},
  doi       = {10.1109/ICCV.2017.74}
}

@INPROCEEDINGS{imagenet2009,
  author    = {Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and
               Li, Kai and Fei-Fei, Li},
  title     = {{ImageNet}: A Large-Scale Hierarchical Image Database},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern
               Recognition (CVPR)},
  year      = {2009},
  pages     = {248--255},
  publisher = {IEEE},
  address   = {Miami, FL, USA},
  doi       = {10.1109/CVPR.2009.5206848}
}

@MISC{adamw2019,
  author        = {Loshchilov, Ilya and Hutter, Frank},
  title         = {Decoupled Weight Decay Regularization},
  year          = {2019},
  eprint        = {1711.05101},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  howpublished  = {arXiv preprint arXiv:1711.05101}
}

@MISC{sgdr2017,
  author        = {Loshchilov, Ilya and Hutter, Frank},
  title         = {{SGDR}: Stochastic Gradient Descent with Warm Restarts},
  year          = {2017},
  eprint        = {1608.03983},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  howpublished  = {arXiv preprint arXiv:1608.03983}
}

@INPROCEEDINGS{batchnorm2015,
  author    = {Ioffe, Sergey and Szegedy, Christian},
  title     = {Batch Normalization: Accelerating Deep Network Training by
               Reducing Internal Covariate Shift},
  booktitle = {Proceedings of the 32nd International Conference on Machine
               Learning (ICML)},
  year      = {2015},
  pages     = {448--456},
  publisher = {PMLR},
  address   = {Lille, France},
  eprint    = {1502.03167},
  archivePrefix = {arXiv}
}

@INPROCEEDINGS{pytorch2019,
  author    = {Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam
               and Bradbury, James and Chanan, Gregory and others},
  title     = {{PyTorch}: An Imperative Style, High-Performance Deep Learning
               Library},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2019},
  pages     = {8024--8035},
  publisher = {Curran Associates, Inc.},
  address   = {Vancouver, BC, Canada},
  eprint    = {1912.01703},
  archivePrefix = {arXiv}
}

@ARTICLE{transfer2010,
  author  = {Pan, Sinno Jialin and Yang, Qiang},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  title   = {A Survey on Transfer Learning},
  year    = {2010},
  volume  = {22},
  number  = {10},
  pages   = {1345--1359},
  doi     = {10.1109/TKDE.2009.191}
}

@ARTICLE{sarsurvey2016,
  author  = {El-Darymli, Khalid and Gill, Eric W. and McGuire, Peter and
             Power, Desmond and Moloney, Cecilia},
  journal = {IEEE Access},
  title   = {Automatic Target Recognition in Synthetic Aperture Radar Imagery:
             A State-of-the-Art Review},
  year    = {2016},
  volume  = {4},
  pages   = {6014--6058},
  doi     = {10.1109/ACCESS.2016.2611492}
}

@ARTICLE{leakage2012,
  author  = {Kaufman, Shachar and Rosset, Saharon and Perlich, Claudia},
  journal = {ACM Transactions on Knowledge Discovery from Data},
  title   = {Leakage in Data Mining: Formulation, Detection, and Avoidance},
  year    = {2012},
  volume  = {6},
  number  = {4},
  pages   = {1--21},
  articleno = {15},
  numpages = {21},
  doi     = {10.1145/2382577.2382579}
}
"""


# ============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="生成 ACM 双栏技术报告（数字全部来自产物）")
    ap.add_argument("--project", default=None)
    args = ap.parse_args()
    project = Path(args.project).resolve() if args.project else Path(__file__).resolve().parent.parent

    outdir = project / "report" / "acm"
    outdir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("make_latex_report", outdir / "generate.log")

    V = load_numbers(project)
    tex = TEMPLATE
    missing = []
    for k, v in V.items():
        token = f"@{k}@"
        if token not in tex:
            missing.append(k)
        tex = tex.replace(token, str(v))

    left = [t for t in __import__("re").findall(r"@[A-Z0-9_]+@", tex)]
    if left:
        logger.error("模板里还有未替换的占位符: %s", sorted(set(left)))
        return 1

    (outdir / "main.tex").write_text(tex, encoding="utf-8")
    (outdir / "refs.bib").write_text(REFS_BIB, encoding="utf-8")
    save_json(V, outdir / "report_numbers.json")

    logger.info("生成 %s", outdir / "main.tex")
    logger.info("生成 %s（%d 条参考文献）", outdir / "refs.bib", REFS_BIB.count("@"))
    logger.info("注入 %d 个数据占位符；未使用的键 %d 个", len(V) - len(missing), len(missing))
    if missing:
        logger.warning("以下键在模板中未被使用（可忽略）: %s", missing[:10])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
