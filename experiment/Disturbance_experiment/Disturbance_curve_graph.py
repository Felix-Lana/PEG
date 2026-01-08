# -*- coding: utf-8 -*-
"""
Publication-ready 3×3 master figure (A–I) for visibility-aware segmentation.

Layout order (row-major) after swapping:
Row1: A B C  (curves)
Row2: D E F  (visibility-aware panels: presence vs v, high-v FPR, high-v presence)
Row3: G H I  (summaries: AUC_Δ_norm, slope, trigger v*)

Directory:
- Single: root_dir/<model>/X_v_mask.png
- Multi : root_dir/<domain>/<model>/X_v_mask.png
GT: gt_dir files start with integer X (e.g., 0.png, 1_xxx.png)
"""

from __future__ import annotations
import os
import re
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# ----------------------------
# Publication style (Science/Nature-like clean)
# ----------------------------
def set_pub_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,

        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "xtick.direction": "out_mask",
        "ytick.direction": "out_mask",

        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        "figure.dpi": 120,
        "savefig.dpi": 300,
    })


def polish_axis(ax, grid_axis: Optional[str] = None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis is None:
        ax.grid(False)
    else:
        ax.grid(True, axis=grid_axis, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.tick_params(pad=2)


# ----------------------------
# I/O
# ----------------------------
def read_index_mask(path: str) -> np.ndarray:
    img = Image.open(path)
    if img.mode not in ("L", "I;16", "P"):
        img = img.convert("L")
    return np.array(img)


def ensure_same_shape(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> None:
    if a.shape[:2] != b.shape[:2]:
        raise ValueError(f"Shape mismatch: {name_a}={a.shape} vs {name_b}={b.shape}")


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den != 0 else 0.0


# ----------------------------
# Metrics
# ----------------------------
@dataclass
class MetricsRow:
    model: str
    sample_x: int
    v: int

    # bias / visibility-aware
    ac_to_lens: float
    lens_fpr: float
    pred_lens_area: float

    # dice/iou kept for completeness (csv)
    dice_bg: float
    dice_cornea: float
    dice_ac: float
    dice_lens: float
    iou_bg: float
    iou_cornea: float
    iou_ac: float
    iou_lens: float


def per_class_dice_iou(gt: np.ndarray, pred: np.ndarray, c: int) -> Tuple[float, float]:
    gt_c = (gt == c)
    pr_c = (pred == c)
    inter = np.logical_and(gt_c, pr_c).sum()
    union = np.logical_or(gt_c, pr_c).sum()
    denom = gt_c.sum() + pr_c.sum()
    dice = safe_div(2.0 * inter, denom)
    iou = safe_div(inter, union)
    return dice, iou


def compute_metrics(gt: np.ndarray, pred: np.ndarray) -> Tuple[float, float, float, Dict[int, Tuple[float, float]]]:
    # labels: 0 bg, 1 cornea, 2 AC, 3 lens
    gt_ac = (gt == 2)
    gt_not_lens = (gt != 3)
    pr_lens = (pred == 3)

    ac_to_lens = safe_div(np.logical_and(gt_ac, pr_lens).sum(), gt_ac.sum())
    lens_fpr = safe_div(np.logical_and(gt_not_lens, pr_lens).sum(), gt_not_lens.sum())
    pred_lens_area = safe_div(pr_lens.sum(), gt.size)

    per = {}
    for c in [0, 1, 2, 3]:
        per[c] = per_class_dice_iou(gt, pred, c)

    return ac_to_lens, lens_fpr, pred_lens_area, per


# ----------------------------
# Dataset scanning
# ----------------------------
PRED_RE = re.compile(r"^(?P<x>\d+)_(?P<v>\d+)_mask\.(png|bmp|tif|tiff|jpg|jpeg)$", re.IGNORECASE)
GT_RE = re.compile(r"^(?P<x>\d+).*?\.(png|bmp|tif|tiff|jpg|jpeg)$", re.IGNORECASE)


def scan_pred_dir(pred_dir: str) -> Dict[Tuple[int, int], str]:
    m: Dict[Tuple[int, int], str] = {}
    for fn in os.listdir(pred_dir):
        mo = PRED_RE.match(fn)
        if not mo:
            continue
        x = int(mo.group("x"))
        v = int(mo.group("v"))
        m[(x, v)] = os.path.join(pred_dir, fn)
    return m


def scan_gt_dir(gt_dir: str) -> Dict[int, str]:
    g: Dict[int, str] = {}
    for fn in os.listdir(gt_dir):
        mo = GT_RE.match(fn)
        if not mo:
            continue
        x = int(mo.group("x"))
        g[x] = os.path.join(gt_dir, fn)
    return g


# ----------------------------
# CSV
# ----------------------------
def write_csv(rows: List[MetricsRow], out_csv: str) -> None:
    import csv
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "model", "sample_x", "v",
            "ac_to_lens", "lens_fpr", "pred_lens_area",
            "dice_bg", "dice_cornea", "dice_ac", "dice_lens",
            "iou_bg", "iou_cornea", "iou_ac", "iou_lens",
        ])
        for r in rows:
            w.writerow([
                r.model, r.sample_x, r.v,
                f"{r.ac_to_lens:.6f}", f"{r.lens_fpr:.6f}", f"{r.pred_lens_area:.6f}",
                f"{r.dice_bg:.6f}", f"{r.dice_cornea:.6f}", f"{r.dice_ac:.6f}", f"{r.dice_lens:.6f}",
                f"{r.iou_bg:.6f}", f"{r.iou_cornea:.6f}", f"{r.iou_ac:.6f}", f"{r.iou_lens:.6f}",
            ])


# ----------------------------
# Plot helpers
# ----------------------------
def mean_ci(values: np.ndarray) -> Tuple[float, float]:
    n = values.size
    if n == 0:
        return 0.0, 0.0
    mu = float(values.mean())
    if n == 1:
        return mu, 0.0
    sd = float(values.std(ddof=1))
    se = sd / np.sqrt(n)
    return mu, 1.96 * se


def mean_curve(rr: List[MetricsRow], V: List[int], attr: str) -> np.ndarray:
    ys = []
    for v in V:
        vv = np.array([getattr(r, attr) for r in rr if r.v == v], dtype=np.float32)
        ys.append(float(vv.mean()) if vv.size else 0.0)
    return np.array(ys, dtype=np.float32)


def auc_value(x: np.ndarray, y: np.ndarray, mode: str) -> float:
    # raw: ∫y dv ; norm: mean(y) over v ; delta_norm: mean(y-y0) over v
    if x.size < 2:
        return 0.0
    rng = float(x[-1] - x[0])
    if rng == 0:
        return 0.0
    if mode == "raw":
        return float(np.trapz(y, x))
    if mode == "norm":
        return float(np.trapz(y, x) / rng)
    if mode == "delta_norm":
        y0 = float(y[0]) if y.size else 0.0
        return float(np.trapz(y - y0, x) / rng)
    raise ValueError(f"Unknown auc_mode: {mode}")


def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    a, _b = np.polyfit(x, y, 1)
    return float(a)


def trigger_v(x: np.ndarray, y: np.ndarray, delta_abs: float) -> Optional[int]:
    # first v where y(v) >= y(0)+delta_abs; None if never
    if x.size == 0:
        return None
    base = float(y[0])
    thr = base + float(delta_abs)
    for i in range(y.size):
        if float(y[i]) >= thr:
            return int(x[i])
    return None


def auto_pad_ylim(ax, mat: np.ndarray, pad_ratio: float = 0.12) -> None:
    vals = mat[np.isfinite(mat)]
    if vals.size == 0:
        return
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmin == vmax:
        pad = abs(vmax) * 0.2 + 1e-6
        ax.set_ylim(vmin - pad, vmax + pad)
        return
    pad = (vmax - vmin) * pad_ratio
    ax.set_ylim(vmin - pad, vmax + pad)


# ----------------------------
# Plot: 3×3 master panel with swapped groups + synced labels
# ----------------------------
def plot_master_3x3_swapped_groups(
    rows: List[MetricsRow],
    out_png: str,
    v_min: int,
    v_max: int,
    show_ci: bool = True,
    trigger_delta_abs: float = 0.05,
    auc_mode: str = "delta_norm",
    presence_tau: float = 0.001,
    high_v0: int = 8,
    save_pdf: bool = False,
) -> None:
    set_pub_style()

    # ---- FIXED model order for bars (and tick labels) ----
    PREFERRED_ORDER = ["UNet", "UNet++", "U2Net", "TransUNet", "nnFormer", "CSWin", "EMCAD", "Ours"]
    models_in_data = sorted(set(r.model for r in rows))
    models = [m for m in PREFERRED_ORDER if m in models_in_data] + [m for m in models_in_data if m not in PREFERRED_ORDER]

    V = list(range(v_min, v_max + 1))
    x_v = np.array(V, dtype=np.float32)

    metric_defs = [
        ("ac_to_lens", "Rate", "(A) AC → Lens confusion"),
        ("lens_fpr", "Rate", "(B) Lens FPR"),
        ("pred_lens_area", "Area / (H·W)", "(C) Predicted lens area ratio"),
    ]
    bar_labels = ["AC→Lens", "Lens FPR", "Pred lens area"]

    fig = plt.figure(figsize=(14.8, 10.0))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.35, 1.05, 1.05], hspace=0.42, wspace=0.32)

    # Row 1: A B C (unchanged)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])

    # Row 2: (D E F) but content = original G H I (visibility-aware)
    axD = fig.add_subplot(gs[1, 0])
    axE = fig.add_subplot(gs[1, 1])
    axF = fig.add_subplot(gs[1, 2])

    # Row 3: (G H I) but content = original D E F (summaries)
    axG = fig.add_subplot(gs[2, 0])
    axH = fig.add_subplot(gs[2, 1])
    axI = fig.add_subplot(gs[2, 2])

    # ---- A/B/C curves + stats for summaries ----
    stats = [dict() for _ in range(3)]  # metric -> model -> (auc, slope, trig)

    for mi, (attr, ylab, title) in enumerate(metric_defs):
        ax = [axA, axB, axC][mi]
        ax.set_title(title)
        ax.set_xlabel("Visibility step v")
        ax.set_ylabel(ylab)
        polish_axis(ax, grid_axis=None)

        for m in models:
            rr = [r for r in rows if r.model == m]
            y = mean_curve(rr, V, attr)

            if mi == 0:
                ax.plot(x_v, y, marker="o", markersize=3, linewidth=1.4, label=m)
            else:
                ax.plot(x_v, y, marker="o", markersize=3, linewidth=1.4)

            if show_ci:
                e = []
                for v in V:
                    vv = np.array([getattr(r, attr) for r in rr if r.v == v], dtype=np.float32)
                    _mu, hw = mean_ci(vv)
                    e.append(hw)
                e = np.array(e, dtype=np.float32)
                ax.fill_between(x_v, y - e, y + e, alpha=0.12)

            stats[mi][m] = (
                auc_value(x_v, y, auc_mode),
                linear_slope(x_v, y),
                trigger_v(x_v, y, trigger_delta_abs),
            )

    # Legend only once
    leg = axA.legend(frameon=False, loc="upper left", ncol=2,
                     handlelength=1.6, columnspacing=0.9, borderaxespad=0.2)
    for lh in leg.legend_handles:
        lh.set_linewidth(1.6)

    # ---- Row 2: D/E/F = visibility-aware panels (content originally G/H/I) ----
    # (D) presence rate vs v
    axD.set_title("(D) Predicted lens presence rate vs v")
    axD.set_xlabel("Visibility step v")
    axD.set_ylabel(f"Presence rate (area > {presence_tau})")
    polish_axis(axD, grid_axis=None)

    presence_rate_by_model: Dict[str, np.ndarray] = {}
    for m in models:
        rr = [r for r in rows if r.model == m]
        y, e = [], []
        for v in V:
            vv = np.array([r.pred_lens_area for r in rr if r.v == v], dtype=np.float32)
            if vv.size == 0:
                y.append(0.0); e.append(0.0); continue
            pres = (vv > presence_tau).astype(np.float32)
            mu = float(pres.mean())
            se = float(pres.std(ddof=1) / np.sqrt(pres.size)) if pres.size > 1 else 0.0
            y.append(mu); e.append(1.96 * se)
        y = np.array(y, dtype=np.float32)
        e = np.array(e, dtype=np.float32)
        presence_rate_by_model[m] = y

        axD.plot(x_v, y, marker="o", markersize=3, linewidth=1.4)
        if show_ci:
            axD.fill_between(x_v, y - e, y + e, alpha=0.12)

    # x positions for bars (must follow fixed model order)
    x_m = np.arange(len(models), dtype=np.float32)

    # (E) mean lens FPR under high v
    axE.set_title(f"(E) Mean lens FPR under high v (v ≥ {high_v0})")
    axE.set_ylabel("Mean lens FPR")
    polish_axis(axE, grid_axis="y")

    valsE = []
    for m in models:
        rr = [r for r in rows if r.model == m and r.v >= high_v0]
        vv = np.array([r.lens_fpr for r in rr], dtype=np.float32)
        valsE.append(float(vv.mean()) if vv.size else 0.0)

    axE.bar(x_m, np.array(valsE, dtype=np.float32), width=0.72, linewidth=0.6, edgecolor="black", alpha=0.92)
    axE.set_xticks(x_m)
    axE.set_xticklabels(models, rotation=35, ha="right")

    # (F) mean presence under high v
    axF.set_title(f"(F) Mean lens presence rate under high v (v ≥ {high_v0})")
    axF.set_ylabel(f"Presence rate (area > {presence_tau})")
    polish_axis(axF, grid_axis="y")

    valsF = []
    for m in models:
        rr = [r for r in rows if r.model == m and r.v >= high_v0]
        vv = np.array([r.pred_lens_area for r in rr], dtype=np.float32)
        if vv.size == 0:
            valsF.append(0.0)
        else:
            pres = (vv > presence_tau).astype(np.float32)
            valsF.append(float(pres.mean()))

    axF.bar(x_m, np.array(valsF, dtype=np.float32), width=0.72, linewidth=0.6, edgecolor="black", alpha=0.92)
    axF.set_xticks(x_m)
    axF.set_xticklabels(models, rotation=35, ha="right")

    # ---- Row 3: G/H/I = summaries (content originally D/E/F) ----
    # Prepare matrices (3 metrics × models)
    group_w = 0.78
    bar_w = group_w / 3.0
    offsets = np.array([-bar_w, 0.0, bar_w], dtype=np.float32)

    auc_mat = np.zeros((3, len(models)), dtype=np.float32)
    slope_mat = np.zeros((3, len(models)), dtype=np.float32)
    trig_mat = np.full((3, len(models)), np.nan, dtype=np.float32)
    trig_na = np.zeros((3, len(models)), dtype=bool)

    for mi in range(3):
        for j, m in enumerate(models):
            auc, slope, trig = stats[mi][m]
            auc_mat[mi, j] = auc
            slope_mat[mi, j] = slope
            if trig is None:
                trig_mat[mi, j] = 0.0
                trig_na[mi, j] = True
            else:
                trig_mat[mi, j] = float(trig)
                trig_na[mi, j] = False

    bar_kw = dict(width=bar_w, linewidth=0.6, edgecolor="black", alpha=0.92)

    def plot_grouped(ax, mat, title, ylabel, show_legend=False):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        polish_axis(ax, grid_axis="y")
        ax.axhline(0.0, linewidth=0.8, color="black", alpha=0.6)

        for mi in range(3):
            ax.bar(x_m + offsets[mi], mat[mi], label=bar_labels[mi], **bar_kw)

        ax.set_xticks(x_m)
        ax.set_xticklabels(models, rotation=35, ha="right")
        auto_pad_ylim(ax, mat)

        if show_legend:
            ax.legend(frameon=False, loc="upper left", ncol=1, handlelength=1.2)

    auc_ylabel = {
        "raw": "AUC (∫y dv)",
        "norm": "AUC_norm (mean over v)",
        "delta_norm": "AUC_Δ_norm (mean excess over v=0)",
    }[auc_mode]

    # Now label them as (G)(H)(I) to keep row-major sequence A..I
    plot_grouped(axG, auc_mat, "(G) AUC across models", auc_ylabel, show_legend=True)
    plot_grouped(axH, slope_mat, "(H) Slope across models", "Slope (linear fit)")

    axI.set_title(f"(I) Trigger v* across models (Δ={trigger_delta_abs})")
    axI.set_ylabel("v*")
    polish_axis(axI, grid_axis="y")
    axI.axhline(0.0, linewidth=0.8, color="black", alpha=0.6)
    for mi in range(3):
        axI.bar(x_m + offsets[mi], trig_mat[mi], label=bar_labels[mi], **bar_kw)
    axI.set_xticks(x_m)
    axI.set_xticklabels(models, rotation=35, ha="right")
    auto_pad_ylim(axI, trig_mat)

    ymaxI = float(np.nanmax(trig_mat)) if np.isfinite(trig_mat).any() else 1.0
    y_naI = 0.02 * (ymaxI + 1.0)
    for mi in range(3):
        for j in range(len(models)):
            if trig_na[mi, j]:
                axI.text(x_m[j] + offsets[mi], y_naI, "NA", ha="center", va="bottom")

    # Save
    fig.subplots_adjust(left=0.045, right=0.995, top=0.965, bottom=0.14)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)

    if save_pdf:
        out_pdf = os.path.splitext(out_png)[0] + ".pdf"
        fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)

    plt.close(fig)


# ----------------------------
# Evaluation core
# ----------------------------
def eval_one_root(root_dir: str, gt_map: Dict[int, str], v_min: int, v_max: int) -> Tuple[List[MetricsRow], int]:
    model_names = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    model_names.sort()
    if len(model_names) == 0:
        raise FileNotFoundError(f"No model subfolders found in: {root_dir}")

    model_pred_maps: Dict[str, Dict[Tuple[int, int], str]] = {}
    for m in model_names:
        pred_dir = os.path.join(root_dir, m)
        pred_map = scan_pred_dir(pred_dir)
        if len(pred_map) == 0:
            raise FileNotFoundError(f"No predictions matched 'X_v_mask.*' in: {pred_dir}")
        model_pred_maps[m] = pred_map

    rows: List[MetricsRow] = []
    missing = 0

    for m in model_names:
        pred_map = model_pred_maps[m]
        for x0, gt_path in sorted(gt_map.items(), key=lambda t: t[0]):
            gt = read_index_mask(gt_path)
            for v in range(v_min, v_max + 1):
                key = (x0, v)
                if key not in pred_map:
                    missing += 1
                    continue
                pred = read_index_mask(pred_map[key])
                ensure_same_shape(gt, pred, f"gt(x={x0})", f"pred({m}, x={x0}, v={v})")

                ac_to_lens, lens_fpr, pred_lens_area, per = compute_metrics(gt, pred)
                (d0, i0) = per[0]
                (d1, i1) = per[1]
                (d2, i2) = per[2]
                (d3, i3) = per[3]

                rows.append(MetricsRow(
                    model=m, sample_x=x0, v=v,
                    ac_to_lens=ac_to_lens, lens_fpr=lens_fpr, pred_lens_area=pred_lens_area,
                    dice_bg=d0, dice_cornea=d1, dice_ac=d2, dice_lens=d3,
                    iou_bg=i0, iou_cornea=i1, iou_ac=i2, iou_lens=i3
                ))

    if len(rows) == 0:
        raise RuntimeError(f"No rows computed for root_dir={root_dir}. Check filenames like X_v_mask.png")
    return rows, missing


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate publication-ready 3×3 master figure with swapped groups and synced labels.")
    parser.add_argument("--root_dir", type=str, default=r"",
                        help="Single: root_dir/<model>/X_v_mask.png ; Multi: root_dir/<domain>/<model>/X_v_mask.png")
    parser.add_argument("--gt_dir", type=str, default=r"",
                        help="GT mask folder; filenames start with integer X (e.g., 0.png, 1_xxx.png)")
    parser.add_argument("--out_dir", type=str, default=r"",)
    parser.add_argument("--v_min", type=int, default=0)
    parser.add_argument("--v_max", type=int, default=10)

    parser.add_argument("--domains", type=str, default="",
                        help="Comma-separated domains under root_dir. Empty -> single.")
    parser.add_argument("--show_ci", action="store_true")
    parser.add_argument("--trigger_delta_abs", type=float, default=0.05)
    parser.add_argument("--auc_mode", type=str, default="delta_norm", choices=["raw", "norm", "delta_norm"])

    parser.add_argument("--presence_tau", type=float, default=0.001,
                        help="Lens present if pred_lens_area > presence_tau. Default 0.001 (0.1%%).")
    parser.add_argument("--high_v0", type=int, default=8,
                        help="High-v region: v >= high_v0. Default 8 (equiv. v>7).")

    parser.add_argument("--save_pdf", action="store_true",
                        help="Also export a PDF version (recommended).")

    args = parser.parse_args()

    gt_map = scan_gt_dir(args.gt_dir)
    if len(gt_map) == 0:
        raise FileNotFoundError(f"No GT found in: {args.gt_dir}")

    os.makedirs(args.out_dir, exist_ok=True)

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    if not domains:
        domains = [""]

    for dom in domains:
        if dom == "":
            this_root = args.root_dir
            tag = "lens_visibility"
            this_out = args.out_dir
        else:
            this_root = os.path.join(args.root_dir, dom)
            tag = dom
            this_out = os.path.join(args.out_dir, dom)

        os.makedirs(this_out, exist_ok=True)

        rows, missing = eval_one_root(this_root, gt_map, args.v_min, args.v_max)

        out_csv = os.path.join(this_out, f"metrics_{tag}.csv")
        out_png = os.path.join(this_out, f"{tag}_panel_3x3_swapped_ABC__GHI_to_DEF.png")


        write_csv(rows, out_csv)
        plot_master_3x3_swapped_groups(
            rows,
            out_png,
            v_min=args.v_min,
            v_max=args.v_max,
            show_ci=args.show_ci,
            trigger_delta_abs=args.trigger_delta_abs,
            auc_mode=args.auc_mode,
            presence_tau=args.presence_tau,
            high_v0=args.high_v0,
            save_pdf=args.save_pdf,
        )

        print(f"[{tag}] Saved:")
        print(" -", out_csv)
        print(" -", out_png)
        if args.save_pdf:
            print(" -", os.path.splitext(out_png)[0] + ".pdf")
        if missing > 0:
            print(f"[{tag}] Warning: missing pairs skipped = {missing}")


if __name__ == "__main__":
    main()
