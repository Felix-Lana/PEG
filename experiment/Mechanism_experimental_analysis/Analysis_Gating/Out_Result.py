# -*- coding: utf-8 -*-
"""
Paper-ready 3-figure script (pub-quality, consistent colormap + annotated quantiles)
===================================================================================

Inputs (same folder):
  - evidence_chain_stats.csv
  - evidence_chain_raw.npz

Outputs:
  - Fig1_effectsize_heatmap_focus_compact.png (+ optional PDF)
  - Fig2_GRI_quantilebands_direction_focus.png (+ optional PDF)
  - Fig3_GRI_exceedance_direction_focus.png (+ optional PDF)

Key improvements vs your current version:
1) Fig1 heatmap uses a FIXED diverging colormap + shared symmetric scale across panels
   -> ensures "same value == same color" (no auto-rescale differences).
2) Fig2 uses quantile-bands (not boxplot) and ANNOTATES medians (numeric values) for each group,
   plus shows per-group sample sizes in the title line (more publishable / interpretable).
3) High-resolution export (PNG dpi=600 by default) + optional PDF vector export.

Note:
- This script is for the *gate proxy* NPZ/CSV format:
    key in NPZ: "{module}__{group}__{metric}"
    CSV columns: module / comparison / metric / delta / p  (or variants auto-detected)
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG (edit here)
# =========================
# Where the CSV/NPZ are located and where figures will be saved

OUT_DIR = r""
IN_DIR  = OUT_DIR

STATS_CSV = os.path.join(IN_DIR, r"\evidence_chain_stats.csv")
RAW_NPZ   = os.path.join(IN_DIR, r"\evidence_chain_raw.npz")

FDR_Q = 0.05

# Optional: add extra "others" modules by effect strength to provide context
ADD_TOPK_OTHERS = 0     # set 0 to disable

# Focus module families
WAVELET_PATTERN = r"(?:^|[._])wavelet[_\.]([0-4])(?:[._]|$)"       # wavelet_0..4
GUIDED_PATTERN  = r"(?:^|[._])guided_attention_hd([1-4])(?:[._]|$)"# guided_attention_hd1..4

# Metrics/Comparisons
METRICS_FIG1 = ["gp_intensity", "out_sparsity"]
METRIC_MAIN  = "gp_intensity"

# Display names for paper figures (do NOT change raw metric keys used in CSV/NPZ)
METRIC_DISPLAY = {
    "gp_intensity": "GRI",   # Gate Response Intensity
    "out_sparsity": "GAS",   # Gate Activation Sparsity
}
COMPS = ["H vs normal", "M vs normal"]
GROUPS = ["normal", "H", "M"]
DIRECTION_COMP = "H vs normal"   # direction split based on H vs normal

# Fig1 readability
MAX_LABEL_LEN = 34
ANNOTATE_IF_ROWS_LEQ = 28  # annotate δ (+ star) when rows not too many

# Fig2 quantiles
Q_LOW_WIDE  = 0.10
Q_HIGH_WIDE = 0.90
Q_LOW_IQR   = 0.25
Q_HIGH_IQR  = 0.75
Q_MED       = 0.50
Q_TAIL      = 0.95

# Fig3 exceedance curve
TAU_POINTS = 90
TAU_LO_Q = 0.01
TAU_HI_Q = 0.99

# Publication export
SAVE_DPI = 600      # PNG export DPI (600 recommended for montage + submission)
SAVE_PDF = True     # also export PDF (vector; best for later layout)
SAVE_SVG = False    # optional
PAD_INCHES = 0.02

# Heatmap style (consistent across panels)
# Match the PM figures' look (purple→green→yellow). With vmin/vmax fixed to [-1,1],
# Cliff's δ is visually comparable across experiments.
HEATMAP_CMAP = "viridis"
# =========================


# ---------- helpers ----------
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR. NaN stays NaN."""
    p = pvals.copy().astype(float)
    q = np.full_like(p, np.nan, dtype=float)
    mask = ~np.isnan(p)
    if mask.sum() == 0:
        return q
    pv = p[mask]
    order = np.argsort(pv)
    pv_sorted = pv[order]
    ranks = np.arange(1, len(pv_sorted) + 1)
    q_sorted = pv_sorted * len(pv_sorted) / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    idx = np.where(mask)[0]
    q_idx = idx[order]
    q[q_idx] = q_sorted
    return q

def load_npz_array(npz, key):
    if key in npz:
        return np.array(npz[key]).astype(np.float32).ravel()
    return np.array([], dtype=np.float32)

def pooled_concat(npz, modules, metric, group):
    arrs = []
    for m in modules:
        k = f"{m}__{group}__{metric}"
        a = load_npz_array(npz, k)
        if a.size > 0:
            arrs.append(a)
    if not arrs:
        return np.array([], dtype=np.float32)
    return np.concatenate(arrs, axis=0)

def smart_shorten(name: str, maxlen: int = 34) -> str:
    """Compact module labels (drop 'sigmoid' to reduce clutter)."""
    s = re.sub(r"_+", "_", str(name))
    toks = re.split(r"[._]", s)
    toks = [t for t in toks if t and t.lower() != "sigmoid"]
    s2 = "_".join(toks)
    if len(s2) <= maxlen:
        return s2
    toks2 = [t for t in re.split(r"[._]", s2) if t]
    for k in [6, 5, 4, 3]:
        cand = "_".join(toks2[-k:])
        if len(cand) <= maxlen:
            return cand
    return "…" + s2[-(maxlen - 1):]


def _auto_text_color(val: float, vmin: float, vmax: float, cmap_name: str) -> str:
    """Pick white/black text color based on the heatmap cell background."""
    try:
        cmap = plt.get_cmap(cmap_name)
        t = 0.5 if vmax == vmin else (val - vmin) / (vmax - vmin)
        t = float(np.clip(t, 0.0, 1.0))
        r, g, b, _ = cmap(t)
        # relative luminance (sRGB)
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return "white" if lum < 0.5 else "black"
    except Exception:
        return "black"

def detect_columns(df: pd.DataFrame):
    """Detect delta and p columns from CSV variants."""
    p_candidates = ["p", "p_mannwhitney", "p_value", "pval", "p_mw", "mw_p", "p_u"]
    d_candidates = ["delta", "cliffs_delta", "cliffs_delta(b_vs_a)", "effect", "effect_size", "cliff_delta"]
    p_col = next((c for c in p_candidates if c in df.columns), None)
    d_col = next((c for c in d_candidates if c in df.columns), None)
    if p_col is None:
        raise ValueError(f"Cannot find p-value column. Columns: {list(df.columns)}")
    if d_col is None:
        raise ValueError(f"Cannot find delta/effect column. Columns: {list(df.columns)}")
    return p_col, d_col

def build_tables(df: pd.DataFrame, modules: list, metrics: list):
    """Build delta/p/q tables indexed by comp, metric, module index."""
    delta = {c: {m: np.full(len(modules), np.nan) for m in metrics} for c in COMPS}
    pval  = {c: {m: np.full(len(modules), np.nan) for m in metrics} for c in COMPS}
    qval  = {c: {m: np.full(len(modules), np.nan) for m in metrics} for c in COMPS}
    for comp in COMPS:
        for metric in metrics:
            for i, mod in enumerate(modules):
                sub = df[(df["comparison"] == comp) & (df["module"] == mod) & (df["metric"] == metric)]
                if len(sub) == 0:
                    continue
                delta[comp][metric][i] = float(sub["delta"].iloc[0])
                pval[comp][metric][i]  = float(sub["p"].iloc[0])
            qval[comp][metric] = bh_fdr(pval[comp][metric])
    return delta, pval, qval

def module_group(modname: str) -> str:
    if re.search(WAVELET_PATTERN, modname):
        return "wavelet"
    if re.search(GUIDED_PATTERN, modname):
        return "guided"
    return "other"

def score_module(i, delta, qval):
    """Rank modules by mechanism relevance (H weighted > M)."""
    sc = 0.0
    for metric in METRICS_FIG1:
        d = delta["H vs normal"][metric][i]
        q = qval["H vs normal"][metric][i]
        if not np.isnan(d):
            sc += abs(d) * (2.0 if (not np.isnan(q) and q < FDR_Q) else 1.0)
    for metric in METRICS_FIG1:
        d = delta["M vs normal"][metric][i]
        q = qval["M vs normal"][metric][i]
        if not np.isnan(d):
            sc += 0.6 * abs(d) * (1.5 if (not np.isnan(q) and q < FDR_Q) else 1.0)
    return sc

def adaptive_left_margin(labels, fontsize):
    max_len = max(len(s) for s in labels) if labels else 10
    left = 0.18 + 0.0045 * max_len * (fontsize / 9.0)
    return float(min(0.44, max(0.18, left)))

def choose_ytick_fontsize(n_rows):
    if n_rows <= 14: return 10
    if n_rows <= 22: return 9
    if n_rows <= 32: return 8
    if n_rows <= 44: return 7
    return 6

def split_by_direction(modules, delta, qval, metric=METRIC_MAIN):
    d = delta[DIRECTION_COMP][metric]
    q = qval[DIRECTION_COMP][metric]
    neg, pos = [], []
    for i, m in enumerate(modules):
        if np.isnan(d[i]) or np.isnan(q[i]) or q[i] >= FDR_Q:
            continue
        if d[i] < 0:
            neg.append(m)
        elif d[i] > 0:
            pos.append(m)
    if len(neg) == 0: neg = modules[:]
    if len(pos) == 0: pos = modules[:]
    return neg, pos

def quantiles(x: np.ndarray, qs: list):
    if x.size == 0:
        return [np.nan] * len(qs)
    return [float(np.quantile(x, q)) for q in qs]

def draw_quantile_sticks(ax, x_positions, q10, q25, q50, q75, q90, q95=None):
    # Use default color cycle; don't hardcode colors
    for i, x in enumerate(x_positions):
        if any(np.isnan(v) for v in [q10[i], q25[i], q50[i], q75[i], q90[i]]):
            continue
        ax.plot([x, x], [q10[i], q90[i]], linewidth=2.2, alpha=0.9)  # P10-P90
        ax.plot([x, x], [q25[i], q75[i]], linewidth=6.4, alpha=0.9)  # IQR
        ax.plot([x], [q50[i]], marker="o", markersize=7)             # median
        if q95 is not None and (not np.isnan(q95[i])):
            ax.plot([x], [q95[i]], marker="x", markersize=7)

def exceedance_curve(x: np.ndarray, tau: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return np.full_like(tau, np.nan, dtype=np.float32)
    return np.array([(x > t).mean() for t in tau], dtype=np.float32)

def _save_pubfig(fig, out_png: str):
    fig.savefig(out_png, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=PAD_INCHES)
    if SAVE_PDF:
        fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=PAD_INCHES)
    if SAVE_SVG:
        fig.savefig(out_png.replace(".png", ".svg"), bbox_inches="tight", pad_inches=PAD_INCHES)


# ---------- Figure 1 ----------
def fig1_effectsize_heatmap(modules, delta, qval):
    """
    Fig1: Effect size heatmap, two panels (H vs normal, M vs normal), columns=GRI/GAS.
    - Shared colormap + shared symmetric scale -> consistent colors across panels.
    - Optional δ numeric annotations + * for FDR significance.
    """
    n = len(modules)
    fig_h = max(4.6, 0.34 * n + 1.8)
    fig_w = 11.2

    yfs = choose_ytick_fontsize(n)
    ylabels = [smart_shorten(m, MAX_LABEL_LEN) for m in modules]
    left_margin = adaptive_left_margin(ylabels, yfs)

    # Build matrices first to compute a shared scale
    mats = {}
    sigs = {}
    vmax_abs = 0.0
    for comp in COMPS:
        mat = np.full((n, len(METRICS_FIG1)), np.nan, dtype=float)
        sig = np.zeros((n, len(METRICS_FIG1)), dtype=bool)
        for i in range(n):
            for j, metric in enumerate(METRICS_FIG1):
                mat[i, j] = delta[comp][metric][i]
                q = qval[comp][metric][i]
                sig[i, j] = (not np.isnan(q)) and (q < FDR_Q)
        mats[comp] = mat
        sigs[comp] = sig
        if np.any(~np.isnan(mat)):
            vmax_abs = max(vmax_abs, float(np.nanmax(np.abs(mat))))
    # Force at least 1.0 for δ range (keeps interpretable diverging scale)
    vmax_abs = max(1.0, vmax_abs)
    vmin, vmax = -vmax_abs, vmax_abs

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(1, 2, wspace=0.10)
    axes = []
    im = None

    for k, comp in enumerate(COMPS):
        ax = fig.add_subplot(gs[0, k])
        axes.append(ax)

        mat = mats[comp]
        sig = sigs[comp]

        im = ax.imshow(mat, aspect="auto", interpolation="nearest",
                       vmin=vmin, vmax=vmax, cmap=HEATMAP_CMAP)

        ax.set_title(f"{comp}", fontsize=12)

        ax.set_xticks(np.arange(len(METRICS_FIG1)))
        ax.set_xticklabels([METRIC_DISPLAY.get(m, m) for m in METRICS_FIG1],
                           rotation=25, ha="right", fontsize=10)

        ax.set_yticks(np.arange(n))
        if k == 0:
            ax.set_yticklabels(ylabels, fontsize=yfs)
        else:
            ax.set_yticklabels([""] * n)
            ax.tick_params(labelleft=False)

        # annotate δ + star when manageable
        if n <= ANNOTATE_IF_ROWS_LEQ:
            for i in range(n):
                for j in range(len(METRICS_FIG1)):
                    v = mat[i, j]
                    if np.isnan(v):
                        continue
                    star = "*" if sig[i, j] else ""
                    tcol = _auto_text_color(float(v), vmin, vmax, HEATMAP_CMAP)
                    ax.text(j, i, f"{v:+.2f}{star}", ha="center", va="center",
                            fontsize=9, fontweight="bold", color=tcol)

    # One shared colorbar for both panels (more consistent)
    cbar = fig.colorbar(im, ax=axes, fraction=0.030, pad=0.02)
    cbar.set_label("Cliff's δ (H/M vs normal)", fontsize=10)

    fig.suptitle("(A) Effect size heatmap (Cliff's δ; * FDR q<0.05)", y=0.995, x=0.5, ha="center", fontsize=14)
    fig.subplots_adjust(left=left_margin)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = os.path.join(OUT_DIR, "Fig1_effectsize_heatmap_focus_compact.png")
    _save_pubfig(fig, out_path)
    plt.close(fig)
    return out_path


# ---------- Figure 2 ----------
def fig2_gri_quantilebands_with_numbers(modules, delta, qval):
    """
    Fig2: GRI quantile bands for direction-pooled groups (suppression/enhancement).
    - Quantile sticks + numeric median annotations for each group (normal/H/M).
    - Shows per-group sample sizes (n) in each subplot title.
    """
    with np.load(RAW_NPZ, allow_pickle=True) as z:
        neg_mods, pos_mods = split_by_direction(modules, delta, qval, metric=METRIC_MAIN)

        fig = plt.figure(figsize=(10.8, 5.0))
        gs = fig.add_gridspec(1, 2, wspace=0.28)

        panels = [
            (neg_mods, "Suppression group (δ<0, q<0.05)"),
            (pos_mods, "Enhancement group (δ>0, q<0.05)"),
        ]

        for c, (mods_group, title) in enumerate(panels):
            ax = fig.add_subplot(gs[0, c])

            pooled = {g: pooled_concat(z, mods_group, metric=METRIC_MAIN, group=g) for g in GROUPS}
            ns = {g: int(pooled[g].size) for g in GROUPS}

            qs = [Q_LOW_WIDE, Q_LOW_IQR, Q_MED, Q_HIGH_IQR, Q_HIGH_WIDE, Q_TAIL]
            qvals_ = {g: quantiles(pooled[g], qs) for g in GROUPS}

            q10 = [qvals_[g][0] for g in GROUPS]
            q25 = [qvals_[g][1] for g in GROUPS]
            q50 = [qvals_[g][2] for g in GROUPS]
            q75 = [qvals_[g][3] for g in GROUPS]
            q90 = [qvals_[g][4] for g in GROUPS]
            q95 = [qvals_[g][5] for g in GROUPS]

            x = np.arange(len(GROUPS))
            draw_quantile_sticks(ax, x, q10, q25, q50, q75, q90, q95=q95)

            # annotate median numeric values (clean; 3 groups only)
            for i, g in enumerate(GROUPS):
                if not np.isnan(q50[i]):
                    ax.text(x[i], q50[i], f"{q50[i]:.3f}", ha="left", va="bottom", fontsize=10)

            ax.set_title(f"{title}\n(n_modules={len(mods_group)}; n={ns['normal']}/{ns['H']}/{ns['M']})", fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(GROUPS, fontsize=11)
            ax.set_ylabel(METRIC_DISPLAY.get(METRIC_MAIN, METRIC_MAIN), fontsize=11)
            ax.grid(True, axis="y", alpha=0.25)

        fig.suptitle("(B) GRI quantile bands (direction-pooled)", y=0.995, x=0.5, ha="center", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        out_path = os.path.join(OUT_DIR, "Fig2_GRI_quantilebands_direction_focus.png")
        _save_pubfig(fig, out_path)
        plt.close(fig)
        return out_path


# ---------- Figure 3 ----------
def fig3_gri_exceedance(modules, delta, qval):
    """
    Fig3: GRI exceedance curves, direction-pooled, 2 subplots.
    Uses normal P1~P99 to set tau range per panel (auto-zoom).
    """
    with np.load(RAW_NPZ, allow_pickle=True) as z:
        neg_mods, pos_mods = split_by_direction(modules, delta, qval, metric=METRIC_MAIN)

        fig = plt.figure(figsize=(11.0, 5.1))
        gs = fig.add_gridspec(1, 2, wspace=0.28)

        panels = [
            (neg_mods, "Suppression group (δ<0, q<0.05)"),
            (pos_mods, "Enhancement group (δ>0, q<0.05)"),
        ]

        for c, (mods_group, title) in enumerate(panels):
            ax = fig.add_subplot(gs[0, c])

            pooled = {g: pooled_concat(z, mods_group, metric=METRIC_MAIN, group=g) for g in GROUPS}
            xN = pooled["normal"]
            if xN.size == 0:
                x_all = np.concatenate([pooled[g] for g in GROUPS if pooled[g].size > 0], axis=0)
                if x_all.size == 0:
                    continue
                xN = x_all

            lo = float(np.quantile(xN, TAU_LO_Q))
            hi = float(np.quantile(xN, TAU_HI_Q))
            if hi <= lo:
                hi = lo + 1e-6
            tau = np.linspace(lo, hi, TAU_POINTS)

            for g in GROUPS:
                y = exceedance_curve(pooled[g], tau)
                ax.plot(tau, y, marker="o", markersize=3, label=g)

            ax.set_title(f"{title}\n(n_modules={len(mods_group)})", fontsize=11)
            ax.set_xlabel("threshold τ (normal P1~P99)", fontsize=11)
            ax.set_ylabel(f"ρ(τ)=P({METRIC_DISPLAY.get(METRIC_MAIN, METRIC_MAIN)} > τ)", fontsize=11)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=10)

        fig.suptitle("(C) GRI exceedance curves (direction-pooled)", y=0.995, x=0.5, ha="center", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        out_path = os.path.join(OUT_DIR, "Fig3_GRI_exceedance_direction_focus.png")
        _save_pubfig(fig, out_path)
        plt.close(fig)
        return out_path


# ---------- main ----------
def main():
    ensure_dir(OUT_DIR)
    if not os.path.exists(STATS_CSV):
        raise FileNotFoundError(f"Missing: {STATS_CSV}")
    if not os.path.exists(RAW_NPZ):
        raise FileNotFoundError(f"Missing: {RAW_NPZ}")

    df0 = pd.read_csv(STATS_CSV)

    # Keep only "gate_proxy" rows if present
    if "type" in df0.columns:
        df0 = df0[df0["type"] == "gate_proxy"].copy()

    p_col, d_col = detect_columns(df0)
    df = df0.copy()
    df["p"] = df[p_col].apply(to_float)
    df["delta"] = df[d_col].apply(to_float)

    for c in ["module", "comparison", "metric", "p", "delta"]:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in CSV. Available: {list(df.columns)}")

    all_modules = sorted(df["module"].dropna().unique().tolist())
    if len(all_modules) == 0:
        raise RuntimeError("No modules found in stats CSV (empty?).")

    # Build for all modules to support top-K others
    delta_all, p_all, q_all = build_tables(df, all_modules, METRICS_FIG1)

    # Focus selection: wavelet + guided
    focus = []
    for m in all_modules:
        if module_group(m) in ("wavelet", "guided"):
            focus.append(m)
    focus = sorted(set(focus), key=lambda x: (0 if module_group(x) == "wavelet" else 1, x))
    if len(focus) == 0:
        raise RuntimeError("No modules matched wavelet_0..4 or guided_attention_hd1..4. Check patterns.")

    others = [m for m in all_modules if m not in focus]
    topk = []
    if ADD_TOPK_OTHERS > 0 and len(others) > 0:
        idx_map = {m: i for i, m in enumerate(all_modules)}
        scored = [(score_module(idx_map[m], delta_all, q_all), m) for m in others]
        scored.sort(key=lambda x: x[0], reverse=True)
        topk = [m for _, m in scored[:ADD_TOPK_OTHERS]]

    selected = focus + topk

    # Subset and rebuild tables for selected
    df_sel = df[df["module"].isin(selected)].copy()
    delta, pval, qval = build_tables(df_sel, selected, METRICS_FIG1)

    # Grouped sorting: wavelet -> guided -> others; within group sort by score desc
    idx_map_sel = {m: i for i, m in enumerate(selected)}

    def sort_within_group(mods):
        scored = [(score_module(idx_map_sel[m], delta, qval), m) for m in mods]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    wavelet_mods = [m for m in selected if module_group(m) == "wavelet"]
    guided_mods  = [m for m in selected if module_group(m) == "guided"]
    other_mods   = [m for m in selected if module_group(m) == "other"]
    final_modules = sort_within_group(wavelet_mods) + sort_within_group(guided_mods) + sort_within_group(other_mods)

    # Remap tables to final order
    old_idx = {m: i for i, m in enumerate(selected)}
    def remap(arr):
        out = np.full(len(final_modules), np.nan, dtype=float)
        for i, m in enumerate(final_modules):
            out[i] = arr[old_idx[m]]
        return out

    delta2 = {c: {m: remap(delta[c][m]) for m in METRICS_FIG1} for c in COMPS}
    qval2  = {c: {m: remap(qval[c][m])  for m in METRICS_FIG1} for c in COMPS}

    f1 = fig1_effectsize_heatmap(final_modules, delta2, qval2)
    f2 = fig2_gri_quantilebands_with_numbers(final_modules, delta2, qval2)
    f3 = fig3_gri_exceedance(final_modules, delta2, qval2)

    print("\n[DONE] Figures saved:")
    print(" ", f1)
    print(" ", f2)
    print(" ", f3)
    if SAVE_PDF:
        print("  (PDF versions also saved alongside PNG.)")


if __name__ == "__main__":
    main()
