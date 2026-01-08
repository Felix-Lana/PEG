# -*- coding: utf-8 -*-
"""
PM Mechanistic Analysis (SPU / FPU / CPU) — Branch Share & Quantile-Threshold Sparsity
====================================================================================
This script is for the Perception Module (PM) implemented in your Perception_Module.py::SFS_Conv
within UNet.py. It hooks EXACTLY:
  wavelet_k.SPU, wavelet_k.FPU, wavelet_k.Cur  (k = 0..4)

Metrics (per image, per wavelet level, per branch):
  1) PRS  (Perception Response Share; normalized intensity / relative contribution)
       PRI_b = mean(|T_b|)
       PRS_b = PRI_b / (PRI_SPU + PRI_FPU + PRI_CPU + 1e-12)
     -> Still "intensity-based", but reveals relative branch contribution.

  2) PASq (Perception Activation Sparsity; quantile-threshold version, more robust)
       eps_{k,b} = Q_q ( |T_{k,b}| ) computed from NORMAL set only (pooled samples)
       PASq = mean( |T| < eps_{k,b} )

Statistics:
  Compare H vs normal and M vs normal using Mann–Whitney U test and Cliff's delta.
  Within each (comparison × metric) block, apply BH-FDR correction (q<0.05).

Outputs (OUT_DIR):
  - pm_branch_raw_v2.npz
  - pm_branch_stats_v2.csv
  - Fig1_PM_effectsize_heatmap_v2.png     (cells show δ (2 decimals) + * for q<0.05)
  - Fig2_PM_PASq_exceedance_curves.png    (PASq exceedance curves; pooled by branch)  [title label: (E)]

Right-click run:
  1) Edit CONFIG below (paths)
  2) Run
"""

import os, csv, inspect, importlib.util
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from PIL import Image
except Exception:
    Image = None

# =========================
# ==========================
# Publication-quality output
# ==========================
PUB_DPI = 600          # raster resolution for PNG
SAVE_PNG = True
SAVE_PDF = False       # set True if you want vector PDF
SAVE_SVG = False       # set True if you want vector SVG

def _save_pubfig(fig, out_path_png: str):
    """Save figure as high-DPI PNG, and optionally PDF/SVG."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(out_path_png) or '.', exist_ok=True)
    if SAVE_PNG:
        fig.savefig(out_path_png, dpi=PUB_DPI, bbox_inches='tight', facecolor='white')
    if SAVE_PDF:
        fig.savefig(out_path_png.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    if SAVE_SVG:
        fig.savefig(out_path_png.replace('.png', '.svg'), bbox_inches='tight', facecolor='white')

# CONFIG (EDIT THESE)
# =========================
MODEL_PY   = r""
CLASS_NAME = "UNet_3Plus_DeepSup"
CKPT_PATH  = r""

OUT_DIR = r""

NORMAL_IMG_DIR = r""
H_IMG_DIR      = r""
M_IMG_DIR      = r""

DEVICE = "cuda"          # "cuda" or "cpu"
BATCH_SIZE = 1           # keep 1 (per-image bookkeeping)
NUM_WORKERS = 0

IN_CHANNELS = 3
RESIZE_HW   = None       # e.g., (256,256) or None
NORM_TO_01  = True

# PASq settings (quantile threshold from normal)
Q_THRESH = 0.10          # e.g., 0.10 means P10
SAMPLE_PER_IMAGE = 4096  # sampled |T| values per normal image to estimate quantile robustly

# PM wavelet stages
WAVELET_LEVELS = 5       # wavelet_0 .. wavelet_4
# =========================

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def load_py_module(path: str):
    spec = importlib.util.spec_from_file_location("dyn_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import model file: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def infer_ckpt_out_channels(ckpt_path: str) -> Optional[int]:
    if not ckpt_path or not os.path.exists(ckpt_path):
        return None
    try:
        sd = torch.load(ckpt_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        for k, v in sd.items():
            if not isinstance(k, str):
                continue
            nk = k[7:] if k.startswith("module.") else k
            if nk in ("outconv1.weight", "final.weight", "outc.weight", "out_conv.weight"):
                if torch.is_tensor(v) and v.ndim >= 1:
                    return int(v.shape[0])
    except Exception:
        return None
    return None


def build_model(model_py: str, class_name: str, ckpt_path: Optional[str] = None) -> nn.Module:
    mod = load_py_module(model_py)
    if not hasattr(mod, class_name):
        raise AttributeError(f"Class '{class_name}' not found in {model_py}")
    cls = getattr(mod, class_name)

    out_ch = infer_ckpt_out_channels(ckpt_path) if ckpt_path else None
    if out_ch is not None:
        try:
            sig = inspect.signature(cls)
            kwargs = {}
            for cand in ["n_classes", "num_classes", "classes", "n_class", "out_ch", "out_channels", "nlabels"]:
                if cand in sig.parameters:
                    kwargs[cand] = out_ch
                    break
            if kwargs:
                return cls(**kwargs)
        except Exception:
            pass
    return cls()


def load_checkpoint_safe(model: nn.Module, ckpt_path: str, device: torch.device):
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    new_sd = {}
    for k, v in sd.items():
        if not isinstance(k, str):
            continue
        nk = k[7:] if k.startswith("module.") else k
        new_sd[nk] = v

    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in new_sd.items():
        if k not in model_state:
            skipped.append((k, "missing_in_model"))
            continue
        if not torch.is_tensor(v):
            skipped.append((k, "not_tensor"))
            continue
        if tuple(v.shape) != tuple(model_state[k].shape):
            skipped.append((k, f"shape_mismatch ckpt={tuple(v.shape)} model={tuple(model_state[k].shape)}"))
            continue
        filtered[k] = v

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if skipped:
        print(f"[LOAD] Skipped keys due to mismatch/missing: {len(skipped)}")
        for k, why in skipped[:10]:
            print(f"  - {k}: {why}")
        if len(skipped) > 10:
            print("  ...")
    if missing:
        print("[LOAD] Missing keys after filtering:", len(missing))
    if unexpected:
        print("[LOAD] Unexpected keys after filtering:", len(unexpected))

    model.to(device)
    model.eval()


def list_images(folder: str) -> List[str]:
    out = []
    if not folder or not os.path.isdir(folder):
        return out
    for root, _dirs, files in os.walk(folder):
        for fn in files:
            if fn.lower().endswith(IMG_EXTS):
                out.append(os.path.join(root, fn))
    out.sort()
    return out


def pil_to_tensor(path: str, in_channels: int, resize_hw, norm01=True):
    if Image is None:
        raise RuntimeError("PIL is required but not available.")
    img = Image.open(path)
    if in_channels == 1:
        img = img.convert("L")
    else:
        img = img.convert("RGB")
    if resize_hw is not None:
        img = img.resize((resize_hw[1], resize_hw[0]), resample=Image.BILINEAR)
    arr = np.array(img).astype(np.float32)
    if in_channels == 1:
        if arr.ndim == 2:
            arr = arr[None, :, :]
        else:
            arr = arr[:, :, 0][None, :, :]
    else:
        arr = np.transpose(arr, (2, 0, 1))
    if norm01:
        arr /= 255.0
    return torch.from_numpy(arr)


@dataclass
class Sample:
    img_path: str
    group: str


class ImgDataset(Dataset):
    def __init__(self, normal_img_dir, h_img_dir, m_img_dir):
        self.samples: List[Sample] = []
        for p in list_images(normal_img_dir):
            self.samples.append(Sample(p, "normal"))
        for p in list_images(h_img_dir):
            self.samples.append(Sample(p, "H"))
        for p in list_images(m_img_dir):
            self.samples.append(Sample(p, "M"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            x = pil_to_tensor(s.img_path, IN_CHANNELS, RESIZE_HW, norm01=NORM_TO_01)
        except Exception:
            return None
        return {"image": x, "group": s.group, "path": s.img_path}


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    img = torch.stack([b["image"] for b in batch], 0)
    group = [b["group"] for b in batch]
    path = [b["path"] for b in batch]
    return {"image": img, "group": group, "path": path}


# ----------------- statistics -----------------
def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.size == 0 or b.size == 0:
        return float("nan")
    gt = 0
    lt = 0
    for x in a:
        gt += int((x > b).sum())
        lt += int((x < b).sum())
    return (gt - lt) / float(a.size * b.size)


def mannwhitney_p(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from scipy.stats import mannwhitneyu
        if a.size < 2 or b.size < 2:
            return float("nan")
        res = mannwhitneyu(a, b, alternative="two-sided")
        return float(res.pvalue)
    except Exception:
        return float("nan")


def bh_fdr(pvals: List[float]) -> List[float]:
    p = np.array(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    mask = ~np.isnan(p)
    if mask.sum() == 0:
        return q.tolist()
    pv = p[mask]
    order = np.argsort(pv)
    pv_sorted = pv[order]
    ranks = np.arange(1, len(pv_sorted) + 1)
    q_sorted = pv_sorted * len(pv_sorted) / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    idx = np.where(mask)[0]
    q_idx = idx[order]
    q[q_idx] = q_sorted
    return q.tolist()


# ----------------- metrics helpers -----------------
def mean_abs(t: torch.Tensor) -> float:
    return float(t.detach().float().abs().mean().item())


def pas_quantile(t: torch.Tensor, eps: float) -> float:
    x = t.detach().float().abs()
    return float((x < eps).float().mean().item())


def sample_abs_values(t: torch.Tensor, k: int, rng: np.random.Generator) -> np.ndarray:
    """Sample up to k values from |t| (flatten)."""
    x = t.detach().float().abs().flatten()
    n = int(x.numel())
    if n == 0:
        return np.empty((0,), dtype=np.float32)
    if n <= k:
        return x.cpu().numpy().astype(np.float32)
    # random indices on CPU for speed
    idx = rng.integers(low=0, high=n, size=k, endpoint=False)
    xs = x[idx].cpu().numpy().astype(np.float32)
    return xs


# ----------------- plotting -----------------
def plot_effectsize_heatmap(mat_H, mat_M, q_H, q_M, row_labels, col_labels, out_path_png,
                            title="(D) Perception Module (PM) branch-wise shift (Cliff's δ; *: q<0.05)"):
    """
    Two-panel heatmap: H vs normal (left) and M vs normal (right).
    - Cell text shows δ (2 decimals) and '*' if BH-FDR q<0.05.
    - Row labels appear ONLY on the left panel to keep the layout compact.
    """
    import matplotlib.pyplot as plt

    row_labels = list(row_labels)
    col_labels = list(col_labels)

    # dynamic height so labels never get cut off
    fig_h = max(5.2, 0.28 * len(row_labels))
    fig = plt.figure(figsize=(12.6, fig_h))

    panels = [("H vs normal", mat_H, q_H), ("M vs normal", mat_M, q_M)]
    im = None
    for i, (ptitle, mat, qmat) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 2, i)
        im = ax.imshow(mat, aspect="auto", interpolation="nearest", vmin=-1, vmax=1)
        ax.set_title(ptitle, fontsize=12)

        ax.set_yticks(np.arange(len(row_labels)))
        if i == 1:
            ax.set_yticklabels(row_labels, fontsize=9)
        else:
            # keep row alignment but avoid duplicated labels on the right panel
            ax.set_yticklabels([])
            ax.tick_params(labelleft=False)

        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, fontsize=10)

        # Annotate cells: δ + significance
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                v = float(mat[r, c])
                qv = float(qmat[r, c]) if qmat is not None else np.nan
                star = "*" if (np.isfinite(qv) and qv < 0.05) else ""
                txt = f"{v:+.2f}{star}"
                # choose text color based on background intensity
                ax.text(c, r, txt, ha="center", va="center",
                        fontsize=9, color=("white" if abs(v) > 0.55 else "black"))

        ax.set_xlabel("")
        ax.set_ylabel("")

    # Centered title across both panels
    fig.suptitle(title, fontsize=16, x=0.5, ha="center", y=0.995)

    # Colorbar (single)
    cax = fig.add_axes([0.92, 0.12, 0.02, 0.76])  # [left, bottom, width, height]
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Cliff's δ (H/M vs normal)", fontsize=11)

    # Tight layout: reserve space for suptitle and colorbar
    fig.tight_layout(rect=[0.02, 0.02, 0.90, 0.965])

    _save_pubfig(fig, out_path_png)
    plt.close(fig)
def plot_prs_quantile_grid(prs_by_level: Dict[int, Dict[str, Dict[str, List[float]]]], out_path: str):
    """Fig.2: PRS quantile curves per wavelet level (rows) and branch (cols).

    prs_by_level[i][branch][group] -> list of PRS values (one per image).
    """
    import matplotlib.pyplot as plt

    branches = ["SPU", "FPU", "CPU"]
    groups = ["normal", "H", "M"]
    levels = sorted(prs_by_level.keys())
    qs = np.linspace(0.05, 0.95, 19)

    nrows = len(levels)
    ncols = len(branches)
    fig_w = 4.0 * ncols
    fig_h = 2.4 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=True)

    # axes can be 1D if nrows==1
    if nrows == 1:
        axes = np.expand_dims(axes, 0)

    # global y-lims per branch (so each column is comparable)
    ylims = {}
    for b in branches:
        allv = []
        for i in levels:
            for g in groups:
                allv.extend(prs_by_level[i][b][g])
        if len(allv) >= 2:
            lo = float(np.quantile(allv, 0.01))
            hi = float(np.quantile(allv, 0.99))
            pad = 0.05 * (hi - lo + 1e-12)
            ylims[b] = (max(0.0, lo - pad), min(1.0, hi + pad))
        else:
            ylims[b] = (0.0, 1.0)

    for r, i in enumerate(levels):
        for c, b in enumerate(branches):
            ax = axes[r, c]
            for g in groups:
                v = np.asarray(prs_by_level[i][b][g], dtype=np.float32)
                if v.size < 2:
                    continue
                qv = np.quantile(v, qs)
                ax.plot(qs, qv, label=f"{g} (n={v.size})")
            if r == 0:
                ax.set_title(f"{b}", fontsize=11)
            if c == 0:
                ax.set_ylabel(f"PM-{i}\nPRS", fontsize=10)
            ax.set_ylim(*ylims[b])
            ax.grid(True, alpha=0.25)
            if r == nrows - 1:
                ax.set_xlabel("quantile")
            # only show legend on the first row, last col to avoid clutter
            if (r == 0) and (c == ncols - 1):
                ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("(E) Perception Module (PM): PRS quantile curves (per level × branch)", fontsize=13, y=1.01)
    fig.tight_layout()
    _save_pubfig(fig, out_path)
    plt.close(fig)


def plot_pasq_exceedance_quantile(pas_by_level: Dict[int, Dict[str, Dict[str, List[float]]]], out_path: str):
    """Fig.3: PASq exceedance curves using *relative* (normal-quantile) thresholds.

    For each (level, branch), define thresholds tau_q = Q_q(PASq_normal).
    Then plot rho(q) = P(PASq_group > tau_q). By construction, rho_normal(q) ~= 1-q.
    """
    import matplotlib.pyplot as plt

    branches = ["SPU", "FPU", "CPU"]
    groups = ["normal", "H", "M"]
    levels = sorted(pas_by_level.keys())
    qs = np.linspace(0.01, 0.99, 99)

    nrows = len(levels)
    ncols = len(branches)
    fig_w = 4.0 * ncols
    fig_h = 2.4 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=True, sharey=True)

    if nrows == 1:
        axes = np.expand_dims(axes, 0)

    for r, i in enumerate(levels):
        for c, b in enumerate(branches):
            ax = axes[r, c]
            v_norm = np.asarray(pas_by_level[i][b]["normal"], dtype=np.float32)
            if v_norm.size < 2:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=9)
                ax.set_axis_off()
                continue

            tau = np.quantile(v_norm, qs)
            for g in groups:
                v = np.asarray(pas_by_level[i][b][g], dtype=np.float32)
                if v.size < 2:
                    continue
                # exceedance over the *normal-defined* thresholds
                rho = np.array([(v > t).mean() for t in tau], dtype=np.float32)
                ax.plot(qs, rho, label=f"{g} (n={v.size})")

            if r == 0:
                ax.set_title(f"{b}", fontsize=11)
            if c == 0:
                ax.set_ylabel(f"PM-{i}\nρ(q)", fontsize=10)
            ax.grid(True, alpha=0.25)
            if r == nrows - 1:
                ax.set_xlabel("q  (threshold = Q_q(PASq_normal))")

            if (r == 0) and (c == ncols - 1):
                ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("(E) Perception Module (PM): PASq exceedance under normal-quantile thresholds", fontsize=13, y=1.01)
    fig.tight_layout()
    _save_pubfig(fig, out_path)
    plt.close(fig)

def main():
    ensure_dir(OUT_DIR)

    print("========== CONFIG SUMMARY ==========")
    print("MODEL_PY:", MODEL_PY)
    print("CLASS_NAME:", CLASS_NAME)
    print("CKPT_PATH:", CKPT_PATH)
    print("OUT_DIR:", OUT_DIR)
    print("DEVICE:", DEVICE)
    print("IMG DIRS (normal/H/M):", NORMAL_IMG_DIR, "|", H_IMG_DIR, "|", M_IMG_DIR)
    print("IN_CHANNELS:", IN_CHANNELS, "RESIZE_HW:", RESIZE_HW)
    print("Q_THRESH:", Q_THRESH, "SAMPLE_PER_IMAGE:", SAMPLE_PER_IMAGE)
    print("WAVELET_LEVELS:", WAVELET_LEVELS)
    print("====================================")

    device = torch.device(DEVICE if (DEVICE == "cpu" or torch.cuda.is_available()) else "cpu")

    model = build_model(MODEL_PY, CLASS_NAME, ckpt_path=CKPT_PATH)
    model.to(device)
    model.eval()

    # dummy forward to init lazy params
    with torch.no_grad():
        try:
            dummy = torch.zeros((1, IN_CHANNELS, 256, 256), dtype=torch.float32, device=device)
            _ = model(dummy)
            print("[INFO] Dummy forward done (model initialized).")
        except Exception as e:
            print("[WARN] Dummy forward failed (continuing):", repr(e))

    load_checkpoint_safe(model, CKPT_PATH, device=device)

    ds = ImgDataset(NORMAL_IMG_DIR, H_IMG_DIR, M_IMG_DIR)
    print(f"[DATA] total images: {len(ds)}")
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_skip_none)

    # resolve exact module names
    mod = dict(model.named_modules())

    branches = ["SPU", "FPU", "CPU"]
    by_level = {i: {"SPU": None, "FPU": None, "CPU": None} for i in range(WAVELET_LEVELS)}
    for i in range(WAVELET_LEVELS):
        root = f"wavelet_{i}"
        # exact names from your SFS_Conv: SPU, FPU, Cur
        for sub, b in [("SPU","SPU"), ("FPU","FPU"), ("Cur","CPU")]:
            name = f"{root}.{sub}"
            if name in mod:
                by_level[i][b] = name

    print("[HOOK TARGETS]")
    for i in range(WAVELET_LEVELS):
        print(f"  wavelet_{i}: SPU={by_level[i]['SPU']} | FPU={by_level[i]['FPU']} | CPU={by_level[i]['CPU']}")

    # hook capture dict
    captures: Dict[str, Optional[torch.Tensor]] = {}

    def make_hook(key: str):
        def _h(_m, _inp, out):
            t = out if torch.is_tensor(out) else None
            if t is None and isinstance(out, (tuple, list)):
                for o in out:
                    if torch.is_tensor(o):
                        t = o
                        break
            captures[key] = t.detach() if t is not None else None
        return _h

    # register hooks
    handles = []
    for i in range(WAVELET_LEVELS):
        for b in branches:
            name = by_level[i][b]
            if name and name in mod:
                key = f"w{i}_{b}"
                captures[key] = None
                handles.append(mod[name].register_forward_hook(make_hook(key)))

    rng = np.random.default_rng(12345)

    # First pass: estimate eps_{i,b} from NORMAL only (pooled sampled |T|)
    eps_map: Dict[Tuple[int,str], float] = {}
    samples_abs: Dict[Tuple[int,str], List[np.ndarray]] = {(i,b): [] for i in range(WAVELET_LEVELS) for b in branches}

    print("[PASS1] Estimating quantile thresholds eps_{i,b} from NORMAL ...")
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            g = batch["group"][0]
            if g != "normal":
                continue
            x = batch["image"].to(device)

            for k in list(captures.keys()):
                captures[k] = None
            try:
                _ = model(x)
            except Exception:
                continue

            for i in range(WAVELET_LEVELS):
                for b in branches:
                    t = captures.get(f"w{i}_{b}", None)
                    if t is None or (not torch.is_tensor(t)) or t.numel() == 0:
                        continue
                    s = sample_abs_values(t, SAMPLE_PER_IMAGE, rng)
                    if s.size:
                        samples_abs[(i,b)].append(s)

    for i in range(WAVELET_LEVELS):
        for b in branches:
            if len(samples_abs[(i,b)]) == 0:
                eps_map[(i,b)] = float("nan")
                continue
            cat = np.concatenate(samples_abs[(i,b)], axis=0)
            eps_map[(i,b)] = float(np.quantile(cat, Q_THRESH))

    print("[PASS1] eps summary (show first 10):")
    cnt = 0
    for i in range(WAVELET_LEVELS):
        for b in branches:
            print(f"  eps(w{i},{b})= {eps_map[(i,b)]:.6g}")
            cnt += 1
            if cnt >= 10:
                break
        if cnt >= 10:
            break

    # Second pass: compute PRS and PASq per image
    groups = ["normal","H","M"]
    metrics = ["PRS", "PASq"]

    store: Dict[Tuple[int,str,str,str], List[float]] = {}
    for i in range(WAVELET_LEVELS):
        for b in branches:
            for m in metrics:
                for g in groups:
                    store[(i,b,m,g)] = []

    prs_pool_by_level = {i: {b: {g: [] for g in groups} for b in branches} for i in range(WAVELET_LEVELS)}
    pas_pool_by_level = {i: {b: {g: [] for g in groups} for b in branches} for i in range(WAVELET_LEVELS)}

    print("[PASS2] Computing PRS and PASq on all groups ...")
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            x = batch["image"].to(device)
            g = batch["group"][0]

            for k in list(captures.keys()):
                captures[k] = None
            try:
                _ = model(x)
            except Exception:
                continue

            for i in range(WAVELET_LEVELS):
                pri_vals = {}
                for b in branches:
                    t = captures.get(f"w{i}_{b}", None)
                    if t is None or (not torch.is_tensor(t)) or t.numel() == 0:
                        pri_vals[b] = None
                    else:
                        pri_vals[b] = mean_abs(t)

                denom = sum(v for v in pri_vals.values() if isinstance(v, float)) + 1e-12

                for b in branches:
                    t = captures.get(f"w{i}_{b}", None)
                    if pri_vals[b] is None or denom <= 0:
                        continue

                    prs = float(pri_vals[b] / denom)
                    eps = eps_map.get((i,b), float("nan"))
                    if t is None or (not torch.is_tensor(t)) or t.numel() == 0 or np.isnan(eps):
                        continue
                    pasq = pas_quantile(t, eps)

                    store[(i,b,"PRS",g)].append(prs)
                    store[(i,b,"PASq",g)].append(pasq)

                    prs_pool_by_level[i][b][g].append(prs)
                    pas_pool_by_level[i][b][g].append(pasq)

    for h in handles:
        try:
            h.remove()
        except Exception:
            pass

    # save NPZ
    npz = {}
    for (i,b,m,g), vals in store.items():
        npz[f"wavelet_{i}__{b}__{m}__{g}"] = np.array(vals, dtype=np.float32)
    npz["meta__Q_THRESH"] = np.array([Q_THRESH], dtype=np.float32)
    npz["meta__SAMPLE_PER_IMAGE"] = np.array([SAMPLE_PER_IMAGE], dtype=np.int32)
    for i in range(WAVELET_LEVELS):
        for b in branches:
            npz[f"meta__eps__w{i}__{b}"] = np.array([eps_map[(i,b)]], dtype=np.float32)

    raw_path = os.path.join(OUT_DIR, "pm_branch_raw_v2.npz")
    np.savez_compressed(raw_path, **npz)

    # stats CSV: for each metric separately, BH-FDR over all wavelet×branch cells within comparison
    rows = []
    comparisons = [("H","normal","H vs normal"), ("M","normal","M vs normal")]
    for B, A, comp_name in comparisons:
        for met in ["PRS","PASq"]:
            pvals = []
            tmp = []
            for i in range(WAVELET_LEVELS):
                for b in branches:
                    a = np.array(store[(i,b,met,A)], dtype=np.float32)
                    bb = np.array(store[(i,b,met,B)], dtype=np.float32)
                    p = mannwhitney_p(bb, a)
                    d = cliffs_delta(bb, a)
                    tmp.append((i, b, met, A, B, a.size, bb.size,
                                float(np.median(a)) if a.size else float("nan"),
                                float(np.median(bb)) if bb.size else float("nan"),
                                p, d))
                    pvals.append(p)
            qvals = bh_fdr(pvals)
            idx = 0
            for (i,b,met,A,B,na,nb,medA,medB,p,d) in tmp:
                rows.append({
                    "comparison": comp_name,
                    "metric": met,
                    "wavelet_level": i,
                    "branch": b,
                    "n_a(normal)": na,
                    "n_b(H/M)": nb,
                    "median_a(normal)": medA,
                    "median_b(H/M)": medB,
                    "p": p,
                    "q": qvals[idx],
                    "cliffs_delta": d,
                })
                idx += 1

    csv_path = os.path.join(OUT_DIR, "pm_branch_stats_v2.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # build heatmap matrices (rows: wavelet0..4 × branch; cols: PRS & PASq)
    row_labels = []
    row_keys = []
    for i in range(WAVELET_LEVELS):
        for b in branches:
            row_labels.append(f"PM-{i}/{b}")
            row_keys.append((i,b))

    col_labels = ["PRS", "PASq"]

    def build_mat(comp: str) -> Tuple[np.ndarray, np.ndarray]:
        delta = np.full((len(row_keys), len(col_labels)), np.nan, dtype=np.float32)
        qmat  = np.full((len(row_keys), len(col_labels)), np.nan, dtype=np.float32)
        rmap = {rk: ri for ri, rk in enumerate(row_keys)}
        cmap = {m: ci for ci, m in enumerate(col_labels)}
        for r in rows:
            if r["comparison"] != comp:
                continue
            rk = (int(r["wavelet_level"]), str(r["branch"]))
            if rk not in rmap:
                continue
            ri = rmap[rk]
            met = str(r["metric"])
            if met not in cmap:
                continue
            ci = cmap[met]
            delta[ri, ci] = float(r["cliffs_delta"])
            qmat[ri, ci]  = float(r["q"]) if r["q"] is not None else np.nan
        return delta, qmat

    delta_H, q_H = build_mat("H vs normal")
    delta_M, q_M = build_mat("M vs normal")

    fig1_path = os.path.join(OUT_DIR, "Fig1_PM_effectsize_heatmap_v2.png")
    plot_effectsize_heatmap(delta_H, delta_M, q_H, q_M, row_labels, col_labels, fig1_path,
                            title="(D) Perception Module (PM) branch-wise shift (Cliff's δ, value shown; *: q<0.05)")
    fig3_path = os.path.join(OUT_DIR, "Fig2_PM_PASq_exceedance_curves.png")
    plot_pasq_exceedance_quantile(pas_pool_by_level, fig3_path)

    print("\n[SANITY] counts per level×branch:")
    for i in range(WAVELET_LEVELS):
        for b in branches:
            for g in groups:
                n_prs = len(prs_pool_by_level[i][b][g])
                n_pas = len(pas_pool_by_level[i][b][g])
                if n_prs or n_pas:
                    print(f"  PM-{i}/{b}/{g}: PRS n={n_prs} | PASq n={n_pas}")

    print("\n[DONE]")
    print("  Raw NPZ :", raw_path)
    print("  Stats CSV:", csv_path)
    print("  Fig1:", fig1_path)
    print("  Fig3:", fig3_path)
    print("\n[NOTE]")
    print("  - PASq uses eps_{w,b} = Q{:.0f}(|T|) computed from NORMAL only.".format(Q_THRESH*100))
    print("  - PRS is normalized intensity share, highlighting relative branch contribution.")
    print("  - If any pooled n=0, your corresponding hook target is missing or not executed.")


if __name__ == "__main__":
    main()