# -*- coding: utf-8 -*-
"""
Target-4 Modules Evidence Statistics (NPZ + CSV + Fig1 heatmap)
==============================================================

This script ONLY computes statistics (Cliff's delta + MWU p-value) for:
  - guided_attention_hd1 ... guided_attention_hd4

No other modules will be written into NPZ/CSV/Fig1.

What we measure (per-image scalars):
  - gp_intensity  (display name in paper: GRI)  : mean(|G|) for gate maps
  - out_sparsity  (display name in paper: GAS)  : fraction of "small" activations:
        * guided_attention gates: mean(G < GATE_TAU)  (G is a sigmoid/gate map)

Outputs:
  - evidence_chain_raw.npz   (keys: "{tag}__{group}__{metric}")
  - evidence_chain_stats.csv
  - Fig1_effectsize_heatmap_9modules.png

You can right-click run: edit only the CONFIG section.
"""

import os
import re
import csv
import math
import importlib.util
import inspect
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from PIL import Image
except Exception:
    Image = None

# =========================
# CONFIG (EDIT THESE)
# =========================
MODEL_PY   = r"F:\Second_Paper_Code_final_version-comparison\a_working\models\UNet.py"
CLASS_NAME = "UNet_3Plus_DeepSup"
CKPT_PATH  = r"/checkpoints/UNet.pt"

OUT_DIR = r"out_mask"

NORMAL_IMG_DIR = r"F:\Second_Paper_Code_final_version-comparison\_MASK_OrdinaryData\_data\images"
H_IMG_DIR      = r"F:\Second_Paper_Code_final_version-comparison\_MASK_HardData1\_data\images"
M_IMG_DIR      = r"F:\Second_Paper_Code_final_version-comparison\_MASK_HardData2\_data\images"

DEVICE = "cuda"  # "cuda" or "cpu"
BATCH_SIZE = 1   # keep 1 to make hooks simpler & deterministic
NUM_WORKERS = 0

# Image preprocessing
IN_CHANNELS = 3          # 1 or 3; if your model expects 1, set 1
RESIZE_HW   = None       # e.g. (256,256) or None to keep original
NORM_TO_01  = True

# Proxy thresholds
EPS = 1e-3          # wavelet small-activation threshold for GAS
GATE_TAU = 0.5      # gate suppression threshold for GAS on gate maps

# Which internal leaves of guided_attention to hook (you can expand if needed)
GUIDED_LEAF_SUFFIXES = [
    "sgate.sigmoid",
    "cgate.sigmoid",
    "cgate.sa.gate",
]

# =========================


# --------- target tags (GUIDED ONLY) ----------
WAVELET_TAGS = []  # disabled: only compute guided_attention_hd1..hd4
GUIDED_TAGS  = [f"guided_attention_hd{i}" for i in range(1, 5)]
TARGET_TAGS  = GUIDED_TAGS

GROUPS = ["normal", "H", "M"]
COMPS  = [("H", "normal", "H vs normal"), ("M", "normal", "M vs normal")]
METRICS = ["gp_intensity", "out_sparsity"]  # keep file compatibility

# Paper display names (used in Fig1)
METRIC_DISPLAY = {"gp_intensity": "GRI", "out_sparsity": "GAS"}


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


# --------- import model dynamically ----------
def load_py_module(path: str):
    spec = importlib.util.spec_from_file_location("dyn_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import model file: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def build_model(model_py: str, class_name: str, ckpt_path: Optional[str] = None) -> nn.Module:
    """
    Build model with best-effort adaptation to checkpoint's output channels.
    Many segmentation codes use args like (n_classes / num_classes / out_ch / n_class / classes).
    If we can infer checkpoint output channels, we try to pass it.
    If not supported, fall back to default ctor().
    """
    mod = load_py_module(model_py)
    if not hasattr(mod, class_name):
        raise AttributeError(f"Class '{class_name}' not found in {model_py}")
    cls = getattr(mod, class_name)

    ckpt_out_ch = None
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            sd = torch.load(ckpt_path, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            # strip module.
            for k, v in sd.items():
                nk = k[7:] if isinstance(k, str) and k.startswith("module.") else k
                if nk in ("outconv1.weight", "final.weight", "outc.weight", "out_conv.weight"):
                    if torch.is_tensor(v) and v.ndim >= 1:
                        ckpt_out_ch = int(v.shape[0])
                        break
        except Exception:
            ckpt_out_ch = None

    # Try to instantiate with inferred out_mask channels if possible
    if ckpt_out_ch is not None:
        try:
            sig = inspect.signature(cls)
            kwargs = {}
            for cand in ["n_classes", "num_classes", "classes", "n_class", "out_ch", "out_channels", "nlabels"]:
                if cand in sig.parameters:
                    kwargs[cand] = ckpt_out_ch
                    break
            if kwargs:
                model = cls(**kwargs)
                return model
        except Exception:
            pass

    # Fallback
    return cls()
def load_checkpoint(model: nn.Module, ckpt_path: str, device: torch.device, strict: bool = False):
    """
    Safe load:
    - strict=False in PyTorch does NOT ignore size mismatches.
    - We filter out_mask keys that are missing or whose tensor shapes do not match.
    This prevents 'size mismatch' errors (e.g., outconv head channels differ).
    """
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    # strip 'module.' if present
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
        # print a few key examples
        for k, why in skipped[:12]:
            print(f"  - {k}: {why}")
        if len(skipped) > 12:
            print("  ...")

    if missing:
        print("[LOAD] Missing keys after filtering:", len(missing))
    if unexpected:
        print("[LOAD] Unexpected keys after filtering:", len(unexpected))

    model.to(device)
    model.eval()

# --------- data ----------
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def list_images(folder: str) -> List[str]:
    out = []
    for root, _dirs, files in os.walk(folder):
        for fn in files:
            if fn.lower().endswith(IMG_EXTS):
                out.append(os.path.join(root, fn))
    out.sort()
    return out


def pil_to_tensor(path: str, in_channels: int, resize_hw):
    if Image is None:
        raise RuntimeError("PIL is required but not available.")
    img = Image.open(path)
    # convert
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
        # HWC -> CHW
        arr = np.transpose(arr, (2, 0, 1))
    if NORM_TO_01:
        arr /= 255.0
    ten = torch.from_numpy(arr)
    return ten


@dataclass
class Sample:
    path: str
    group: str


class ImgFolderDataset(Dataset):
    def __init__(self, normal_dir: str, h_dir: str, m_dir: str):
        self.samples: List[Sample] = []
        for p in list_images(normal_dir):
            self.samples.append(Sample(p, "normal"))
        for p in list_images(h_dir):
            self.samples.append(Sample(p, "H"))
        for p in list_images(m_dir):
            self.samples.append(Sample(p, "M"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            x = pil_to_tensor(s.path, IN_CHANNELS, RESIZE_HW)
        except Exception as e:
            # skip unreadable images
            return None
        return {"image": x, "group": s.group, "path": s.path}


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    # BATCH_SIZE is 1; keep simple
    img = torch.stack([b["image"] for b in batch], 0)
    group = [b["group"] for b in batch]
    path = [b["path"] for b in batch]
    return {"image": img, "group": group, "path": path}


# --------- stats helpers ----------
def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    # a: groupB, b: groupA (we'll follow "B vs A" sign if needed; here use (B - A) style)
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.size == 0 or b.size == 0:
        return float("nan")
    # O(n*m) is fine for per-image scalars (<= few hundreds)
    gt = 0
    lt = 0
    for x in a:
        gt += int((x > b).sum())
        lt += int((x < b).sum())
    return (gt - lt) / float(a.size * b.size)


def mannwhitney_p(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from scipy.stats import mannwhitneyu
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


# --------- hook aggregation ----------
class TagAggregator:
    """
    Collect per-image tensors for each TAG, then compute two scalars per TAG:
      gp_intensity (GRI) and out_sparsity (GAS)
    """
    def __init__(self):
        self._buf: Dict[str, List[torch.Tensor]] = {t: [] for t in TARGET_TAGS}

    def clear(self):
        for t in TARGET_TAGS:
            self._buf[t].clear()

    def add_tensor(self, tag: str, t: torch.Tensor):
        if tag not in self._buf:
            return
        if not torch.is_tensor(t):
            return
        if t.numel() == 0:
            return
        self._buf[tag].append(t.detach())

    def compute_scalars(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for tag in TARGET_TAGS:
            ts = self._buf[tag]
            if len(ts) == 0:
                out[tag] = {"gp_intensity": float("nan"), "out_sparsity": float("nan")}
                continue
            v = torch.cat([x.float().reshape(-1) for x in ts], dim=0)
            if tag.startswith("wavelet_"):
                gri = torch.mean(torch.abs(v)).item()
                gas = torch.mean((torch.abs(v) < EPS).float()).item()
            else:
                # guided_attention gate maps (expected in [Analysis_Gating,1])
                gri = torch.mean(torch.abs(v)).item()
                gas = torch.mean((v < GATE_TAU).float()).item()
            out[tag] = {"gp_intensity": float(gri), "out_sparsity": float(gas)}
        return out


def build_hook_plan(model: nn.Module) -> Tuple[Dict[str, str], List[str]]:
    """
    Returns:
      - name_to_tag: module_full_name -> TAG
      - hook_names: list of module names to register hooks on

    Plan:
      - guided tags: hook internal leaves that match GUIDED_LEAF_SUFFIXES inside each hd1..4
        and map them to the parent TAG guided_attention_hdX
    """
    name_to_tag: Dict[str, str] = {}

    # 1) wavelet roots
    for t in WAVELET_TAGS:
        for n, _m in model.named_modules():
            if n == t:
                name_to_tag[n] = t

    # 2) guided leaves -> tag
    pat_hd = re.compile(r"(?:^|[._])guided_attention_hd([1-4])(?:$|[._])")
    suffixes = tuple(GUIDED_LEAF_SUFFIXES)

    for n, _m in model.named_modules():
        m = pat_hd.search(n)
        if not m:
            continue
        hd = m.group(1)
        tag = f"guided_attention_hd{hd}"
        # only hook specific leaves
        for suf in suffixes:
            if n.endswith(suf):
                name_to_tag[n] = tag
                break

    hook_names = list(name_to_tag.keys())
    return name_to_tag, hook_names


def register_hooks(model: nn.Module, name_to_tag: Dict[str, str], agg: TagAggregator):
    handles = []
    mod_dict = dict(model.named_modules())

    def make_hook(modname: str):
        tag = name_to_tag[modname]
        def _hook(_m, _inp, out):
            # out_mask may be tuple/list; keep tensor parts
            if torch.is_tensor(out):
                agg.add_tensor(tag, out)
            elif isinstance(out, (tuple, list)):
                for o in out:
                    if torch.is_tensor(o):
                        agg.add_tensor(tag, o)
        return _hook

    for name in name_to_tag.keys():
        if name not in mod_dict:
            continue
        h = mod_dict[name].register_forward_hook(make_hook(name))
        handles.append(h)
    return handles


# --------- plotting (Fig1 only, guided_attention_hd1..hd4) ----------
def fig1_heatmap(stats_rows: List[dict], out_path: str):
    """
    stats_rows: rows for 4 guided tags x 2 comps x 2 metrics with fields:
      module, comparison, metric, delta, q
    """
    import matplotlib.pyplot as plt

    # fixed ordering
    modules = WAVELET_TAGS + GUIDED_TAGS
    comps = ["H vs normal", "M vs normal"]
    metrics = METRICS

    # build matrices
    delta = {c: np.full((len(modules), len(metrics)), np.nan, dtype=float) for c in comps}
    sig   = {c: np.zeros((len(modules), len(metrics)), dtype=bool) for c in comps}

    for r in stats_rows:
        mod = r["module"]
        comp = r["comparison"]
        met = r["metric"]
        if mod not in modules or comp not in comps or met not in metrics:
            continue
        i = modules.index(mod)
        j = metrics.index(met)
        delta[comp][i, j] = float(r["delta"]) if r["delta"] != "" else float("nan")
        qv = r.get("q", "")
        try:
            sig[comp][i, j] = (float(qv) < 0.05)
        except Exception:
            sig[comp][i, j] = False

    fig = plt.figure(figsize=(9.2, 5.2))
    gs = fig.add_gridspec(1, 2, wspace=0.25)

    for k, comp in enumerate(comps):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(delta[comp], aspect="auto")
        ax.set_title(f"Cliff's δ | {comp}", fontsize=12)

        ax.set_xticks(np.arange(len(metrics)))
        ax.set_xticklabels([METRIC_DISPLAY[m] for m in metrics], rotation=25, ha="right")

        ax.set_yticks(np.arange(len(modules)))
        if k == 0:
            ax.set_yticklabels(modules, fontsize=10)
        else:
            ax.set_yticklabels([""] * len(modules))

        # annotate
        for i in range(len(modules)):
            for j in range(len(metrics)):
                v = delta[comp][i, j]
                if np.isnan(v):
                    continue
                star = "*" if sig[comp][i, j] else ""
                ax.text(j, i, f"{v:+.2f}{star}", ha="center", va="center", fontsize=9)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Cliff's δ")

    fig.suptitle("Effect sizes for guided_attention_hd1..hd4 (GRI/GAS)", y=0.98, fontsize=14)
    fig.text(0.5, 0.01, "GRI=mean(|G|); GAS=mean(G<Analysis_Gating.5); *: BH-FDR q<Analysis_Gating.05",
             ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main():
    ensure_dir(OUT_DIR)

    print("========== CONFIG SUMMARY ==========")
    print("MODEL_PY:", MODEL_PY)
    print("CLASS_NAME:", CLASS_NAME)
    print("CKPT_PATH:", CKPT_PATH)
    print("OUT_DIR:", OUT_DIR)
    print("DEVICE:", DEVICE)
    print("NORMAL_IMG_DIR:", NORMAL_IMG_DIR)
    print("H_IMG_DIR:", H_IMG_DIR)
    print("M_IMG_DIR:", M_IMG_DIR)
    print("IN_CHANNELS:", IN_CHANNELS, " RESIZE_HW:", RESIZE_HW)
    print("EPS:", EPS, " GATE_TAU:", GATE_TAU)
    print("GUIDED_LEAF_SUFFIXES:", GUIDED_LEAF_SUFFIXES)
    print("====================================")

    device = torch.device(DEVICE if (DEVICE == "cpu" or torch.cuda.is_available()) else "cpu")

    model = build_model(MODEL_PY, CLASS_NAME, ckpt_path=CKPT_PATH)

    # Initialize potential lazy modules with a dummy forward
    dummy = torch.zeros((1, IN_CHANNELS, 256, 256), dtype=torch.float32, device=device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        try:
            _ = model(dummy)
            print("[INFO] Dummy forward done (model initialized).")
        except Exception as e:
            print("[WARN] Dummy forward failed (continuing):", repr(e))

    load_checkpoint(model, CKPT_PATH, device=device, strict=False)

    # Dataset
    ds = ImgFolderDataset(NORMAL_IMG_DIR, H_IMG_DIR, M_IMG_DIR)
    print(f"[DATA] total images: {len(ds)}")
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                        collate_fn=collate_skip_none)

    # Hooks: only guided gate leaves; output tags are exactly 4 (hd1..hd4)
    name_to_tag, hook_names = build_hook_plan(model)
    print(f"[HOOK] hook modules: {len(hook_names)}  | output tags: {len(TARGET_TAGS)}")
    # Count per tag
    tag_count = {t: 0 for t in TARGET_TAGS}
    for n, t in name_to_tag.items():
        tag_count[t] += 1
    for t in TARGET_TAGS:
        print(f"  - {t}: hooks={tag_count[t]}")
    if any(tag_count[t] == 0 for t in GUIDED_TAGS):
        print("[WARN] Some guided_attention_hd* have Analysis_Gating hooks. Your model leaf names may differ.")
        print("       Adjust GUIDED_LEAF_SUFFIXES to match actual module names printed above/logs.")

    agg = TagAggregator()
    handles = register_hooks(model, name_to_tag, agg)

    # Per-group arrays (per-image scalars)
    raw: Dict[str, Dict[str, List[float]]] = {t: {g: [] for g in GROUPS} for t in TARGET_TAGS}
    raw2: Dict[str, Dict[str, List[float]]] = {t: {g: [] for g in GROUPS} for t in TARGET_TAGS}

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            x = batch["image"].to(device, non_blocking=True)
            g = batch["group"][0]
            agg.clear()
            try:
                _ = model(x)
            except Exception as e:
                print("[WARN] forward failed for", batch["path"][0], ":", repr(e))
                continue

            scal = agg.compute_scalars()
            # store scalars
            for t in TARGET_TAGS:
                v1 = scal[t]["gp_intensity"]
                v2 = scal[t]["out_sparsity"]
                if not (math.isnan(v1) or math.isinf(v1)):
                    raw[t][g].append(v1)
                if not (math.isnan(v2) or math.isinf(v2)):
                    raw2[t][g].append(v2)

    # Remove hooks
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass

    # Save NPZ raw arrays (compat keys)
    npz_dict = {}
    for t in TARGET_TAGS:
        for g in GROUPS:
            npz_dict[f"{t}__{g}__gp_intensity"] = np.array(raw[t][g], dtype=np.float32)
            npz_dict[f"{t}__{g}__out_sparsity"] = np.array(raw2[t][g], dtype=np.float32)

    raw_npz_path = os.path.join(OUT_DIR, "evidence_chain_raw.npz")
    np.savez_compressed(raw_npz_path, **npz_dict)

    # Build CSV stats
    rows = []
    # We apply BH-FDR within each (comparison, metric) across the 9 modules
    for met in METRICS:
        for b, a, comp_name in COMPS:
            pvals = []
            tmp = []
            for t in TARGET_TAGS:
                arr_a = np.array(raw[t][a] if met == "gp_intensity" else raw2[t][a], dtype=np.float32)
                arr_b = np.array(raw[t][b] if met == "gp_intensity" else raw2[t][b], dtype=np.float32)
                p = mannwhitney_p(arr_b, arr_a)
                d = cliffs_delta(arr_b, arr_a)  # B vs A
                med_a = float(np.median(arr_a)) if arr_a.size else float("nan")
                med_b = float(np.median(arr_b)) if arr_b.size else float("nan")
                tmp.append((t, comp_name, met, arr_a.size, arr_b.size, med_a, med_b, p, d))
                pvals.append(p)

            qvals = bh_fdr(pvals)
            for i, (t, comp_name, met, na, nb, med_a, med_b, p, d) in enumerate(tmp):
                rows.append({
                    "module": t,
                    "comparison": comp_name,
                    "metric": met,
                    "n_a": na, "n_b": nb,
                    "median_a(normal)": med_a,
                    "median_b(H/M)": med_b,
                    "p": p,
                    "q": qvals[i],
                    "delta": d,
                })

    csv_path = os.path.join(OUT_DIR, "evidence_chain_stats.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Fig1 only (guided_attention_hd1..hd4)
    fig1_path = os.path.join(OUT_DIR, "Fig1_effectsize_heatmap_9modules.png")
    fig1_heatmap(rows, fig1_path)

    print("\n[DONE]")
    print("  Raw NPZ :", raw_npz_path)
    print("  Stats CSV:", csv_path)
    print("  Fig1:", fig1_path)
    print("\n[NOTE] Output contains ONLY guided module tags:", ", ".join(TARGET_TAGS))


if __name__ == "__main__":
    main()