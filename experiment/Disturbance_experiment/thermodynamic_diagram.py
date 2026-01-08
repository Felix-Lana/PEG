from __future__ import annotations
import os, re, argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

PRED_RE = re.compile(r"^(?P<x>\d+)_(?P<v>\d+)_mask\.(png|bmp|tif|tiff|jpg|jpeg)$", re.IGNORECASE)
GT_RE   = re.compile(r"^(?P<x>\d+).*?\.(png|bmp|tif|tiff|jpg|jpeg)$", re.IGNORECASE)


def read_index_png(path: str) -> np.ndarray:
    img = Image.open(path)
    if img.mode not in ("L", "I;16", "P"):
        img = img.convert("L")
    return np.array(img)


def read_gray_img(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.array(img)


def scan_gt_dir(gt_dir: str) -> dict[int, str]:
    m: dict[int, str] = {}
    for fn in os.listdir(gt_dir):
        mo = GT_RE.match(fn)
        if mo:
            x = int(mo.group("x"))
            m[x] = os.path.join(gt_dir, fn)
    return m


def scan_pred_dir(pred_dir: str) -> dict[tuple[int, int], str]:
    m: dict[tuple[int, int], str] = {}
    for fn in os.listdir(pred_dir):
        mo = PRED_RE.match(fn)
        if mo:
            x = int(mo.group("x"))
            v = int(mo.group("v"))
            m[(x, v)] = os.path.join(pred_dir, fn)
    return m


def save_heatmap(
    heat: np.ndarray,
    out_path: str,
    bg: np.ndarray | None = None,
    with_colorbar: bool = False,
    dpi: int = 220,
    overlay_alpha: float = 0.6,
):
    """
    Paper-friendly export:
      - no title
      - no axes
      - tight crop (no surrounding whitespace)
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_axis_off()

    if bg is not None:
        ax.imshow(bg, interpolation="nearest")
        im = ax.imshow(heat, alpha=overlay_alpha, interpolation="nearest")  # overlay
    else:
        im = ax.imshow(heat, interpolation="nearest")

    if with_colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def mean_background(img_dir: str, img_fmt: str, xs: list[int], v: int) -> np.ndarray | None:
    if not img_dir:
        return None
    acc = None
    n = 0
    for x in xs:
        p = os.path.join(img_dir, img_fmt.format(x=x, v=v))
        if not os.path.exists(p):
            continue
        im = read_gray_img(p).astype(np.float32)
        if acc is None:
            acc = np.zeros_like(im, dtype=np.float32)
        acc += im
        n += 1
    if acc is None or n == 0:
        return None
    bg = (acc / n)
    return bg


def main():
    ap = argparse.ArgumentParser("Make spatial heatmaps for bias analysis (AC->Lens etc.)")

    ap.add_argument("--root_dir", type=str, default=r"",help="")
    ap.add_argument("--gt_dir", type=str,default=r"",help="")
    ap.add_argument("--out_dir", type=str,default=r"1")
    ap.add_argument("--v_list", type=str, default="0,7,8,10", help="comma list, e.g. 0,5,10 or 0-10")
    ap.add_argument("--img_dir", type=str, default=None,help="optional: perturbed OCT images folder for overlay")
    ap.add_argument("--img_fmt", type=str, default="{x}_{v}.png", help="template inside img_dir")
    ap.add_argument("--with_colorbar", action="store_true",help="Draw colorbar (default: off for clean paper layout).")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--overlay_alpha", type=float, default=0.6)

    args = ap.parse_args()

    # parse v_list
    if "-" in args.v_list:
        a, b = args.v_list.split("-")
        v_list = list(range(int(a), int(b) + 1))
    else:
        v_list = [int(s.strip()) for s in args.v_list.split(",") if s.strip()]

    gt_map = scan_gt_dir(args.gt_dir)
    if not gt_map:
        raise FileNotFoundError(f"No OCTA files found in {args.gt_dir}")
    xs_all = sorted(gt_map.keys())

    model_names = sorted([d for d in os.listdir(args.root_dir) if os.path.isdir(os.path.join(args.root_dir, d))])
    if not model_names:
        raise FileNotFoundError(f"No model folders in {args.root_dir}")

    # load one OCTA for shape
    gt0 = read_index_png(gt_map[xs_all[0]])
    H, W = gt0.shape

    for model in model_names:
        pred_dir = os.path.join(args.root_dir, model)
        pred_map = scan_pred_dir(pred_dir)
        if not pred_map:
            print(f"[SKIP] {model}: no pred files matched X_Y_mask.*")
            continue

        for v in v_list:
            # accumulators
            sum_ac = np.zeros((H, W), dtype=np.float32)           # count of OCTA==B2
            sum_err_ac_lens = np.zeros((H, W), dtype=np.float32)  # count of (OCTA==B2 & Pred==3)
            sum_pred_lens = np.zeros((H, W), dtype=np.float32)    # count of Pred==3
            used = 0

            for x in xs_all:
                if (x, v) not in pred_map:
                    continue
                gt = read_index_png(gt_map[x])
                pred = read_index_png(pred_map[(x, v)])
                if gt.shape != (H, W) or pred.shape != (H, W):
                    raise ValueError(f"shape mismatch at model={model} x={x} v={v}")

                ac = (gt == 2)
                pred_lens = (pred == 3)
                sum_ac += ac.astype(np.float32)
                sum_err_ac_lens += (ac & pred_lens).astype(np.float32)
                sum_pred_lens += pred_lens.astype(np.float32)
                used += 1

            if used == 0:
                print(f"[WARN] {model} v={v}: no samples used")
                continue

            eps = 1e-6
            heat_ac_to_lens = sum_err_ac_lens / (sum_ac + eps)  # conditional prob within AC
            heat_pred_lens = sum_pred_lens / float(used)        # occupancy prob

            # optional background
            bg = mean_background(args.img_dir, args.img_fmt, xs_all, v=v)

            # save (no titles; tight cropped)
            outA = os.path.join(args.out_dir, "AC_to_Lens", f"{model}_v{v:02d}.png")
            outB = os.path.join(args.out_dir, "PredLens",   f"{model}_v{v:02d}.png")

            save_heatmap(
                heat_ac_to_lens, outA, bg=bg,
                with_colorbar=args.with_colorbar, dpi=args.dpi, overlay_alpha=args.overlay_alpha
            )
            save_heatmap(
                heat_pred_lens, outB, bg=bg,
                with_colorbar=args.with_colorbar, dpi=args.dpi, overlay_alpha=args.overlay_alpha
            )

            # difference map relative to v=0 (optional)
            if v != 0:
                sum0 = np.zeros((H, W), dtype=np.float32)
                used0 = 0
                for x in xs_all:
                    if (x, 0) not in pred_map:
                        continue
                    pred0 = read_index_png(pred_map[(x, 0)])
                    sum0 += (pred0 == 3).astype(np.float32)
                    used0 += 1
                if used0 > 0:
                    heat0 = sum0 / float(used0)
                    delta = heat_pred_lens - heat0
                    outD = os.path.join(args.out_dir, "DeltaPredLens", f"{model}_v{v:02d}_minus_v00.png")
                    save_heatmap(
                        delta, outD, bg=bg,
                        with_colorbar=args.with_colorbar, dpi=args.dpi, overlay_alpha=args.overlay_alpha
                    )

            print(f"[OK] model={model} v={v} used={used}")

    print("Done. Heatmaps saved to:", args.out_dir)


if __name__ == "__main__":
    main()
