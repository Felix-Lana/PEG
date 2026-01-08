# -*- coding: utf-8 -*-


from __future__ import annotations

import os
import argparse
from typing import Optional, Iterable, Tuple, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import torch
except Exception:
    torch = None


DEFAULT_COLORS_20 = [
    (0, 0, 0),        # Analysis_Gating: background
    (230, 25, 75),    # B1: red
    (60, 180, 75),    # B2: green
    (255, 225, 25),   # 3: yellow
    (0, 130, 200),    # 4: blue
    (245, 130, 48),   # 5: orange
    (145, 30, 180),   # 6: purple
    (70, 240, 240),   # 7: cyan
    (240, 50, 230),   # 8: magenta
    (210, 245, 60),   # 9: lime
    (250, 190, 212),  # 10: pink
    (0, 128, 128),    # 11: teal
    (220, 190, 255),  # 12: lavender
    (170, 110, 40),   # 13: brown
    (255, 250, 200),  # 14: beige
    (128, 0, 0),      # 15: maroon
    (170, 255, 195),  # 16: mint
    (128, 128, 0),    # 17: olive
    (255, 215, 180),  # 18: apricot
    (0, 0, 128),      # 19: navy
]


def make_palette(num_classes: int, colors: Optional[List[Tuple[int, int, int]]] = None) -> List[int]:

    if colors is None:
        colors = DEFAULT_COLORS_20

    pal = []
    for i in range(256):
        r, g, b = colors[i % len(colors)] if i < num_classes else (0, 0, 0)
        pal.extend([int(r), int(g), int(b)])
    return pal



def colorize_index_mask(index_mask: np.ndarray, num_classes: int, to_rgb: bool = False) -> Image.Image:
    if index_mask.ndim != 2:
        raise ValueError("index_mask:2D (H,W)")
    arr = np.asarray(index_mask)
    if not np.issubdtype(arr.dtype, np.integer):
        arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    pal = make_palette(num_classes)
    im_p = Image.fromarray(arr, mode="P")
    im_p.putpalette(pal)
    return im_p.convert("RGB") if to_rgb else im_p


def overlay_on_image(image: Image.Image, color_mask_rgb: Image.Image, alpha: float = 0.5) -> Image.Image:
    if image.size != color_mask_rgb.size:
        color_mask_rgb = color_mask_rgb.resize(image.size, Image.NEAREST)
    base = image.convert("RGB")
    return Image.blend(base, color_mask_rgb, alpha)


def make_legend(
    num_classes: int,
    class_names: Optional[Iterable[str]] = None,
    swatch_size: Tuple[int, int] = (28, 18),
    margin: int = 8,
    cols: int = 1
) -> Image.Image:
    pal = make_palette(num_classes)
    colors = [tuple(pal[i * 3:(i + 1) * 3]) for i in range(num_classes)]
    names = list(class_names) if class_names is not None else [f"class_{i}" for i in range(num_classes)]

    rows = (num_classes + cols - 1) // cols
    cell_w = max(180, swatch_size[0] + 120)
    cell_h = max(swatch_size[1], 18) + 6
    W = cols * cell_w + (cols + 1) * margin
    H = rows * cell_h + (rows + 1) * margin

    legend = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(legend)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for idx in range(num_classes):
        r = idx // cols
        c = idx % cols
        x0 = margin + c * cell_w
        y0 = margin + r * cell_h

        # swatch
        sx0, sy0 = x0, y0 + (cell_h - swatch_size[1]) // 2
        sx1, sy1 = sx0 + swatch_size[0], sy0 + swatch_size[1]
        draw.rectangle([sx0, sy0, sx1, sy1], fill=colors[idx], outline=(0, 0, 0))

        # text
        text = f"{idx}: {names[idx] if idx < len(names) else f'class_{idx}'}"
        tx, ty = sx1 + 8, y0 + (cell_h - 14) // 2
        draw.text((tx, ty), text, fill=(0, 0, 0), font=font)

    return legend


# ========================= IO 辅助 =========================

def load_index_mask(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
        if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW -> HW
            arr = arr[0]
        return arr.astype(np.int64) if np.issubdtype(arr.dtype, np.integer) else arr
    if ext in (".pt", ".pth"):
        if torch is None:
            raise RuntimeError(".pt/.pth: Need torch")
        t = torch.load(path, map_location="cpu")
        if isinstance(t, dict) and "_mask" in t:
            t = t["_mask"]
        if hasattr(t, "detach"):
            t = t.detach().cpu()
        arr = t.numpy() if hasattr(t, "numpy") else np.asarray(t)
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = arr[0]
        return arr
    # image
    im = Image.open(path)
    if im.mode in ("P", "L", "I"):
        return np.array(im)
    return np.array(im.convert("L"))


def find_matching_image(mask_path: str, image_dir: str, exts: Tuple[str, ...]) -> Optional[str]:
    base = os.path.splitext(os.path.basename(mask_path))[0]
    for e in exts:
        cand = os.path.join(image_dir, base + e)
        if os.path.isfile(cand):
            return cand
    return None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_image(im: Image.Image, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    im.save(path)


def iter_from_list_file(list_file: str) -> List[Tuple[str, Optional[str]]]:
    pairs: List[Tuple[str, Optional[str]]] = []
    with open(list_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if "," in s:
                parts = [p.strip() for p in s.split(",") if p.strip()]
            else:
                parts = [p.strip() for p in s.split() if p.strip()]
            if len(parts) == 1:
                pairs.append((parts[0], None))
            else:
                pairs.append((parts[0], parts[1]))
    return pairs


def iter_from_mask_dir(mask_dir: str, valid_exts: Tuple[str, ...]) -> List[str]:
    files = []
    for fn in os.listdir(mask_dir):
        p = os.path.join(mask_dir, fn)
        if os.path.isfile(p) and os.path.splitext(fn)[1].lower() in valid_exts:
            files.append(p)
    files.sort()
    return files


# ========================= CLI =========================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="")
    p.add_argument("--_mask", type=str, default=r"", help=" _mask")
    p.add_argument("--image", type=str, default=None, help="single pic")
    p.add_argument("--mask_dir", type=str, default=r"F:\PaperCode\Pic\data\masks", help="_mask_batch")
    p.add_argument("--image_dir", type=str, default=None, help="Pic_batch")
    p.add_argument("--list_file", type=str, default=None, help="")
    p.add_argument("--out_dir", type=str, default=r"F:\PaperCode\Pic\data\GT_color")
    p.add_argument("--num_classes", type=int, default=4, help="")
    p.add_argument("--alpha", type=float, default=0.5, help="")
    p.add_argument("--legend", action="store_true", help="")
    p.add_argument("--image_exts", type=str, default=".jpg,.jpeg,.png,.bmp,.tif,.tiff", help="")

    return p


def main():
    args = build_argparser().parse_args()
    ensure_dir(args.out_dir)

    mode = None
    if args.list_file:
        mode = "list"
    elif args.mask_dir:
        mode = "dir"
    elif args.mask:
        mode = "single"
    else:
        raise SystemExit("请提供 --list_file 或 --mask_dir 或 --_mask 其中之一。")

    if args.legend:
        legend = make_legend(args.num_classes)
        out_legend = os.path.join(args.out_dir, "legend.png")
        save_image(legend, out_legend)
        print("Saved:", out_legend)

    image_exts = tuple([e.strip().lower() for e in args.image_exts.split(",") if e.strip()])

    jobs: List[Tuple[str, Optional[str]]] = []
    if mode == "single":
        jobs = [(args.mask, args.image)]
    elif mode == "list":
        jobs = iter_from_list_file(args.list_file)
    elif mode == "dir":
        mask_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy", ".pt", ".pth")
        masks = iter_from_mask_dir(args.mask_dir, mask_exts)
        for m in masks:
            img = None
            if args.image_dir:
                img = find_matching_image(m, args.image_dir, image_exts)
            jobs.append((m, img))

    n_ok, n_fail = 0, 0
    for mask_path, image_path in jobs:
        try:
            if not mask_path or not os.path.isfile(mask_path):
                raise FileNotFoundError(f"_mask 不存在: {mask_path}")

            base = os.path.splitext(os.path.basename(mask_path))[0]

            idx_mask = load_index_mask(mask_path)
            color_mask = colorize_index_mask(idx_mask, args.num_classes, to_rgb=False)  # P 模式更适合保存
            out_color = os.path.join(args.out_dir, f"{base}_color.png")
            save_image(color_mask, out_color)
            print("Saved:", out_color)

            if image_path:
                if not os.path.isfile(image_path):
                    raise FileNotFoundError(f"image 不存在: {image_path}")
                img = Image.open(image_path)
                overlay = overlay_on_image(img, color_mask.convert("RGB"), alpha=args.alpha)
                out_overlay = os.path.join(args.out_dir, f"{base}_overlay_a{args.alpha:.2f}.png")
                save_image(overlay, out_overlay)
                print("Saved:", out_overlay)

            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"[WARN] {mask_path} failed: {e}")

    print(f"Done. ok={n_ok}, fail={n_fail}")


if __name__ == "__main__":
    main()
