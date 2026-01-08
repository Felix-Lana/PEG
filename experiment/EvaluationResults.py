import os
import csv
import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy.ndimage import binary_erosion, distance_transform_edt


PRED_DIR = r""
GT_DIR   = r""

OUT_DIR = r""

NUM_CLASSES = 4
IGNORE_INDEX = 0
SPACING = (1.0, 1.0)


PRED_SUFFIX_TO_STRIP = "_mask"

SKIP_CLASS_IF_GT_EMPTY = True
SKIP_INF_IN_MEAN_HD95 = True

MAKE_PLOTS = True
PLOT_KIND = "both"  # "box" / "violin" / "both"
# ===============================================================


def ensure_dir(p: str) -> None:
    if p and (not os.path.exists(p)):
        os.makedirs(p, exist_ok=True)


def read_mask_class_id(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(f"Cannot read (empty file?): {path}")

    m = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Cannot decode image: {path}")

    if m.ndim == 3:
        raise ValueError(
            f"Mask is {m.shape[2]}-channel. Expected single-channel class-id _mask (Analysis_Gating..{NUM_CLASSES-1}). "
            f"If it's color-coded, convert to class ids first."
        )
    return m.astype(np.int64)


def _safe_div(n: float, d: float, eps: float = 1e-7) -> float:
    return float((n + eps) / (d + eps))


def binary_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray, eps: float = 1e-7) -> Dict[str, float]:
    pred_bin = pred_bin.astype(bool)
    gt_bin = gt_bin.astype(bool)

    tp = np.logical_and(pred_bin, gt_bin).sum()
    fp = np.logical_and(pred_bin, ~gt_bin).sum()
    fn = np.logical_and(~pred_bin, gt_bin).sum()

    pred_sum = pred_bin.sum()
    gt_sum = gt_bin.sum()
    inter = tp
    union = pred_sum + gt_sum - inter

    if pred_sum == 0 and gt_sum == 0:
        return {"dice": 1.0, "iou": 1.0, "prec": 1.0, "rec": 1.0}

    dice = _safe_div(2.0 * inter, pred_sum + gt_sum, eps)
    iou = _safe_div(inter, union, eps)
    prec = _safe_div(tp, tp + fp, eps)
    rec = _safe_div(tp, tp + fn, eps)
    return {"dice": dice, "iou": iou, "prec": prec, "rec": rec}


def hd95_binary(pred_bin: np.ndarray,
                gt_bin: np.ndarray,
                spacing: Tuple[float, float] = (1.0, 1.0)) -> float:
    pred_bin = pred_bin.astype(bool)
    gt_bin = gt_bin.astype(bool)

    if pred_bin.sum() == 0 and gt_bin.sum() == 0:
        return 0.0
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return float("inf")

    structure = np.ones((3, 3), dtype=bool)
    pred_er = binary_erosion(pred_bin, structure=structure, border_value=0)
    gt_er = binary_erosion(gt_bin, structure=structure, border_value=0)

    pred_surf = np.logical_and(pred_bin, ~pred_er)
    gt_surf = np.logical_and(gt_bin, ~gt_er)

    dt_gt = distance_transform_edt(~gt_surf, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_surf, sampling=spacing)

    d_pred_to_gt = dt_gt[pred_surf]
    d_gt_to_pred = dt_pred[gt_surf]

    if d_pred_to_gt.size == 0 and d_gt_to_pred.size == 0:
        return 0.0
    if d_pred_to_gt.size == 0 or d_gt_to_pred.size == 0:
        return float("inf")

    all_d = np.concatenate([d_pred_to_gt, d_gt_to_pred], axis=0)
    return float(np.percentile(all_d, 95))


def multiclass_eval(pred: np.ndarray,
                    gt: np.ndarray,
                    num_classes: int = 4,
                    ignore_index: Optional[int] = None,
                    spacing: Tuple[float, float] = (1.0, 1.0),
                    eps: float = 1e-7,
                    skip_inf_in_mean_hd95: bool = True,
                    skip_class_if_gt_empty: bool = True) -> Tuple[Dict[int, Dict[str, float]], Dict[str, float]]:

    assert pred.shape == gt.shape, f"Shape mismatch: pred {pred.shape}, gt {gt.shape}"
    pred = pred.astype(np.int64)
    gt = gt.astype(np.int64)

    valid_classes = [c for c in range(num_classes) if c != ignore_index]
    per_class: Dict[int, Dict[str, float]] = {}

    for c in valid_classes:
        pred_c = (pred == c)
        gt_c = (gt == c)
        gt_cnt = int(gt_c.sum())

        if skip_class_if_gt_empty and gt_cnt == 0:
            per_class[c] = {
                "skipped": True,
                "gt_pixels": gt_cnt,
                "dice": float("nan"),
                "iou": float("nan"),
                "prec": float("nan"),
                "rec": float("nan"),
                "hd95": float("nan"),
            }
            continue

        m = binary_metrics(pred_c, gt_c, eps=eps)
        h = hd95_binary(pred_c, gt_c, spacing=spacing)

        per_class[c] = {
            "skipped": False,
            "gt_pixels": gt_cnt,
            "dice": m["dice"],
            "iou": m["iou"],
            "prec": m["prec"],
            "rec": m["rec"],
            "hd95": h,
        }

    def mean_of(key: str) -> float:
        vals = []
        for c in valid_classes:
            if per_class[c]["skipped"]:
                continue
            v = per_class[c][key]
            if key == "hd95" and skip_inf_in_mean_hd95 and (not np.isfinite(v)):
                continue
            if np.isfinite(v):
                vals.append(v)
        if len(vals) == 0:
            return float("nan")
        return float(np.mean(vals))

    summary = {
        "mean_dice": mean_of("dice"),
        "mean_iou": mean_of("iou"),
        "mean_prec": mean_of("prec"),
        "mean_rec": mean_of("rec"),
        "mean_hd95": mean_of("hd95"),
    }
    return per_class, summary


def list_mask_files(folder: str) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    out = []
    for fn in os.listdir(folder):
        p = os.path.join(folder, fn)
        if os.path.isfile(p) and os.path.splitext(fn)[1].lower() in exts:
            out.append(p)
    out.sort()
    return out


def pred_to_gt_path(pred_path: str, gt_dir: str) -> str:
    base = os.path.splitext(os.path.basename(pred_path))[0]
    if PRED_SUFFIX_TO_STRIP and base.endswith(PRED_SUFFIX_TO_STRIP):
        base = base[: -len(PRED_SUFFIX_TO_STRIP)]
    # 默认 GT 是 .png；如果你 GT 后缀不同，可在这里扩展
    return os.path.join(gt_dir, base + ".png")


def nan_summary_stats(x: List[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in x if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "median": np.nan, "q1": np.nan, "q3": np.nan}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "q1": float(np.percentile(arr, 25)),
        "q3": float(np.percentile(arr, 75)),
    }


def make_plots_from_csvs(per_image_rows: List[Dict[str, float]],
                         per_class_rows: List[Dict[str, float]],
                         out_plot_dir: str) -> None:

    ensure_dir(out_plot_dir)

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib not available, skip plotting. err={repr(e)}")
        return

    # -------- per-image 分布 --------
    metrics = ["mean_dice", "mean_iou", "mean_prec", "mean_rec"]
    labels = ["Dice", "IoU", "Prec", "Rec"]

    data = []
    for k in metrics:
        vals = [r[k] for r in per_image_rows if (k in r) and np.isfinite(r[k])]
        data.append(vals)

    if PLOT_KIND in ("box", "both"):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.boxplot(data, labels=labels, showfliers=True)
        ax.set_title("Per-image Metric Distribution (Boxplot)")
        ax.set_ylabel("Value")
        fig.tight_layout()
        fig.savefig(os.path.join(out_plot_dir, "per_image_boxplot.png"), dpi=300)
        plt.close(fig)

    if PLOT_KIND in ("violin", "both"):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.violinplot(data, showmeans=True, showmedians=True)
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_title("Per-image Metric Distribution (Violin)")
        ax.set_ylabel("Value")
        fig.tight_layout()
        fig.savefig(os.path.join(out_plot_dir, "per_image_violinplot.png"), dpi=300)
        plt.close(fig)


    valid_class_rows = [r for r in per_class_rows if (not r.get("skipped", False))]

    class_ids = sorted({int(r["class_id"]) for r in valid_class_rows})
    class_dice_data = []
    class_labels = []
    for c in class_ids:
        vals = [r["dice"] for r in valid_class_rows if int(r["class_id"]) == c and np.isfinite(r["dice"])]
        class_dice_data.append(vals)
        class_labels.append(f"Class {c}")

    if len(class_ids) > 0:
        if PLOT_KIND in ("box", "both"):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.boxplot(class_dice_data, labels=class_labels, showfliers=True)
            ax.set_title("Per-class Dice Distribution (Boxplot)")
            ax.set_ylabel("Dice")
            fig.tight_layout()
            fig.savefig(os.path.join(out_plot_dir, "per_class_dice_boxplot.png"), dpi=300)
            plt.close(fig)

        if PLOT_KIND in ("violin", "both"):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.violinplot(class_dice_data, showmeans=True, showmedians=True)
            ax.set_xticks(np.arange(1, len(class_labels) + 1))
            ax.set_xticklabels(class_labels)
            ax.set_title("Per-class Dice Distribution (Violin)")
            ax.set_ylabel("Dice")
            fig.tight_layout()
            fig.savefig(os.path.join(out_plot_dir, "per_class_dice_violinplot.png"), dpi=300)
            plt.close(fig)


def run_folder_eval():
    ensure_dir(OUT_DIR)
    out_plot_dir = os.path.join(OUT_DIR, "plots")

    out_csv_per_image = os.path.join(OUT_DIR, "per_image_summary.csv")
    out_csv_per_class = os.path.join(OUT_DIR, "per_image_per_class_detail.csv")

    pred_files = list_mask_files(PRED_DIR)
    if len(pred_files) == 0:
        raise RuntimeError(f"No prediction masks found in: {PRED_DIR}")

    image_rows: List[Dict[str, float]] = []
    per_class_rows: List[Dict[str, float]] = []
    class_pool = {c: {"dice": [], "iou": [], "prec": [], "rec": [], "hd95": []}
                  for c in range(NUM_CLASSES) if c != IGNORE_INDEX}

    ok, miss, failed = 0, 0, 0

    for pred_path in pred_files:
        gt_path = pred_to_gt_path(pred_path, GT_DIR)
        if not os.path.exists(gt_path):
            miss += 1
            print(f"[MISS GT] pred={os.path.basename(pred_path)} -> gt not found: {gt_path}")
            continue

        try:
            pred = read_mask_class_id(pred_path)
            gt = read_mask_class_id(gt_path)

            per_class, summary = multiclass_eval(
                pred, gt,
                num_classes=NUM_CLASSES,
                ignore_index=IGNORE_INDEX,
                spacing=SPACING,
                skip_inf_in_mean_hd95=SKIP_INF_IN_MEAN_HD95,
                skip_class_if_gt_empty=SKIP_CLASS_IF_GT_EMPTY
            )

            base = os.path.splitext(os.path.basename(pred_path))[0]
            image_rows.append({
                "id": base,
                "pred_path": pred_path,
                "gt_path": gt_path,
                **summary
            })

            for c, m in per_class.items():
                row = {
                    "id": base,
                    "class_id": int(c),
                    "skipped": bool(m["skipped"]),
                    "gt_pixels": int(m["gt_pixels"]),
                    "dice": float(m["dice"]) if np.isfinite(m["dice"]) else float("nan"),
                    "iou": float(m["iou"]) if np.isfinite(m["iou"]) else float("nan"),
                    "prec": float(m["prec"]) if np.isfinite(m["prec"]) else float("nan"),
                    "rec": float(m["rec"]) if np.isfinite(m["rec"]) else float("nan"),
                    "hd95": float(m["hd95"]) if np.isfinite(m["hd95"]) else (float("inf") if m["hd95"] == float("inf") else float("nan")),
                }
                per_class_rows.append(row)

            # 记录每类的指标（用于算全数据集每类平均）
            for c, m in per_class.items():
                if m["skipped"]:
                    continue
                class_pool[c]["dice"].append(m["dice"])
                class_pool[c]["iou"].append(m["iou"])
                class_pool[c]["prec"].append(m["prec"])
                class_pool[c]["rec"].append(m["rec"])
                if (not SKIP_INF_IN_MEAN_HD95) or np.isfinite(m["hd95"]):
                    class_pool[c]["hd95"].append(m["hd95"])

            ok += 1
        except Exception as e:
            failed += 1
            print(f"[FAIL] {pred_path}  <->  {gt_path}  -> {repr(e)}")

    if ok == 0:
        raise RuntimeError("No valid pred/gt pairs evaluated.")

    # ========== 数据集级平均（用于论文）==========
    def mean_over_images(key: str) -> float:
        vals = [r[key] for r in image_rows if (key in r) and np.isfinite(r[key])]
        return float(np.mean(vals)) if len(vals) else float("nan")

    dataset_summary = {
        "mean_dice": mean_over_images("mean_dice"),
        "mean_iou":  mean_over_images("mean_iou"),
        "mean_prec": mean_over_images("mean_prec"),
        "mean_rec":  mean_over_images("mean_rec"),
        "mean_hd95": mean_over_images("mean_hd95"),
    }

    # 每类平均
    per_class_mean = {}
    for c, pool in class_pool.items():
        per_class_mean[c] = {
            k: (float(np.mean(v)) if len(v) else float("nan"))
            for k, v in pool.items()
        }
        per_class_mean[c]["n_images_used"] = int(len(pool["dice"]))

    # ========== 输出到控制台 ==========
    print("\n================= DATASET SUMMARY (macro over classes per image, then mean over images) =================")
    for k, v in dataset_summary.items():
        print(f"{k}: {v:.4f}")

    print("\n================= PER-CLASS MEAN (mean over images where class is used) =================")
    for c in sorted(per_class_mean.keys()):
        m = per_class_mean[c]
        print(f"class {c} | n={m['n_images_used']} | "
              f"Dice={m['dice']:.4f} IoU={m['iou']:.4f} Prec={m['prec']:.4f} Rec={m['rec']:.4f} HD95={m['hd95']:.4f}")

    print("\n================= PER-IMAGE DISTRIBUTION STATS (for paper text) =================")
    for k in ["mean_dice", "mean_iou", "mean_prec", "mean_rec", "mean_hd95"]:
        vals = [r[k] for r in image_rows if (k in r) and np.isfinite(r[k])]
        st = nan_summary_stats(vals)
        print(f"{k}: n={st['n']} mean={st['mean']:.4f} std={st['std']:.4f} median={st['median']:.4f} "
              f"q1={st['q1']:.4f} q3={st['q3']:.4f}")

    print("\n================= STATS =================")
    print(f"total_pred={len(pred_files)}, ok_pairs={ok}, miss_gt={miss}, failed={failed}")

    # ========== 写 CSV ==========
    # 1) per-image 汇总
    with open(out_csv_per_image, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "pred_path", "gt_path", "mean_dice", "mean_iou", "mean_prec", "mean_rec", "mean_hd95"]
        )
        writer.writeheader()
        for r in image_rows:
            writer.writerow(r)
    print(f"\nSaved per-image summary CSV to: {out_csv_per_image}")

    with open(out_csv_per_class, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "class_id", "skipped", "gt_pixels", "dice", "iou", "prec", "rec", "hd95"]
        )
        writer.writeheader()
        for r in per_class_rows:
            writer.writerow(r)
    print(f"Saved per-image per-class detail CSV to: {out_csv_per_class}")

    if MAKE_PLOTS:
        make_plots_from_csvs(image_rows, per_class_rows, out_plot_dir)
        print(f"\nSaved plots to: {out_plot_dir}")


if __name__ == "__main__":
    run_folder_eval()
