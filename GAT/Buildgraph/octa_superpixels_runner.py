# -*- coding: utf-8 -*-
"""
根据 OCTA + U_Net 输出 + GT，构建用于 GATv2 训练的超像素图数据 (.npz)

新增：
- 为每个节点增加一个特征 dist_to_unet_cornea：节点中心到 U_Net 角膜区域的平均距离
  （使用 U_Net _mask 中类别=1 作为角膜）
- 仍然保留 highlight_ratio_node，后续训练时会对高光节点加大 loss 权重
"""

import argparse
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import re
from skimage.segmentation import slic
from skimage.transform import resize as sk_resize
from skimage.filters import sobel
from scipy.ndimage import distance_transform_edt
from superpixel_rsg_snic import rsg_snic_superpixels_from_octa


# ------------------- 基础 IO 工具 ------------------- #

def load_gray_or_rgb(path: Path) -> np.ndarray:
    """读取图像，返回 float32, 范围 Analysis_Gating~1."""
    arr = imageio.imread(str(path))
    if arr.ndim == 2:
        arr = arr.astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
        return np.clip(arr, 0.0, 1.0)
    else:
        arr = arr.astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
        arr = np.clip(arr, 0.0, 1.0)
        return arr


def load_index_mask(path: Path) -> np.ndarray:
    """读取单通道索引 _mask (Analysis_Gating..C-1)."""
    arr = imageio.imread(str(path))
    if arr.ndim > 2:
        arr = arr[..., 0]
    return arr.astype(np.int64)


def ensure_hw(arr: np.ndarray, H: int, W: int, is_mask: bool = False) -> np.ndarray:
    """调整到 (H,W) 大小；_mask 用最近邻，其他用双线性。"""
    if arr.shape[:2] == (H, W):
        return arr
    order = 0 if is_mask else 1
    aa = False if is_mask else True
    out = sk_resize(arr, (H, W), order=order, preserve_range=True, anti_aliasing=aa).astype(arr.dtype)
    return out


def load_unet_probs(prob_path: Path, H: int, W: int, C: int) -> np.ndarray:
    """
    从 .npy 读取 U_Net softmax 概率。
    支持形状 (H,W,C) 或 (C,H,W) 或 (N,C,H,W) 取第一张。
    返回 float32, shape=(H,W,C)，并做归一化。
    """
    arr = np.load(str(prob_path), allow_pickle=True)
    arr = np.array(arr, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        if arr.shape[0] == C and arr.shape[1] == H and arr.shape[2] == W:
            arr = np.transpose(arr, (1, 2, 0))
        elif arr.shape[0] == H and arr.shape[1] == W and arr.shape[2] == C:
            pass
        else:
            arr = ensure_hw(arr, H, W, is_mask=False)
    else:
        raise ValueError(f"Unsupported unet prob shape: {arr.shape}")

    s = arr.sum(axis=-1, keepdims=True)
    arr = arr / np.maximum(s, 1e-8)
    return np.clip(arr, 1e-8, 1.0)


# ------------------- 超像素 & 图构建 ------------------- #

def build_superpixels_from_octa(
    octa: np.ndarray,
    n_segments: int = 1500,
    compactness: float = 8.0,
    sigma: float = 0.0,
    method: str = "rsg_snic",
    coarse_mask: np.ndarray = None,
    unc_map: np.ndarray = None,
) -> np.ndarray:
    """
    根据 OCTA 生成超像素 sp_index (H,W)，标签连续 Analysis_Gating..N-1。

    method:
      - 'rsg_snic'：使用 RSG+SNIC（推荐）
      - 'slic'    ：使用传统 SLIC
    """
    if method.lower() == "rsg_snic":
        sp = rsg_snic_superpixels_from_octa(
            octa_image=octa,
            coarse_mask=coarse_mask,
            uncertainty=unc_map,
            num_superpixels=n_segments,
            random_state=123,
        )
        return sp.astype(np.int64)

    # fallback: SLIC
    arr = octa.astype(np.float32)
    if arr.ndim == 2:
        arr_ch = arr[..., None]
    else:
        arr_ch = arr

    sp = slic(
        arr_ch,
        n_segments=int(n_segments),
        compactness=float(compactness),
        sigma=float(sigma),
        start_label=0,
        enforce_connectivity=True,
        channel_axis=-1,
    ).astype(np.int64)

    uniq = np.unique(sp)
    lut = {int(u): i for i, u in enumerate(uniq)}
    for u, i in lut.items():
        sp[sp == u] = i
    return sp.astype(np.int64)


def compute_node_stats(sp_index: np.ndarray,
                       img_octa: np.ndarray,
                       unet_mask: np.ndarray,
                       unet_probs: np.ndarray,
                       unc_map: np.ndarray,
                       gt_mask: np.ndarray,
                       num_classes: int,
                       err_thresh: float = 0.2) -> dict:
    """
    计算每个超像素节点的各种统计量：
    - x: 节点特征（不含 GT）
    - y_soft, y_hard: GT 分布 & 多数表决标签
    - highlight_ratio_node, mean_U_node
    - error_ratio_node, incorrect_node, node_weight
    - centers: 超像素质心 (cx,cy)，已经归一化到 [Analysis_Gating,1]
    - 新增: dist_to_unet_cornea_node: 到 U_Net 角膜区域的平均距离（Analysis_Gating~1）
    """
    H, W = sp_index.shape
    N = sp_index.max() + 1
    C = num_classes

    # --- 基本图像特征 ---
    if img_octa.ndim == 2:
        img_gray = img_octa
    else:
        img_gray = img_octa.mean(axis=-1)

    grad = sobel(img_gray)  # Analysis_Gating~1 左右
    hi_thr = float(np.percentile(img_gray, 90.0))
    highlight = (img_gray >= hi_thr).astype(np.float32)

    # --- U_Net 概率 ---
    if unet_probs is None:
        probs_u = np.zeros((H, W, C), dtype=np.float32)
        m = np.clip(unet_mask, 0, C - 1)
        for c in range(C):
            probs_u[..., c] = (m == c).astype(np.float32)
    else:
        probs_u = unet_probs

    # --- 不确定性 ---
    if unc_map is None:
        maxp = probs_u.max(axis=-1)
        unc = 1.0 - maxp
    else:
        unc = unc_map.astype(np.float32)
        if unc.max() > 1.5:
            unc = unc / 255.0
        unc = np.clip(unc, 0.0, 1.0)
        if unc.shape != (H, W):
            unc = ensure_hw(unc, H, W, is_mask=False)

    # --- GT & U_Net hard _mask ---
    gt = np.clip(gt_mask, 0, C - 1).astype(np.int64)
    unet_pred = np.clip(unet_mask, 0, C - 1).astype(np.int64)

    # --- 构造“到 U_Net 角膜区域的距离” ---
    # 假设 类别 1 = 角膜（背景=Analysis_Gating，角膜=1，前房=2，晶状体=3）
    CORNEA_LABEL = 1
    cornea_mask = (unet_mask == CORNEA_LABEL)
    if cornea_mask.any():
        # distance_transform_edt 计算的是到最近“Analysis_Gating”的距离，
        # 所以我们对 ~cornea_mask 做距离变换，即：“非角膜”到角膜的距离
        # cornea 像素位置距离=Analysis_Gating
        dist_raw = distance_transform_edt(~cornea_mask)
    else:
        # 万一某张图没有角膜，统一设一个大值，后面归一化后 ~1
        dist_raw = np.full((H, W), np.sqrt(H * H + W * W), dtype=np.float32)

    # 归一化到 [Analysis_Gating,1]（用图像对角线作为最大距离近似）
    max_d = np.sqrt(H * H + W * W) + 1e-6
    dist_norm = (dist_raw / max_d).astype(np.float32)

    # --- 展平 ---
    flat_sp = sp_index.reshape(-1)
    flat_gray = img_gray.reshape(-1)
    flat_grad = grad.reshape(-1)
    flat_hi = highlight.reshape(-1)
    flat_unc = unc.reshape(-1)
    flat_gt = gt.reshape(-1)
    flat_unet_pred = unet_pred.reshape(-1)
    flat_probs_u = probs_u.reshape(-1, C)
    flat_dist_cornea = dist_norm.reshape(-1)

    node_pixels = [[] for _ in range(N)]
    for idx, nid in enumerate(flat_sp):
        node_pixels[int(nid)].append(idx)

    mean_int = np.zeros(N, dtype=np.float32)
    std_int = np.zeros(N, dtype=np.float32)
    grad_mean = np.zeros(N, dtype=np.float32)
    highlight_ratio = np.zeros(N, dtype=np.float32)
    area = np.zeros(N, dtype=np.float32)
    p_unet_mean = np.zeros((N, C), dtype=np.float32)
    unc_mean = np.zeros(N, dtype=np.float32)
    unc_max = np.zeros(N, dtype=np.float32)
    dist_cornea_node = np.zeros(N, dtype=np.float32)

    y_soft = np.zeros((N, C), dtype=np.float32)
    y_hard = np.zeros(N, dtype=np.int64)

    error_ratio = np.zeros(N, dtype=np.float32)
    incorrect = np.zeros(N, dtype=np.int64)
    node_weight = np.zeros(N, dtype=np.float32)

    centers = np.zeros((N, 2), dtype=np.float32)

    for nid in range(N):
        idxs = node_pixels[nid]
        if len(idxs) == 0:
            continue
        idxs = np.array(idxs, dtype=np.int64)

        area[nid] = float(len(idxs))

        vals = flat_gray[idxs]
        mean_int[nid] = float(vals.mean())
        std_int[nid] = float(vals.std())

        gvals = flat_grad[idxs]
        grad_mean[nid] = float(gvals.mean())

        hvals = flat_hi[idxs]
        highlight_ratio[nid] = float(hvals.mean())

        uvals = flat_unc[idxs]
        unc_mean[nid] = float(uvals.mean())
        unc_max[nid] = float(uvals.max())

        # U_Net 概率均值
        pvals = flat_probs_u[idxs]
        p_unet_mean[nid, :] = pvals.mean(axis=0)

        # 到角膜区域的平均距离
        dvals = flat_dist_cornea[idxs]
        dist_cornea_node[nid] = float(dvals.mean())

        # GT 分布
        gt_vals = flat_gt[idxs]
        for c in range(C):
            y_soft[nid, c] = float((gt_vals == c).mean())
        y_hard[nid] = int(np.argmax(y_soft[nid]))

        # U_Net 错误比例
        un_vals = flat_unet_pred[idxs]
        error_ratio[nid] = float((un_vals != gt_vals).mean())
        incorrect[nid] = int(error_ratio[nid] > err_thresh)

        # 节点权重（粗略考虑错误率 + 不确定性）
        w_err = min(1.0, error_ratio[nid] / 0.5)
        w_unc = min(1.0, max(0.0, unc_mean[nid] - 0.2) / 0.5)
        node_weight[nid] = 0.5 * w_err + 0.5 * w_unc

        # 质心（归一化到 Analysis_Gating~1）
        ys = idxs % W
        xs = idxs // W
        cx = float(xs.mean() / max(H - 1, 1))
        cy = float(ys.mean() / max(W - 1, 1))
        centers[nid, 0] = cx
        centers[nid, 1] = cy

    area_norm = area / float(H * W + 1e-8)

    # --- 构建节点特征 x ---
    # [mean_int, std_int, grad_mean, highlight_ratio, area_norm,
    #  cx, cy, dist_cornea_node, unc_mean, unc_max, p_unet_mean(C 维)]
    feat_list = [
        mean_int[:, None],
        std_int[:, None],
        grad_mean[:, None],
        highlight_ratio[:, None],
        area_norm[:, None],
        centers,                       # (N,2)
        dist_cornea_node[:, None],     # 新增：到角膜区域的距离
        unc_mean[:, None],
        unc_max[:, None],
        p_unet_mean,
    ]
    x = np.concatenate(feat_list, axis=1).astype(np.float32)

    # y_soft 归一化
    y_soft_sum = y_soft.sum(axis=1, keepdims=True)
    y_soft = y_soft / np.maximum(y_soft_sum, 1e-8)

    return dict(
        x=x,
        y_soft=y_soft.astype(np.float32),
        y_hard=y_hard.astype(np.int64),
        highlight_ratio_node=highlight_ratio.astype(np.float32),
        mean_U_node=unc_mean.astype(np.float32),
        centers=centers.astype(np.float32),
        error_ratio_node=error_ratio.astype(np.float32),
        incorrect_node=incorrect.astype(np.int64),
        node_weight=node_weight.astype(np.float32),
        dist_to_unet_cornea=dist_cornea_node.astype(np.float32),
        sp_index=sp_index.astype(np.int64),
    )


def build_edge_index_from_sp(sp_index: np.ndarray) -> np.ndarray:
    """从 superpixel index map 构建邻接边（4-邻接）。"""
    H, W = sp_index.shape
    sp = sp_index
    edges = set()

    for i in range(H):
        for j in range(W):
            nid = int(sp[i, j])
            if j + 1 < W:
                nid_r = int(sp[i, j + 1])
                if nid_r != nid:
                    edges.add((nid, nid_r))
                    edges.add((nid_r, nid))
            if i + 1 < H:
                nid_d = int(sp[i + 1, j])
                if nid_d != nid:
                    edges.add((nid, nid_d))
                    edges.add((nid_d, nid))

    if not edges:
        return np.zeros((2, 0), dtype=np.int64)

    edges = np.array(list(edges), dtype=np.int64)
    edges = np.unique(edges, axis=0)
    edge_index = edges.T
    return edge_index


# ------------------- 文件路径匹配部分 ------------------- #

def extract_prefix(filename: str) -> str:
    """从文件名中提取数字前缀部分，例如 '105.png' -> '105'"""
    match = re.match(r"^\d+", filename)
    if match:
        return match.group(0)
    else:
        raise ValueError(f"文件名 '{filename}' 无法提取数字前缀")


def find_matching_file(dir_path: Path, stem: str, exts):
    """
    在 dir_path 下，根据文件名 stem 和扩展名列表 exts 找对应文件。
    比如 stem='894', exts=['.png']。
    """
    for ext in exts:
        fp = dir_path / f"{stem}{ext}"
        print(f"[DEBUG] 查找文件：{fp}")
        if fp.exists():
            return fp
    return None


# ------------------- 主要处理部分 ------------------- #

def process_single_case(octa_path: Path,
                        gt_path: Path,
                        unet_mask_path: Path,
                        unet_prob_base: Path,
                        unc_base: Path,
                        out_npz_path: Path,
                        C: int,
                        args):
    """处理一张图像：构建超像素图并保存 npz。"""
    octa = load_gray_or_rgb(octa_path)
    gt = load_index_mask(gt_path)
    unet_mask = load_index_mask(unet_mask_path)

    H, W = gt.shape
    if octa.shape[:2] != (H, W):
        octa = ensure_hw(octa, H, W, is_mask=False)
    if unet_mask.shape != (H, W):
        unet_mask = ensure_hw(unet_mask, H, W, is_mask=True)

    unet_probs = None
    if unet_prob_base:
        prob_filename = extract_prefix(octa_path.stem)
        print(f"[DEBUG] 查找 U_Net 概率文件：{prob_filename}")
        prob_path = find_matching_file(unet_prob_base, prob_filename, exts=['_prob.npy'])
        if prob_path:
            print(f"[INFO] 找到 U_Net prob 文件: {prob_path}")
            unet_probs = load_unet_probs(prob_path, H, W, C)
        else:
            print(f"[Warn] 未找到对应的 U_Net prob 文件: {prob_filename}, 将用 hard _mask 近似 one-hot")

    unc = None
    if unc_base:
        unc_filename = extract_prefix(octa_path.stem)
        print(f"[DEBUG] 查找不确定性图文件：{unc_filename}")
        unc_path = find_matching_file(unc_base, unc_filename, exts=['_uncertainty.png'])
        if unc_path:
            print(f"[INFO] 找到不确定性图文件: {unc_path}")
            unc = imageio.imread(str(unc_path))
            if unc.ndim > 2:
                unc = unc[..., 0]
            unc = unc.astype(np.float32)
            if unc.shape != (H, W):
                unc = ensure_hw(unc, H, W, is_mask=False)
        else:
            print(f"[Info] 未找到 {unc_filename} 对应的不确定性图, 使用 1-max_prob 近似")

    coarse_fg = (unet_mask > 0).astype(np.uint8)
    sp = build_superpixels_from_octa(
        octa,
        n_segments=args.n_segments,
        compactness=args.compactness,
        sigma=0.0,
        method=args.sp_method,
        coarse_mask=coarse_fg,
        unc_map=unc,
    )

    stats = compute_node_stats(
        sp_index=sp,
        img_octa=octa,
        unet_mask=unet_mask,
        unet_probs=unet_probs,
        unc_map=unc,
        gt_mask=gt,
        num_classes=C,
        err_thresh=0.2,
    )

    x = stats["x"]
    y_soft = stats["y_soft"]
    y_hard = stats["y_hard"]
    highlight_ratio_node = stats["highlight_ratio_node"]
    mean_U_node = stats["mean_U_node"]
    centers = stats["centers"]
    error_ratio_node = stats["error_ratio_node"]
    incorrect_node = stats["incorrect_node"]
    node_weight = stats["node_weight"]
    dist_to_unet_cornea = stats["dist_to_unet_cornea"]
    sp_index = stats["sp_index"]

    edge_index = build_edge_index_from_sp(sp_index)

    out_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_npz_path),
        x=x.astype(np.float32),
        edge_index=edge_index.astype(np.int64),
        y=y_hard.astype(np.int64),
        y_soft=y_soft.astype(np.float32),
        centers=centers.astype(np.float32),
        highlight_ratio_node=highlight_ratio_node.astype(np.float32),
        mean_U_node=mean_U_node.astype(np.float32),
        error_ratio_node=error_ratio_node.astype(np.float32),
        incorrect_node=incorrect_node.astype(np.int64),
        node_weight=node_weight.astype(np.float32),
        dist_to_unet_cornea=dist_to_unet_cornea.astype(np.float32),
        sp_index=sp_index.astype(np.int64),
    )

    print(f"[OK] {octa_path.name} -> {out_npz_path.name} | "
          f"nodes={x.shape[0]}, edges={edge_index.shape[1]}, feat_dim={x.shape[1]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--octa_img", type=str,default=r"F:\Second_Paper_Code_final_version-comparison\0_UNET_3+_accept\data\images\704.png",help="OCTA 原图 (灰度或RGB) 文件 或 文件夹")

    ap.add_argument("--gt_mask", type=str,default=r"F:\Second_Paper_Code_final_version-comparison\0_UNET_3+_accept\data\masks\704.png",help="GT 掩码 (单通道索引; Analysis_Gating..C-1) 文件 或 文件夹")

    ap.add_argument("--unet_mask", type=str,default=r"F:\Second_Paper_Code_final_version-comparison\0_UNET_3+_accept\SAM_修补结果图\Net_masks\704_mask.png",help="U_Net 预测掩码 (单通道索引; *_mask.png) 文件 或 文件夹")

    ap.add_argument("--unet_prob_npy", type=str,default=r"",help="(可选) U_Net softmax 概率 .npy 文件 或 文件夹")

    ap.add_argument("--unet_unc_png", type=str,default=r"",help="(可选) U_Net 不确定性图 文件 或 文件夹")

    ap.add_argument("--n_segments", type=int, default=1500,
                    help="超像素数量")
    ap.add_argument("--compactness", type=float, default=8.0,
                    help="SLIC 的 compactness 参数（RSG+SNIC 时可忽略）")
    ap.add_argument("--classes", type=int, default=4,
                    help="类别数 C")
    ap.add_argument("--out_npz", type=str,default=r"F:\Second_Paper_Code_final_version-comparison\0_UNET_3+_accept\SAM_修补结果图\npz",help="输出 graph .npz 文件 或 文件夹")

    ap.add_argument("--sp_method", type=str, default="rsg_snic",
                    choices=["rsg_snic", "slic"],
                    help="超像素方法：rsg_snic（推荐）或 slic")

    args = ap.parse_args()

    octa_path = Path(args.octa_img)
    gt_path = Path(args.gt_mask)
    unet_mask_path = Path(args.unet_mask)
    prob_base = Path(args.unet_prob_npy) if args.unet_prob_npy else None
    unc_base = Path(args.unet_unc_png) if args.unet_unc_png else None
    out_npz_path = Path(args.out_npz)

    C = int(args.classes)

    # 单张图像
    if octa_path.is_file():
        if not gt_path.is_file():
            raise ValueError("octa_img 是文件时，gt_mask 也必须是单个文件路径")
        if not unet_mask_path.is_file():
            raise ValueError("octa_img 是文件时，unet_mask 也必须是单个文件路径")

        if out_npz_path.suffix.lower() != ".npz":
            out_npz_path = out_npz_path / f"{octa_path.stem}.npz"

        process_single_case(octa_path,
                            gt_path,
                            unet_mask_path,
                            prob_base,
                            unc_base,
                            out_npz_path,
                            C,
                            args)
        return

    # 批量处理
    if not octa_path.is_dir():
        raise ValueError(f"octa_img 路径既不是文件也不是文件夹: {octa_path}")

    if not gt_path.is_dir() or not unet_mask_path.is_dir():
        raise ValueError("当 octa_img 是文件夹时，gt_mask 和 unet_mask 也必须是文件夹")

    out_npz_path.mkdir(parents=True, exist_ok=True)

    exts_img = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    octa_files = sorted([p for p in octa_path.iterdir()
                         if p.suffix.lower() in exts_img])

    print(f"[Info] 在 {octa_path} 中找到 {len(octa_files)} 张 OCTA 图像")

    for img_path in octa_files:
        stem = img_path.stem

        gt_file = find_matching_file(gt_path, stem, exts_img)
        unet_mask_file = find_matching_file(unet_mask_path, stem + "_mask", exts_img)

        if gt_file is None:
            print(f"[Warn] 找不到 {stem} 对应的 GT, 跳过")
            continue
        if unet_mask_file is None:
            print(f"[Warn] 找不到 {stem}_mask 对应的 U_Net _mask, 跳过")
            continue

        out_file = out_npz_path / f"{stem}.npz"

        process_single_case(
            img_path,
            gt_file,
            unet_mask_file,
            prob_base,
            unc_base,
            out_file,
            C,
            args
        )

    print("[Done] 所有图像处理完成.")


if __name__ == "__main__":
    main()
