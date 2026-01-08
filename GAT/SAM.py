import argparse
from pathlib import Path
import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from segment_anything import sam_model_registry, SamPredictor

try:
    from scipy import ndimage as ndi
except Exception:
    ndi = None

from GAT.GATv2 import GATv2Net


def build_gat_model(in_channels: int, num_classes: int, args, device: torch.device):

    try:
        model = GATv2Net(
            in_dim=in_channels,
            num_hidden=args.hid,
            num_classes=num_classes,
            num_layers=1,
            heads=[args.heads1, args.heads2],
            feat_drop=args.dropout,
            attn_drop=args.dropout,
        ).to(device)
        print("[INFO]:1")
    except TypeError:
        # __init__(self, in_dim, hid_dim, num_classes, heads1, heads2, dropout)
        model = GATv2Net(
            in_channels,
            args.hid,
            num_classes,
            args.heads1,
            args.heads2,
            args.dropout,
        ).to(device)
        print("[INFO]:2")
    return model



LOCKED_CLASSES = (1, 2, 3)
PALETTE = np.array([
    [0, 0, 0],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [255, 0, 255],
], dtype=np.uint8)



def colorize_index(idx_img: np.ndarray, palette: np.ndarray) -> np.ndarray:
    h, w = idx_img.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    max_idx = min(palette.shape[0], int(idx_img.max()) + 1)
    for i in range(max_idx):
        out[idx_img == i] = palette[i]
    return out


def load_gray_or_index_any(path: Path) -> np.ndarray:
    img = imageio.imread(path)
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if img.shape[2] >= 3:
            # RGB or RGBA -> 灰度
            return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.uint8)
    raise ValueError(f"Unsupported image shape: {img.shape}")


def _resize_probs_to_hw(probs: np.ndarray, target_hw):
    H, W = target_hw
    if probs.shape[0] == H and probs.shape[1] == W:
        return probs
    out = []
    for c in range(probs.shape[2]):
        ch = probs[..., c]
        ch_r = cv2.resize(ch, (W, H), interpolation=cv2.INTER_NEAREST)
        out.append(ch_r)
    out = np.stack(out, axis=-1)
    return out


def load_unet_probs(p_unet_prob: Path,
                    p_unet_pred: Path,
                    num_classes: int,
                    target_hw=None) -> np.ndarray:

    if p_unet_prob is not None and p_unet_prob.exists():
        probs = np.load(str(p_unet_prob))
        if probs.ndim == 3 and probs.shape[0] == num_classes:
            probs = np.transpose(probs, (1, 2, 0))
        elif probs.ndim == 3 and probs.shape[2] == num_classes:
            pass
        else:
            raise ValueError(f"U-Net prob .npy shape invalid: {probs.shape}, num_classes={num_classes}")
        if target_hw is not None:
            probs = _resize_probs_to_hw(probs, target_hw)
    else:
        m = load_gray_or_index_any(p_unet_pred)
        if target_hw is not None and (m.shape[0] != target_hw[0] or m.shape[1] != target_hw[1]):
            m = cv2.resize(m, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
        H, W = m.shape[:2]
        probs = np.zeros((H, W, num_classes), dtype=np.float32)
        for c in range(num_classes):
            probs[..., c] = (m == c).astype(np.float32)

    s = probs.sum(axis=-1, keepdims=True)
    s[s == 0] = 1.0
    probs = probs / s
    return probs.astype(np.float32)



def load_gat_probs_from_npz_and_ckpt(
    npz_path: Path,
    ckpt_path: Path,
    device: torch.device,
    args,
    num_classes: int,
):

    arr = np.load(str(npz_path))
    x_np = arr["x"].astype(np.float32)                  # (N, F)
    edge_index_np = arr["edge_index"].astype(np.int64)  # (2, E)

    x = torch.from_numpy(x_np).to(device)
    edge_index = torch.from_numpy(edge_index_np).to(device)

    in_channels = x_np.shape[1]
    model = build_gat_model(in_channels, num_classes, args, device)
    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        out = model(x, edge_index)   # (N, C)
        node_probs = F.softmax(out, dim=-1).cpu().numpy().astype(np.float32)

    return node_probs, arr["sp_index"].astype(np.int64)  # 也把 sp_index 带出来


def expand_node_probs_to_pixels(node_probs: np.ndarray,
                                sp_index: np.ndarray,
                                strict_sp_id: bool = True) -> np.ndarray:

    N, C = node_probs.shape
    max_id = int(sp_index.max())
    if strict_sp_id:
        if sp_index.min() != 0 or max_id != (N - 1):
            raise ValueError(
                f"[STRICT] sp_index id range [min={sp_index.min()}, max={max_id}] != [Analysis_Gating,{N-1}] from GAT graph."
            )
        idx = sp_index
    else:
        if max_id >= N:
            print(f"[WARN] sp_map max id {max_id} >= num nodes {N}, clipping to N-1.")
        idx = np.clip(sp_index, 0, N - 1)
    return node_probs[idx]



def find_bg_islands_and_cracks(pred_u: np.ndarray,
                               locked_classes=LOCKED_CLASSES) -> np.ndarray:

    fg = np.isin(pred_u, locked_classes)
    bg = ~fg
    island = np.zeros_like(bg, dtype=bool)
    cracks = np.zeros_like(bg, dtype=bool)

    if ndi is not None:
        lbl, n = ndi.label(bg.astype(np.uint8))
        if n > 0:
            ids = np.arange(1, n + 1)
            touch = np.zeros(n + 1, dtype=bool)
            touch[0] = True
            edge_mask = np.zeros_like(bg, dtype=bool)
            edge_mask[0, :] = True
            edge_mask[-1, :] = True
            edge_mask[:, 0] = True
            edge_mask[:, -1] = True
            edge_lbl = lbl[edge_mask]
            touch[edge_lbl] = True
            for i in ids:
                if not touch[i]:
                    island[lbl == i] = True

        fg_dil = ndi.binary_dilation(fg, iterations=1)
        fg_close = ndi.binary_erosion(fg_dil, iterations=1)
        cracks = fg_close & (~fg) & bg

    return (island | cracks) & bg



def get_click_from_user(image_rgb: np.ndarray):

    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    window_name = "SAM"
    points = []
    img_show = img_bgr.copy()

    def redraw():
        img_show[:] = img_bgr
        for (x, y) in points:
            cv2.circle(img_show, (x, y), 4, (0, 0, 255), -1)  # 小红点
        cv2.imshow(window_name, img_show)

    def on_mouse(event, x, y, flags, param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print(f"[INFO] Add click points: (x={x}, y={y})")
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if points:
                removed = points.pop()
                print(f"[INFO] Cancel click point: {removed}")
                redraw()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    redraw()

    print("Right-click to undo the last point. Press any key on the keyboard to end and start the segmentation.")
    while True:
        cv2.imshow(window_name, img_show)
        key = cv2.waitKey(20) & 0xFF
        if key != 255:
            break

    cv2.destroyWindow(window_name)
    print(f"[INFO] find {len(points)} points: {points}")
    return points


def run_sam_on_click(click_img_rgb: np.ndarray,
                     sam_img_rgb: np.ndarray,
                     sam_ckpt: str,
                     model_type: str,
                     device: str,
                     points_xy):

    Hc, Wc = click_img_rgb.shape[:2]
    Hs, Ws = sam_img_rgb.shape[:2]
    if (Hc, Wc) != (Hs, Ws):
        raise ValueError(f"click_image size {Hc, Wc} and SAM size {Hs, Ws} inconformity！")

    if not points_xy:
        raise ValueError("points_xy empty")

    sam = sam_model_registry[model_type](checkpoint=sam_ckpt)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    predictor.set_image(sam_img_rgb)

    input_point = np.array(points_xy, dtype=np.float32)   # (K,2)
    input_label = np.ones(len(points_xy), dtype=np.int32) # (K,)

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    mask = masks[0]   # (H,W), bool
    return mask.astype(bool)


# ------------------- 主流程 ------------------- #

def main():
    ap = argparse.ArgumentParser( description="" )

    ap.add_argument("--unet_pred_png", type=str, default=r"",help="")
    ap.add_argument("--unet_prob_npy", type=str, default="", help="U-Net.npy")
    ap.add_argument("--graph_npz", type=str, default=r"", help="Pic superpixel graph npz（x, edge_index, sp_index）")
    ap.add_argument("--gat_ckpt", type=str, default=r"val0.100143.pt", help="GAT checkpoint")
    ap.add_argument("--out_png", type=str, default=r"",help="")
    ap.add_argument("--classes", type=int, default=4, help="")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--hole_fg_ratio", type=float, default=0.3, help="")
    ap.add_argument("--unet_conf_thresh", type=float, default=1.1, help="")
    ap.add_argument("--hid", type=int, default=64)
    ap.add_argument("--heads1", type=int, default=4)
    ap.add_argument("--heads2", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.2)

    # === SAM 点击相关 ===
    ap.add_argument("--sam_ckpt", type=str,  default=r"sam_.pth",  help="")
    ap.add_argument("--sam_model_type", type=str, default="vit_b",help="")
    ap.add_argument("--sam_image", type=str, default=r"", help="")
    ap.add_argument("--enable_click", type=lambda s: s.lower() in ["true", "1", "yes"], default=True, help="")


    ap.add_argument("--save_color", type=lambda s: s.lower() in ["true", "1", "yes"], default=True)
    ap.add_argument("--save_debug", type=lambda s: s.lower() in ["true", "1", "yes"], default=True)
    ap.add_argument("--strict_sp_id", type=lambda s: s.lower() in ["true", "1", "yes"], default=True)
    ap.add_argument("--strict_lock", type=lambda s: s.lower() in ["true", "1", "yes"], default=True)
    ap.add_argument("--save_probs_npy", action="store_true")

    args = ap.parse_args()
    device = torch.device(args.device)

    p_unet_pred = Path(args.unet_pred_png);  assert p_unet_pred.exists(), f"unet_pred_png not found: {p_unet_pred}"
    p_graph = Path(args.graph_npz);          assert p_graph.exists(),     f"graph_npz not found: {p_graph}"
    p_gat_ckpt = Path(args.gat_ckpt);       assert p_gat_ckpt.exists(),  f"gat_ckpt not found: {p_gat_ckpt}"
    p_unet_prob = Path(args.unet_prob_npy) if args.unet_prob_npy else None
    p_out = Path(args.out_png)

    node_probs, sp_index = load_gat_probs_from_npz_and_ckpt(
        p_graph,
        p_gat_ckpt,
        device=device,
        args=args,
        num_classes=args.classes,
    )
    H_sp, W_sp = sp_index.shape
    target_hw = (H_sp, W_sp)

    probs_u = load_unet_probs(p_unet_prob, p_unet_pred, args.classes, target_hw=target_hw)
    pred_u = np.argmax(probs_u, axis=-1).astype(np.uint8)
    has_soft_unet = p_unet_prob is not None and p_unet_prob.exists()

    pixel_probs_g = expand_node_probs_to_pixels(node_probs, sp_index, strict_sp_id=args.strict_sp_id)
    probs_g = pixel_probs_g.astype(np.float32)
    pred_g = np.argmax(probs_g, axis=-1).astype(np.uint8)

    unet_color = colorize_index(pred_u, PALETTE)

    if args.enable_click:
        points = get_click_from_user(unet_color)

        if args.sam_image:
            sam_img_bgr = cv2.imread(args.sam_image)
            if sam_img_bgr is None:
                raise FileNotFoundError(args.sam_image)
            sam_img_rgb = cv2.cvtColor(sam_img_bgr, cv2.COLOR_BGR2RGB)
            if sam_img_rgb.shape[:2] != unet_color.shape[:2]:
                H, W = unet_color.shape[:2]
                print(f"[WARN] sam_image size {sam_img_rgb.shape[:2]} and UNet/sp_index size {unet_color.shape[:2]} inconformity")
                sam_img_rgb = cv2.resize(sam_img_rgb, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            sam_img_rgb = unet_color

        click_mask = run_sam_on_click(
            click_img_rgb=unet_color,
            sam_img_rgb=sam_img_rgb,
            sam_ckpt=args.sam_ckpt,
            model_type=args.sam_model_type,
            device=args.device,
            points_xy=points,
        )
        print("[INFO] SAM shape:", click_mask.shape)
    else:
        click_mask = None

    bg_priority_all = find_bg_islands_and_cracks(pred_u, locked_classes=LOCKED_CLASSES)

    if args.enable_click and click_mask is not None:
        bg_priority = (pred_u == 0) & click_mask
        print(f"[INFO] Click mode：Number of candidate background pixels: {bg_priority.sum()}")
    else:
        bg_priority = bg_priority_all
        print(f"[INFO] No SAM click, perform automatic full-image fusion and the number of candidate pixels: {bg_priority.sum()}")

    pred_f = pred_u.copy()

    if ndi is not None:
        lbl_hole, n_hole = ndi.label(bg_priority.astype(np.uint8))
        print(f"[INFO] The number of connected regions of the small holes that need to be processed: {n_hole}")

        for comp_id in range(1, n_hole + 1):
            comp_mask = (lbl_hole == comp_id)
            if comp_mask.sum() == 0:
                continue

            if has_soft_unet:
                comp_probs_u = probs_u[comp_mask]  # (P, C)
                comp_conf_u = np.mean(np.max(comp_probs_u, axis=-1))
                if comp_conf_u > args.unet_conf_thresh:
                    continue

            comp_probs_g = probs_g[comp_mask]       # (P, C)
            p_fg_pixels = 1.0 - comp_probs_g[..., 0]   # 1 - p_g(bg)
            fg_ratio = float(np.mean(p_fg_pixels))
            if fg_ratio < args.hole_fg_ratio:
                continue

            dil = ndi.binary_dilation(comp_mask, iterations=1)
            ring = dil & (~comp_mask)

            ring_labels = pred_u[ring]
            ring_locked = ring_labels[np.isin(ring_labels, LOCKED_CLASSES)]

            dominant = None
            if ring_locked.size > 0:
                uniq, counts = np.unique(ring_locked, return_counts=True)
                dominant = int(uniq[np.argmax(counts)])
            else:
                dil2 = ndi.binary_dilation(comp_mask, iterations=3)
                neigh = dil2 & (~comp_mask)
                neigh_labels = pred_u[neigh]
                neigh_fg = neigh_labels[neigh_labels != 0]
                if neigh_fg.size > 0:
                    uniq, counts = np.unique(neigh_fg, return_counts=True)
                    dominant = int(uniq[np.argmax(counts)])
                else:
                    continue

            pred_f[comp_mask] = dominant
    else:
        pred_f[bg_priority] = pred_g[bg_priority]

    if args.enable_click and click_mask is not None and ndi is not None:
        labels_in_click = pred_f[click_mask]
        labels_fg = labels_in_click[np.isin(labels_in_click, LOCKED_CLASSES)]
        if labels_fg.size > 0:
            uniq, counts = np.unique(labels_fg, return_counts=True)
            click_dominant = int(uniq[np.argmax(counts)])
            print(f"[INFO] Click on the main foreground category of the area: {click_dominant}")
        else:
            click_dominant = None

        if click_dominant is not None:
            mis_priority = (
                click_mask &
                np.isin(pred_f, LOCKED_CLASSES) &
                (pred_f != click_dominant)
            )
            print(f"[INFO] Number of pixels suspected to be wrongly classified in the clicked area: {mis_priority.sum()}")

            if mis_priority.any():
                lbl_mis, n_mis = ndi.label(mis_priority.astype(np.uint8))
                print(f"[INFO] The number of connected regions for the errors to be processed: {n_mis}")

                for comp_id in range(1, n_mis + 1):
                    comp_mask = (lbl_mis == comp_id)
                    if comp_mask.sum() == 0:
                        continue

                    cur_labels = pred_f[comp_mask]
                    u, c = np.unique(cur_labels, return_counts=True)
                    cur_class = int(u[np.argmax(c)])

                    if cur_class == click_dominant:
                        continue

                    comp_probs_g = probs_g[comp_mask]  # (P, C)
                    mean_dom = float(np.mean(comp_probs_g[:, click_dominant]))
                    mean_cur = float(np.mean(comp_probs_g[:, cur_class]))

                    print(f"[DEBUG] 区域 {comp_id}: cur={cur_class}, mean_p_dom={mean_dom:.3f}, mean_p_cur={mean_cur:.3f}")

                    if mean_dom <= mean_cur:
                        continue

                    if has_soft_unet:
                        comp_probs_u = probs_u[comp_mask]  # (P, C)
                        mean_conf_cur = float(np.mean(comp_probs_u[:, cur_class]))
                        if mean_conf_cur > 0.9:
                            continue

                    pred_f[comp_mask] = click_dominant
        else:
            print("[INFO] no foreground category in the clicked area.")

    use_gat = (pred_f != pred_u)

    purple_color = np.array([185, 132, 213], dtype=np.uint8)

    if ndi is not None:
        use_gat_binary = use_gat.astype(np.uint8)
        eroded = ndi.binary_erosion(use_gat_binary, iterations=1)
        border = use_gat & (~eroded)
    else:
        H, W = use_gat.shape
        border = np.zeros_like(use_gat, dtype=bool)
        for yy in range(1, H - 1):
            for xx in range(1, W - 1):
                if not use_gat[yy, xx]:
                    continue
                if not use_gat[yy - 1, xx] or not use_gat[yy + 1, xx] \
                        or not use_gat[yy, xx - 1] or not use_gat[yy, xx + 1]:
                    border[yy, xx] = True

    color_fusion = colorize_index(pred_f, PALETTE)
    color_border = color_fusion.copy()
    color_border[border] = purple_color

    imageio.imwrite(p_out, pred_f)
    print(f"[OK] saved: {p_out.resolve()}")

    if args.save_color:
        imageio.imwrite(p_out.with_name(p_out.stem + "_color.png"), color_border)

    if args.save_debug:
        imageio.imwrite(p_out.with_name(p_out.stem + "_changed.png"),
                        ((pred_f != pred_u).astype(np.uint8) * 255))
        imageio.imwrite(p_out.with_name(p_out.stem + "_unet.png"), pred_u)
        imageio.imwrite(p_out.with_name(p_out.stem + "_gat.png"), pred_g)

    print("[DONE] Accomplish。")


if __name__ == "__main__":
    main()
