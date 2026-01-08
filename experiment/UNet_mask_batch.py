# -*- coding: utf-8 -*-


import os
import sys
from typing import Tuple, Dict, Any, List, Optional
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision.transforms import functional as TF
from models.UNet import UNet_3Plus_DeepSup as UNet


NUM_CLASSES: int = 4 
MODEL_PATH: str = r"F:\PaperCode\checkpoints\UNet.pt"  
INPUT_DIR: str = r"F:\PaperCode\Pic\data\image"
OUT_DIR: str = r"/Pic/data/out_mask"
IMG_MODE = "RGB" 
IMG_SIZE = 256
MEAN = (0.5, 0.5, 0.5)
STD  = (0.5, 0.5, 0.5)
TEMP = 1.0


RECURSIVE = False                
KEEP_ORIGINAL_SIZE = True       


SAVE_NPY = False                 

AUTO_SUPPRESS_CLASS3 = True       
CLASS3_ID = 3
CLASS3_MIN_PIXELS = 50          
CLASS3_MIN_RATIO  = 0.0005      


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.append(CUR_DIR)

def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _load_state_dict_flex(model: torch.nn.Module, ckpt_obj: Any) -> None:
    state = None
    if isinstance(ckpt_obj, dict) and ckpt_obj and all(isinstance(k, str) for k in ckpt_obj.keys()):
        if any(torch.is_tensor(v) for v in ckpt_obj.values()):
            state = ckpt_obj
    if state is None and isinstance(ckpt_obj, dict):
        for key in ("state_dict", "model_state", "model"):
            maybe = ckpt_obj.get(key, None)
            if isinstance(maybe, dict):
                state = maybe
                break
    if state is None:
        raise ValueError("No find state_dict)

    new_state = {}
    for k, v in state.items():
        nk = k[7:] if isinstance(k, str) and k.startswith("module.") else k
        new_state[nk] = v

    try:
        model.load_state_dict(new_state, strict=True)
    except Exception:
        print("[WARN] Strict matching failed")
        model.load_state_dict(new_state, strict=False)


def load_model(model_path: str) -> Tuple[torch.nn.Module, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(
        in_channels=3,
        n_classes=4,
        feature_scale=4,
        is_deconv=True,
        is_batchnorm=True
    ).to(device)


    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No find pt/pth：{model_path}")

    ckpt = torch.load(model_path, map_location=device)
    _load_state_dict_flex(model, ckpt)
    model.eval()
    return model, device


def list_images(input_dir: str, recursive: bool = False) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = []
    if recursive:
        for root, _, files in os.walk(input_dir):
            for fn in files:
                if os.path.splitext(fn)[1].lower() in exts:
                    paths.append(os.path.join(root, fn))
    else:
        for fn in os.listdir(input_dir):
            p = os.path.join(input_dir, fn)
            if os.path.isfile(p) and os.path.splitext(fn)[1].lower() in exts:
                paths.append(p)

    paths.sort()
    return paths


def preprocess_image(img_path: str, device: torch.device) -> Tuple[torch.Tensor, Tuple[int, int], str]:
    img = Image.open(img_path).convert(IMG_MODE)
    w, h = img.size
    orig_hw = (h, w)
    img_rs = TF.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=TF.InterpolationMode.BILINEAR)
    img_t = TF.to_tensor(img_rs)
    if IMG_MODE == "RGB" and img_t.shape[0] != 3:
        raise ValueError(f"exp RGB: {tuple(img_t.shape)}")
    if IMG_MODE == "L" and img_t.shape[0] != 1:
        img_t = img_t[:1, :, :]
    img_t = TF.normalize(img_t, mean=MEAN[:img_t.shape[0]], std=STD[:img_t.shape[0]])
    return img_t.unsqueeze(0).to(device), orig_hw, img_path


def infer_single_image(model: torch.nn.Module, img_t: torch.Tensor) -> Dict[str, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        logits = model(img_t)
        logits = logits / TEMP
        probs = torch.softmax(logits, dim=1)          # [B,C,H,W]
        preds = torch.argmax(probs, dim=1)            # [B,H,W]
    return {"logits": logits, "probs": probs, "preds": preds}


def infer_single_image(model: torch.nn.Module, img_t: torch.Tensor) -> Dict[str, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        out = model(img_t)

        if isinstance(out, (tuple, list)):
            logits = None
            for x in out:
                if torch.is_tensor(x) and x.dim() == 4:
                    logits = x
                    break
            if logits is None:
                raise TypeError(f"model output is tuple/list but no 4D tensor found: {[type(x) for x in out]}")
        elif isinstance(out, dict):
            logits = out.get("logits", None)
            if logits is None:
                for k in ("out_Heat map_integration", "pred", "seg"):
                    if k in out and torch.is_tensor(out[k]):
                        logits = out[k]
                        break
            if logits is None:
                raise TypeError(f"model output is dict but no logits-like key found: {list(out.keys())}")
        else:
            logits = out
        logits = logits / float(TEMP)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
    return {"logits": logits, "probs": probs, "preds": preds}



def maybe_suppress_class3(preds: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    if not AUTO_SUPPRESS_CLASS3 or NUM_CLASSES < 4:
        return preds

    b, h, w = preds.shape
    class3_area = (preds == CLASS3_ID).sum().item()
    min_area = max(CLASS3_MIN_PIXELS, int(CLASS3_MIN_RATIO * h * w))

    if class3_area < min_area:
        probs_3 = probs[:, :3, :, :]
        preds_3 = torch.argmax(probs_3, dim=1)  # [B,H,W]
        preds = torch.where(preds == CLASS3_ID, preds_3, preds)

    return preds


def resize_mask_to_original(preds: torch.Tensor, orig_hw: Tuple[int, int]) -> torch.Tensor:
    if not KEEP_ORIGINAL_SIZE:
        return preds
    oh, ow = orig_hw
    x = preds.unsqueeze(1).float()  # [B,B1,H,W]
    x = F.interpolate(x, size=(oh, ow), mode="nearest")
    return x.squeeze(1).long()


def save_index_mask(pred_mask: torch.Tensor, src_path: str, out_dir: str, save_npy: bool = False) -> str:
    pm = pred_mask[0].detach().cpu().numpy().astype(np.uint8)
    ensure_out_dir(out_dir)
    src_name = os.path.splitext(os.path.basename(src_path))[0]
    out_png = os.path.join(out_dir, f"{src_name}_mask.png")
    Image.fromarray(pm, mode="L").save(out_png)

    if save_npy:
        out_npy = os.path.join(out_dir, f"{src_name}_mask.npy")
        np.save(out_npy, pm)

    return out_png


def run_folder():
    if not os.path.isdir(INPUT_DIR):
        raise NotADirectoryError(f"INPUT_DIR 不是文件夹：{INPUT_DIR}")

    model, device = load_model(MODEL_PATH)
    img_paths = list_images(INPUT_DIR, recursive=RECURSIVE)
    if len(img_paths) == 0:
        raise RuntimeError(f" {INPUT_DIR} :No find。")
    try:
        from tqdm import tqdm
        iterator = tqdm(img_paths, desc="Infer", ncols=100)
    except Exception:
        iterator = img_paths

    ok, failed = 0, 0
    for p in iterator:
        try:
            img_t, orig_hw, _ = preprocess_image(p, device)
            out = infer_single_image(model, img_t)
            preds = out["preds"]
            probs = out["probs"]
            preds = maybe_suppress_class3(preds, probs)
            preds = resize_mask_to_original(preds, orig_hw)
            out_dir = OUT_DIR
            if RECURSIVE:
                rel = os.path.relpath(os.path.dirname(p), INPUT_DIR)
                out_dir = os.path.join(OUT_DIR, rel)
            save_index_mask(preds, p, out_dir=out_dir, save_npy=SAVE_NPY)
            ok += 1
        except Exception as e:
            failed += 1
            print(f"[FAIL] {p} -> {repr(e)}")

    print(f"\nDone. total={len(img_paths)}, ok={ok}, failed={failed}")
    print(f"Masks saved to: {OUT_DIR}")


if __name__ == "__main__":
    run_folder()
