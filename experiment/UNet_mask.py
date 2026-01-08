# -*- coding: utf-8 -*-

import os
import sys
from typing import Tuple, Dict, Any
import torch
from PIL import Image
import numpy as np
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from models.UNet import UNet_3Plus_DeepSup as UNet




NUM_CLASSES: int = 4
MODEL_PATH: str = r""
OUT_DIR: str = r""
TEST_IMAGE_PATH = r""

IMG_MODE = "RGB"
IMG_SIZE = 256
MEAN = (0.5, 0.5, 0.5)
STD  = (0.5, 0.5, 0.5)
TEMP = 1.0



CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.append(CUR_DIR)


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
        raise ValueError("NO: state_dict（state_dict / model_state / model）")

    new_state = {}
    for k, v in state.items():
        nk = k[7:] if isinstance(k, str) and k.startswith("module.") else k
        new_state[nk] = v

    try:
        model.load_state_dict(new_state, strict=True)
    except Exception:
        print("[WARN] False，strict=False")
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
        raise FileNotFoundError(f"NO pth/pt：{model_path}")

    ckpt = torch.load(model_path, map_location=device)
    _load_state_dict_flex(model, ckpt)
    model.eval()
    return model, device


def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def preprocess_image(img_path: str, device: torch.device) -> torch.Tensor:
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"NO Pic：{img_path}")
    img = Image.open(img_path).convert(IMG_MODE)
    img = TF.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=TF.InterpolationMode.BILINEAR)
    img_t = TF.to_tensor(img)

    # 通道检查
    if IMG_MODE == "RGB" and img_t.shape[0] != 3:
        raise ValueError(f"exp RGB，but shape: {tuple(img_t.shape)}")
    if IMG_MODE == "L" and img_t.shape[0] != 1:
        img_t = img_t[:1, :, :]

    img_t = TF.normalize(img_t, mean=MEAN[:img_t.shape[0]], std=STD[:img_t.shape[0]])
    return img_t.unsqueeze(0).to(device)


def infer_single_image(model: torch.nn.Module, img_t: torch.Tensor) -> Dict[str, torch.Tensor]:

    model.eval()
    with torch.no_grad():
        logits = model(img_t)       # [B, NUM_CLASSES, H, W]
        logits = logits / TEMP

        probs = torch.softmax(logits, dim=1)  # [B, NUM_CLASSES, H, W]

        preds = torch.argmax(probs, dim=1)    # [B, H, W]

        probs_safe = probs.clamp_min(1e-8)
        entropy = -(probs_safe * probs_safe.log()).sum(dim=1)  # [B, H, W]
        entropy_norm = entropy / np.log(NUM_CLASSES)

    return {
        "logits": logits,
        "probs": probs,
        "entropy": entropy_norm,
        "feature": None,
        "preds": preds,
    }


def save_index_mask(pred_mask: torch.Tensor, src_path: str, save_npy: bool = False) -> str:

    pm = pred_mask[0].detach().cpu().numpy().astype(np.uint8)
    ensure_out_dir(OUT_DIR)
    src_name = os.path.splitext(os.path.basename(src_path))[0]
    out_png = os.path.join(OUT_DIR, f"{src_name}_mask.png")
    Image.fromarray(pm, mode="L").save(out_png)

    if save_npy:
        out_npy = os.path.join(OUT_DIR, f"{src_name}_mask.npy")
        np.save(out_npy, pm)
    return out_png


# ========= PyTest =========
def test_single_image():
    if MODEL_PATH.startswith("FF:"):
        raise AssertionError("MODEL_PATH 'FF:' 'F:'。")
    model, device = load_model(MODEL_PATH)
    img_t = preprocess_image(TEST_IMAGE_PATH, device)
    out = infer_single_image(model, img_t)
    assert "preds" in out and out["preds"].ndim == 3, "False"
    save_path = save_index_mask(out["preds"], TEST_IMAGE_PATH, save_npy=False)
    print(f"Testing complete. Mask saved to: {save_path}")


if __name__ == "__main__":
    model, device = load_model(MODEL_PATH)
    img_t = preprocess_image(TEST_IMAGE_PATH, device)
    out = infer_single_image(model, img_t)
    save_path = save_index_mask(out["preds"], TEST_IMAGE_PATH, save_npy=False)
    print(f"Done. Mask saved to: {save_path}")
