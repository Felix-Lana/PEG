import os
import sys
import torch
from PIL import Image
import numpy as np
from torchvision.transforms import functional as TF
from models.UNet import UNet_3Plus_DeepSup as UNet
from tqdm import tqdm
from typing import Any, Dict, Tuple



NUM_CLASSES: int = 4
IMG_MODE = "RGB"
IMG_SIZE = 256
MEAN = (0.5, 0.5, 0.5)
STD  = (0.5, 0.5, 0.5)
TEMP = 1.0
TEST_IMAGE_PATH = r""
MODEL_PATH: str = r""  # 模型路径
OUT_DIR: str = r""


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
        raise ValueError("NO find state_dict（state_dict / model_state / model）")

    new_state = {}
    for k, v in state.items():
        nk = k[7:] if isinstance(k, str) and k.startswith("module.") else k
        new_state[nk] = v

    try:
        model.load_state_dict(new_state, strict=True)
    except Exception as e_strict:
        print("[WARN] False，strict=False")
        model.load_state_dict(new_state, strict=False)


def load_model(model_path: str) -> Tuple[torch.nn.Module, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        in_channels=3 if IMG_MODE == "RGB" else 1,
        n_classes=NUM_CLASSES,
        feature_scale=2,
        is_deconv=True,
        is_batchnorm=True
    ).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"NO pth or pt：{model_path}")

    ckpt = torch.load(model_path, map_location=device)
    _load_state_dict_flex(model, ckpt)
    model.eval()  # 设置模型为评估模式
    return model, device


def _color_map_for_classes(nc: int) -> np.ndarray:
    base = np.array([
        [0, 0, 0], [0, 255, 0], [255, 0, 0], [0, 0, 255],
        [255, 255, 0], [255, 0, 255], [0, 255, 255], [128, 128, 128],
        [255, 128, 0], [128, 0, 255],
    ], dtype=np.uint8)
    if nc <= len(base):
        return base[:nc]
    rng = np.random.RandomState(42)
    extra = rng.randint(0, 256, size=(nc - len(base), 3), dtype=np.uint8)
    return np.concatenate([base, extra], axis=0)


def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_color_preview(input_tensor: torch.Tensor, pred_mask: torch.Tensor, src_path: str) -> str:
    mean = torch.tensor(MEAN, dtype=input_tensor.dtype, device=input_tensor.device).view(1, -1, 1, 1)
    std = torch.tensor(STD, dtype=input_tensor.dtype, device=input_tensor.device).view(1, -1, 1, 1)
    x = torch.clamp(input_tensor * std + mean, 0.0, 1.0)
    x_np = (x[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    cm = _color_map_for_classes(NUM_CLASSES)
    pm = pred_mask[0].cpu().numpy().astype(np.int64)
    color_mask = cm[pm]
    H, W = x_np.shape[:2]
    canvas = np.zeros((H, W * 2, 3), dtype=np.uint8)
    canvas[:, :W] = x_np
    canvas[:, W:] = color_mask
    ensure_out_dir(OUT_DIR)
    src_name = os.path.splitext(os.path.basename(src_path))[0]
    out_path = os.path.join(OUT_DIR, f"{src_name}_preview.png")
    Image.fromarray(canvas).save(out_path)
    return out_path


def preprocess_image(img_path: str, device: torch.device) -> torch.Tensor:
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"NO find Pic：{img_path}")
    img = Image.open(img_path).convert(IMG_MODE)
    img = TF.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=TF.InterpolationMode.BILINEAR)
    img_t = TF.to_tensor(img)
    if IMG_MODE == "RGB" and img_t.shape[0] != 3:
        raise ValueError(f"exp RGB，but shape {tuple(img_t.shape)}")
    if IMG_MODE == "L" and img_t.shape[0] != 1:
        img_t = img_t[:1, :, :]
    img_t = TF.normalize(img_t, mean=MEAN[:img_t.shape[0]], std=STD[:img_t.shape[0]])
    return img_t.unsqueeze(0).to(device)


def infer_single_image(model: torch.nn.Module, img_t: torch.Tensor) -> Dict[str, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        logits = model(img_t)
        preds = torch.argmax(logits, dim=1)
    return {"logits": logits, "preds": preds}


def test_single_image():
    if MODEL_PATH.startswith("FF:"):
        raise AssertionError("")
    model, device = load_model(MODEL_PATH)
    img_t = preprocess_image(TEST_IMAGE_PATH, device)
    out = infer_single_image(model, img_t)
    assert "preds" in out and out["preds"].ndim == 3, "Output False"
    save_path = save_color_preview(img_t, out["preds"], TEST_IMAGE_PATH)
    print(f"Testing complete. Preview saved to: {save_path}")


if __name__ == "__main__":
    model, device = load_model(MODEL_PATH)
    img_t = preprocess_image(TEST_IMAGE_PATH, device)
    out = infer_single_image(model, img_t)
    save_path = save_color_preview(img_t, out["preds"], TEST_IMAGE_PATH)
    print(f"Done. Preview saved to: {save_path}")
