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


IMG_MODE = "RGB"
IMG_SIZE = 256
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)
TEMP = 1.0

TEST_IMAGE_PATH = r""


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

    model_state = model.state_dict()
    filtered_state = {k: v for k, v in new_state.items() if k in model_state}

    try:
        model.load_state_dict(filtered_state, strict=False)
    except Exception as e:
        print("[WARN] find error")
        print(f"Error: {e}")
        model.load_state_dict(filtered_state, strict=False)


def load_model(model_path: str) -> Tuple[torch.nn.Module, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        in_ch=3 if IMG_MODE == "RGB" else 1,
        out_ch=NUM_CLASSES,
        enable_dropout=False,
        dropout_p=0.1,
    ).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No pth or pt：{model_path}")

    ckpt = torch.load(model_path, map_location=device)
    _load_state_dict_flex(model, ckpt)
    model.eval()
    return model, device


def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def is_image_file(fname: str) -> bool:
    ext = os.path.splitext(fname)[1].lower()
    return ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def preprocess_image(img_path: str, device: torch.device) -> torch.Tensor:
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"NO find pic：{img_path}")
    img = Image.open(img_path).convert(IMG_MODE)
    img = TF.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BILINEAR)
    img_t = TF.to_tensor(img)

    # 通道检查
    if IMG_MODE == "RGB" and img_t.shape[0] != 3:
        raise ValueError(f"exp RGB，but shape {tuple(img_t.shape)}")
    if IMG_MODE == "L" and img_t.shape[0] != 1:
        img_t = img_t[:1, :, :]

    img_t = TF.normalize(img_t, mean=MEAN[:img_t.shape[0]], std=STD[:img_t.shape[0]])
    return img_t.unsqueeze(0).to(device)


def infer_single_image(model: torch.nn.Module, img_t: torch.Tensor) -> Dict[str, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        logits, probs, entropy_norm, d2 = model(
            img_t, return_feature_map=True, temperature=TEMP
        )
        preds = torch.argmax(logits, dim=1)  # [B,H,W]
    return {"logits": logits, "probs": probs, "entropy": entropy_norm, "feature": d2, "preds": preds}


def save_probability_npy(probs: torch.Tensor, src_path: str) -> str:

    probs = probs[0].detach().cpu().numpy()  # 取出 batch 中的第一个样本
    ensure_out_dir(OUT_DIR)
    src_name = os.path.splitext(os.path.basename(src_path))[0]
    out_npy = os.path.join(OUT_DIR, f"{src_name}_prob.npy")
    np.save(out_npy, probs)
    return out_npy


def save_uncertainty_image(uncertainty: torch.Tensor, src_path: str) -> str:

    uncertainty = (uncertainty[0].detach().cpu().numpy() * 255).astype(np.uint8)
    ensure_out_dir(OUT_DIR)
    src_name = os.path.splitext(os.path.basename(src_path))[0]
    out_png = os.path.join(OUT_DIR, f"{src_name}_uncertainty.png")
    Image.fromarray(uncertainty, mode="L").save(out_png)
    return out_png


def infer_folder(
        img_dir: str,
        model: torch.nn.Module,
        device: torch.device,
        save_npy: bool = True,
) -> None:

    if not os.path.isdir(img_dir):
        raise NotADirectoryError(f"Not a valid folder path：{img_dir}")

    ensure_out_dir(OUT_DIR)

    files = sorted(os.listdir(img_dir))
    if not files:
        print(f"[WARN] The folder is empty：{img_dir}")
        return

    cnt = 0
    for fname in files:
        if not is_image_file(fname):
            continue

        src_path = os.path.join(img_dir, fname)
        try:
            img_t = preprocess_image(src_path, device)
            out = infer_single_image(model, img_t)

            prob_save_path = save_probability_npy(out["probs"], src_path)

            unc_save_path = save_uncertainty_image(out["entropy"], src_path)

            cnt += 1
            print(f"[{cnt}] {src_path} -> {prob_save_path}, {unc_save_path}")
        except Exception as e:
            print(f"[ERROR] 处理 {src_path} 时出错: {e}")

    print(f"文件夹推理完成，共处理 {cnt} 张图像。输出目录：{OUT_DIR}")


if __name__ == "__main__":

    model, device = load_model(MODEL_PATH)
    input_path = TEST_IMAGE_PATH

    if os.path.isdir(input_path):
        print(f"[INFO] batch processing：{input_path}")
        infer_folder(input_path, model, device, save_npy=True)
    else:
        print(f"[INFO] single Pic processing：{input_path}")
        img_t = preprocess_image(input_path, device)
        out = infer_single_image(model, img_t)

        prob_save_path = save_probability_npy(out["probs"], input_path)
        unc_save_path = save_uncertainty_image(out["entropy"], input_path)

        print(f"Done. Probability saved to: {prob_save_path}, Uncertainty saved to: {unc_save_path}")
