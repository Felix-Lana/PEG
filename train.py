import os
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.UNet import UNet_3Plus_DeepSup as UNet
from tqdm import tqdm

# ===========================
# Loss: Multi-class CE + Soft Dice
# ===========================
def soft_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    include_background: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Soft Dice loss for multi-class segmentation.
    - logits: [B,C,H,W]
    - targets: [B,H,W] with values in [Analysis_Gating..C-1]
    """
    probs = F.softmax(logits, dim=1)  # [B,C,H,W]
    # one-hot: [B,H,W,C] -> [B,C,H,W]
    tgt_1h = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()

    if not include_background:
        probs = probs[:, 1:, ...]
        tgt_1h = tgt_1h[:, 1:, ...]

    dims = (0, 2, 3)  # sum over batch and spatial dims
    intersection = (probs * tgt_1h).sum(dims)
    denom = probs.sum(dims) + tgt_1h.sum(dims)
    dice = (2.0 * intersection + eps) / (denom + eps)

    return 1.0 - dice.mean()


class MCEPlusDiceLoss(nn.Module):
    """L = L_MCE + λ * L_Dice"""
    def __init__(
        self,
        num_classes: int,
        dice_weight: float = 0.5,
        include_background: bool = False,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.dice_weight = float(dice_weight)
        self.include_background = bool(include_background)
        self.mce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Deep-supervision heads may differ in resolution; align to GT if needed.
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)

        loss_mce = self.mce(logits, targets)
        loss_dice = soft_dice_loss(
            logits, targets,
            num_classes=self.num_classes,
            include_background=self.include_background,
        )
        return loss_mce + self.dice_weight * loss_dice



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class CorneaDataset(Dataset):

    def __init__(self, image_dir, mask_dir, img_size=None):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

        self.image_files = [
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(exts)
        ]
        self.image_files.sort()  # 保持有序

        if len(self.image_files) == 0:
            raise RuntimeError(f"No images found in {self.image_dir} with extensions {exts}.")

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.img_size is not None:
            img = img.resize(self.img_size, resample=Image.BILINEAR)
            mask = mask.resize(self.img_size, resample=Image.NEAREST)

        img = self.img_transform(img)  # [3,H,W]
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))  # [H,W]

        return img, mask



def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)  # [B,H,W]

        optimizer.zero_grad()
        outputs = model(images)

        if isinstance(outputs, (list, tuple)):
            loss = 0
            for out in outputs:
                loss += criterion(out, masks)
            loss = loss / len(outputs)
            final_output = outputs[-1]
        else:
            loss = criterion(outputs, masks)
            final_output = outputs

        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss



def validate(model, dataloader, criterion, device, num_classes=4):
    model.eval()
    running_loss = 0.0
    correct_pixels = 0
    total_pixels = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", leave=False)
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            if isinstance(outputs, (list, tuple)):
                loss = 0
                for out in outputs:
                    loss += criterion(out, masks)
                loss = loss / len(outputs)
                final_output = outputs[-1]
            else:
                loss = criterion(outputs, masks)
                final_output = outputs

            running_loss += loss.item() * images.size(0)


            preds = torch.argmax(final_output, dim=1)  # [B,H,W]
            correct_pixels += (preds == masks).sum().item()
            total_pixels += masks.numel()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    val_loss = running_loss / len(dataloader.dataset)
    pixel_acc = correct_pixels / total_pixels
    return val_loss, pixel_acc



def main():
    set_seed(42)

    image_dir = r"/data_noise_total\images"
    mask_dir = r"/data_noise_total\masks"
    img_size = (256, 256)


    dataset = CorneaDataset(image_dir=image_dir, mask_dir=mask_dir, img_size=img_size)

    val_ratio = 0.2
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


    batch_size = 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    num_classes = 4

    model = UNet(
        in_channels=3,
        n_classes=num_classes,
        feature_scale=4,
        is_deconv=True,
        is_batchnorm=True
    ).to(device)

    criterion = MCEPlusDiceLoss(num_classes=num_classes, dice_weight=0.5, include_background=False)
    initial_lr = 1.2e-4
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)


    num_epochs = 150
    lr_min = 5e-6
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr_min)

    save_path = r"F:\PaperCode\checkpoints"
    os.makedirs(save_path, exist_ok=True)

    best_val_loss = float("inf")
    best_train_loss = float("inf")
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_pixel_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"[Epoch {epoch:03d}/{num_epochs}] "
              f"LR: {current_lr:.6f} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Pixel Acc: {val_pixel_acc * 100:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_path = os.path.join(save_path, f"unet_epoch_{epoch}_bestValLoss_{val_loss:.4f}.pt")
            torch.save(model.state_dict(), model_save_path)
            print(f"  --> loss best: {model_save_path}")



    print("Finish！")


if __name__ == "__main__":
    main()
