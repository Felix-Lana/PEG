# -*- coding: utf-8 -*-


import os
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from GAT.GATv2 import GATv2Net




class SuperpixelGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        self.root_dir = Path(root)
        super().__init__(root, transform, pre_transform)
        self.data_list = self.load_graphs_from_npz(self.root_dir)
        self.data, self.slices = self.collate(self.data_list)

    @staticmethod
    def load_graphs_from_npz(root_dir: Path):
        data_list = []
        files = sorted(root_dir.glob("*.npz"))
        print(f"[INFO] 在 {root_dir} 中发现 {len(files)} 个图文件")
        for fp in files:
            arr = np.load(str(fp))
            x = torch.from_numpy(arr["x"]).float()                  # (N, F)
            edge_index = torch.from_numpy(arr["edge_index"]).long() # (2, E)
            y = torch.from_numpy(arr["y"]).long()                   # (N,)

            highlight_ratio = torch.from_numpy(arr["highlight_ratio_node"]).float()
            node_weight = torch.from_numpy(arr["node_weight"]).float()
            dist_to_unet_cornea = torch.from_numpy(arr["dist_to_unet_cornea"]).float()

            data = Data(
                x=x,
                edge_index=edge_index,
                y=y,
                highlight_ratio=highlight_ratio,
                node_weight=node_weight,
                dist_to_unet_cornea=dist_to_unet_cornea,
            )
            data_list.append(data)
        return data_list



def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weight(dataset, num_classes: int):
    counts = np.zeros(num_classes, dtype=np.int64)
    for data in dataset:
        y = data.y.cpu().numpy().astype(np.int64)
        for c in range(num_classes):
            counts[c] += np.sum(y == c)
    counts = counts + 1
    freq = counts / counts.sum()
    w = 1.0 / freq
    w = w / w.mean()
    print("[INFO] class_weight:", w)
    return torch.from_numpy(w.astype(np.float32))


def node_confusion_matrix(all_y_true, all_y_pred, num_classes: int):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(all_y_true, all_y_pred):
        cm[t, p] += 1
    return cm



def train_one_epoch(model, loader, optimizer, device, class_weight, alpha_hl: float):
    model.train()
    total_loss = 0.0
    n_samples = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)   # 你的 GATv2Net 接口：out_mask (N, num_classes)
        y = data.y                             # (N,)

        loss_vec = F.cross_entropy(
            out,
            y,
            reduction="none",
            weight=class_weight.to(device) if class_weight is not None else None,
        )


        w_hl = 1.0 + alpha_hl * data.highlight_ratio.to(device)  # (N,)
        if hasattr(data, "node_weight"):
            w_hl = w_hl * data.node_weight.to(device)

        loss = (loss_vec * w_hl).mean()

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * data.num_nodes
        n_samples += data.num_nodes

    return total_loss / max(1, n_samples)


@torch.no_grad()
def eval_model(model, loader, device, class_weight=None, alpha_hl: float = 0.0):
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_true = []
    all_pred = []

    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)
        y = data.y

        loss_vec = F.cross_entropy(
            out,
            y,
            reduction="none",
            weight=class_weight.to(device) if class_weight is not None else None,
        )
        if alpha_hl > 0 and hasattr(data, "highlight_ratio"):
            w_hl = 1.0 + alpha_hl * data.highlight_ratio.to(device)
            if hasattr(data, "node_weight"):
                w_hl = w_hl * data.node_weight.to(device)
            loss = (loss_vec * w_hl).mean()
        else:
            loss = loss_vec.mean()

        total_loss += float(loss.item()) * data.num_nodes
        n_samples += data.num_nodes

        pred = out.argmax(dim=-1)
        all_true.append(y.cpu().numpy())
        all_pred.append(pred.cpu().numpy())

    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)
    acc = (all_true == all_pred).mean()

    return total_loss / max(1, n_samples), float(acc), all_true, all_pred



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=r"F:\Second_Paper_Code_final_version\GAT\npz", help="npz")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min_lr", type=float, default=2e-7)
    ap.add_argument("--decay_gamma", type=float, default=0.999774)
    ap.add_argument("--hid", type=int, default=64)
    ap.add_argument("--heads1", type=int, default=4)
    ap.add_argument("--heads2", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--test_split", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", type=str, default=r"", help="")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--alpha_hl", type=float, default=1.5, help="")
    ap.add_argument("--num_classes", type=int, default=4)

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    dataset = SuperpixelGraphDataset(args.data_dir)

    N = len(dataset)
    n_val = int(N * args.val_split)
    n_test = int(N * args.test_split)
    n_train = N - n_val - n_test
    train_set, val_set, test_set = random_split(
        dataset,
        lengths=[n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"[INFO] Dataset split: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    class_weight = compute_class_weight(train_set, num_classes=args.num_classes)

    loader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    loader_test = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    sample_data = dataset[0]
    in_channels = sample_data.x.size(1)
    num_classes = args.num_classes


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
        print("[INFO] using the GATv2Net keyword parameters")
    except TypeError:
        model = GATv2Net(
            in_channels,          # in_dim
            args.hid,             # hidden
            num_classes,          # out_dim
            args.heads1,
            args.heads2,
            args.dropout,
        ).to(device)
        print("[INFO] using the GATv2Net position parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_gamma)

    best_val_loss = float("inf")
    best_state_path = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            loader_train,
            optimizer,
            device,
            class_weight=class_weight,
            alpha_hl=args.alpha_hl,
        )

        val_loss, val_acc, _, _ = eval_model(
            model,
            loader_val,
            device,
            class_weight=class_weight,
            alpha_hl=args.alpha_hl,
        )
        test_loss, test_acc, _, _ = eval_model(
            model,
            loader_test,
            device,
            class_weight=class_weight,
            alpha_hl=args.alpha_hl,
        )

        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f} | "
              f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_path = os.path.join(
                args.save_dir,
                f"best_epoch{epoch:04d}_val{val_loss:.6f}.pt"
            )
            torch.save(model.state_dict(), best_state_path)
            print(f"[INFO] New best model saved at epoch {epoch} to {best_state_path}")

        scheduler.step()

    if best_state_path is not None:
        print(f"[INFO] Loading best model from {best_state_path}")
        model.load_state_dict(torch.load(best_state_path, map_location=device))

    _, _, y_val_true, y_val_pred = eval_model(
        model, loader_val, device, class_weight=class_weight, alpha_hl=args.alpha_hl
    )
    _, _, y_test_true, y_test_pred = eval_model(
        model, loader_test, device, class_weight=class_weight, alpha_hl=args.alpha_hl
    )

    cm_val = node_confusion_matrix(y_val_true, y_val_pred, num_classes=num_classes)
    cm_test = node_confusion_matrix(y_test_true, y_test_pred, num_classes=num_classes)

    print("Validation set node-level confusion matrix:")
    print(cm_val)
    print("Test set node-level confusion matrix:")
    print(cm_test)


if __name__ == "__main__":
    main()
