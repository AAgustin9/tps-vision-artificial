import argparse
import base64
import io
import json
import random
import zlib
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def decode_bitmap_to_full_mask(ann_path: Path) -> np.ndarray:
    ann = json.loads(ann_path.read_text())
    h = ann["size"]["height"]
    w = ann["size"]["width"]
    full_mask = np.zeros((h, w), dtype=np.uint8)

    for obj in ann.get("objects", []):
        bm = obj.get("bitmap")
        if bm is None:
            continue
        origin_x, origin_y = bm["origin"]
        compressed = base64.b64decode(bm["data"])
        decoded = zlib.decompress(compressed)
        rgba = np.array(Image.open(io.BytesIO(decoded)).convert("RGBA"))
        local_mask = (rgba[:, :, 3] > 0).astype(np.uint8)
        mh, mw = local_mask.shape

        y2 = min(origin_y + mh, h)
        x2 = min(origin_x + mw, w)
        y1 = max(origin_y, 0)
        x1 = max(origin_x, 0)

        ly1 = y1 - origin_y
        lx1 = x1 - origin_x
        ly2 = ly1 + (y2 - y1)
        lx2 = lx1 + (x2 - x1)

        if y1 < y2 and x1 < x2:
            full_mask[y1:y2, x1:x2] = np.maximum(full_mask[y1:y2, x1:x2], local_mask[ly1:ly2, lx1:lx2])

    return full_mask


def compute_dataset_mean_std(img_paths):
    s1 = np.zeros(3, dtype=np.float64)
    s2 = np.zeros(3, dtype=np.float64)
    n = 0
    for p in img_paths:
        arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        arr = arr.reshape(-1, 3)
        s1 += arr.sum(axis=0)
        s2 += (arr ** 2).sum(axis=0)
        n += arr.shape[0]
    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


class PolypDataset(Dataset):
    def __init__(self, pairs, image_size=(256, 256), mean=None, std=None, augment=False):
        self.pairs = pairs
        self.image_size = image_size
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, ann_path = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        mask = decode_bitmap_to_full_mask(ann_path)
        mask_img = Image.fromarray(mask * 255)

        image = image.resize(self.image_size, Image.BILINEAR)
        mask_img = mask_img.resize(self.image_size, Image.NEAREST)

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        mask_np = (np.asarray(mask_img, dtype=np.uint8) > 0).astype(np.float32)

        if self.augment:
            if random.random() < 0.5:
                image_np = np.fliplr(image_np).copy()
                mask_np = np.fliplr(mask_np).copy()
            if random.random() < 0.5:
                image_np = np.flipud(image_np).copy()
                mask_np = np.flipud(mask_np).copy()

        image_np = (image_np - self.mean) / self.std
        image_t = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).float()
        return image_t, mask_t


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=(64, 128, 256, 512)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        c = in_channels
        for f in features:
            self.downs.append(DoubleConv(c, f))
            c = f

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(f * 2, f))

        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i // 2]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)

        return self.final(x)


def dice_iou_from_logits(logits, targets, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    targets = targets.float()

    inter = (preds * targets).sum(dim=(1, 2, 3))
    pred_area = preds.sum(dim=(1, 2, 3))
    true_area = targets.sum(dim=(1, 2, 3))
    union = pred_area + true_area - inter

    dice = ((2 * inter + eps) / (pred_area + true_area + eps)).mean().item()
    iou = ((inter + eps) / (union + eps)).mean().item()
    return dice, iou


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    dices, ious = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            d, i = dice_iou_from_logits(logits, y)
            dices.append(d)
            ious.append(i)
    n = len(loader.dataset)
    return total_loss / n, float(np.mean(dices)), float(np.mean(ious))


@dataclass
class RunArtifacts:
    output_dir: Path
    figures_dir: Path
    predictions_dir: Path


def save_preprocessing_examples(pairs, mean, std, out_path, image_size=(256, 256), n=3):
    idxs = np.linspace(0, len(pairs) - 1, n, dtype=int)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = np.array([axes])

    for row, i in enumerate(idxs):
        img_path, ann_path = pairs[i]
        image = Image.open(img_path).convert("RGB")
        mask = decode_bitmap_to_full_mask(ann_path)
        mask_img = Image.fromarray(mask * 255)

        image = image.resize(image_size, Image.BILINEAR)
        mask_img = mask_img.resize(image_size, Image.NEAREST)

        img_np = np.asarray(image, dtype=np.float32) / 255.0
        norm = (img_np - mean) / std
        vis = np.clip((norm * std + mean), 0, 1)

        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title("Original + resize")
        axes[row, 1].imshow(mask_img, cmap="gray")
        axes[row, 1].set_title("Mascara")
        axes[row, 2].imshow(vis)
        axes[row, 2].set_title("Normalizada")
        for c in range(3):
            axes[row, c].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_training_curves(history, out_path):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["val_dice"], label="Val Dice")
    axes[1].set_title("Dice")
    axes[1].legend()

    axes[2].plot(epochs, history["val_iou"], label="Val IoU")
    axes[2].set_title("IoU")
    axes[2].legend()

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def denormalize(img_t, mean, std):
    x = img_t.cpu().numpy().transpose(1, 2, 0)
    x = (x * std) + mean
    return np.clip(x, 0, 1)


def save_test_predictions(model, loader, device, mean, std, out_dir, max_items=8):
    model.eval()
    saved = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float().cpu()

            for b in range(x.shape[0]):
                if saved >= max_items:
                    return
                img = denormalize(x[b].cpu(), mean, std)
                gt = y[b, 0].cpu().numpy()
                pr = preds[b, 0].numpy()

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(img)
                axes[0].set_title("Imagen test")
                axes[1].imshow(gt, cmap="gray")
                axes[1].set_title("Mascara real")
                axes[2].imshow(pr, cmap="gray")
                axes[2].set_title("Prediccion")
                for ax in axes:
                    ax.axis("off")
                plt.tight_layout()
                fig.savefig(out_dir / f"pred_{saved:02d}.png", dpi=150)
                plt.close(fig)
                saved += 1


def main(args):
    set_seed(args.seed)

    root = Path(args.dataset_root)
    img_dir = root / "ds" / "img"
    ann_dir = root / "ds" / "ann"

    img_paths = sorted(img_dir.glob("*.png"), key=lambda p: int(p.stem))
    pairs = [(p, ann_dir / f"{p.name}.json") for p in img_paths]
    pairs = [(i, a) for i, a in pairs if a.exists()]

    train_pairs, test_pairs = train_test_split(pairs, test_size=args.test_size, random_state=args.seed)
    train_pairs, val_pairs = train_test_split(train_pairs, test_size=args.val_size, random_state=args.seed)

    train_img_paths = [p for p, _ in train_pairs]
    mean, std = compute_dataset_mean_std(train_img_paths)

    train_ds = PolypDataset(train_pairs, image_size=(args.size, args.size), mean=mean, std=std, augment=True)
    val_ds = PolypDataset(val_pairs, image_size=(args.size, args.size), mean=mean, std=std, augment=False)
    test_ds = PolypDataset(test_pairs, image_size=(args.size, args.size), mean=mean, std=std, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    out = Path(args.output_dir)
    figs = out / "figures"
    preds = out / "predictions"
    figs.mkdir(parents=True, exist_ok=True)
    preds.mkdir(parents=True, exist_ok=True)

    save_preprocessing_examples(pairs, mean, std, figs / "preprocessing_examples.png", image_size=(args.size, args.size))

    if args.device == "auto":
        backend = "mps" if torch.backends.mps.is_available() else "cpu"
        device = torch.device(backend)
    else:
        device = torch.device(args.device)
    model = UNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running += loss.item() * x.size(0)
            if step % args.log_every == 0:
                print(f"Epoch {epoch:02d} Step {step:03d}/{len(train_loader):03d} loss={loss.item():.4f}", flush=True)

        train_loss = running / len(train_loader.dataset)
        val_loss, val_dice, val_iou = evaluate(model, val_loader, device, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out / "best_unet.pt")

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_dice={val_dice:.4f} | val_iou={val_iou:.4f}"
        )

    model.load_state_dict(torch.load(out / "best_unet.pt", map_location=device))
    test_loss, test_dice, test_iou = evaluate(model, test_loader, device, criterion)

    save_training_curves(history, figs / "training_curves.png")
    save_test_predictions(model, test_loader, device, mean, std, preds, max_items=8)

    summary = {
        "dataset": "CVC-ClinicDB",
        "num_total": len(pairs),
        "num_train": len(train_pairs),
        "num_val": len(val_pairs),
        "num_test": len(test_pairs),
        "image_size": args.size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "test_loss": test_loss,
        "test_dice": test_dice,
        "test_iou": test_iou,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "history.json").write_text(json.dumps(history, indent=2))

    print("\nEntrenamiento y evaluacion finalizados.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "auto"])
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    main(args)
