"""
Trains CarClassifierResNet (ResNet50, 6 classes) to match accidentModel.py.
Saves checkpoint as {"model": state_dict} — matches load_accident_model().

Dataset expected layout:
  datasets/accident/<split>/<class_name>/image.jpg
  where split ∈ {train, val, test} and class_name ∈ {0,1,2,3,4,5}
  (or any 6 folder names — they become class_0 … class_5 alphabetically)

Usage:
    python scripts/train_accident.py
"""

import argparse, shutil
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Import the model definition from the project
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "deployment"))
from accidentModel import CarClassifierResNet


NUM_CLASSES  = 6
BATCH_SIZE   = 32
EPOCHS       = 30
LR           = 1e-4
WEIGHT_DECAY = 1e-4
DATA_ROOT    = Path("datasets/accident")
OUTPUT_PATH  = Path("models/accident/accident_model.pth")


# ── Transforms ───────────────────────────────────────────────────────────────

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Training loop ─────────────────────────────────────────────────────────────

def accuracy(outputs, targets):
    preds = outputs.argmax(dim=1)
    return (preds == targets).float().mean().item()


def run_epoch(model, loader, criterion, optimizer, device, *, train: bool):
    model.train() if train else model.eval()
    total_loss, total_acc = 0.0, 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            total_acc  += accuracy(outputs, labels)
    n = len(loader)
    return total_loss / n, total_acc / n


def find_dataset_splits(root: Path):
    """Auto-detect train/val splits in the dataset folder."""
    # Auto-resolve nested 'data' folder if it exists
    if (root / "data").exists() and any((root / "data" / split).exists() for split in ["train", "Train", "training"]):
        root = root / "data"

    for split in ["train", "Train", "training"]:
        if (root / split).exists():
            train_dir = root / split
            break
    else:
        train_dir = root   # flat layout

    for split in ["val", "valid", "validation", "Val"]:
        if (root / split).exists():
            val_dir = root / split
            break
    else:
        # Create an 80/20 split if no val folder exists
        val_dir = None

    return train_dir, val_dir


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_dir, val_dir = find_dataset_splits(DATA_ROOT)
    train_ds = datasets.ImageFolder(str(train_dir), transform=TRAIN_TRANSFORM)
    n_classes = len(train_ds.classes)
    print(f"Classes ({n_classes}): {train_ds.classes}")

    if val_dir is None:
        # 80/20 random split
        n_val = int(0.2 * len(train_ds))
        n_tr  = len(train_ds) - n_val
        train_ds, val_ds = torch.utils.data.random_split(
            train_ds, [n_tr, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        val_ds.dataset.transform = VAL_TRANSFORM
    else:
        val_ds = datasets.ImageFolder(str(val_dir), transform=VAL_TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    model = CarClassifierResNet(num_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    # Phase 1 — only train the head and layer4
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    patience_counter = 0
    PATIENCE = 8

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        scheduler.step()

        print(f"Epoch {epoch:03d}  "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f}  "
              f"val loss={vl_loss:.4f} acc={vl_acc:.4f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            patience_counter = 0
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            # Safe-retrieve classes attribute from dataset (handles direct ImageFolder and Subset wrappers)
            dataset_obj = train_loader.dataset
            classes_list = None
            if hasattr(dataset_obj, "classes"):
                classes_list = dataset_obj.classes
            elif hasattr(dataset_obj, "dataset") and hasattr(dataset_obj.dataset, "classes"):
                classes_list = dataset_obj.dataset.classes

            torch.save({"model": model.state_dict(),
                        "classes": classes_list,
                        "epoch": epoch,
                        "val_acc": vl_acc},
                       OUTPUT_PATH)
            print(f"  [OK] Saved best model (val_acc={vl_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"\n[OK] Accident model -> {OUTPUT_PATH}  (best val acc = {best_val_acc:.4f})")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch-size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-4)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
