#    Train a baseline image classifier using torchvision's ResNet-18 on the prepared
#    train/val split. Saves the best checkpoint (by val accuracy) together with the
#    class list for later inference.

import os, time
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# -------------------- CONFIG --------------------

DATA_DIR = "data/processed/cv_split" # Root with 'train/' and 'val/' subfolders
BATCH_SIZE = 16                      # Batch size for training/validation
EPOCHS = 5                           # Number of epochs
LR = 1e-3                            # Learning rate for Adam
OUT_DIR = "models"                   # Where to save checkpoints

# -------------------- DEVICE SELECTOR --------------------

def get_device():
    # Pick the best available accelerator: Apple MPS (Metal), CUDA, or CPU
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# -------------------- DATALOADERS --------------------

def get_loaders(data_dir: str, size: int = 224) -> Tuple[DataLoader, DataLoader]:
    #    Create train/val DataLoaders with basic augmentations for train and
    #    deterministic resize for validation.
    #
    #   Parameters
    #   ----------
    #   data_dir : str
    #       Root folder with 'train' and 'val' subfolders (class subdirs inside).
    #   size : int
    #       Target image size for Resize (height, width).
    #
    #   Returns
    #   -------
    #   train_dl : DataLoader
    #   val_dl   : DataLoader
    #   classes  : list[str]  (class names in deterministic order)

    train_tfms = transforms.Compose([
        transforms.Resize((size, size)),            # Scaling to a fixed size
        transforms.RandomHorizontalFlip(),          # Random horizontal reflection
        transforms.RandomRotation(10),              # A small random twist
        transforms.ToTensor(),                      # In tensor [0,1]
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])

    # num_workers=2 is a safe default; adjust based on your CPU

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tfms)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_tfms)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return train_dl, val_dl, train_ds.classes

# -------------------- MODEL --------------------

def build_model(num_classes: int):
    #   Load a ResNet-18 with ImageNet weights and replace the final FC layer
    #   with a fresh Linear head of size (in_features -> num_classes).
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    #   Fine-tune the whole network (all params trainable). For feature-extract, you could freeze all but the final layer
    for param in model.parameters():
        param.requires_grad = True  # fine-tune весь

    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f, num_classes)
    return model

# -------------------- EVALUATION --------------------

def evaluate(model, dl, device):
    # Compute top-1 accuracy on the given DataLoader.
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / max(1,total)

# -------------------- TRAINING LOOP --------------------

def main():
    #   Orchestrate training:
    #    - build loaders/model/optimizer/loss
    #    - train for EPOCHS
    #    - evaluate each epoch
    #    - save the best checkpoint with class names

    device = get_device()
    print("Device:", device)

    train_dl, val_dl, classes = get_loaders(DATA_DIR)
    print("Classes:", classes)

    model = build_model(num_classes=len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    best_acc, best_path = 0.0, None
    for epoch in range(1, EPOCHS+1):
        model.train()
        t0 = time.time()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
        train_time = time.time() - t0
        
        # Post-Epoch Evaluation
        val_acc = evaluate(model, val_dl, device)
        print(f"Epoch {epoch}/{EPOCHS} | val_acc={val_acc:.3f} | time={train_time:.1f}s")

        # Save best checkpoint (state_dict + class names)
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(OUT_DIR, f"baseline_resnet18_acc{best_acc:.3f}.pt")
            torch.save({
                "state_dict": model.state_dict(),
                "classes": classes
            }, best_path)
            print("Saved:", best_path)

    print("Best val_acc:", best_acc)

if __name__ == "__main__":
    main()
