#    1Train a baseline image classifier using torchvision's ResNet-18 on the prepared
#    train/val split. Saves the best checkpoint (by val accuracy) together with the
#    class list for later inference.

import os, time
from pathlib import Path
from typing import Tuple
from collections import Counter

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# -------------------- CONFIG --------------------

DATA_DIR = "data/processed/cv_split" # Root with 'train/' and 'val/' subfolders
BATCH_SIZE = 16                      # Batch size for training/validation
EPOCHS = 25                          # Number of epochs
LR = 1e-3                            # Learning rate for Adam
OUT_DIR = "models"                   # Where to save checkpoints
PATIENCE = 8                         # NEW: Early stopping (stop if there is no improvement for 5 epochs)

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
    # Modified: added ImageNet normalization + more augmentation !
    # Modified: added aggressive augmentation to prevent overfitting

    train_tfms = transforms.Compose([
        transforms.Resize((256, 256)),              # Scaling to a fixed size
        transforms.RandomCrop((224, 224)),          # ------------------------------
        transforms.RandomHorizontalFlip(p=0.5),          # Random horizontal reflection
        transforms.RandomRotation(20),              # A small random twist(WAS 10 WE PUT 20)
        transforms.ColorJitter(                     # NEW
            brightness = 0.3,
            contrast = 0.3,
            saturation = 0.2,
            hue = 0.1),
        transforms.RandomAffine(                    # NEW shift
            degrees = 10,
            translate = (0.15, 0.15),
            scale = (0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),  # New
        transforms.RandomGrayscale(p=0.1),  # New
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),  # New
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # NEW
    ])

    # num_workers=2 is a safe default; adjust based on your CPU

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tfms)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_tfms)
    test_ds = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=val_tfms)  # NEW test

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)  # NEW test
    return train_dl, val_dl, test_dl, train_ds.classes

# -------------------- MODEL --------------------

def build_model(num_classes: int):
    #   Load a ResNet-18 with ImageNet weights and replace the final FC layer
    #   with a fresh Linear head of size (in_features -> num_classes).
    #   Modified: freeze early layers (faster + less overfitting)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) # Freeze early layers (only train layer4 and fc)
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False
    
    in_f = model.fc.in_features

    # Add Dropout for regularization
    model.fc = nn.Sequential(
        nn.Dropout(0.3),  # Dropout 30%
        nn.Linear(in_f, num_classes)
    )
    
    return model

# -------------------- CLASS WEIGHTS --------------------
def get_class_weights(train_dl, device):
    """
    New calculate class weights for imbalanced dataset
    """
    class_counts = Counter()
    for _, labels in train_dl:
        class_counts.update(labels.tolist())
    
    total = sum(class_counts.values())
    num_classes = len(class_counts)
    weights = []
    
    for i in range(num_classes):
        count = class_counts.get(i, 1)
        weight = total / (num_classes * count)
        weights.append(weight)
    
    return torch.FloatTensor(weights).to(device)


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

    train_dl, val_dl, test_dl, classes = get_loaders(DATA_DIR)
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Train batches: {len(train_dl)}, Val batches: {len(val_dl)}, Test batches: {len(test_dl)}")

    # Count samples per class
    train_samples = len(train_dl.dataset)
    val_samples = len(val_dl.dataset)
    test_samples = len(test_dl.dataset)
    print(f"Samples: train={train_samples}, val={val_samples}, test={test_samples}")

    model = build_model(num_classes=len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    
    # NEW: Class weights for imbalanced dataset
    class_weights = get_class_weights(train_dl, device)
    print(f"\nClass weights:")
    for i, (cls, weight) in enumerate(zip(classes, class_weights)):
        print(f"  {cls:<15} {weight:.3f}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)

      # NEW: Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=0.5, patience=4
    )

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    best_acc, best_path = 0.0, None
    no_improve = 0  # NEW: for early stopping
    
    print("\n" + "="*70)
    print("🚀 Starting training...")
    print("="*70)

    for epoch in range(1, EPOCHS + 1):
        # Training
        model.train()
        train_loss = 0.0
        t0 = time.time()
        
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        
        train_time = time.time() - t0
        avg_train_loss = train_loss / len(train_dl)
        
        # Post-Epoch Evaluation
        val_acc = evaluate(model, val_dl, device)
        train_acc = evaluate(model, train_dl, device)  # NEW to see overfitting
        
        # NEW Learning rate scheduler step
        scheduler.step(val_acc)
        current_lr = opt.param_groups[0]['lr']

        # Calculate overfitting gap
        gap = train_acc - val_acc
        gap_indicator = "🔥" if gap > 0.15 else "✅" if gap < 0.08 else "⚠️"
        
        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"loss={avg_train_loss:.3f} | "
              f"train={train_acc:.3f} | "
              f"val={val_acc:.3f} | "
              f"gap={gap:.3f} {gap_indicator} | "
              f"lr={current_lr:.6f} | "
              f"time={train_time:.1f}s")

        # Save best checkpoint (state_dict + class names)
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(OUT_DIR, f"resnet18_acc{best_acc:.3f}_e{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "classes": classes,
                "val_acc": val_acc,
                "train_acc": train_acc
            }, best_path)
            print(f"  ✅ Saved: {best_path}")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n⚠️  Early stopping: no improvement for {PATIENCE} epochs")
                break
    # NEW Test set evaluation
    print("\n" + "="*70)
    print("🧪 Evaluating on test set...")
    print("="*70)

    if best_path:
        # Load best model
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        
        test_acc = evaluate(model, test_dl, device)
        train_acc = checkpoint["train_acc"]
        val_acc = checkpoint["val_acc"]
        
        print(f"\n📊 Best model (epoch {checkpoint['epoch']}):")
        print(f"  Train accuracy: {train_acc:.1%}")
        print(f"  Val accuracy:   {val_acc:.1%}")
        print(f"  Test accuracy:  {test_acc:.1%}")

        # Overfitting check
        train_val_gap = train_acc - val_acc
        val_test_gap = abs(val_acc - test_acc)

        print(f"\n📈 Gaps:")
        print(f"  Train-Val:  {train_val_gap:.1%}", end="")
        if train_val_gap > 0.15:
            print(" 🔥 HIGH - overfitting detected!")
            print("     → Try: more augmentation, more data, or stronger regularization")
        elif train_val_gap > 0.10:
            print(" ⚠️  MODERATE - some overfitting")
            print("     → Consider: more augmentation or dropout")
        else:
            print(" ✅ LOW - good generalization!")
        
        print(f"  Val-Test:   {val_test_gap:.1%}", end="")
        if val_test_gap > 0.10:
            print(" ⚠️  Val-Test mismatch")
        else:
            print(" ✅ Good consistency")
        
        print("\n" + "="*70)
        print(f"✅ Training complete!")
        print(f"📁 Best model: {best_path}")
        print(f"🎯 Final test accuracy: {test_acc:.1%}")
        print("="*70)

if __name__ == "__main__":
    main()
