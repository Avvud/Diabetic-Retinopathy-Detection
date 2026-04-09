"""
=============================================================================
EfficientNet-B3 Training Pipeline for Diabetic Retinopathy Classification
=============================================================================
Dataset  : gaussian_filtered_images  (5 classes)
           No_DR | Mild | Moderate | Severe | Proliferate_DR
GPU      : NVIDIA RTX 3050 6GB Laptop GPU
Features :
  - EfficientNet-B3 (pretrained, staged fine-tuning)
  - Mixed-precision training  (torch.cuda.amp)
  - Gradient accumulation     (effective batch = 32)
  - AdamW + CosineAnnealingLR with differential LRs
  - Dropout + weight-decay regularisation
  - Accuracy / F1 / Confusion-Matrix / ROC-AUC evaluation
  - Grad-CAM visualisation
  - Ensemble utilities (simple & weighted averaging)
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import copy
import random
import time
import warnings
from pathlib import Path

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms
from torchvision.models import EfficientNet_B3_Weights

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    """Central configuration – change values here, not scattered across code."""

    # ── Paths ───────────────────────────────────────────────────────────────
    DATA_DIR        = Path("gaussian_filtered_images/gaussian_filtered_images")
    OUTPUT_DIR      = Path("outputs_efficientnet_b3")

    # ── Classes (torchvision ImageFolder uses alphabetical order) ───────────
    CLASS_NAMES     = ["Mild", "Moderate", "No_DR", "Proliferate_DR", "Severe"]
    NUM_CLASSES     = 5

    # ── Input ───────────────────────────────────────────────────────────────
    IMAGE_SIZE      = 300          # EfficientNet-B3 native; reduce to 256 if OOM
    IMAGENET_MEAN   = [0.485, 0.456, 0.406]
    IMAGENET_STD    = [0.229, 0.224, 0.225]

    # ── Training ─────────────────────────────────────────────────────────────
    BATCH_SIZE      = 8            # hardware batch (fits ~6 GB VRAM at 300×300)
    ACCUM_STEPS     = 4            # effective batch = 8 × 4 = 32
    EPOCHS_FROZEN   = 5            # classifier-head only
    EPOCHS_FINETUNE = 20           # full fine-tuning
    LR_CLASSIFIER   = 1e-4
    LR_FINETUNE     = 1e-5
    WEIGHT_DECAY    = 1e-4
    DROPOUT         = 0.45
    LABEL_SMOOTHING = 0.1          # softens targets → better calibration

    # ── Reproducibility ──────────────────────────────────────────────────────
    SEED            = 42

    # ── Hardware ─────────────────────────────────────────────────────────────
    DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Windows uses 'spawn' for multiprocessing; set > 0 only if confident
    NUM_WORKERS     = 0

    # ── Checkpoint ───────────────────────────────────────────────────────────
    CHECKPOINT_PATH = OUTPUT_DIR / "best_efficientnet_b3.pth"


CFG = Config()
CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    """Seed everything for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False   # True speeds training but breaks repro


set_seed(CFG.SEED)
print(f"[INFO] Using device : {CFG.DEVICE}")
if CFG.DEVICE.type == "cuda":
    print(f"[INFO] GPU          : {torch.cuda.get_device_name(0)}")
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[INFO] VRAM         : {total_vram:.1f} GB")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATASET HELPERS
#
#  WHY module-level class?
#    Windows multiprocessing uses 'spawn' (not 'fork'), so DataLoader worker
#    processes pickle every object they need. Classes defined INSIDE a function
#    (i.e. local classes) CANNOT be pickled → AttributeError / EOFError.
#    Defining TransformSubset at module level solves this completely.
# ─────────────────────────────────────────────────────────────────────────────

class TransformSubset(torch.utils.data.Dataset):
    """
    Wraps a torch.utils.data.Subset and applies a per-split transform.

    ImageFolder loads images as PIL Images.  We defer the transform so
    train / val / test each get their own augmentation pipeline without
    duplicating the underlying files.
    """

    def __init__(self, subset: Subset, transform: transforms.Compose):
        self.subset    = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        img, label = self.subset[idx]   # PIL Image, int
        return self.transform(img), label


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DATA PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def get_transforms(split: str, image_size: int = CFG.IMAGE_SIZE) -> transforms.Compose:
    """
    Medically safe augmentations:
      ✓ Horizontal flip (eyes are roughly symmetric)
      ✓ Mild rotation ±15° (small camera tilt)
      ✓ Conservative colour jitter (lighting variation)
      ✗ Vertical flip / aggressive crops (could destroy iris structure)
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size + 20, image_size + 20)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.1, hue=0.0),
            transforms.ToTensor(),
            transforms.Normalize(CFG.IMAGENET_MEAN, CFG.IMAGENET_STD),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(CFG.IMAGENET_MEAN, CFG.IMAGENET_STD),
        ])


def build_dataloaders(
    data_dir:   Path  = CFG.DATA_DIR,
    batch_size: int   = CFG.BATCH_SIZE,
    image_size: int   = CFG.IMAGE_SIZE,
    val_split:  float = 0.15,
    test_split: float = 0.10,
) -> dict:
    """
    1. Loads dataset with ImageFolder (no transform at this stage).
    2. Reproducibly splits into train / val / test.
    3. Wraps each split with TransformSubset for per-split augmentations.
    4. Applies WeightedRandomSampler on train to handle class imbalance.
    5. Returns DataLoaders + class_names.
    """
    # ── Load raw dataset (PIL Images) ────────────────────────────────────────
    full_dataset = datasets.ImageFolder(root=str(data_dir))
    n_total = len(full_dataset)

    # ── Reproducible split ───────────────────────────────────────────────────
    indices = list(range(n_total))
    random.shuffle(indices)                    # already seeded
    n_test  = int(n_total * test_split)
    n_val   = int(n_total * val_split)
    n_train = n_total - n_val - n_test

    train_idx = indices[:n_train]
    val_idx   = indices[n_train : n_train + n_val]
    test_idx  = indices[n_train + n_val :]

    # ── Wrap with per-split transforms ───────────────────────────────────────
    train_set = TransformSubset(Subset(full_dataset, train_idx),
                                get_transforms("train", image_size))
    val_set   = TransformSubset(Subset(full_dataset, val_idx),
                                get_transforms("val",   image_size))
    test_set  = TransformSubset(Subset(full_dataset, test_idx),
                                get_transforms("test",  image_size))

    # ── Weighted sampler (over-sample minority classes) ───────────────────────
    train_labels  = [full_dataset.targets[i] for i in train_idx]
    class_counts  = np.bincount(train_labels, minlength=CFG.NUM_CLASSES)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[l] for l in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_labels), replacement=True
    )

    # ── DataLoaders ──────────────────────────────────────────────────────────
    # persistent_workers=True crashes when num_workers=0 on Windows
    loader_kwargs = dict(
        num_workers=CFG.NUM_WORKERS,
        pin_memory=(CFG.DEVICE.type == "cuda"),
        persistent_workers=(CFG.NUM_WORKERS > 0),
    )

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              sampler=sampler, **loader_kwargs)
    val_loader   = DataLoader(val_set,   batch_size=batch_size,
                              shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_set,  batch_size=batch_size,
                              shuffle=False, **loader_kwargs)

    print(f"[DATA] Classes : {full_dataset.classes}")
    print(f"[DATA] Total   : {n_total}  |  Train: {n_train}  "
          f"Val: {n_val}  Test: {n_test}")
    print(f"[DATA] Class counts (all): "
          f"{dict(zip(full_dataset.classes, class_counts))}")

    return {
        "train":       train_loader,
        "val":         val_loader,
        "test":        test_loader,
        "class_names": full_dataset.classes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MODEL
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    num_classes:     int   = CFG.NUM_CLASSES,
    dropout:         float = CFG.DROPOUT,
    freeze_backbone: bool  = True,
) -> nn.Module:
    """
    Loads pretrained EfficientNet-B3 and replaces the classifier head.

    New head:
        AdaptiveAvgPool (built-in)
        → Dropout(0.45)
        → Linear(1536 → 512)
        → SiLU  (EfficientNet's native activation)
        → Dropout(0.225)
        → Linear(512 → num_classes)

    Two dropout layers because:
      - Heavy dropout after pooled features fights overfitting on small
        medical datasets.
      - Light dropout before output keeps gradient flow stable.
    """
    weights = EfficientNet_B3_Weights.IMAGENET1K_V1
    model   = models.efficientnet_b3(weights=weights)

    # ── Freeze / unfreeze backbone ───────────────────────────────────────────
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
        print("[MODEL] Backbone FROZEN – training classifier head only.")
    else:
        for param in model.parameters():
            param.requires_grad = True
        print("[MODEL] All layers TRAINABLE (fine-tuning mode).")

    # ── Replace head ─────────────────────────────────────────────────────────
    in_features = model.classifier[1].in_features   # 1536 for B3
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 512),
        nn.SiLU(),
        nn.Dropout(p=dropout / 2),
        nn.Linear(512, num_classes),
    )

    return model.to(CFG.DEVICE)


def unfreeze_backbone(model: nn.Module,
                      lr_finetune: float = CFG.LR_FINETUNE) -> optim.AdamW:
    """
    Unfreezes entire backbone and returns a new optimizer with
    differential learning rates:
      - Backbone  : lr_finetune        (small, preserves pretrained features)
      - Head      : lr_finetune × 10   (larger, adapts quickly to new classes)
    """
    for param in model.features.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW([
        {"params": model.features.parameters(),   "lr": lr_finetune},
        {"params": model.classifier.parameters(), "lr": lr_finetune * 10},
    ], weight_decay=CFG.WEIGHT_DECAY)

    print(f"[MODEL] Backbone UNFROZEN | "
          f"backbone LR={lr_finetune}, head LR={lr_finetune * 10}")
    return optimizer


# ─────────────────────────────────────────────────────────────────────────────
# 5.  TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:       nn.Module,
    loader:      DataLoader,
    optimizer:   optim.Optimizer,
    criterion:   nn.Module,
    scaler:      GradScaler,
    accum_steps: int = CFG.ACCUM_STEPS,
) -> tuple:
    """
    One training epoch.

    Key techniques:
      • Mixed-precision  : autocast() → FP16 forward, FP32 master weights
      • Gradient accum   : accumulate gradients over `accum_steps` batches
                           before stepping → effective batch = batch × accum
      • Grad clipping    : clips norm ≤ 1.0 to prevent exploding gradients
    """
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    optimizer.zero_grad()   # zero once; accumulate over multiple batches

    for step, (images, labels) in enumerate(loader):
        images = images.to(CFG.DEVICE, non_blocking=True)
        labels = labels.to(CFG.DEVICE, non_blocking=True)

        # ── Forward (FP16) ───────────────────────────────────────────────────
        with autocast():
            logits = model(images)
            # Divide by accum_steps so gradients average (not sum)
            loss   = criterion(logits, labels) / accum_steps

        # ── Backward ─────────────────────────────────────────────────────────
        scaler.scale(loss).backward()

        # ── Optimizer step after every accum_steps batches ───────────────────
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # ── Progress reporting (every 20 batches) ────────────────────────────
        if (step + 1) % 20 == 0 or (step + 1) == len(loader):
            print(f"    - Step {step+1:>4}/{len(loader)} | Loss: {loss.item()*accum_steps:.4f}", end="\r")

        running_loss += loss.item() * accum_steps   # recover true loss scale
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
) -> tuple:
    """Validation loop – no gradients computed."""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(CFG.DEVICE, non_blocking=True)
        labels = labels.to(CFG.DEVICE, non_blocking=True)

        with autocast():
            logits = model(images)
            loss   = criterion(logits, labels)

        # ── Progress reporting (every 25 batches) ────────────────────────────
        if (len(all_labels) // loader.batch_size + 1) % 25 == 0:
             print(f"    - Val Step {len(all_labels)//loader.batch_size+1:>3}/{len(loader)}", end="\r")

        running_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


def train(
    model:      nn.Module,
    loaders:    dict,
    optimizer:  optim.Optimizer,
    criterion:  nn.Module,
    scheduler,
    scaler:     GradScaler,
    num_epochs: int,
    phase_name: str = "Training",
) -> dict:
    """
    Full training loop for one phase (frozen-backbone or fine-tuning).
    Saves best checkpoint by validation accuracy.
    Returns history dict for later plotting.
    """
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    print(f"\n{'='*60}")
    print(f"  Phase : {phase_name}  ({num_epochs} epochs)")
    print(f"{'='*60}")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], optimizer, criterion, scaler
        )
        val_loss, val_acc = validate(model, loaders["val"], criterion)

        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # ── Save best checkpoint ──────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, CFG.CHECKPOINT_PATH)
            marker = " ★ saved"
        else:
            marker = ""

        print(
            f"[{phase_name}] Epoch {epoch:>3}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"{elapsed:.1f}s{marker}"
        )

        # ── VRAM snapshot (epoch 1 only, useful for OOM debugging) ────────────
        if CFG.DEVICE.type == "cuda" and epoch == 1:
            alloc = torch.cuda.memory_allocated() / 1e9
            resvd = torch.cuda.memory_reserved()  / 1e9
            print(f"  [MEM] Allocated: {alloc:.2f} GB | Reserved: {resvd:.2f} GB")

    model.load_state_dict(best_weights)
    print(f"\n[{phase_name}] Best Val Accuracy: {best_val_acc:.4f}")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# 6.  EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:       nn.Module,
    loader:      DataLoader,
    class_names: list,
    split:       str = "Test",
) -> dict:
    """
    Comprehensive evaluation:
      • Accuracy
      • Macro F1-score
      • Confusion matrix (plot saved to disk)
      • ROC-AUC (One-vs-Rest, macro average)
      • Per-class classification report
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(CFG.DEVICE, non_blocking=True)
        with autocast():
            logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    all_probs  = np.array(all_probs)
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro")
    cm  = confusion_matrix(all_labels, all_preds)

    # ROC-AUC requires one-hot encoded labels
    y_bin = label_binarize(all_labels, classes=list(range(len(class_names))))
    try:
        auc = roc_auc_score(y_bin, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")   # edge case: missing class in split

    print(f"\n{'='*60}")
    print(f"  {split} Results")
    print(f"{'='*60}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro F1  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\n{classification_report(all_labels, all_preds, target_names=class_names)}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix – EfficientNet-B3 ({split})", fontsize=14)
    plt.tight_layout()
    save_path = CFG.OUTPUT_DIR / f"confusion_matrix_{split.lower()}.png"
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"  Confusion matrix saved → {save_path}")

    return {
        "accuracy": acc, "f1": f1, "auc": auc, "cm": cm,
        "probs": all_probs, "labels": all_labels, "preds": all_preds,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7.  VISUALISATION – TRAINING CURVES
# ─────────────────────────────────────────────────────────────────────────────

def plot_history(history_frozen: dict, history_finetune: dict) -> None:
    """Plots combined loss & accuracy curves for both training phases."""

    def _concat(key):
        return history_frozen[key] + history_finetune[key]

    frozen_boundary = len(history_frozen["train_loss"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (train_key, val_key), ylabel in zip(
        axes,
        [("train_loss", "val_loss"), ("train_acc", "val_acc")],
        ["Loss", "Accuracy"],
    ):
        ax.plot(_concat(train_key), label=f"Train {ylabel}")
        ax.plot(_concat(val_key),   label=f"Val {ylabel}")
        ax.axvline(x=frozen_boundary - 1, color="grey", linestyle="--",
                   alpha=0.7, label="Unfreeze backbone")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"EfficientNet-B3 – {ylabel}")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = CFG.OUTPUT_DIR / "training_curves.png"
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"[PLOT] Training curves saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  GRAD-CAM
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Maps for EfficientNet-B3.

    Usage:
        cam = GradCAM(model, target_layer=model.features[-1])
        heatmap = cam(image_tensor)     # (H_feat, W_feat) in [0, 1]
        cam.remove_hooks()
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self._gradients   = None
        self._activations = None
        self._fwd_handle  = target_layer.register_forward_hook(self._fwd_hook)
        self._bwd_handle  = target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, input, output):
        self._activations = output.detach()

    def _bwd_hook(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def remove_hooks(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __call__(self, image: torch.Tensor,
                 class_idx: int | None = None) -> np.ndarray:
        """
        Args:
            image     : (1, C, H, W) tensor (already on device)
            class_idx : target class; None → predicted class
        Returns:
            heatmap   : (H_feat, W_feat) in [0, 1]
        """
        self.model.eval()
        image  = image.to(CFG.DEVICE)
        output = self.model(image)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1.0
        output.backward(gradient=one_hot)

        # Weight feature maps by gradient importance
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        heatmap = (weights * self._activations).sum(dim=1).squeeze()  # (H, W)
        heatmap = torch.relu(heatmap).cpu().numpy()

        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        return heatmap


def visualise_gradcam(
    model:       nn.Module,
    image_path:  str,
    class_names: list,
    image_size:  int = CFG.IMAGE_SIZE,
) -> None:
    """Loads one image, runs Grad-CAM, and saves a 3-panel overlay figure."""
    from PIL import Image

    tfm     = get_transforms("val", image_size)
    img_pil = Image.open(image_path).convert("RGB")
    img_t   = tfm(img_pil).unsqueeze(0)

    target_layer = model.features[-1]
    cam     = GradCAM(model, target_layer)
    heatmap = cam(img_t)
    cam.remove_hooks()

    # ── Overlay ───────────────────────────────────────────────────────────────
    img_np  = np.array(img_pil.resize((image_size, image_size)))
    hmap_rs = cv2.resize(heatmap, (image_size, image_size))
    hmap_rgb = cm.jet(hmap_rs)[:, :, :3]
    overlay  = np.clip(0.55 * img_np / 255.0 + 0.45 * hmap_rgb, 0, 1)

    # ── Prediction ────────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad(), autocast():
        logits = model(img_t.to(CFG.DEVICE))
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred  = probs.argmax()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_np);               axes[0].set_title("Original");   axes[0].axis("off")
    axes[1].imshow(hmap_rs, cmap="jet");  axes[1].set_title("Grad-CAM");   axes[1].axis("off")
    axes[2].imshow(overlay);              axes[2].set_title(
        f"Overlay\nPred: {class_names[pred]} ({probs[pred]*100:.1f}%)"
    ); axes[2].axis("off")

    plt.suptitle("Grad-CAM – EfficientNet-B3", fontsize=14)
    plt.tight_layout()
    out = CFG.OUTPUT_DIR / "gradcam_overlay.png"
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"[GRADCAM] Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  ENSEMBLE UTILITIES  (drop-in with ResNet50 / DenseNet121)
# ─────────────────────────────────────────────────────────────────────────────

class EnsemblePredictor:
    """
    Combines probability outputs from multiple models.

    Example:
        ensemble = EnsemblePredictor([
            (effb3_model,   2.0),   # B3 gets double weight
            (resnet50,      1.0),
            (densenet121,   1.0),
        ])
        probs = ensemble.predict(image_batch, mode="weighted")
        preds = probs.argmax(axis=1)
    """

    def __init__(self, models_and_weights: list, device=CFG.DEVICE):
        self.models = [(m.eval().to(device), w) for m, w in models_and_weights]
        self.device = device

    @torch.no_grad()
    def predict(self, images: torch.Tensor,
                mode: str = "weighted") -> np.ndarray:
        """
        Returns (N, num_classes) probability array.

        mode='simple'   → unweighted mean
        mode='weighted' → weights normalised to sum to 1
        """
        images     = images.to(self.device)
        prob_list  = []
        weight_list = []

        for model, weight in self.models:
            with autocast():
                logits = model(images)
            prob_list.append(torch.softmax(logits, dim=1).cpu().numpy())
            weight_list.append(weight)

        prob_stack = np.stack(prob_list, axis=0)   # (n_models, N, C)

        if mode == "simple":
            return prob_stack.mean(axis=0)

        w = np.array(weight_list) / sum(weight_list)
        return (prob_stack * w[:, None, None]).sum(axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# 10.  OOM DEBUGGING REFERENCE
# ─────────────────────────────────────────────────────────────────────────────

OOM_TIPS = """
╔══════════════════════════════════════════════════════════════════╗
║            CUDA Out-of-Memory Debugging Checklist               ║
╠══════════════════════════════════════════════════════════════════╣
║ 1. Reduce CFG.IMAGE_SIZE  300 → 256                             ║
║ 2. Reduce CFG.BATCH_SIZE  8   → 4                               ║
║ 3. Increase CFG.ACCUM_STEPS to keep effective batch constant     ║
║ 4. torch.cuda.empty_cache() after each epoch in the train loop  ║
║ 5. Set CFG.NUM_WORKERS = 0 to isolate DataLoader issues         ║
║ 6. print(torch.cuda.memory_summary()) to inspect allocations    ║
║ 7. Freeze more backbone blocks (only unfreeze last 2–3)         ║
╚══════════════════════════════════════════════════════════════════╝
"""


# ─────────────────────────────────────────────────────────────────────────────
# 11.  MAIN – ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(OOM_TIPS)

    # ── Data ─────────────────────────────────────────────────────────────────
    loaders     = build_dataloaders()
    class_names = loaders["class_names"]

    criterion = nn.CrossEntropyLoss(label_smoothing=CFG.LABEL_SMOOTHING)
    scaler    = GradScaler()

    # =========================================================================
    #  PHASE 1: Classifier head only (backbone frozen)
    # =========================================================================
    model = build_model(freeze_backbone=True)

    optimizer_frozen = optim.AdamW(
        model.classifier.parameters(),
        lr=CFG.LR_CLASSIFIER,
        weight_decay=CFG.WEIGHT_DECAY,
    )
    scheduler_frozen = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_frozen, T_max=CFG.EPOCHS_FROZEN, eta_min=1e-6
    )

    history_frozen = train(
        model, loaders, optimizer_frozen, criterion,
        scheduler_frozen, scaler,
        num_epochs=CFG.EPOCHS_FROZEN,
        phase_name="Frozen Backbone",
    )

    # =========================================================================
    #  PHASE 2: Full fine-tuning (all layers)
    # =========================================================================
    optimizer_ft = unfreeze_backbone(model)
    scheduler_ft = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_ft, T_max=CFG.EPOCHS_FINETUNE, eta_min=1e-7
    )

    history_ft = train(
        model, loaders, optimizer_ft, criterion,
        scheduler_ft, scaler,
        num_epochs=CFG.EPOCHS_FINETUNE,
        phase_name="Fine-Tuning",
    )

    # =========================================================================
    #  EVALUATION on held-out test set
    # =========================================================================
    # Load best weights saved during training
    model.load_state_dict(
        torch.load(CFG.CHECKPOINT_PATH, weights_only=True)
    )
    results = evaluate(model, loaders["test"], class_names, split="Test")

    # =========================================================================
    #  PLOTS
    # =========================================================================
    plot_history(history_frozen, history_ft)

    # =========================================================================
    #  GRAD-CAM demo (uncomment and point to a real image)
    # =========================================================================
    # sample = "gaussian_filtered_images/gaussian_filtered_images/No_DR/example.jpg"
    # visualise_gradcam(model, sample, class_names)

    print("\n[DONE] Training complete.")
    print(f"       Best model → {CFG.CHECKPOINT_PATH.resolve()}")
    return model, results


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
