"""
Training Module — Full training pipeline with evaluation and visualization.

Usage: python run.py train

Features:
- Label smoothing to prevent overconfident predictions
- MixUp augmentation for on-the-fly feature-space regularization
- Adam optimizer with ReduceLROnPlateau scheduler
- Early stopping to prevent overfitting
- Class-weighted loss for imbalanced data
- Confusion matrix and training history plots
- Best checkpoint saving with gesture label map
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from utils.helpers import ensure_dirs, get_device
from src.model import build_model


# ─── Dataset Loading ──────────────────────────────────────────

def load_processed_data():
    """Load preprocessed train/val/test splits."""
    data = {}
    for split in ["train", "val", "test"]:
        X_path = os.path.join(config.PROCESSED_DATA_DIR, f"X_{split}.npy")
        y_path = os.path.join(config.PROCESSED_DATA_DIR, f"y_{split}.npy")

        if not os.path.exists(X_path):
            raise FileNotFoundError(
                f"Processed data not found. Run 'python run.py process' first."
            )

        data[f"X_{split}"] = np.load(X_path)
        data[f"y_{split}"] = np.load(y_path)

    return data


def create_dataloaders(data, batch_size=config.BATCH_SIZE):
    """Create PyTorch DataLoaders from numpy arrays."""
    loaders = {}
    for split in ["train", "val", "test"]:
        X = torch.FloatTensor(data[f"X_{split}"])
        y = torch.LongTensor(data[f"y_{split}"])

        dataset = TensorDataset(X, y)
        sampler = None
        shuffle = (split == "train")

        if split == "train" and config.USE_WEIGHTED_SAMPLER:
            class_counts = np.bincount(data["y_train"], minlength=config.NUM_GESTURES)
            class_weights = np.zeros_like(class_counts, dtype=np.float32)
            nonzero = class_counts > 0
            class_weights[nonzero] = 1.0 / class_counts[nonzero]
            sample_weights = class_weights[data["y_train"]]
            sampler = WeightedRandomSampler(
                weights=torch.DoubleTensor(sample_weights),
                num_samples=len(sample_weights),
                replacement=True,
            )
            shuffle = False

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=0,       # Windows compatibility
            pin_memory=torch.cuda.is_available(),
        )

    return loaders


# ─── MixUp Augmentation ──────────────────────────────────────

def mixup_data(x, y, alpha=0.2):
    """
    Apply MixUp augmentation: blend pairs of training samples.

    This creates "virtual" training examples by interpolating between
    two random samples, which acts as powerful regularization.

    Args:
        x: input batch (B, seq_len, features)
        y: labels (B,)
        alpha: MixUp strength (0 = no mixup, higher = stronger blending)
    Returns:
        mixed_x, y_a, y_b, lam (lambda blend factor)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute MixUp loss as weighted blend of two targets."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ─── Training Loop ────────────────────────────────────────────

def train_one_epoch(model, dataloader, criterion, optimizer, device, use_mixup=True):
    """Train for one epoch with optional MixUp. Returns average loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        if use_mixup and np.random.random() > 0.3:  # Apply MixUp 70% of the time
            mixed_x, y_a, y_b, lam = mixup_data(X_batch, y_batch, alpha=config.MIXUP_ALPHA)
            outputs = model(mixed_x)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)

            # For accuracy tracking, use the original (un-mixed) predictions
            _, predicted = torch.max(outputs, 1)
            correct += (lam * (predicted == y_a).float() +
                        (1 - lam) * (predicted == y_b).float()).sum().item()
        else:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()

        loss.backward()

        # Gradient clipping to prevent exploding gradients in LSTM
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        total += y_batch.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model. Returns loss, accuracy, all predictions and labels."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


# ─── Visualization ────────────────────────────────────────────

def plot_training_history(history, save_path):
    """Plot and save training/validation loss and accuracy curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(history["train_loss"]) + 1)

        # Loss plot
        ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
        ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.set_title("Training & Validation Loss", fontsize=14)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Accuracy plot
        ax2.plot(epochs, history["train_acc"], "b-", label="Train Acc", linewidth=2)
        ax2.plot(epochs, history["val_acc"], "r-", label="Val Acc", linewidth=2)
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Accuracy", fontsize=12)
        ax2.set_title("Training & Validation Accuracy", fontsize=14)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [OK] Training history saved: {save_path}")
    except ImportError:
        print("  [!] matplotlib not available — skipping plot")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save a confusion matrix heatmap."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-6)

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".1%",
            cmap="YlOrRd",
            xticklabels=class_names,
            yticklabels=class_names,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )
        ax.set_xlabel("Predicted", fontsize=13)
        ax.set_ylabel("Actual", fontsize=13)
        ax.set_title("Confusion Matrix (Normalized)", fontsize=15)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [OK] Confusion matrix saved: {save_path}")
    except ImportError:
        print("  [!] matplotlib/seaborn not available — skipping plot")


# ─── Main Training Pipeline ──────────────────────────────────

def train_model():
    """
    Full training pipeline:
    1. Load processed data
    2. Build model
    3. Train with early stopping + MixUp + label smoothing
    4. Evaluate on test set
    5. Save model + visualizations
    """
    ensure_dirs()
    device = get_device()

    print("\n" + "=" * 60)
    print("  SIGN LANGUAGE TRANSLATOR — MODEL TRAINING")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/5] Loading processed data...")
    data = load_processed_data()
    loaders = create_dataloaders(data)

    for split in ["train", "val", "test"]:
        print(f"  {split:>5}: {data[f'X_{split}'].shape[0]:>5} samples, "
              f"shape {data[f'X_{split}'].shape}")

    # Verify label range
    y_all = np.concatenate([data["y_train"], data["y_val"], data["y_test"]])
    print(f"\n  Label range: [{y_all.min()}, {y_all.max()}] "
          f"(expected [0, {config.NUM_GESTURES - 1}])")
    if y_all.max() >= config.NUM_GESTURES:
        print(f"  [X] FATAL: Labels exceed gesture count! "
              f"Re-run 'python run.py process' with updated config.")
        return

    # Step 2: Build model
    print("\n[2/5] Building model...")
    model = build_model(device)

    y_train = data["y_train"]
    class_counts = np.bincount(y_train, minlength=config.NUM_GESTURES)
    majority_baseline = class_counts.max() / class_counts.sum()

    print(f"\n  Per-class training samples:")
    for i, count in enumerate(class_counts):
        if i < len(config.GESTURES):
            print(f"    {config.GESTURES[i]:>10}: {count}")
    print(f"  Majority-class baseline: {majority_baseline:.1%}")
    print(f"  Weighted sampler: {'enabled' if config.USE_WEIGHTED_SAMPLER else 'disabled'}")

    criterion = nn.CrossEntropyLoss(
        label_smoothing=config.LABEL_SMOOTHING,
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Step 3: Training loop with early stopping
    print(f"\n[3/5] Training for up to {config.EPOCHS} epochs...")
    print(f"  Early stopping patience: {config.EARLY_STOP_PATIENCE} epochs")
    print(f"  Label smoothing: {config.LABEL_SMOOTHING}")
    print(f"  MixUp: {'enabled' if config.USE_MIXUP else 'disabled'}")
    print(f"  {'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | "
          f"{'Val Loss':>8} | {'Val Acc':>7} | {'LR':>10} | {'Time':>6}")
    print("  " + "-" * 72)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }

    best_val_acc = 0.0
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    for epoch in range(1, config.EPOCHS + 1):
        epoch_start = time.time()

        # Train (with MixUp)
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, use_mixup=config.USE_MIXUP
        )

        # Validate (no MixUp, no augmentation — clean evaluation)
        val_loss, val_acc, _, _ = evaluate(
            model, loaders["val"], criterion, device
        )

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - epoch_start

        print(f"  {epoch:>5}  | {train_loss:>10.4f} | {train_acc:>8.1%} | "
              f"{val_loss:>8.4f} | {val_acc:>6.1%} | {current_lr:>10.6f} | {elapsed:>5.1f}s",
              end="")

        # Check for improvement
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            # Save best model checkpoint with gesture labels
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "val_loss": val_loss,
                "gestures": config.GESTURES,
                "num_gestures": config.NUM_GESTURES,
            }, config.MODEL_PATH)
            print(" * best")
        else:
            patience_counter += 1
            print(f" (patience: {patience_counter}/{config.EARLY_STOP_PATIENCE})")

            if patience_counter >= config.EARLY_STOP_PATIENCE:
                print(f"\n  [!] Early stopping at epoch {epoch}")
                break

    print(f"\n  Best model: epoch {best_epoch}, val_acc={best_val_acc:.1%}")

    # Step 4: Evaluate on test set
    print("\n[4/5] Evaluating on test set...")

    # Load best model for evaluation
    checkpoint = torch.load(config.MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Use unweighted loss for fair test evaluation
    test_criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, loaders["test"], test_criterion, device
    )

    print(f"  Test Accuracy: {test_acc:.1%}")
    print(f"  Test Loss:     {test_loss:.4f}")

    # Detailed classification report
    present_classes = sorted(set(test_labels))
    present_names = [config.GESTURES[i] for i in present_classes]

    print(f"\n  Classification Report:")
    print(classification_report(
        test_labels, test_preds,
        labels=present_classes,
        target_names=present_names,
        digits=3,
    ))

    # F1 Score
    f1 = f1_score(test_labels, test_preds, average="weighted")
    print(f"  Weighted F1 Score: {f1:.3f}")

    # Step 5: Save visualizations
    print("\n[5/5] Saving visualizations...")
    plot_training_history(history, config.TRAINING_HISTORY_PATH)
    plot_confusion_matrix(
        test_labels, test_preds, present_names, config.CONFUSION_MATRIX_PATH
    )

    print(f"\n{'=' * 60}")
    print(f"  Training complete!")
    print(f"  Model saved:    {config.MODEL_PATH}")
    print(f"  Test accuracy:  {test_acc:.1%}")
    print(f"  F1 score:       {f1:.3f}")
    print(f"  Next step:      python run.py run")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    train_model()
