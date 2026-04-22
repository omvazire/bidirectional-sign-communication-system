"""
Diagnostic Script — Verify pipeline consistency and model health.

Usage: python run.py diagnose

Checks:
1. Label map consistency between config, processed data, and model checkpoint
2. Normalization consistency (data range checks)
3. Per-class accuracy breakdown
4. Data leakage detection (checks if val/test overlap with train)
5. Model confidence distribution analysis
"""

import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.model import GestureRecognitionModel


def check_raw_data_coverage():
    """Check whether each gesture has enough raw sequences for live robustness."""
    print("\n-- Raw Data Coverage Check --")
    issues = 0
    counts = {}

    for gesture in config.GESTURES:
        gesture_dir = os.path.join(config.RAW_DATA_DIR, gesture)
        if os.path.exists(gesture_dir):
            counts[gesture] = len([f for f in os.listdir(gesture_dir) if f.endswith(".npy")])
        else:
            counts[gesture] = 0

    for gesture in config.GESTURES:
        count = counts[gesture]
        status = "OK"
        if count == 0:
            status = "MISSING"
            issues += 1
        elif count < config.RAW_DATA_WARNING_THRESHOLD:
            status = "LOW"
            issues += 1
        print(f"  {gesture:>10}: {count:>2} sequences [{status}]")

    print("\n  Confusion-sensitive pairs:")
    for anchor, guarded in config.CONFUSION_GUARDS.items():
        for other in guarded:
            if other not in counts:
                continue
            gap = counts.get(anchor, 0) - counts.get(other, 0)
            print(f"    {anchor:>10} vs {other:<10}: {counts.get(anchor, 0):>2} vs {counts.get(other, 0):>2} (gap {gap:+d})")

    if issues:
        print(
            f"  [!] Low-count gestures often look correct in the small test split but fail more often live. "
            f"Target at least {config.RAW_DATA_WARNING_THRESHOLD} clean sequences per gesture."
        )

    return issues


def check_label_consistency():
    """Verify labels match between config, saved label map, and model checkpoint."""
    print("\n-- Label Consistency Check --")
    issues = 0

    # Check label_map.json
    label_map_path = os.path.join(config.PROCESSED_DATA_DIR, "label_map.json")
    if os.path.exists(label_map_path):
        with open(label_map_path, "r") as f:
            saved_map = json.load(f)
        saved_gestures = [saved_map[str(i)] for i in range(len(saved_map))]

        if saved_gestures == config.GESTURES:
            print("  [OK] label_map.json matches config.GESTURES")
        else:
            print("  [!!] MISMATCH: label_map.json differs from config.GESTURES!")
            print(f"       Saved:   {saved_gestures}")
            print(f"       Config:  {config.GESTURES}")
            issues += 1
    else:
        print("  [!] label_map.json not found. Run 'python run.py process' to create it.")
        issues += 1

    # Check model checkpoint
    if os.path.exists(config.MODEL_PATH):
        checkpoint = torch.load(config.MODEL_PATH, map_location="cpu", weights_only=True)
        model_gestures = checkpoint.get("gestures", None)

        if model_gestures is None:
            print("  [!] Model checkpoint does not contain gesture list.")
            issues += 1
        elif model_gestures == config.GESTURES:
            print("  [OK] Model checkpoint gestures match config.GESTURES")
        else:
            print("  [!!] MISMATCH: Model was trained with different gestures!")
            print(f"       Model:   {model_gestures}")
            print(f"       Config:  {config.GESTURES}")
            issues += 1

        # Check model output size
        model = GestureRecognitionModel()
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
            out_features = model.classifier[-1].out_features
            if out_features == config.NUM_GESTURES:
                print(f"  [OK] Model output size ({out_features}) matches NUM_GESTURES ({config.NUM_GESTURES})")
            else:
                print(f"  [!!] Model output size ({out_features}) != NUM_GESTURES ({config.NUM_GESTURES})")
                print(f"       You MUST retrain: python run.py train")
                issues += 1
        except RuntimeError as e:
            print(f"  [!!] Cannot load model state_dict: {e}")
            print(f"       Model architecture likely changed. Retrain required.")
            issues += 1
    else:
        print("  [!] No model checkpoint found. Train first: python run.py train")
        issues += 1

    return issues


def check_data_ranges():
    """Verify data normalization is correct and consistent."""
    print("\n-- Data Range Check --")
    issues = 0

    for split in ["train", "val", "test"]:
        X_path = os.path.join(config.PROCESSED_DATA_DIR, f"X_{split}.npy")
        y_path = os.path.join(config.PROCESSED_DATA_DIR, f"y_{split}.npy")

        if not os.path.exists(X_path):
            print(f"  [!] {split} data not found.")
            issues += 1
            continue

        X = np.load(X_path)
        y = np.load(y_path)

        print(f"\n  {split.upper()}:")
        print(f"    Shape:    X={X.shape}, y={y.shape}")
        print(f"    X range:  [{X.min():.4f}, {X.max():.4f}]")
        print(f"    X mean:   {X.mean():.4f}")
        print(f"    X std:    {X.std():.4f}")
        print(f"    Y range:  [{y.min()}, {y.max()}]")

        # Check for issues
        if X.max() > 5.0 or X.min() < -5.0:
            print(f"    [!!] Data range seems too wide. Possible double-normalization.")
            issues += 1

        if y.max() >= config.NUM_GESTURES:
            print(f"    [!!] Label {y.max()} exceeds NUM_GESTURES ({config.NUM_GESTURES})!")
            issues += 1

        # Check zero-frame ratio
        zero_frames = (np.abs(X).sum(axis=2) == 0).sum()
        total_frames = X.shape[0] * X.shape[1]
        zero_pct = zero_frames / total_frames * 100
        print(f"    Zero frames: {zero_frames}/{total_frames} ({zero_pct:.1f}%)")
        if zero_pct > 10:
            print(f"    [!] High zero-frame ratio suggests poor data collection quality.")
            issues += 1

        # Per-class distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"    Classes: {len(unique)}, samples: {dict(zip(unique, counts))}")

    return issues


def check_data_leakage():
    """Check for potential data leakage between train/val/test splits."""
    print("\n-- Data Leakage Check --")
    issues = 0

    try:
        X_train = np.load(os.path.join(config.PROCESSED_DATA_DIR, "X_train.npy"))
        X_val = np.load(os.path.join(config.PROCESSED_DATA_DIR, "X_val.npy"))
        X_test = np.load(os.path.join(config.PROCESSED_DATA_DIR, "X_test.npy"))
    except FileNotFoundError:
        print("  [!] Processed data not found. Run 'python run.py process' first.")
        return 1

    # Check for exact duplicates between val/test and train
    # Reshape to 2D for comparison
    train_flat = X_train.reshape(X_train.shape[0], -1)
    val_flat = X_val.reshape(X_val.shape[0], -1)
    test_flat = X_test.reshape(X_test.shape[0], -1)

    # Check val vs train
    val_in_train = 0
    for i in range(len(val_flat)):
        # Check if this exact sequence exists in training set
        diffs = np.abs(train_flat - val_flat[i]).sum(axis=1)
        if np.any(diffs < 1e-6):
            val_in_train += 1

    if val_in_train > 0:
        print(f"  [!!] {val_in_train}/{len(val_flat)} val samples are EXACT duplicates of train!")
        print(f"       This is data leakage. Re-run 'python run.py process'.")
        issues += 1
    else:
        print(f"  [OK] No exact val-train duplicates found.")

    # Check test vs train
    test_in_train = 0
    for i in range(len(test_flat)):
        diffs = np.abs(train_flat - test_flat[i]).sum(axis=1)
        if np.any(diffs < 1e-6):
            test_in_train += 1

    if test_in_train > 0:
        print(f"  [!!] {test_in_train}/{len(test_flat)} test samples are EXACT duplicates of train!")
        issues += 1
    else:
        print(f"  [OK] No exact test-train duplicates found.")

    # Check for near-duplicates (augmented versions)
    # A rough check: if val/test samples are very close (L2 < threshold) to train samples
    near_dupes_val = 0
    for i in range(min(len(val_flat), 50)):  # Sample check for speed
        diffs = np.sqrt(((train_flat - val_flat[i]) ** 2).sum(axis=1))
        if np.any(diffs < 0.5):  # Very close
            near_dupes_val += 1

    checked = min(len(val_flat), 50)
    if near_dupes_val > checked * 0.3:
        print(f"  [!] {near_dupes_val}/{checked} checked val samples are suspiciously close to train.")
        print(f"       Possible augmented-clone leakage.")
        issues += 1
    else:
        print(f"  [OK] No suspicious near-duplicates in val set ({near_dupes_val}/{checked} close).")

    return issues


def check_model_confidence():
    """Run model on test set and analyze prediction confidence distribution."""
    print("\n-- Model Confidence Analysis --")

    try:
        X_test = np.load(os.path.join(config.PROCESSED_DATA_DIR, "X_test.npy"))
        y_test = np.load(os.path.join(config.PROCESSED_DATA_DIR, "y_test.npy"))
    except FileNotFoundError:
        print("  [!] Test data not found.")
        return 1

    if not os.path.exists(config.MODEL_PATH):
        print("  [!] No model checkpoint found.")
        return 1

    # Load model
    model = GestureRecognitionModel()
    checkpoint = torch.load(config.MODEL_PATH, map_location="cpu", weights_only=True)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError:
        print("  [!] Cannot load model (architecture mismatch). Retrain required.")
        return 1

    model.eval()

    # Run inference on test set
    confidences = []
    correct_confidences = []
    wrong_confidences = []
    predictions = []

    with torch.no_grad():
        for i in range(len(X_test)):
            x = torch.FloatTensor(X_test[i]).unsqueeze(0)
            pred_class, confidence, probs = model.predict_with_confidence(x)

            confidences.append(confidence)
            predictions.append(pred_class)

            if pred_class == y_test[i]:
                correct_confidences.append(confidence)
            else:
                wrong_confidences.append(confidence)

    confidences = np.array(confidences)
    accuracy = np.mean(np.array(predictions) == y_test)

    print(f"  Test Accuracy: {accuracy:.1%}")
    print(f"  Confidence Stats:")
    print(f"    Overall:   mean={confidences.mean():.3f}, std={confidences.std():.3f}")
    print(f"    Correct:   mean={np.mean(correct_confidences):.3f} ({len(correct_confidences)} samples)")
    if wrong_confidences:
        print(f"    Wrong:     mean={np.mean(wrong_confidences):.3f} ({len(wrong_confidences)} samples)")
    else:
        print(f"    Wrong:     none (all correct)")

    # Per-class accuracy
    print(f"\n  Per-Class Accuracy:")
    for class_idx in range(config.NUM_GESTURES):
        mask = y_test == class_idx
        if mask.sum() > 0:
            class_acc = np.mean(np.array(predictions)[mask] == class_idx)
            class_conf = confidences[mask].mean()
            name = config.GESTURES[class_idx] if class_idx < len(config.GESTURES) else f"Class_{class_idx}"
            status = "OK" if class_acc >= 0.7 else "LOW"
            print(f"    {name:>10}: acc={class_acc:.0%}, conf={class_conf:.2f}  [{status}]")

    # Warn about overconfidence
    if confidences.mean() > 0.95 and accuracy < 0.9:
        print(f"\n  [!!] Model is OVERCONFIDENT (high confidence but low accuracy).")
        print(f"       This suggests overfitting. Retrain with more data.")

    return 0


def run_diagnostics():
    """Run all diagnostic checks and report overall health."""
    print("=" * 60)
    print("  PIPELINE DIAGNOSTIC REPORT")
    print("=" * 60)
    print(f"  Active gestures: {config.NUM_GESTURES}")
    print(f"  Gestures: {config.GESTURES}")
    print(f"  Hands mode: {config.MAX_NUM_HANDS} hands ({config.INPUT_FEATURES} features/frame)")
    print(f"  Sequence length: {config.SEQUENCE_LENGTH} frames")

    total_issues = 0
    total_issues += check_raw_data_coverage()
    total_issues += check_label_consistency()
    total_issues += check_data_ranges()
    total_issues += check_data_leakage()
    total_issues += check_model_confidence()

    print("\n" + "=" * 60)
    if total_issues == 0:
        print("  [OK] ALL CHECKS PASSED -- Pipeline is healthy!")
    else:
        print(f"  [!!] {total_issues} ISSUE(S) FOUND -- See details above.")
        print(f"  Recommended: Re-run 'python run.py process' then 'python run.py train'")
    print("=" * 60 + "\n")

    return total_issues


if __name__ == "__main__":
    run_diagnostics()
