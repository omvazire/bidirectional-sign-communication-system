"""
Data Processing Module — Preprocessing + Augmentation Pipeline.

Usage: python run.py process

Loads raw landmark sequences, validates them, and splits into train/val/test
sets. Augmentation is applied ONLY to the training set to prevent data leakage.

IMPORTANT: Raw data is already normalized during collection (normalize_landmarks).
           We do NOT re-normalize here to avoid double normalization, which would
           cause a train/inference mismatch.
"""

import os
import sys
import json
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from utils.helpers import ensure_dirs


# ─── Augmentation Functions ───────────────────────────────────

def augment_noise(sequence, std=None):
    """
    Add Gaussian noise to simulate sensor jitter and hand tremor.

    Args:
        sequence: np.ndarray (seq_len, features)
        std: noise standard deviation (default from config)
    Returns:
        Augmented sequence
    """
    if std is None:
        std = config.AUGMENT_NOISE_STD
    noise = np.random.normal(0, std, sequence.shape).astype(np.float32)
    return sequence + noise


def augment_time_stretch(sequence, rate=None):
    """
    Time-stretch a sequence by resampling to simulate speed variations.

    Args:
        sequence: np.ndarray (seq_len, features)
        rate: stretch factor (>1 = slower, <1 = faster). Random if None.
    Returns:
        Resampled sequence of original length
    """
    if rate is None:
        low, high = config.AUGMENT_TIME_STRETCH_RANGE
        rate = np.random.uniform(low, high)

    seq_len, features = sequence.shape
    new_len = int(seq_len * rate)
    new_len = max(new_len, 2)  # Minimum 2 frames for interpolation

    # Resample using linear interpolation
    old_indices = np.linspace(0, seq_len - 1, new_len)
    new_indices = np.linspace(0, new_len - 1, seq_len)

    stretched = np.zeros_like(sequence)
    for f in range(features):
        stretched_feature = np.interp(old_indices, np.arange(seq_len), sequence[:, f])
        stretched[:, f] = np.interp(new_indices, np.arange(new_len), stretched_feature)

    return stretched.astype(np.float32)


def augment_mirror(sequence):
    """
    Mirror landmarks horizontally by flipping x-coordinates.
    For dual-hand data, this also swaps the left and right hand slots,
    since mirroring makes the left hand become the right hand and vice versa.

    Each hand has 63 features (21 landmarks × 3 coords).
    x-coordinates are at indices 0, 3, 6, ..., 60 within each hand's block.
    """
    mirrored = sequence.copy()
    fph = config.FEATURES_PER_HAND  # 63

    # Swap left hand [0:63] and right hand [63:126]
    left_half = mirrored[:, :fph].copy()
    right_half = mirrored[:, fph:].copy()
    mirrored[:, :fph] = right_half
    mirrored[:, fph:] = left_half

    # Negate x-coordinates in BOTH hand blocks
    # x-coords are at indices 0, 3, 6, ..., 60 within each 63-feature block
    for hand_start in [0, fph]:
        mirrored[:, hand_start::3] = -mirrored[:, hand_start::3]

    return mirrored


def augment_scale(sequence, scale_range=(0.85, 1.15)):
    """
    Randomly scale all coordinates to simulate distance variation.
    """
    scale = np.random.uniform(*scale_range)
    return (sequence * scale).astype(np.float32)


# ─── Main Processing Pipeline ─────────────────────────────────

def load_raw_data():
    """
    Load all raw sequences and create labels.

    Only loads gestures that are in config.GESTURES (the active gesture set).
    Uses the gesture's index in config.GESTURES as the label, ensuring
    consistency between training and inference.

    Returns:
        sequences: list of np.ndarray (seq_len, features)
        labels: list of int (gesture class index)
        gesture_counts: dict of gesture_name -> count
    """
    sequences = []
    labels = []
    gesture_counts = {}

    for gesture_idx, gesture in enumerate(config.GESTURES):
        gesture_dir = os.path.join(config.RAW_DATA_DIR, gesture)

        if not os.path.exists(gesture_dir):
            print(f"  [!] Warning: No data directory for gesture '{gesture}'")
            gesture_counts[gesture] = 0
            continue

        files = sorted([f for f in os.listdir(gesture_dir) if f.endswith(".npy")])

        if len(files) == 0:
            print(f"  [!] Warning: No sequences found for '{gesture}'")
            gesture_counts[gesture] = 0
            continue

        loaded = 0
        for fname in files:
            filepath = os.path.join(gesture_dir, fname)
            seq = np.load(filepath)

            if seq.shape != (config.SEQUENCE_LENGTH, config.INPUT_FEATURES):
                print(f"  [!] Skipping {filepath}: unexpected shape {seq.shape}")
                continue

            sequences.append(seq)
            labels.append(gesture_idx)
            loaded += 1

        gesture_counts[gesture] = loaded

    return sequences, labels, gesture_counts


def apply_augmentation(X, y):
    """
    Apply data augmentation to increase dataset size and variety.

    IMPORTANT: This must ONLY be called on the training set.
    Applying augmentation before splitting causes data leakage.

    Augmentations:
    1. Gaussian noise (2 variants per sample)
    2. Time stretch (1 variant per sample)
    3. Mirror (1 variant per sample, if enabled)
    4. Scale variation (1 variant per sample)

    This ~5-6x the original dataset.
    """
    aug_X = list(X)
    aug_y = list(y)

    num_original = len(X)

    for i in range(num_original):
        seq = X[i]
        label = y[i]

        # 1. Noise augmentation (2 variants)
        aug_X.append(augment_noise(seq))
        aug_y.append(label)

        aug_X.append(augment_noise(seq, std=config.AUGMENT_NOISE_STD * 2))
        aug_y.append(label)

        # 2. Time stretch
        aug_X.append(augment_time_stretch(seq))
        aug_y.append(label)

        # 3. Mirror
        if config.AUGMENT_MIRROR:
            aug_X.append(augment_mirror(seq))
            aug_y.append(label)

        # 4. Scale
        aug_X.append(augment_scale(seq))
        aug_y.append(label)

    return np.array(aug_X, dtype=np.float32), np.array(aug_y, dtype=np.int64)


def process_data():
    """
    Full data processing pipeline:
    1. Load raw data (already normalized during collection)
    2. Split into train/val/test FIRST (prevents data leakage)
    3. Augment ONLY the training set
    4. Save processed arrays + label map
    """
    ensure_dirs()
    print("\n" + "=" * 60)
    print("  SIGN LANGUAGE TRANSLATOR — DATA PROCESSING")
    print("=" * 60)

    # Step 1: Load raw data
    print("\n[1/4] Loading raw data...")
    sequences, labels, gesture_counts = load_raw_data()

    if len(sequences) == 0:
        print("[X] No data found! Run 'python run.py collect' first.")
        return

    print(f"  Loaded {len(sequences)} sequences across {len([v for v in gesture_counts.values() if v > 0])} gestures")
    print()

    # Show per-gesture counts and warn about low counts
    min_count = float('inf')
    low_count_gestures = []
    for gesture, count in gesture_counts.items():
        status = "OK" if count >= 20 else "LOW" if count > 0 else "MISSING"
        marker = "  " if status == "OK" else "! " if status == "LOW" else "X "
        print(f"  {marker}{gesture:>10}: {count} sequences  [{status}]")
        if count > 0:
            min_count = min(min_count, count)
        if 0 < count < config.MIN_SEQUENCES_PER_GESTURE:
            low_count_gestures.append((gesture, count))

    if min_count < config.MIN_SEQUENCES_PER_GESTURE:
        print(
            f"\n  [!] WARNING: Some gestures have <{config.MIN_SEQUENCES_PER_GESTURE} sequences. "
            f"Consider collecting more data."
        )

    if low_count_gestures:
        print(f"\n  [X] Not enough data for a reliable stratified split.")
        print(f"      Each active gesture should have at least {config.MIN_SEQUENCES_PER_GESTURE} sequences.")
        print(f"      Either collect more data or temporarily reduce the active gesture list in config.py.")
        for gesture, count in low_count_gestures:
            print(f"      - {gesture}: {count} sequences")
        return

    # Convert to numpy
    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    print(f"\n  Dataset shape: X={X.shape}, y={y.shape}")

    # Verify data is already normalized (quick sanity check)
    print(f"  Data range: [{X.min():.3f}, {X.max():.3f}], mean={X.mean():.3f}")

    # Step 2: Split FIRST, then augment (CRITICAL: prevents data leakage)
    print("\n[2/4] Splitting dataset (BEFORE augmentation)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"  Train: {X_train.shape[0]} raw samples")
    print(f"  Val:   {X_val.shape[0]} raw samples (NO augmentation)")
    print(f"  Test:  {X_test.shape[0]} raw samples (NO augmentation)")

    # Step 3: Augment ONLY the training set
    print("\n[3/4] Augmenting training set ONLY...")
    original_train_count = len(X_train)
    X_train, y_train = apply_augmentation(X_train, y_train)
    print(f"  Train: {original_train_count} -> {len(X_train)} samples "
          f"({len(X_train)/original_train_count:.1f}x)")
    print(f"  Val/Test: untouched (clean evaluation data)")

    # Step 4: Save processed data + label map
    print("\n[4/4] Saving processed data...")
    np.save(os.path.join(config.PROCESSED_DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(config.PROCESSED_DATA_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(config.PROCESSED_DATA_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(config.PROCESSED_DATA_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(config.PROCESSED_DATA_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(config.PROCESSED_DATA_DIR, "y_test.npy"), y_test)

    # Save label map for consistency verification during inference
    label_map = {idx: name for idx, name in enumerate(config.GESTURES)}
    label_map_path = os.path.join(config.PROCESSED_DATA_DIR, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"  Label map saved: {label_map_path}")

    print(f"  Saved to: {config.PROCESSED_DATA_DIR}")

    print(f"\n{'=' * 60}")
    print(f"  Processing complete!")
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"  Next step: python run.py train")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    process_data()
