"""
Project status and readiness report.

Usage: python run.py status

Gives a lightweight snapshot of:
- raw data collection progress
- processed dataset artifacts
- model assets
- recommended next pipeline step
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def _count_sequences(gesture):
    """Count raw .npy sequence files for a gesture."""
    gesture_dir = os.path.join(config.RAW_DATA_DIR, gesture)
    if not os.path.exists(gesture_dir):
        return 0
    return len([f for f in os.listdir(gesture_dir) if f.endswith(".npy")])


def _artifact_status(path):
    """Return a compact existence marker for a file path."""
    return "[OK]" if os.path.exists(path) else "[ ]"


def _recommend_next_step(total_raw, processed_ready, model_ready):
    """Suggest the most useful next command based on current state."""
    if total_raw == 0:
        return "python run.py collect"
    if not processed_ready:
        return "python run.py process"
    if not model_ready:
        return "python run.py train"
    return "python run.py diagnose  (then python run.py run)"


def run_status():
    """Print a project readiness report."""
    print("\n" + "=" * 60)
    print("  SIGN LANGUAGE TRANSLATOR -- PROJECT STATUS")
    print("=" * 60)

    print("\n[1/4] Gesture configuration")
    print(f"  Active gestures:     {config.NUM_GESTURES}")
    print(f"  Sequence length:     {config.SEQUENCE_LENGTH} frames")
    print(f"  Target per gesture:  {config.NUM_SEQUENCES} sequences")
    print(f"  Input features:      {config.INPUT_FEATURES} per frame")

    print("\n[2/4] Raw data collection progress")
    gesture_counts = {gesture: _count_sequences(gesture) for gesture in config.GESTURES}
    total_raw = sum(gesture_counts.values())
    gestures_started = sum(count > 0 for count in gesture_counts.values())
    gestures_complete = sum(count >= config.NUM_SEQUENCES for count in gesture_counts.values())
    target_total = config.NUM_GESTURES * config.NUM_SEQUENCES

    print(f"  Total collected:     {total_raw}/{target_total}")
    print(f"  Gestures started:    {gestures_started}/{config.NUM_GESTURES}")
    print(f"  Gestures complete:   {gestures_complete}/{config.NUM_GESTURES}")

    for gesture, count in gesture_counts.items():
        if count >= config.NUM_SEQUENCES:
            state = "READY"
        elif count >= config.RAW_DATA_WARNING_THRESHOLD:
            state = "GOOD"
        elif count > 0:
            state = "LOW"
        else:
            state = "MISSING"
        print(f"    {gesture:>10}: {count:>2}/{config.NUM_SEQUENCES}  [{state}]")

    low_coverage = [gesture for gesture, count in gesture_counts.items() if 0 < count < config.RAW_DATA_WARNING_THRESHOLD]
    if low_coverage:
        print(
            f"  [!] Live accuracy risk: {', '.join(low_coverage)} still have fewer than "
            f"{config.RAW_DATA_WARNING_THRESHOLD} raw sequences."
        )

    print("\n[3/4] Artifacts")
    processed_files = [
        os.path.join(config.PROCESSED_DATA_DIR, "X_train.npy"),
        os.path.join(config.PROCESSED_DATA_DIR, "y_train.npy"),
        os.path.join(config.PROCESSED_DATA_DIR, "X_val.npy"),
        os.path.join(config.PROCESSED_DATA_DIR, "y_val.npy"),
        os.path.join(config.PROCESSED_DATA_DIR, "X_test.npy"),
        os.path.join(config.PROCESSED_DATA_DIR, "y_test.npy"),
        os.path.join(config.PROCESSED_DATA_DIR, "label_map.json"),
    ]
    processed_ready = all(os.path.exists(path) for path in processed_files)
    model_ready = os.path.exists(config.MODEL_PATH)

    print(f"  {_artifact_status(os.path.join(config.MODELS_DIR, 'hand_landmarker.task'))} hand_landmarker.task")
    for path in processed_files:
        print(f"  {_artifact_status(path)} {os.path.basename(path)}")
    print(f"  {_artifact_status(config.MODEL_PATH)} {os.path.basename(config.MODEL_PATH)}")
    print(f"  {_artifact_status(config.TRAINING_HISTORY_PATH)} {os.path.basename(config.TRAINING_HISTORY_PATH)}")
    print(f"  {_artifact_status(config.CONFUSION_MATRIX_PATH)} {os.path.basename(config.CONFUSION_MATRIX_PATH)}")

    label_map_path = os.path.join(config.PROCESSED_DATA_DIR, "label_map.json")
    if os.path.exists(label_map_path):
        try:
            with open(label_map_path, "r", encoding="utf-8") as f:
                saved_map = json.load(f)
            saved_gestures = [saved_map[str(i)] for i in range(len(saved_map))]
            if saved_gestures == config.GESTURES:
                print("  [OK] label_map.json matches config.GESTURES")
            else:
                print("  [!] label_map.json exists but does not match current config")
        except (OSError, KeyError, json.JSONDecodeError):
            print("  [!] label_map.json exists but could not be validated")

    print("\n[4/4] Recommended next step")
    next_step = _recommend_next_step(total_raw, processed_ready, model_ready)
    print(f"  Next command: {next_step}")

    if total_raw == 0:
        print("  Note: the project is scaffolded correctly, but no training data has been recorded yet.")
    elif not processed_ready:
        print("  Note: you have raw data; preprocessing is the next milestone.")
    elif not model_ready:
        print("  Note: processed splits exist; the training checkpoint is the missing piece.")
    else:
        print("  Note: the full pipeline artifacts are present. Run diagnostics before live inference.")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_status()
