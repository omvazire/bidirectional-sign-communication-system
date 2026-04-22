"""
Data Collection Module -- Guided Webcam Recording for Gesture Sequences.

Usage: python run.py collect

Records hand landmark sequences for each gesture class. Provides a guided UI
with countdown timers, visual feedback, and resume support.
"""

import os
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from utils.helpers import (
    ensure_dirs, create_hand_detector, detect_hands, extract_landmarks,
    preprocess_landmark_frame, draw_hand_landmarks, draw_info_panel,
    draw_progress_bar,
)


def count_existing_sequences(gesture):
    """Count how many sequences already exist for a gesture."""
    gesture_dir = os.path.join(config.RAW_DATA_DIR, gesture)
    if not os.path.exists(gesture_dir):
        return 0
    return len([f for f in os.listdir(gesture_dir) if f.endswith(".npy")])


def collect_data():
    """
    Main data collection pipeline.

    For each gesture, records NUM_SEQUENCES sequences of SEQUENCE_LENGTH frames.
    Each frame captures 21 hand landmarks (63 values).
    Supports resuming -- skips already collected sequences.
    """
    ensure_dirs()
    print("\n" + "=" * 60)
    print("  SIGN LANGUAGE TRANSLATOR -- DATA COLLECTION")
    print("=" * 60)
    print(f"\n  Gestures:  {config.NUM_GESTURES}")
    print(f"  Sequences: {config.NUM_SEQUENCES} per gesture")
    print(f"  Frames:    {config.SEQUENCE_LENGTH} per sequence")
    print(f"\n  Controls:")
    print(f"    [S] Start recording a sequence")
    print(f"    [N] Skip to next gesture")
    print(f"    [Q] Quit and save progress")
    print("=" * 60 + "\n")

    # Initialize webcam
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print("[X] Error: Cannot open webcam. Check camera connection.")
        return

    # Initialize MediaPipe HandLandmarker (Tasks API)
    detector = create_hand_detector()

    # Timestamp tracker for VIDEO mode
    frame_timestamp_ms = 0

    total_collected = 0
    total_needed = config.NUM_GESTURES * config.NUM_SEQUENCES

    try:
        for gesture_idx, gesture in enumerate(config.GESTURES):
            existing = count_existing_sequences(gesture)
            start_seq = existing
            skip_gesture = False

            if existing >= config.NUM_SEQUENCES:
                print(f"[OK] {gesture}: All {config.NUM_SEQUENCES} sequences already collected. Skipping.")
                total_collected += config.NUM_SEQUENCES
                continue

            if existing > 0:
                print(f"[i] {gesture}: Resuming from sequence {existing + 1}/{config.NUM_SEQUENCES}")

            for seq_idx in range(start_seq, config.NUM_SEQUENCES):
                if skip_gesture:
                    break

                # -- Waiting Phase: Show instructions --
                waiting = True
                while waiting:
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    frame = cv2.flip(frame, 1)  # Mirror for natural interaction
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    frame_timestamp_ms += 33  # ~30 FPS
                    result = detect_hands(detector, rgb_frame, frame_timestamp_ms)

                    # Draw hand landmarks
                    draw_hand_landmarks(frame, result)

                    # Check which hands are detected
                    hand_detected = (result.hand_landmarks and len(result.hand_landmarks) > 0)
                    num_hands = len(result.hand_landmarks) if result.hand_landmarks else 0

                    if num_hands == 2:
                        hand_status = "Both Hands Detected (L+R)"
                    elif num_hands == 1:
                        # Show which hand
                        if result.handedness and len(result.handedness) > 0:
                            h_label = result.handedness[0][0].category_name
                            hand_status = f"{h_label} Hand Detected"
                        else:
                            hand_status = "1 Hand Detected"
                    else:
                        hand_status = "No Hand -- Show Your Hand(s)"
                    hand_color = (0, 255, 100) if hand_detected else (0, 80, 255)

                    # Info panel
                    progress = (total_collected + seq_idx) / total_needed
                    info_lines = [
                        (f"Gesture: {gesture} ({gesture_idx + 1}/{config.NUM_GESTURES})",
                         (255, 255, 255)),
                        (f"Sequence: {seq_idx + 1}/{config.NUM_SEQUENCES}  |  {hand_status}",
                         hand_color),
                        (f"Press [S] to record  |  [N] skip gesture  |  [Q] quit",
                         (180, 180, 180)),
                    ]
                    draw_info_panel(frame, info_lines, position="top")

                    # Overall progress bar
                    draw_progress_bar(frame, progress, frame.shape[0] - 15)

                    # Large gesture name in center
                    h, w = frame.shape[:2]
                    text_size = cv2.getTextSize(gesture.upper(), cv2.FONT_HERSHEY_SIMPLEX,
                                                2.0, 3)[0]
                    text_x = (w - text_size[0]) // 2
                    text_y = h // 2
                    cv2.putText(frame, gesture.upper(), (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3,
                                cv2.LINE_AA)

                    cv2.imshow("Sign Language Translator -- Data Collection", frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('s') or key == ord('S'):
                        if hand_detected:
                            waiting = False
                        else:
                            # Flash red warning
                            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 8)
                            cv2.imshow("Sign Language Translator -- Data Collection", frame)
                            cv2.waitKey(200)
                    elif key == ord('n') or key == ord('N'):
                        skip_gesture = True
                        waiting = False
                    elif key == ord('q') or key == ord('Q'):
                        print(f"\n[i] Collection paused. {total_collected + seq_idx} sequences saved.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                if skip_gesture:
                    print(f"[->] Skipping {gesture}")
                    break

                # -- Countdown Phase --
                for countdown in range(3, 0, -1):
                    start_time = time.time()
                    while time.time() - start_time < 1.0:
                        ret, frame = cap.read()
                        if not ret:
                            continue
                        frame = cv2.flip(frame, 1)
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        frame_timestamp_ms += 33
                        result = detect_hands(detector, rgb_frame, frame_timestamp_ms)
                        draw_hand_landmarks(frame, result)

                        # Countdown display
                        h, w = frame.shape[:2]
                        cv2.putText(frame, str(countdown), (w // 2 - 30, h // 2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 200, 255), 5,
                                    cv2.LINE_AA)
                        info = [(f"Get ready for: {gesture}", (0, 200, 255))]
                        draw_info_panel(frame, info, position="top")

                        cv2.imshow("Sign Language Translator -- Data Collection", frame)
                        cv2.waitKey(1)

                # -- Recording Phase --
                sequence = []
                frames_recorded = 0
                missed_frames = 0

                while frames_recorded < config.SEQUENCE_LENGTH:
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    frame = cv2.flip(frame, 1)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    frame_timestamp_ms += 33
                    result = detect_hands(detector, rgb_frame, frame_timestamp_ms)

                    landmarks = extract_landmarks(result)

                    if landmarks is not None:
                        normalized = preprocess_landmark_frame(landmarks)
                        sequence.append(normalized)
                        frames_recorded += 1
                    else:
                        # Use zero vector for missed frames
                        missed_frames += 1
                        if missed_frames > config.SEQUENCE_LENGTH:
                            break
                        sequence.append(np.zeros(config.INPUT_FEATURES, dtype=np.float32))
                        frames_recorded += 1

                    # Draw recording UI
                    draw_hand_landmarks(frame, result)

                    # Green border to indicate recording
                    h, w = frame.shape[:2]
                    cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 100), 4)

                    rec_progress = frames_recorded / config.SEQUENCE_LENGTH
                    info = [
                        (f"RECORDING: {gesture}  [{frames_recorded}/{config.SEQUENCE_LENGTH}]",
                         (0, 255, 100)),
                    ]
                    draw_info_panel(frame, info, position="top")
                    draw_progress_bar(frame, rec_progress, 80, color=(0, 255, 100))

                    cv2.imshow("Sign Language Translator -- Data Collection", frame)
                    cv2.waitKey(1)

                # Save sequence
                if len(sequence) == config.SEQUENCE_LENGTH:
                    sequence_array = np.array(sequence, dtype=np.float32)
                    save_path = os.path.join(
                        config.RAW_DATA_DIR, gesture, f"{seq_idx}.npy"
                    )
                    np.save(save_path, sequence_array)
                    print(f"  [{gesture}] Sequence {seq_idx + 1}/{config.NUM_SEQUENCES} saved "
                          f"({missed_frames} missed frames)")
                else:
                    print(f"  [{gesture}] Sequence {seq_idx + 1} FAILED -- too many missed frames. Retry.")

            total_collected += count_existing_sequences(gesture)

    except KeyboardInterrupt:
        print("\n[i] Collection interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()

    print(f"\n{'=' * 60}")
    print(f"  Collection complete!")
    print(f"  Total sequences: {total_collected}")
    print(f"  Next step: python run.py process")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    collect_data()
