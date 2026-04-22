"""
Shared utility functions for the Sign Language Translator.
Handles directory setup, landmark extraction, normalization, and drawing.

Uses the modern MediaPipe Tasks API (HandLandmarker) instead of the
deprecated mp.solutions.hands API.
"""

import os
import numpy as np
import cv2
import torch
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ── Hand Landmark Connections (for drawing) ───────────────────
# The 21 MediaPipe hand landmarks and their connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm
]

# Landmark colors by finger group
LANDMARK_COLORS = {
    'thumb':  (0, 255, 255),   # Yellow
    'index':  (0, 200, 100),   # Green
    'middle': (255, 150, 0),   # Blue-ish
    'ring':   (200, 100, 255), # Purple
    'pinky':  (100, 200, 255), # Light blue
    'wrist':  (255, 255, 255), # White
}


# ── Directory Management ─────────────────────────────────────

def ensure_dirs():
    """Create all required project directories if they don't exist."""
    dirs = [
        config.DATA_DIR,
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.MODELS_DIR,
    ]
    for gesture in config.GESTURES:
        dirs.append(os.path.join(config.RAW_DATA_DIR, gesture))

    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"[OK] Directories verified ({len(dirs)} paths)")


# ── Device Detection ─────────────────────────────────────────

def get_device():
    """Auto-detect CUDA GPU or fallback to CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[OK] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[i] Using CPU (no CUDA GPU detected)")
    return device


# ── MediaPipe Hand Detection (Tasks API) ─────────────────────

# Path to the hand landmarker model
HAND_MODEL_PATH = os.path.join(config.MODELS_DIR, "hand_landmarker.task")


def create_hand_detector():
    """
    Create a MediaPipe HandLandmarker using the Tasks API.

    Returns a HandLandmarker object. Caller is responsible for closing it.
    """
    if not os.path.exists(HAND_MODEL_PATH):
        raise FileNotFoundError(
            f"Hand landmarker model not found at {HAND_MODEL_PATH}.\n"
            f"Download it with:\n"
            f"  python -c \"import urllib.request; urllib.request.urlretrieve("
            f"'https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            f"hand_landmarker/float16/1/hand_landmarker.task', "
            f"'{HAND_MODEL_PATH}')\""
        )

    base_options = mp_python.BaseOptions(
        model_asset_path=HAND_MODEL_PATH
    )
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=config.MAX_NUM_HANDS,
        min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=config.MIN_TRACKING_CONFIDENCE,
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)
    print("[OK] Hand landmarker initialized (Tasks API)")
    return detector


def detect_hands(detector, frame_rgb, timestamp_ms):
    """
    Run hand detection on a frame using the Tasks API.

    Args:
        detector: HandLandmarker instance
        frame_rgb: RGB numpy array (uint8)
        timestamp_ms: Frame timestamp in milliseconds

    Returns:
        result: HandLandmarkerResult with .hand_landmarks and .handedness
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect_for_video(mp_image, timestamp_ms)
    return result


def extract_landmarks(result):
    """
    Extract hand landmarks from a HandLandmarkerResult for BOTH hands.

    Returns a 126-feature vector: [left_hand_63 | right_hand_63].
    Each hand's slot is filled based on MediaPipe's handedness label.
    Missing hands are zero-padded.

    NOTE: MediaPipe's handedness is from the camera's perspective (mirrored),
    so we swap Left<->Right to match the user's actual hands when the
    frame is already mirrored (flipped) for natural interaction.

    Args:
        result: HandLandmarkerResult from detect_hands()

    Returns:
        np.ndarray of shape (126,) — [left_63 | right_63],
        or None if no hands are detected.
    """
    if not result.hand_landmarks or len(result.hand_landmarks) == 0:
        return None

    # Initialize with zeros (missing hands stay as zero-padded)
    combined = np.zeros(config.INPUT_FEATURES, dtype=np.float32)  # 126

    for i, hand in enumerate(result.hand_landmarks):
        # Extract 63 features for this hand
        landmarks = []
        for lm in hand:
            landmarks.extend([lm.x, lm.y, lm.z])
        hand_features = np.array(landmarks, dtype=np.float32)

        # Determine which slot (left=0:63, right=63:126)
        # MediaPipe returns handedness from camera's perspective,
        # but since we mirror the frame, "Left" from MediaPipe = user's left hand
        if result.handedness and i < len(result.handedness):
            label = result.handedness[i][0].category_name
            # After mirroring: MediaPipe "Left" = user's left hand
            if label == "Left":
                combined[:config.FEATURES_PER_HAND] = hand_features
            else:  # "Right"
                combined[config.FEATURES_PER_HAND:] = hand_features
        else:
            # Fallback: first hand -> left slot, second -> right slot
            if i == 0:
                combined[:config.FEATURES_PER_HAND] = hand_features
            else:
                combined[config.FEATURES_PER_HAND:] = hand_features

    return combined


def normalize_landmarks(landmarks):
    """
    Normalize landmarks relative to each hand's wrist (landmark 0).

    For the 126-feature vector, normalizes each hand independently:
    - Left hand [0:63]: centered on left wrist, scaled to unit box
    - Right hand [63:126]: centered on right wrist, scaled to unit box
    - Zero-padded (absent) hands are left as zeros.

    Args:
        landmarks: np.ndarray of shape (126,)
    Returns:
        np.ndarray of shape (126,) -- normalized
    """
    if landmarks is None:
        return None

    result = landmarks.copy()
    fph = config.FEATURES_PER_HAND  # 63

    for hand_idx in range(config.NUM_HANDS):
        start = hand_idx * fph
        end = start + fph
        hand_data = result[start:end]

        # Skip normalization for absent (zero-padded) hands
        if np.abs(hand_data).sum() < 1e-8:
            continue

        lm = hand_data.reshape(config.NUM_KEYPOINTS, config.FEATURES_PER_KEYPOINT)

        # Center relative to wrist (index 0)
        wrist = lm[0].copy()
        lm -= wrist

        # Scale to unit bounding box
        max_val = np.max(np.abs(lm))
        if max_val > 0:
            lm /= max_val

        result[start:end] = lm.flatten()

    return result


def preprocess_landmark_frame(landmarks):
    """
    Shared preprocessing used by both training data collection and inference.

    The contract is intentionally simple:
    1. extract raw landmarks
    2. normalize exactly once
    3. return float32 features
    """
    normalized = normalize_landmarks(landmarks)
    if normalized is None:
        return None
    return normalized.astype(np.float32)


def normalize_sequence(sequence):
    """
    Normalize a full sequence of landmarks.

    Args:
        sequence: np.ndarray of shape (seq_len, 63)
    Returns:
        np.ndarray of shape (seq_len, 63) -- normalized
    """
    normalized = np.zeros_like(sequence)
    for i in range(len(sequence)):
        result = normalize_landmarks(sequence[i])
        if result is not None:
            normalized[i] = result
    return normalized


# ── Drawing Utilities ─────────────────────────────────────────

def draw_hand_landmarks(frame, result):
    """
    Draw hand skeleton overlay on the frame using detection results.
    Uses different colors for left hand (cyan) and right hand (orange).

    Args:
        frame: BGR OpenCV image
        result: HandLandmarkerResult from detect_hands()
    Returns:
        frame: Annotated frame
    """
    if not result.hand_landmarks:
        return frame

    h, w = frame.shape[:2]

    # Color schemes per hand
    HAND_COLORS = {
        'Left':  {'bone': (230, 180, 0),   'joint': (255, 220, 50)},   # Cyan-ish
        'Right': {'bone': (0, 150, 255),   'joint': (50, 200, 255)},   # Orange-ish
    }

    for hand_idx, hand_landmarks in enumerate(result.hand_landmarks):
        # Determine handedness
        if result.handedness and hand_idx < len(result.handedness):
            label = result.handedness[hand_idx][0].category_name  # "Left" or "Right"
        else:
            label = "Left" if hand_idx == 0 else "Right"

        colors = HAND_COLORS.get(label, HAND_COLORS['Left'])

        # Draw connections (bones)
        for start_idx, end_idx in HAND_CONNECTIONS:
            start = hand_landmarks[start_idx]
            end = hand_landmarks[end_idx]

            start_pt = (int(start.x * w), int(start.y * h))
            end_pt = (int(end.x * w), int(end.y * h))

            cv2.line(frame, start_pt, end_pt, colors['bone'], 2, cv2.LINE_AA)

        # Draw landmarks (joints)
        for idx, lm in enumerate(hand_landmarks):
            cx, cy = int(lm.x * w), int(lm.y * h)

            if idx == 0:
                radius = 6
            else:
                radius = 4

            cv2.circle(frame, (cx, cy), radius, colors['joint'], -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), radius + 1, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw hand label (L/R) near wrist
        wrist = hand_landmarks[0]
        wx, wy = int(wrist.x * w), int(wrist.y * h)
        hand_label = "L" if label == "Left" else "R"
        cv2.putText(frame, hand_label, (wx - 10, wy + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['joint'], 2, cv2.LINE_AA)

    return frame


def draw_info_panel(frame, text_lines, position="top"):
    """
    Draw a semi-transparent info panel with text on the frame.

    Args:
        frame: OpenCV image (BGR)
        text_lines: list of (text, color) tuples
        position: 'top' or 'bottom'
    """
    h, w = frame.shape[:2]
    panel_height = 40 * len(text_lines) + 20

    if position == "top":
        y_start = 0
    else:
        y_start = h - panel_height

    # Semi-transparent dark overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y_start), (w, y_start + panel_height),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Draw text lines
    for i, (text, color) in enumerate(text_lines):
        y = y_start + 30 + i * 40
        cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2, cv2.LINE_AA)

    return frame


def draw_progress_bar(frame, progress, y_pos, color=(0, 255, 100)):
    """Draw a progress bar on the frame."""
    h, w = frame.shape[:2]
    bar_width = w - 40
    filled = int(bar_width * progress)

    cv2.rectangle(frame, (20, y_pos), (20 + bar_width, y_pos + 8),
                  (60, 60, 60), -1)
    cv2.rectangle(frame, (20, y_pos), (20 + filled, y_pos + 8),
                  color, -1)

    return frame


def draw_confidence_bar(frame, confidence, label, y_pos):
    """Draw a labeled confidence bar."""
    if confidence > 0.8:
        color = (0, 255, 100)   # Green
    elif confidence > 0.5:
        color = (0, 200, 255)   # Yellow
    else:
        color = (0, 80, 255)    # Red

    bar_width = 200
    filled = int(bar_width * confidence)

    cv2.putText(frame, f"{label}: {confidence:.0%}", (15, y_pos - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (15, y_pos), (15 + bar_width, y_pos + 10),
                  (60, 60, 60), -1)
    cv2.rectangle(frame, (15, y_pos), (15 + filled, y_pos + 10),
                  color, -1)

    return frame
