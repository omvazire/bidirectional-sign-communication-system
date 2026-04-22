"""
Central configuration for the Sign Language Translator.
All tunable parameters are defined here for easy experimentation.
"""

import os

# ─── Project Root ───────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ─── Gesture Definitions ───────────────────────────────────────
# Keep this list small while the pipeline is still being stabilized.
# A strong 8-12 gesture system is much more valuable than a noisy 20+ gesture one.
GESTURES = [
    # Core (original 10)
    "Hello", "Yes", "No", "Thanks", "Sorry",
    "Help", "Stop", "Water", "Eat", "Please",
    # Expanded (10 new)
    "Goodbye", "I", "You", "Come", "Go",
    "Good", "Bad", "Want", "Love", "Home",
]
NUM_GESTURES = len(GESTURES)
MIN_SEQUENCES_PER_GESTURE = 10  # Minimum needed for a reliable stratified split

# ─── Data Collection Parameters ────────────────────────────────
SEQUENCE_LENGTH = 25          # Frames per sequence (shorter = faster inference)
NUM_SEQUENCES = 35            # Sequences per gesture (more data for 20 classes)
NUM_KEYPOINTS = 21            # MediaPipe hand landmarks
FEATURES_PER_KEYPOINT = 3     # x, y, z coordinates
FEATURES_PER_HAND = NUM_KEYPOINTS * FEATURES_PER_KEYPOINT  # 63
NUM_HANDS = 2                 # Both hands (left + right)
INPUT_FEATURES = FEATURES_PER_HAND * NUM_HANDS  # 126 (63 per hand)

# ─── MediaPipe Parameters ──────────────────────────────────────
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5
MAX_NUM_HANDS = 2             # Both hands for reliable detection

# ─── Model Hyperparameters ─────────────────────────────────────
HIDDEN_SIZE = 128             # Smaller model generalizes better on limited data
NUM_LAYERS = 2                # Stacked Bi-LSTM layers
DROPOUT = 0.3                 # Moderate dropout for stability
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4           # L2 regularization
BATCH_SIZE = 16               # Smaller batches help on small datasets
EPOCHS = 80
EARLY_STOP_PATIENCE = 10
LABEL_SMOOTHING = 0.1         # Prevents overconfident predictions
USE_WEIGHTED_SAMPLER = True   # Oversample minority classes during training
USE_MIXUP = False             # Keep training simple until the baseline is stable
MIXUP_ALPHA = 0.2

# ─── Real-Time Detection Parameters ───────────────────────────
PREDICTION_THRESHOLD = 0.75   # Min confidence to accept prediction
CUSTOM_THRESHOLDS = {         # Higher thresholds for confusable gesture pairs
    "Hello": 0.85, "Goodbye": 0.85,   # Both involve waving
    "I": 0.85, "You": 0.85,           # Both involve pointing
    "Come": 0.80, "Go": 0.80,         # Opposite directions
    "Good": 0.85, "Bad": 0.85,        # Similar start position
    "Water": 0.84,                    # Can absorb Thanks/Please when tracking is noisy
}
PREDICTION_MARGIN = 0.04      # Generic winner-vs-runner-up confidence gap
CUSTOM_MARGIN_THRESHOLDS = {
    "Water": 0.10,
    "Goodbye": 0.12,
}
CONFUSION_GUARDS = {
    "Water": ["Thanks", "Please", "Goodbye"],
    "Goodbye": ["Please", "Thanks", "Water"],
}
LIVE_TOP_K = 3
RAW_DATA_WARNING_THRESHOLD = 24

# -- Smoothing & Stability --
SMOOTHING_WINDOW = 7          # Recent high-confidence predictions for majority vote
MIN_MAJORITY_RATIO = 0.6      # Majority vote confidence needed to trust a label
COOLDOWN_FRAMES = 15          # Frames to wait after accepting a word

# Require N stable majority-vote frames before accepting
MIN_CONSECUTIVE_FRAMES = 4
RESET_AFTER_HAND_LOSS_FRAMES = 10

CAMERA_INDEX = 0              # Default webcam
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Web runtime performance tuning
WEB_CAPTURE_WIDTH = 320       # Keep the browser preview light enough for hosted CPU runtimes
WEB_CAPTURE_HEIGHT = 240
WEB_CAMERA_FRAME_RATE = 18
WEB_MAX_INFERENCE_WIDTH = 320
WEB_MAX_INFERENCE_HEIGHT = 240
WEB_DETECTION_INTERVAL_MS = 60  # Roughly 16 FPS keeps tracking responsive without overloading the page
WEB_MIN_HAND_DETECTION_CONFIDENCE = 0.55
WEB_MIN_HAND_PRESENCE_CONFIDENCE = 0.45
WEB_MIN_HAND_TRACKING_CONFIDENCE = 0.45
WEB_JPEG_QUALITY = 60         # Browser canvas JPEG quality (0-100)
WEB_SUCCESS_DELAY_MS = 15     # Small gap after a successful request; effective FPS stays adaptive
WEB_RETRY_DELAY_MS = 120      # Backoff after transient network/server errors

# ─── TTS Parameters ───────────────────────────────────────────
TTS_RATE = 150                # Words per minute
TTS_VOLUME = 0.9              # Volume 0.0 to 1.0

# ─── Directory Paths ──────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "gesture_model.pth")
CONFUSION_MATRIX_PATH = os.path.join(MODELS_DIR, "confusion_matrix.png")
TRAINING_HISTORY_PATH = os.path.join(MODELS_DIR, "training_history.png")

# ─── Augmentation Parameters ─────────────────────────────────
AUGMENT_NOISE_STD = 0.01      # Gaussian noise standard deviation
AUGMENT_TIME_STRETCH_RANGE = (0.9, 1.1)  # Speed variation range
AUGMENT_MIRROR = True         # Flip x-coordinates
