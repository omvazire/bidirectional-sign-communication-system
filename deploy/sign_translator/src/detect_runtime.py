"""
Stable real-time detection loop for gesture recognition.

Design goals:
- share the exact same landmark preprocessing used during data collection
- reject weak predictions early
- smooth predictions with a recent-history majority vote
- only accept a gesture after repeated stable agreement
"""

import collections
import os
import sys
import textwrap
import threading
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.model import load_model
from src.sentence_builder import SentenceBuilder
from src.tts_runtime import get_tts_engine, shutdown_tts
from utils.helpers import (
    create_hand_detector,
    detect_hands,
    draw_confidence_bar,
    draw_hand_landmarks,
    draw_info_panel,
    extract_landmarks,
    get_device,
    preprocess_landmark_frame,
)


_BROWSER_MODEL = None
_BROWSER_DEVICE = None
_BROWSER_MODEL_LOCK = threading.Lock()
_BROWSER_INFERENCE_LOCK = threading.Lock()
_BROWSER_WARMUP_LOCK = threading.Lock()
_BROWSER_WARMUP_STARTED = False
_BROWSER_WARMUP_ERROR = ""


def _load_shared_browser_model():
    """Load the browser inference model once per process and reuse it across sessions."""
    global _BROWSER_DEVICE, _BROWSER_MODEL, _BROWSER_WARMUP_ERROR

    with _BROWSER_MODEL_LOCK:
        if _BROWSER_MODEL is None:
            _BROWSER_DEVICE = get_device()
            _BROWSER_MODEL = load_model(device=_BROWSER_DEVICE)
            _BROWSER_WARMUP_ERROR = ""

    return _BROWSER_DEVICE, _BROWSER_MODEL


def warm_browser_runtime():
    """Synchronously warm the shared browser model cache."""
    global _BROWSER_WARMUP_ERROR

    try:
        _load_shared_browser_model()
        return True
    except Exception as exc:
        _BROWSER_WARMUP_ERROR = str(exc)
        print(f"[X] Browser runtime warmup failed: {exc}")
        return False


def ensure_browser_runtime_warmup():
    """Kick off background browser-model warmup once per process."""
    global _BROWSER_WARMUP_STARTED

    with _BROWSER_WARMUP_LOCK:
        if _BROWSER_WARMUP_STARTED:
            return
        _BROWSER_WARMUP_STARTED = True

    threading.Thread(
        target=warm_browser_runtime,
        name="sign-browser-runtime-warmup",
        daemon=True,
    ).start()


class RealTimeDetector:
    """Reliable real-time detector built for stable predictions first."""

    def __init__(self, speech_mode="server"):
        print("\n" + "=" * 60)
        print("  SIGN LANGUAGE TRANSLATOR -- REAL-TIME DETECTION")
        print("=" * 60)

        if speech_mode not in {"server", "browser", "disabled"}:
            raise ValueError(f"Unsupported speech mode: {speech_mode}")

        self.speech_mode = speech_mode

        self.device = None
        self.model = None
        self.detector = None
        self.browser_runtime_ready = False

        if self.speech_mode == "browser":
            print("\n[1/4] Preparing shared browser inference runtime...")
            ensure_browser_runtime_warmup()
        else:
            self.device = get_device()
            print("\n[1/4] Loading model...")
            self.model = load_model(device=self.device)

        if self.speech_mode == "browser":
            print("[2/4] Browser hand tracking runs client-side...")
        else:
            print("[2/4] Initializing hand detector...")
            self.detector = create_hand_detector()

        print("[3/4] Initializing sentence builder...")
        self.sentence_builder = SentenceBuilder()

        print("[4/4] Initializing text-to-speech...")
        self.tts = get_tts_engine() if self.speech_mode == "server" else None

        self.frame_buffer = collections.deque(maxlen=config.SEQUENCE_LENGTH)
        self.prediction_history = collections.deque(maxlen=config.SMOOTHING_WINDOW)
        self.fps_counter = collections.deque(maxlen=30)

        self.consecutive_prediction = None
        self.consecutive_count = 0
        self.cooldown_counter = 0
        self.frame_timestamp_ms = 0

        self.current_prediction = None
        self.current_raw_prediction = None
        self.current_confidence = 0.0
        self.current_margin = 0.0
        self.current_top_candidates = []
        self.current_input_variant = "observed"
        self.frames_since_hand_lost = 0
        self.is_detecting = False
        self.hands_status = "NONE"
        self.last_accepted_gesture = ""
        self.last_guard_reason = ""
        self.last_guard_banner_at = 0.0
        self.window_name = "Sign Language Translator"
        self.banner_text = "Show a gesture to begin."
        self.banner_expires_at = time.time() + 3.0
        self.state_lock = threading.RLock()
        self.web_camera = None
        self.web_streaming = False
        self.pinned_sentence = ""
        self.pinned_sentence_source_raw = ""

        if self.speech_mode == "browser":
            print("[OK] Browser speech mode enabled")
        elif self.speech_mode == "disabled":
            print("[OK] Speech disabled for this runtime")

        print("\n  Controls:")
        print("    [SPACE]  Speak the current sentence")
        print("    [C]      Clear the current sentence")
        print("    [Z]      Undo last word")
        print("    [Q/ESC]  Quit")
        print("=" * 60 + "\n")

    def _ensure_browser_runtime_ready(self):
        """Attach this session runtime to the shared browser model cache."""
        if self.speech_mode != "browser":
            return

        if self.model is None or self.device is None:
            self.device, self.model = _load_shared_browser_model()

        self.browser_runtime_ready = True

    def _reset_prediction_state(self, clear_buffer=False):
        """Reset transient runtime state after hand loss or manual clear."""
        if clear_buffer:
            self.frame_buffer.clear()
        self.prediction_history.clear()
        self.consecutive_prediction = None
        self.consecutive_count = 0
        self.current_prediction = None
        self.current_raw_prediction = None
        self.current_confidence = 0.0
        self.current_margin = 0.0
        self.current_top_candidates = []
        self.current_input_variant = "observed"
        self.last_guard_reason = ""

    def _set_banner(self, text, duration=2.5):
        """Show a short-lived status banner in the live UI."""
        self.banner_text = text
        self.banner_expires_at = time.time() + duration

    def _wrap_sentence(self, sentence, width=40):
        """Wrap long sentences so the OpenCV UI stays readable."""
        if not sentence:
            return [""]
        return textwrap.wrap(sentence, width=width) or [sentence]

    def _get_display_sentence(self):
        """Return the sentence that should stay visible in the UI."""
        current_sentence = self.sentence_builder.get_corrected_sentence()
        current_raw_sentence = self.sentence_builder.get_raw_sentence()

        if self.pinned_sentence:
            if current_raw_sentence == self.pinned_sentence_source_raw:
                return self.pinned_sentence
            if not current_sentence:
                return self.pinned_sentence

        return current_sentence

    def _get_speech_status(self):
        """Return a UI-friendly speech backend status label."""
        if self.speech_mode == "server" and self.tts is not None:
            return self.tts.get_status()
        if self.speech_mode == "browser":
            return "Browser"
        return "Disabled"

    def _is_speaking(self):
        """Return whether a speech backend is actively speaking."""
        return bool(self.speech_mode == "server" and self.tts and self.tts.is_speaking())

    def _update_hands_status(self, result):
        """Update the short hands-visible status used in the UI."""
        if not result.hand_landmarks:
            self.hands_status = "NONE"
            return

        labels = []
        for i in range(len(result.hand_landmarks)):
            if result.handedness and i < len(result.handedness):
                labels.append(result.handedness[i][0].category_name[0])
            else:
                labels.append("?")

        if "L" in labels and "R" in labels:
            self.hands_status = "L+R"
        elif "L" in labels:
            self.hands_status = "L"
        elif "R" in labels:
            self.hands_status = "R"
        else:
            self.hands_status = "?"

    def _threshold_for(self, gesture_name):
        """Return the configured confidence threshold for a gesture."""
        return config.CUSTOM_THRESHOLDS.get(gesture_name, config.PREDICTION_THRESHOLD)

    def _margin_threshold_for(self, gesture_name):
        """Return the minimum winner-vs-runner-up margin for a gesture."""
        return config.CUSTOM_MARGIN_THRESHOLDS.get(gesture_name, config.PREDICTION_MARGIN)

    def _format_top_candidates(self, probabilities):
        """Return the top runtime candidates for UI/debugging."""
        top_indices = np.argsort(probabilities)[::-1][:config.LIVE_TOP_K]
        return [
            {
                "gesture": config.GESTURES[idx],
                "confidence": round(float(probabilities[idx]), 4),
                "label": f"{float(probabilities[idx]):.0%}",
            }
            for idx in top_indices
        ]

    def _passes_runtime_guards(self, top_gesture, top_confidence, second_gesture, second_confidence):
        """Reject shaky predictions before they enter the smoothing pipeline."""
        if top_confidence < self._threshold_for(top_gesture):
            return False, f"{top_gesture} below confidence threshold."

        guard_list = config.CONFUSION_GUARDS.get(top_gesture, [])
        if second_gesture and guard_list and second_gesture in guard_list:
            margin = top_confidence - second_confidence
            if margin < self._margin_threshold_for(top_gesture):
                return False, f"{top_gesture} too close to {second_gesture}."

        return True, ""

    def _note_guard_rejection(self, reason):
        """Show a short banner for repeated ambiguous live predictions without spamming."""
        if not reason:
            return

        self.last_guard_reason = reason
        now = time.time()
        if now - self.last_guard_banner_at >= 1.2:
            self._set_banner("Hold the gesture steady for a clearer match.", duration=1.3)
            self.last_guard_banner_at = now

    def _predict_sequence(self, sequence):
        """Run model inference for one candidate sequence."""
        if self.speech_mode == "browser":
            self._ensure_browser_runtime_ready()
            with _BROWSER_INFERENCE_LOCK:
                input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                predicted_class, confidence, probabilities = self.model.predict_with_confidence(input_tensor)
        else:
            input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            predicted_class, confidence, probabilities = self.model.predict_with_confidence(input_tensor)
        return {
            "predicted_class": predicted_class,
            "gesture": config.GESTURES[predicted_class],
            "confidence": confidence,
            "probabilities": probabilities,
        }

    def _canonicalize_single_hand_sequence(self, sequence, target_slot="left"):
        """
        Move all single-hand frames into one consistent slot.

        This guards against live handedness flicker where the same gesture jumps
        between left/right slots across frames even though the hand tracking dots look correct.
        """
        canonical = sequence.copy()
        fph = config.FEATURES_PER_HAND

        for frame_idx in range(len(canonical)):
            left = canonical[frame_idx, :fph]
            right = canonical[frame_idx, fph:]
            left_active = np.abs(left).sum() > 1e-6
            right_active = np.abs(right).sum() > 1e-6

            if left_active and right_active:
                continue
            if not left_active and not right_active:
                continue

            if target_slot == "left" and right_active and not left_active:
                canonical[frame_idx, :fph] = right
                canonical[frame_idx, fph:] = 0.0
            elif target_slot == "right" and left_active and not right_active:
                canonical[frame_idx, fph:] = left
                canonical[frame_idx, :fph] = 0.0

        return canonical

    def _sequence_single_hand_score(self, sequence):
        """Measure how strongly the buffered sequence behaves like a single-hand gesture."""
        fph = config.FEATURES_PER_HAND
        single_hand_frames = 0
        both_hand_frames = 0

        for frame in sequence:
            left_active = np.abs(frame[:fph]).sum() > 1e-6
            right_active = np.abs(frame[fph:]).sum() > 1e-6

            if left_active and right_active:
                both_hand_frames += 1
            elif left_active or right_active:
                single_hand_frames += 1

        return single_hand_frames, both_hand_frames

    def _candidate_sequences_for_prediction(self, sequence):
        """Generate sequence variants to handle single-hand slot instability."""
        candidates = [("observed", sequence)]
        single_hand_frames, both_hand_frames = self._sequence_single_hand_score(sequence)

        if single_hand_frames >= max(8, config.SEQUENCE_LENGTH // 2) and both_hand_frames <= 2:
            candidates.append(("single-left", self._canonicalize_single_hand_sequence(sequence, "left")))
            candidates.append(("single-right", self._canonicalize_single_hand_sequence(sequence, "right")))

        return candidates

    def process_frame(self, frame, draw_annotations=True):
        """Process one frame from the webcam and update detector state."""
        with self.state_lock:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            self.frame_timestamp_ms += 33
            result = detect_hands(self.detector, rgb_frame, self.frame_timestamp_ms)

            if draw_annotations:
                draw_hand_landmarks(frame, result)
            self._update_hands_status(result)

            landmarks = extract_landmarks(result)
            if landmarks is not None:
                processed = preprocess_landmark_frame(landmarks)
                self.frame_buffer.append(processed)
                self.is_detecting = True
                self.frames_since_hand_lost = 0
            else:
                self.frames_since_hand_lost += 1
                self.is_detecting = False
                if self.frames_since_hand_lost > config.RESET_AFTER_HAND_LOSS_FRAMES:
                    self._reset_prediction_state(clear_buffer=True)

            if len(self.frame_buffer) == config.SEQUENCE_LENGTH and self.is_detecting:
                self._predict()

            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1

        return frame

    def process_browser_frame(self, frame):
        """Process one browser-uploaded frame and return the latest state."""
        self._process_stream_frame(frame, draw_overlay=False, draw_annotations=False)
        return self.get_state()

    def process_browser_landmarks(self, landmarks, hands_status="NONE"):
        """Process browser-supplied hand landmarks and return the latest state."""
        if self.speech_mode == "browser" and self.model is None:
            self._ensure_browser_runtime_ready()

        landmark_array = None
        if landmarks is not None:
            landmark_array = np.asarray(landmarks, dtype=np.float32)
            if landmark_array.shape != (config.INPUT_FEATURES,):
                raise ValueError(
                    f"Expected {config.INPUT_FEATURES} landmark values, got {landmark_array.shape}."
                )

        with self.state_lock:
            self.fps_counter.append(time.time())
            self.hands_status = hands_status or "NONE"

            if landmark_array is not None:
                processed = preprocess_landmark_frame(landmark_array)
                self.frame_buffer.append(processed)
                self.is_detecting = True
                self.frames_since_hand_lost = 0
            else:
                self.hands_status = "NONE"
                self.frames_since_hand_lost += 1
                self.is_detecting = False
                if self.frames_since_hand_lost > config.RESET_AFTER_HAND_LOSS_FRAMES:
                    self._reset_prediction_state(clear_buffer=True)

            if len(self.frame_buffer) == config.SEQUENCE_LENGTH and self.is_detecting:
                self._predict()

            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1

        return self.get_state()

    def _predict(self):
        """Run model inference and stabilize the output before accepting it."""
        sequence = np.array(list(self.frame_buffer), dtype=np.float32)
        best_variant_name = "observed"
        best_result = None

        for variant_name, candidate_sequence in self._candidate_sequences_for_prediction(sequence):
            candidate_result = self._predict_sequence(candidate_sequence)
            if best_result is None or candidate_result["confidence"] > best_result["confidence"]:
                best_result = candidate_result
                best_variant_name = variant_name

        raw_gesture = best_result["gesture"]
        raw_confidence = best_result["confidence"]
        probabilities = best_result["probabilities"]
        self.current_raw_prediction = raw_gesture
        self.current_confidence = raw_confidence
        self.current_top_candidates = self._format_top_candidates(probabilities)
        self.current_input_variant = best_variant_name
        second_gesture = self.current_top_candidates[1]["gesture"] if len(self.current_top_candidates) > 1 else ""
        second_confidence = self.current_top_candidates[1]["confidence"] if len(self.current_top_candidates) > 1 else 0.0
        self.current_margin = max(0.0, raw_confidence - second_confidence)

        passes_guards, guard_reason = self._passes_runtime_guards(
            raw_gesture,
            raw_confidence,
            second_gesture,
            second_confidence,
        )
        if not passes_guards:
            self.current_prediction = None
            self.prediction_history.clear()
            self.consecutive_prediction = None
            self.consecutive_count = 0
            self._note_guard_rejection(guard_reason)
            return

        self.last_guard_reason = ""

        self.prediction_history.append(raw_gesture)

        majority_gesture, majority_count = collections.Counter(self.prediction_history).most_common(1)[0]
        majority_ratio = majority_count / len(self.prediction_history)

        if majority_ratio < config.MIN_MAJORITY_RATIO:
            self.current_prediction = None
            self.consecutive_prediction = None
            self.consecutive_count = 0
            return

        if majority_gesture == self.consecutive_prediction:
            self.consecutive_count += 1
        else:
            self.consecutive_prediction = majority_gesture
            self.consecutive_count = 1

        self.current_prediction = majority_gesture

        if (
            self.consecutive_count >= config.MIN_CONSECUTIVE_FRAMES
            and self.cooldown_counter == 0
        ):
            self._accept_prediction(majority_gesture)

    def _accept_prediction(self, gesture):
        """Accept a stable gesture and append it to the sentence."""
        added = self.sentence_builder.add_word(gesture)
        if added:
            self.cooldown_counter = config.COOLDOWN_FRAMES
            self.prediction_history.clear()
            self.consecutive_prediction = None
            self.consecutive_count = 0
            self.last_accepted_gesture = gesture
            self._set_banner(f'Accepted "{gesture}"')
            print(f"  -> Added: {gesture}")

    def draw_ui(self, frame):
        """Draw the runtime UI on top of the frame."""
        with self.state_lock:
            h, w = frame.shape[:2]

            status = "DETECTING" if self.is_detecting else "NO HAND"
            status_color = (0, 255, 100) if self.is_detecting else (0, 80, 255)
            hands_text = f"Hands: {self.hands_status}"
            pred_text = f"Stable: {self.current_prediction}" if self.current_prediction else "Stable: ---"
            raw_text = (
                f"Raw: {self.current_raw_prediction} ({self.current_confidence:.0%})"
                if self.current_raw_prediction
                else "Raw: ---"
            )
            buf_status = f"Buffer: {len(self.frame_buffer)}/{config.SEQUENCE_LENGTH}"
            vote_status = f"Vote: {len(self.prediction_history)}/{config.SMOOTHING_WINDOW}"
            consec_status = f"Stable: {self.consecutive_count}/{config.MIN_CONSECUTIVE_FRAMES}"
            tts_status = f"TTS: {self._get_speech_status()}"
            last_text = f"Last accepted: {self.last_accepted_gesture or '---'}"
            alt_text = "Top: " + " | ".join(
                f"{candidate['gesture']} {candidate['label']}"
                for candidate in self.current_top_candidates[: config.LIVE_TOP_K]
            ) if self.current_top_candidates else "Top: ---"
            variant_text = f"Input layout: {self.current_input_variant}"

            top_lines = [
                ("AI Sign Language Translator", (255, 255, 255)),
                (f"Status: {status}  |  {hands_text}  |  {buf_status}", status_color),
                (f"{pred_text}  |  {raw_text}", (0, 255, 255) if self.current_prediction else (170, 170, 170)),
                (f"{vote_status}  |  {consec_status}  |  {tts_status}", (180, 180, 180)),
                (alt_text, (170, 205, 255)),
                (variant_text, (145, 185, 225)),
                (last_text, (150, 210, 255)),
            ]
            draw_info_panel(frame, top_lines, position="top")

            if self.current_confidence > 0 and self.is_detecting:
                label = self.current_prediction or self.current_raw_prediction or "---"
                draw_confidence_bar(frame, self.current_confidence, label, y_pos=190)

            if self.cooldown_counter > 0:
                progress = self.cooldown_counter / config.COOLDOWN_FRAMES
                cv2.rectangle(
                    frame,
                    (15, 210),
                    (15 + int(220 * progress), 216),
                    (255, 165, 0),
                    -1,
                )

            sentence = self._get_display_sentence()
            wrapped_sentence = self._wrap_sentence(sentence or "(waiting for gestures...)")
            bottom_lines = [(f"Sentence ({self.sentence_builder.get_word_count()} words):", (255, 255, 255))]
            for line in wrapped_sentence[:2]:
                bottom_lines.append((line, (230, 230, 230)))
            bottom_lines.append(("[SPACE] Speak  |  [C] Clear  |  [Z] Undo  |  [Q] Quit", (140, 140, 140)))
            draw_info_panel(frame, bottom_lines, position="bottom")

            if time.time() < self.banner_expires_at and self.banner_text:
                overlay = frame.copy()
                box_w = min(w - 80, 420)
                box_h = 42
                x1 = (w - box_w) // 2
                y1 = h // 2 - 30
                cv2.rectangle(overlay, (x1, y1), (x1 + box_w, y1 + box_h), (25, 25, 25), -1)
                cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
                cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), (0, 200, 255), 2)
                cv2.putText(
                    frame,
                    self.banner_text,
                    (x1 + 16, y1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            if self._is_speaking():
                cv2.putText(
                    frame,
                    "SPEAKING...",
                    (w - 180, 46),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 200, 255),
                    2,
                    cv2.LINE_AA,
                )

            if self.fps_counter:
                fps = len(self.fps_counter) / (self.fps_counter[-1] - self.fps_counter[0] + 1e-6)
                cv2.putText(
                    frame,
                    f"FPS: {fps:.0f}",
                    (w - 120, h - 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (100, 255, 100),
                    2,
                    cv2.LINE_AA,
                )

        return frame

    def _process_stream_frame(self, frame, draw_overlay=True, draw_annotations=True):
        """Apply the full runtime pipeline to a camera frame."""
        frame = cv2.flip(frame, 1)
        with self.state_lock:
            self.fps_counter.append(time.time())
        frame = self.process_frame(frame, draw_annotations=draw_annotations)
        if draw_overlay:
            return self.draw_ui(frame)
        return frame

    def get_state(self):
        """Return a browser-friendly snapshot of the current detector state."""
        with self.state_lock:
            browser_runtime_ready = (
                self.speech_mode != "browser"
                or self.browser_runtime_ready
                or _BROWSER_MODEL is not None
            )

            if self.speech_mode == "browser" and not browser_runtime_ready:
                status = "WARMING UP"
            else:
                status = "DETECTING" if self.is_detecting else "NO HAND"
            sentence = self._get_display_sentence()
            return {
                "status": status,
                "runtime_ready": browser_runtime_ready,
                "hands": self.hands_status,
                "prediction": self.current_prediction or "",
                "raw_prediction": self.current_raw_prediction or "",
                "confidence": f"{self.current_confidence:.0%}" if self.current_confidence else "0%",
                "confidence_value": round(float(self.current_confidence), 4),
                "margin_value": round(float(self.current_margin), 4),
                "input_variant": self.current_input_variant,
                "sentence": sentence,
                "word_count": self.sentence_builder.get_word_count(),
                "tts_status": self._get_speech_status(),
                "is_speaking": self._is_speaking(),
                "last_accepted": self.last_accepted_gesture or "",
                "top_candidates": list(self.current_top_candidates),
                "guard_reason": self.last_guard_reason,
                "banner": self.banner_text if time.time() < self.banner_expires_at else "",
                "runtime_error": _BROWSER_WARMUP_ERROR if self.speech_mode == "browser" else "",
                "buffer_fill": len(self.frame_buffer),
                "buffer_target": config.SEQUENCE_LENGTH,
                "vote_fill": len(self.prediction_history),
                "vote_target": config.SMOOTHING_WINDOW,
                "cooldown_remaining": self.cooldown_counter,
            }

    def trigger_speak(self):
        """Queue the current sentence for speech from the web UI."""
        with self.state_lock:
            sentence = self.sentence_builder.get_final_sentence()
            if not sentence:
                self._set_banner("No sentence to speak.")
                return {"success": False, "speech_text": ""}

            print(f'\n  >> Speaking: "{sentence}"')
            if self.speech_mode == "browser":
                self.pinned_sentence = sentence
                self.pinned_sentence_source_raw = self.sentence_builder.get_raw_sentence()
                self._reset_prediction_state(clear_buffer=False)
                self._set_banner("Sentence ready for browser speech. Use Clear to remove it.")
                return {"success": True, "speech_text": sentence}

            if self.speech_mode == "server" and self.tts and self.tts.speak(sentence):
                self.pinned_sentence = sentence
                self.pinned_sentence_source_raw = self.sentence_builder.get_raw_sentence()
                self._reset_prediction_state(clear_buffer=False)
                self._set_banner("Sentence queued for speech. Use Clear to remove it.")
                return {"success": True, "speech_text": ""}

            self._set_banner("TTS unavailable. Sentence kept.", duration=3.0)
            return {"success": False, "speech_text": ""}

    def trigger_clear(self):
        """Clear the current sentence and queued TTS items."""
        with self.state_lock:
            self.sentence_builder.clear()
            if self.tts is not None:
                self.tts.clear_queue()
            self.pinned_sentence = ""
            self.pinned_sentence_source_raw = ""
            self._reset_prediction_state(clear_buffer=False)
            self._set_banner("Sentence cleared.")
            print("  [Cleared]")
            return True

    def trigger_undo(self):
        """Undo the last accepted word."""
        with self.state_lock:
            if self.sentence_builder.undo():
                self._set_banner("Removed last word.")
                print("  [Undo]")
                return True

            self._set_banner("Nothing to undo.")
            return False

    def generate_frames(self):
        """Yield an MJPEG stream for the browser UI."""
        from utils.camera_stream import ThreadedCamera

        cap = None
        with self.state_lock:
            self.web_streaming = True

        try:
            cap = ThreadedCamera(
                src=config.CAMERA_INDEX,
                width=config.FRAME_WIDTH,
                height=config.FRAME_HEIGHT,
            )
            cap.start()
            with self.state_lock:
                self.web_camera = cap

            print("[OK] Web video stream started")

            while True:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.02)
                    continue

                frame = self._process_stream_frame(frame, draw_overlay=False)
                ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if not ok:
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )

        except GeneratorExit:
            pass
        finally:
            if cap is not None:
                cap.release()
            with self.state_lock:
                self.web_camera = None
                self.web_streaming = False
            print("[i] Web video stream stopped")

    def close(self):
        """Release long-lived detector resources."""
        with self.state_lock:
            if self.web_camera is not None:
                self.web_camera.release()
                self.web_camera = None
            if self.detector is not None:
                detector = self.detector
                self.detector = None
                try:
                    detector.close()
                except RuntimeError as exc:
                    print(f"[i] Detector shutdown skipped during interpreter teardown: {exc}")
        if self.speech_mode == "server":
            shutdown_tts()

    def handle_key(self, key):
        """Handle keyboard controls for the runtime loop."""
        if key in (ord("q"), ord("Q"), 27):
            return False

        if key == ord(" "):
            self.trigger_speak()

        elif key in (ord("c"), ord("C")):
            self.trigger_clear()

        elif key in (ord("z"), ord("Z")):
            self.trigger_undo()

        return True

    def run(self):
        """Start the webcam loop and run live detection."""
        from utils.camera_stream import ThreadedCamera

        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, config.FRAME_WIDTH, config.FRAME_HEIGHT)

            splash = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
            cv2.putText(
                splash,
                "Starting camera...",
                (60, config.FRAME_HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(self.window_name, splash)
            cv2.waitKey(1)
        except cv2.error as exc:
            print(f"[X] OpenCV UI error: {exc}")
            print("[!] The detector uses a desktop OpenCV window, not a browser UI.")
            return

        cap = ThreadedCamera(
            src=config.CAMERA_INDEX,
            width=config.FRAME_WIDTH,
            height=config.FRAME_HEIGHT,
        )

        try:
            cap.start()
        except ValueError as exc:
            print(f"[X] Error: Cannot open webcam. {exc}")
            return

        print("[OK] Webcam opened. Starting detection...\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[!] Frame capture failed. Retrying...")
                    continue

                frame = self._process_stream_frame(frame, draw_overlay=True)

                cv2.imshow(self.window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if not self.handle_key(key):
                    break

        except KeyboardInterrupt:
            print("\n[i] Detection stopped by user.")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.close()

        history = self.sentence_builder.get_history()
        if history:
            print("\n  Session Summary:")
            print(f"  Sentences spoken: {len(history)}")
            for i, sentence in enumerate(history, 1):
                print(f"    {i}. {sentence}")

        print(f"\n{'=' * 60}")
        print("  Session ended. Goodbye!")
        print(f"{'=' * 60}\n")


def run_detection():
    """Entry point for the stable detection system."""
    detector = RealTimeDetector()
    detector.run()


if __name__ == "__main__":
    run_detection()
