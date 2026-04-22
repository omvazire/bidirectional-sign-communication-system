"""
Flask Blueprint version of the sign language translator app.

This wraps the original app.py logic into a Blueprint that can be mounted
at any URL prefix (e.g. /sign-translator) by the unified app.py.

All ML models and detector logic are completely unchanged.
"""

import atexit
import os
import secrets
import sys
import threading
import time

import cv2
import numpy as np
from flask import (
    Blueprint,
    jsonify,
    make_response,
    render_template,
    request,
    send_file,
    url_for,
)
from werkzeug.exceptions import RequestEntityTooLarge

# ── Ensure the sign_translator package root is on sys.path ──────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SIGN_ROOT = os.path.dirname(_THIS_DIR)  # sign_translator/
if _SIGN_ROOT not in sys.path:
    sys.path.insert(0, _SIGN_ROOT)

import config  # noqa: E402
from src.detect_runtime import RealTimeDetector, ensure_browser_runtime_warmup  # noqa: E402

# ── Constants ───────────────────────────────────────────────────────────
SESSION_COOKIE_NAME = "sign_translator_session"
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "900"))
MAX_ACTIVE_SESSIONS = int(os.getenv("MAX_ACTIVE_SESSIONS", "6"))
MAX_FRAME_BYTES = int(os.getenv("MAX_FRAME_BYTES", str(2 * 1024 * 1024)))
HAND_LANDMARKER_MODEL_PATH = os.path.join(config.MODELS_DIR, "hand_landmarker.task")
MEDIAPIPE_TASKS_VISION_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12"


# ── Session management (same as original app.py) ───────────────────────
class BrowserSessionRuntime:
    """Track one browser session's detector and last activity time."""

    def __init__(self, session_id):
        self.session_id = session_id
        self.detector = RealTimeDetector(speech_mode="browser")
        self.last_seen = time.time()

    def touch(self):
        self.last_seen = time.time()

    def close(self):
        self.detector.close()


_sessions = {}
_sessions_lock = threading.Lock()


def _cleanup_expired_sessions(force=False):
    now = time.time()
    expired = []
    with _sessions_lock:
        for session_id, runtime in list(_sessions.items()):
            if force or (now - runtime.last_seen) > SESSION_TTL_SECONDS:
                expired.append(_sessions.pop(session_id))
        while len(_sessions) > MAX_ACTIVE_SESSIONS:
            oldest_session_id = min(_sessions, key=lambda sid: _sessions[sid].last_seen)
            expired.append(_sessions.pop(oldest_session_id))
    for runtime in expired:
        runtime.close()


def _build_session_response(payload, session_id=None, status_code=200, clear_cookie=False):
    response = make_response(jsonify(payload), status_code)
    if clear_cookie:
        response.delete_cookie(SESSION_COOKIE_NAME)
    elif session_id is not None:
        response.set_cookie(
            SESSION_COOKIE_NAME,
            session_id,
            max_age=SESSION_TTL_SECONDS,
            samesite="Lax",
            secure=request.is_secure,
        )
    return response


def _get_or_create_runtime():
    ensure_browser_runtime_warmup()
    _cleanup_expired_sessions()
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    with _sessions_lock:
        runtime = _sessions.get(session_id) if session_id else None
        if runtime is None:
            session_id = secrets.token_urlsafe(18)
            runtime = BrowserSessionRuntime(session_id)
            _sessions[session_id] = runtime
            print(f"[OK] Browser session created: {session_id}")
        runtime.touch()
        return session_id, runtime


def _get_existing_runtime():
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if not session_id:
        return None, None
    _cleanup_expired_sessions()
    with _sessions_lock:
        runtime = _sessions.get(session_id)
        if runtime is not None:
            runtime.touch()
        return session_id, runtime


def _decode_uploaded_frame():
    uploaded_file = request.files.get("frame")
    if uploaded_file is not None:
        raw_bytes = uploaded_file.read()
    else:
        raw_bytes = request.get_data(cache=False)
    if not raw_bytes:
        raise ValueError("Uploaded frame was empty.")
    encoded = np.frombuffer(raw_bytes, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode uploaded frame.")
    return frame


def _prepare_frame_for_inference(frame):
    height, width = frame.shape[:2]
    width_scale = config.WEB_MAX_INFERENCE_WIDTH / max(width, 1)
    height_scale = config.WEB_MAX_INFERENCE_HEIGHT / max(height, 1)
    scale = min(width_scale, height_scale, 1.0)
    if scale >= 1.0:
        return frame
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def _parse_landmarks_payload():
    payload = request.get_json(silent=True)
    if payload is None:
        raise ValueError("Expected a JSON landmarks payload.")
    landmarks = payload.get("landmarks")
    hands_status = str(payload.get("hands", "NONE"))
    if landmarks is None:
        return None, hands_status
    if not isinstance(landmarks, list):
        raise ValueError("Landmarks payload must be a flat list.")
    if len(landmarks) != config.INPUT_FEATURES:
        raise ValueError(
            f"Expected {config.INPUT_FEATURES} landmark values, got {len(landmarks)}."
        )
    try:
        values = [float(value) for value in landmarks]
    except (TypeError, ValueError) as exc:
        raise ValueError("Landmark values must be numeric.") from exc
    return values, hands_status


def shutdown_sessions():
    _cleanup_expired_sessions(force=True)


atexit.register(shutdown_sessions)


# ── Blueprint definition ───────────────────────────────────────────────
sign_bp = Blueprint(
    "sign_translator",
    __name__,
    template_folder=os.path.join(_SIGN_ROOT, "templates"),
    static_folder=os.path.join(_SIGN_ROOT, "static"),
    static_url_path="/static",
)


@sign_bp.after_request
def add_no_store_headers(response):
    if (
        request.blueprint == sign_bp.name
        and request.endpoint
        and request.endpoint.startswith(f"{sign_bp.name}.api_")
    ):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response


@sign_bp.route("/")
def index():
    """Serve the sign translator browser UI."""
    ensure_browser_runtime_warmup()
    command_endpoint = url_for("sign_translator.api_speak").rsplit("/", 1)[0]
    runtime_config = {
        "captureWidth": config.WEB_CAPTURE_WIDTH,
        "captureHeight": config.WEB_CAPTURE_HEIGHT,
        "cameraFrameRate": config.WEB_CAMERA_FRAME_RATE,
        "detectionIntervalMs": config.WEB_DETECTION_INTERVAL_MS,
        "jpegQuality": config.WEB_JPEG_QUALITY / 100.0,
        "successDelayMs": config.WEB_SUCCESS_DELAY_MS,
        "retryDelayMs": config.WEB_RETRY_DELAY_MS,
        "minHandDetectionConfidence": config.WEB_MIN_HAND_DETECTION_CONFIDENCE,
        "minHandPresenceConfidence": config.WEB_MIN_HAND_PRESENCE_CONFIDENCE,
        "minHandTrackingConfidence": config.WEB_MIN_HAND_TRACKING_CONFIDENCE,
        "featuresPerHand": config.FEATURES_PER_HAND,
        "inputFeatures": config.INPUT_FEATURES,
        "landmarksEndpoint": url_for("sign_translator.api_landmarks"),
        "sessionEndpoint": url_for("sign_translator.api_session"),
        "commandEndpoint": command_endpoint,
        "tasksVisionUrl": f"{MEDIAPIPE_TASKS_VISION_URL}/vision_bundle.mjs",
        "wasmRoot": f"{MEDIAPIPE_TASKS_VISION_URL}/wasm",
        "handLandmarkerModelPath": url_for("sign_translator.browser_hand_landmarker_model"),
    }
    return render_template(
        "index.html",
        runtime_config=runtime_config,
        stylesheet_url=url_for("sign_translator.static", filename="style.css"),
    )


@sign_bp.route("/browser-assets/hand_landmarker.task")
def browser_hand_landmarker_model():
    return send_file(
        HAND_LANDMARKER_MODEL_PATH,
        mimetype="application/octet-stream",
        conditional=True,
        max_age=3600,
    )


@sign_bp.route("/healthz")
def healthz():
    _cleanup_expired_sessions()
    with _sessions_lock:
        active_sessions = len(_sessions)
    return jsonify({"status": "ok", "active_sessions": active_sessions})


@sign_bp.errorhandler(RequestEntityTooLarge)
def request_too_large(_exc):
    max_size_kb = max(1, MAX_FRAME_BYTES // 1024)
    return _build_session_response(
        {
            "success": False,
            "error": f"Uploaded payload exceeds the {max_size_kb} KB request limit.",
        },
        status_code=413,
    )


@sign_bp.route("/api/session", methods=["POST"])
def api_session():
    try:
        session_id, runtime = _get_or_create_runtime()
        return _build_session_response(
            {"success": True, "state": runtime.detector.get_state()},
            session_id=session_id,
        )
    except Exception as exc:
        return _build_session_response({"success": False, "error": str(exc)}, status_code=503)


@sign_bp.route("/api/session", methods=["DELETE"])
def api_session_delete():
    session_id, runtime = _get_existing_runtime()
    if runtime is None:
        return _build_session_response({"success": True}, clear_cookie=True)
    with _sessions_lock:
        _sessions.pop(session_id, None)
    runtime.close()
    print(f"[i] Browser session closed: {session_id}")
    return _build_session_response({"success": True}, clear_cookie=True)


@sign_bp.route("/api/frame", methods=["POST"])
def api_frame():
    try:
        session_id, runtime = _get_or_create_runtime()
        frame = _prepare_frame_for_inference(_decode_uploaded_frame())
        state = runtime.detector.process_browser_frame(frame)
        return _build_session_response(
            {"success": True, "state": state},
            session_id=session_id,
        )
    except ValueError as exc:
        return _build_session_response({"success": False, "error": str(exc)}, status_code=400)
    except Exception as exc:
        return _build_session_response({"success": False, "error": str(exc)}, status_code=500)


@sign_bp.route("/api/landmarks", methods=["POST"])
def api_landmarks():
    try:
        session_id, runtime = _get_or_create_runtime()
        landmarks, hands_status = _parse_landmarks_payload()
        state = runtime.detector.process_browser_landmarks(landmarks, hands_status=hands_status)
        return _build_session_response(
            {"success": True, "state": state},
            session_id=session_id,
        )
    except ValueError as exc:
        return _build_session_response({"success": False, "error": str(exc)}, status_code=400)
    except Exception as exc:
        return _build_session_response({"success": False, "error": str(exc)}, status_code=500)


@sign_bp.route("/api/state")
def api_state():
    session_id, runtime = _get_existing_runtime()
    if runtime is None:
        return _build_session_response({"success": False, "error": "No active session."}, status_code=404)
    return _build_session_response(
        {"success": True, "state": runtime.detector.get_state()},
        session_id=session_id,
    )


@sign_bp.route("/api/command/speak", methods=["POST"])
def api_speak():
    session_id, runtime = _get_or_create_runtime()
    result = runtime.detector.trigger_speak()
    return _build_session_response(
        {
            "success": result["success"],
            "speech_text": result["speech_text"],
            "state": runtime.detector.get_state(),
        },
        session_id=session_id,
    )


@sign_bp.route("/api/command/clear", methods=["POST"])
def api_clear():
    session_id, runtime = _get_or_create_runtime()
    return _build_session_response(
        {
            "success": runtime.detector.trigger_clear(),
            "state": runtime.detector.get_state(),
        },
        session_id=session_id,
    )


@sign_bp.route("/api/command/undo", methods=["POST"])
def api_undo():
    session_id, runtime = _get_or_create_runtime()
    return _build_session_response(
        {
            "success": runtime.detector.trigger_undo(),
            "state": runtime.detector.get_state(),
        },
        session_id=session_id,
    )
