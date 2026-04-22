"""
Flask web app for the deployable browser-based sign language translator UI.
"""

import atexit
import os
import secrets
import sys
import threading
import time

import cv2
import numpy as np
from flask import Flask, jsonify, make_response, render_template, request, send_file, url_for
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.middleware.proxy_fix import ProxyFix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.detect_runtime import RealTimeDetector, ensure_browser_runtime_warmup


SESSION_COOKIE_NAME = "sign_translator_session"
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "900"))
MAX_ACTIVE_SESSIONS = int(os.getenv("MAX_ACTIVE_SESSIONS", "6"))
MAX_FRAME_BYTES = int(os.getenv("MAX_FRAME_BYTES", str(2 * 1024 * 1024)))
HAND_LANDMARKER_MODEL_PATH = os.path.join(config.MODELS_DIR, "hand_landmarker.task")
MEDIAPIPE_VISION_BUNDLE = "vendor/mediapipe/tasks-vision/vision_bundle.mjs"
MEDIAPIPE_WASM_ROOT = "vendor/mediapipe/tasks-vision/wasm"


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


app = Flask(__name__, template_folder="../templates", static_folder="../static")
app.config["MAX_CONTENT_LENGTH"] = MAX_FRAME_BYTES
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

_sessions = {}
_sessions_lock = threading.Lock()


def _cleanup_expired_sessions(force=False):
    """Close expired in-memory browser sessions."""
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
    """Return a JSON response and keep the session cookie in sync."""
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
    """Fetch the current browser runtime or create one lazily."""
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
    """Return the current browser runtime if it still exists."""
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
    """Decode an uploaded JPEG/PNG frame into an OpenCV BGR image."""
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
    """Downscale uploaded frames to the runtime's preferred inference size."""
    height, width = frame.shape[:2]
    width_scale = config.WEB_MAX_INFERENCE_WIDTH / max(width, 1)
    height_scale = config.WEB_MAX_INFERENCE_HEIGHT / max(height, 1)
    scale = min(width_scale, height_scale, 1.0)

    if scale >= 1.0:
        return frame

    new_size = (
        max(1, int(width * scale)),
        max(1, int(height * scale)),
    )
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def _parse_landmarks_payload():
    """Parse and validate the browser landmarks payload."""
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
    """Release all detector resources on process exit."""
    _cleanup_expired_sessions(force=True)


atexit.register(shutdown_sessions)


@app.after_request
def add_no_store_headers(response):
    """Disable caching for the live API routes."""
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response


@app.route("/")
def index():
    """Serve the browser UI."""
    ensure_browser_runtime_warmup()
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
        "landmarksEndpoint": "/api/landmarks",
        "sessionEndpoint": "/api/session",
        "commandEndpoint": "/api/command",
        "tasksVisionUrl": url_for("static", filename=MEDIAPIPE_VISION_BUNDLE),
        "wasmRoot": url_for("static", filename=MEDIAPIPE_WASM_ROOT),
        "handLandmarkerModelPath": url_for("browser_hand_landmarker_model"),
    }
    return render_template(
        "index.html",
        runtime_config=runtime_config,
        stylesheet_url=url_for("static", filename="style.css"),
    )


@app.route("/browser-assets/hand_landmarker.task")
def browser_hand_landmarker_model():
    """Serve the hand landmarker model to the browser runtime."""
    return send_file(
        HAND_LANDMARKER_MODEL_PATH,
        mimetype="application/octet-stream",
        conditional=True,
        max_age=3600,
    )


@app.route("/healthz")
def healthz():
    """Lightweight health check for platform probes."""
    _cleanup_expired_sessions()
    with _sessions_lock:
        active_sessions = len(_sessions)
    return jsonify({"status": "ok", "active_sessions": active_sessions})


@app.errorhandler(RequestEntityTooLarge)
def request_too_large(_exc):
    """Return a JSON error instead of Flask's default HTML 413 page."""
    max_size_kb = max(1, MAX_FRAME_BYTES // 1024)
    return _build_session_response(
        {
            "success": False,
            "error": f"Uploaded payload exceeds the {max_size_kb} KB request limit.",
        },
        status_code=413,
    )


@app.route("/api/session", methods=["POST"])
def api_session():
    """Create or resume a browser session."""
    try:
        session_id, runtime = _get_or_create_runtime()
        return _build_session_response(
            {"success": True, "state": runtime.detector.get_state()},
            session_id=session_id,
        )
    except Exception as exc:
        return _build_session_response({"success": False, "error": str(exc)}, status_code=503)


@app.route("/api/session", methods=["DELETE"])
def api_session_delete():
    """Close the current browser session eagerly."""
    session_id, runtime = _get_existing_runtime()
    if runtime is None:
        return _build_session_response({"success": True}, clear_cookie=True)

    with _sessions_lock:
        _sessions.pop(session_id, None)
    runtime.close()
    print(f"[i] Browser session closed: {session_id}")
    return _build_session_response({"success": True}, clear_cookie=True)


@app.route("/api/frame", methods=["POST"])
def api_frame():
    """Process one browser webcam frame and return updated detector state."""
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


@app.route("/api/landmarks", methods=["POST"])
def api_landmarks():
    """Process browser-side hand landmarks and return updated detector state."""
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


@app.route("/api/state")
def api_state():
    """Return the latest detector state for the current browser session."""
    session_id, runtime = _get_existing_runtime()
    if runtime is None:
        return _build_session_response({"success": False, "error": "No active session."}, status_code=404)

    return _build_session_response(
        {"success": True, "state": runtime.detector.get_state()},
        session_id=session_id,
    )


@app.route("/api/command/speak", methods=["POST"])
def api_speak():
    """Prepare the sentence for browser speech."""
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


@app.route("/api/command/clear", methods=["POST"])
def api_clear():
    """Clear the active sentence."""
    session_id, runtime = _get_or_create_runtime()
    return _build_session_response(
        {
            "success": runtime.detector.trigger_clear(),
            "state": runtime.detector.get_state(),
        },
        session_id=session_id,
    )


@app.route("/api/command/undo", methods=["POST"])
def api_undo():
    """Undo the last accepted word."""
    session_id, runtime = _get_or_create_runtime()
    return _build_session_response(
        {
            "success": runtime.detector.trigger_undo(),
            "state": runtime.detector.get_state(),
        },
        session_id=session_id,
    )


def create_app():
    """WSGI entrypoint for production servers."""
    return app


def run_server():
    """Start the Flask server for the browser UI."""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))

    print("=" * 60)
    print("  SIGN LANGUAGE TRANSLATOR -- WEB UI")
    print(f"  Open: http://127.0.0.1:{port}")
    print("=" * 60)
    app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    run_server()
