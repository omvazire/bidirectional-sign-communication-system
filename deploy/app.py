"""
Unified Flask application for the AI-Powered Communication System.

Serves three components from a single process on one port (7860):
  /                   → Dashboard hub
  /sign-translator/   → Sign language translator (Flask Blueprint)
  /text-to-sign/      → Text-to-sign 3D studio (pre-built React static files)
"""

import os
from flask import Flask, send_from_directory
from werkzeug.middleware.proxy_fix import ProxyFix

# ---------------------------------------------------------------------------
# Create the main Flask application
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)


# ---------------------------------------------------------------------------
# 1.  Dashboard Hub  –  serves the launcher/hub landing page
# ---------------------------------------------------------------------------
HUB_DIR = os.path.join(BASE_DIR, "hub")


@app.route("/")
def hub_index():
    """Serve the main dashboard page."""
    return send_from_directory(HUB_DIR, "index.html")


@app.route("/hub/<path:filename>")
def hub_static(filename):
    """Serve hub static assets (CSS, JS, favicon)."""
    return send_from_directory(HUB_DIR, filename)


# ---------------------------------------------------------------------------
# 2.  Sign Translator  –  mount existing Flask app as a Blueprint
# ---------------------------------------------------------------------------
from sign_translator.src.app_blueprint import MAX_FRAME_BYTES, sign_bp  # noqa: E402

app.config["MAX_CONTENT_LENGTH"] = MAX_FRAME_BYTES
app.register_blueprint(sign_bp, url_prefix="/sign-translator")


# ---------------------------------------------------------------------------
# 3.  Text-to-Sign Studio  –  serve pre-built React production files
# ---------------------------------------------------------------------------
TEXT_TO_SIGN_DIR = None
for candidate in (
    os.path.join(BASE_DIR, "text_to_sign", "dist"),
    os.path.join(BASE_DIR, "text_to_sign_dist"),
):
    if os.path.isdir(candidate):
        TEXT_TO_SIGN_DIR = candidate
        break

if TEXT_TO_SIGN_DIR is None:
    raise FileNotFoundError("Text-to-Sign build output was not found.")


@app.route("/text-to-sign/")
def text_to_sign_index():
    """Serve the React app entry point."""
    return send_from_directory(TEXT_TO_SIGN_DIR, "index.html")


@app.route("/text-to-sign/<path:filename>")
def text_to_sign_static(filename):
    """Serve React app static assets."""
    return send_from_directory(TEXT_TO_SIGN_DIR, filename)


# ---------------------------------------------------------------------------
# 4.  Health check
# ---------------------------------------------------------------------------
@app.route("/healthz")
def healthz():
    """Lightweight health probe for Hugging Face Spaces."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# WSGI entry point for Gunicorn
# ---------------------------------------------------------------------------
def create_app():
    return app


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    print("=" * 60)
    print("  AI POWERED COMMUNICATION SYSTEM FOR MUTE AND DEAF")
    print(f"  Dashboard : http://127.0.0.1:{port}")
    print(f"  Sign Trans: http://127.0.0.1:{port}/sign-translator/")
    print(f"  Text->Sign : http://127.0.0.1:{port}/text-to-sign/")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
