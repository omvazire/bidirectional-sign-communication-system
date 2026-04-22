from __future__ import annotations

import json
import mimetypes
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


HOST = os.getenv("COMM_HUB_HOST", "127.0.0.1")
PORT = int(os.getenv("COMM_HUB_PORT", "8090"))

LAUNCHER_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = LAUNCHER_DIR.parent
STATIC_DIR = LAUNCHER_DIR / "static"
LOG_DIR = LAUNCHER_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

NPM_COMMAND = "npm.cmd" if os.name == "nt" else "npm"
COMMON_HEADERS = {
    "Cache-Control": "no-store",
    "Pragma": "no-cache",
}


@dataclass
class ProcessRecord:
    process: subprocess.Popen
    log_path: Path
    started_at: float


PROJECTS = {
    "text-to-sign": {
        "id": "text-to-sign",
        "name": "Text To Sign Studio",
        "tagline": "React + Vite + 3D signing workspace",
        "folder": WORKSPACE_DIR / "sign-speak-text-to-sign--main",
        "url": f"http://{HOST}:5173",
        "health_url": f"http://{HOST}:5173",
        "command": [
            "node",
            str((WORKSPACE_DIR / "sign-speak-text-to-sign--main" / "node_modules" / "vite" / "bin" / "vite.js").resolve()),
            "--configLoader",
            "native",
            "--host",
            HOST,
            "--port",
            "5173",
            "--strictPort",
        ],
        "env": {},
        "launch_hint": "Runs the existing Vite app from its own folder without moving any files.",
        "install_hint": "Run `npm install` inside `sign-speak-text-to-sign--main` first.",
    },
    "sign-translator": {
        "id": "sign-translator",
        "name": "Sign Translator Web",
        "tagline": "Python browser-based real-time sign recognition",
        "folder": WORKSPACE_DIR / "test2",
        "url": f"http://{HOST}:5000",
        "health_url": f"http://{HOST}:5000/healthz",
        "command": [sys.executable, "run.py", "web"],
        "env": {
            "HOST": HOST,
            "PORT": "5000",
        },
        "launch_hint": "Runs the existing Python web app from its own folder and keeps its project structure intact.",
        "install_hint": "Make sure the Python packages from `requirements.txt` are installed in the Python you're using for the launcher.",
    },
}

_process_lock = threading.Lock()
_processes: dict[str, ProcessRecord] = {}


def _read_log_tail(log_path: Path, limit: int = 20) -> list[str]:
    if not log_path.exists():
        return []

    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []

    return lines[-limit:]


def _service_ready(url: str) -> bool:
    request = Request(url, headers={"User-Agent": "communication-hub"})

    try:
        with urlopen(request, timeout=1.5) as response:
            return 200 <= response.status < 500
    except HTTPError as exc:
        return 200 <= exc.code < 500
    except URLError:
        return False
    except TimeoutError:
        return False


def _get_record(project_id: str) -> ProcessRecord | None:
    with _process_lock:
        record = _processes.get(project_id)
        if record and record.process.poll() is not None:
            _processes.pop(project_id, None)
            return None
        return record


def _project_status(project_id: str) -> dict:
    project = PROJECTS[project_id]
    record = _get_record(project_id)
    reachable = _service_ready(project["health_url"])
    running = reachable or record is not None
    status = "running" if reachable else "starting" if record else "stopped"
    log_path = record.log_path if record else LOG_DIR / f"{project_id}.log"

    status_payload = {
        "id": project["id"],
        "name": project["name"],
        "tagline": project["tagline"],
        "folder": str(project["folder"]),
        "url": project["url"],
        "status": status,
        "reachable": reachable,
        "tracked": record is not None,
        "running": running,
        "startedAt": record.started_at if record else None,
        "launchHint": project["launch_hint"],
        "installHint": project["install_hint"],
        "logTail": _read_log_tail(log_path),
    }

    if record and record.process.poll() is not None:
        status_payload["exitCode"] = record.process.poll()

    return status_payload


def _dependency_warning(project_id: str) -> str | None:
    project = PROJECTS[project_id]

    if not project["folder"].exists():
        return f"Project folder not found: {project['folder']}"

    if project_id == "text-to-sign":
        node_modules = project["folder"] / "node_modules"
        if not node_modules.exists():
            return (
                "Dependencies are missing for the Vite project. "
                "Run `npm install` inside `sign-speak-text-to-sign--main`, then try launching again."
            )

    return None


def _start_project(project_id: str) -> tuple[dict, int]:
    project = PROJECTS[project_id]
    current_status = _project_status(project_id)
    if current_status["reachable"]:
        current_status["message"] = "Project is already running."
        return current_status, HTTPStatus.OK

    if current_status["tracked"]:
        current_status["message"] = "Project is already starting."
        return current_status, HTTPStatus.ACCEPTED

    warning = _dependency_warning(project_id)
    if warning:
        payload = _project_status(project_id)
        payload["error"] = warning
        return payload, HTTPStatus.CONFLICT

    log_path = LOG_DIR / f"{project_id}.log"
    log_file = log_path.open("a", encoding="utf-8")
    log_file.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting {project['name']}\n")
    log_file.flush()

    env = os.environ.copy()
    env.update(project["env"])

    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    try:
        process = subprocess.Popen(
            project["command"],
            cwd=project["folder"],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            creationflags=creationflags,
        )
    except FileNotFoundError:
        log_file.close()
        payload = _project_status(project_id)
        payload["error"] = (
            f"Could not find the launcher command `{project['command'][0]}`. "
            "Make sure it is installed and available in PATH."
        )
        return payload, HTTPStatus.INTERNAL_SERVER_ERROR
    except OSError as exc:
        log_file.close()
        payload = _project_status(project_id)
        payload["error"] = f"Unable to start the project: {exc}"
        return payload, HTTPStatus.INTERNAL_SERVER_ERROR

    with _process_lock:
        _processes[project_id] = ProcessRecord(
            process=process,
            log_path=log_path,
            started_at=time.time(),
        )

    log_file.close()
    payload = _project_status(project_id)
    payload["message"] = "Launch requested."
    return payload, HTTPStatus.ACCEPTED


class LauncherHandler(BaseHTTPRequestHandler):
    server_version = "CommunicationHub/1.0"

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/index.html":
            self._serve_static("index.html")
            return

        if self.path == "/api/projects":
            self._send_json(
                {"projects": [_project_status(project_id) for project_id in PROJECTS]},
                HTTPStatus.OK,
            )
            return

        if self.path.startswith("/api/projects/"):
            project_id = self.path.split("/")[3]
            if project_id not in PROJECTS:
                self._send_json({"error": "Unknown project"}, HTTPStatus.NOT_FOUND)
                return

            self._send_json({"project": _project_status(project_id)}, HTTPStatus.OK)
            return

        if self.path.startswith("/static/"):
            relative_path = self.path.replace("/static/", "", 1)
            self._serve_static(relative_path)
            return

        self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path.startswith("/api/projects/") and self.path.endswith("/launch"):
            project_id = self.path.split("/")[3]
            if project_id not in PROJECTS:
                self._send_json({"error": "Unknown project"}, HTTPStatus.NOT_FOUND)
                return

            payload, status_code = _start_project(project_id)
            self._send_json({"project": payload}, status_code)
            return

        self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def _serve_static(self, relative_path: str) -> None:
        target = (STATIC_DIR / relative_path).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.exists():
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return

        content_type, _ = mimetypes.guess_type(target.name)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        for header, value in COMMON_HEADERS.items():
            self.send_header(header, value)
        self.end_headers()
        self.wfile.write(target.read_bytes())

    def _send_json(self, payload: dict, status_code: int) -> None:
        response = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response)))
        for header, value in COMMON_HEADERS.items():
            self.send_header(header, value)
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args) -> None:
        return


def run() -> None:
    print("=" * 72)
    print(" AI POWERED COMMUNICATIONS SYSTEM FOR MUTE AND DEAF")
    print(f" Dashboard: http://{HOST}:{PORT}")
    print("=" * 72)
    server = ThreadingHTTPServer((HOST, PORT), LauncherHandler)
    server.serve_forever()


if __name__ == "__main__":
    run()
