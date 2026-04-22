---
title: AI Communication System for Mute and Deaf
emoji: 🤟
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
fullWidth: true
header: mini
short_description: Sign translator and 3D text-to-sign studio.
---

# AI-Powered Communication System for Mute and Deaf

A unified platform that combines two powerful tools for bridging communication gaps:

1. **Sign Translator Web** — Real-time sign language recognition using browser webcam, Bi-LSTM + attention classifier, sentence building, and browser speech playback.
2. **Text-to-Sign Studio** — Convert typed English text into 3D animated sign language using React Three.js and hand models.

## Architecture

One Docker container serves everything on port 7860:

- `/` — Dashboard hub with links to both workspaces
- `/sign-translator/` — Browser-based sign language translator
- `/text-to-sign/` — 3D text-to-sign animation studio

## Tech Stack

- **Backend**: Python 3.11, Flask, PyTorch (CPU), MediaPipe, Gunicorn
- **Frontend (Sign Translator)**: Vanilla JS, browser-side MediaPipe hand tracking, Web Speech API
- **Frontend (Text-to-Sign)**: React 19, Three.js, React Three Fiber, Vite, TailwindCSS v4
- **Deployment**: Multi-stage Docker (Node for React build, Python for runtime)

## Local Docker Test

```bash
docker build -t comm-system .
docker run -p 7860:7860 comm-system
```

Then open `http://localhost:7860`.

## Deploy To Hugging Face Spaces

This `deploy` folder is the correct root for the unified Hugging Face Docker Space.

Use these settings when you create the Space:

- SDK: `Docker`
- Visibility: your choice
- Hardware: `CPU Basic` is the safest starting point
- Space root: the contents of this `deploy` folder

### What To Upload

Push the contents of this folder to the Space repository:

- `README.md`
- `Dockerfile`
- `requirements.txt`
- `app.py`
- `hub/`
- `sign_translator/`
- `text_to_sign/`
- `.dockerignore`
- `.gitattributes`

Do not push the parent workspace root. The Space should be built from `deploy`, not from `mute&deaf pp`.

### First Publish Flow

```bash
git lfs install
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
```

Then copy everything from this `deploy` folder into the cloned Space folder, commit, and push:

```bash
git add .
git commit -m "Deploy unified communication system"
git push
```

### Important Notes For Spaces

- Large files such as `.pth`, `.task`, and `.glb` are already covered by `.gitattributes` for Git LFS.
- Keep Gunicorn at `1` worker because browser session state is stored in memory.
- Browser camera access will work on the public Space URL because Hugging Face serves over HTTPS.
- The first build can take a while because PyTorch, MediaPipe, and the React bundle are built inside Docker.

## Deployment Notes

- Keep the Gunicorn worker count at `1`. The sign-translator browser session state lives in-memory, so multiple workers would split active sessions across processes.
- The unified Flask app is proxy-aware for HTTPS deployments, which keeps browser session cookies correct behind platforms like Hugging Face Spaces, Render, Railway, or Nginx.
- The Text-to-Sign app is mounted under `/text-to-sign/`, so public assets and 3D models must be loaded with the deployment base path rather than root-level `/models/...` URLs.

## Environment Variables

Copy `.env.example` if you want to override runtime defaults:

- `PORT`: container port, defaults to `7860`
- `SESSION_TTL_SECONDS`: idle timeout for sign-translator browser sessions, defaults to `900`
- `MAX_ACTIVE_SESSIONS`: maximum concurrent in-memory translator sessions per process, defaults to `6`
- `MAX_FRAME_BYTES`: request size limit for uploaded frame/landmark payloads, defaults to `2097152`

## Build From Repo Root

If you are building from the repository root instead of changing into the `deploy` folder first:

```bash
docker build -t comm-system ./deploy
docker run -p 7860:7860 comm-system
```
