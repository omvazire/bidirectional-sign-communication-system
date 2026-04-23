<h1 align="center">🤟 Bidirectional Sign Communication System</h1>

<p align="center">
  A unified communication platform that bridges the gap between sign-language users and non-sign users.
</p>

<p align="center">
  <a href="https://bidirectional-sign-communication-system.onrender.com" target="_blank">
    <b>Live Demo</b>
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Backend-Flask%20%7C%20Python-blue" />
  <img src="https://img.shields.io/badge/Frontend-React%20%7C%20Three.js-61dafb" />
  <img src="https://img.shields.io/badge/Deployment-Docker%20%7C%20Render-orange" />
</p>

<hr>

<h2>🎓 Academic Project Details</h2>

<ul>
  <li><b>Project Type:</b> Third Year Engineering Project</li>
  <li><b>College:</b> Konkan Gyanpeeth College of Engineering, Karjat</li>
  <li><b>Branch:</b> Artificial Intelligence and Data Science (AIDS)</li>
</ul>

<h3>👥 Team Members</h3>

<ul>
  <li>Shubham Brijesh Tripathi</li>
  <li>Om Prabhakar Vazire</li>
  <li>Ayush Shailendra Vishwakarma</li>
</ul>

<hr>

<h2>📌 Overview</h2>

<p>
This project is a <b>bidirectional sign communication system</b> built as a single deployed application with two main modules:
</p>

<ul>
  <li><b>Sign Translator</b> — converts live hand gestures into text and speech.</li>
  <li><b>Text-to-Sign Studio</b> — converts typed English text into animated ASL-style 3D hand poses.</li>
</ul>

<p>
The app is packaged with Docker and served from one Flask entry point. It uses browser webcam access, MediaPipe hand tracking, a Bi-LSTM + attention gesture model, sentence building logic, and a React + Three.js studio for 3D sign animation.
</p>

<hr>

<h2>✨ Key Features</h2>

<ul>
  <li>Real-time sign language recognition from webcam input</li>
  <li>Sentence building with duplicate filtering and basic grammar correction</li>
  <li>Browser speech input support in the text-to-sign interface</li>
  <li>Text-to-sign animation with 3D hand rendering</li>
  <li>Quick phrase buttons for common words and greetings</li>
  <li>Play, pause, reset, and speed controls for sign animation playback</li>
  <li>Responsive dashboard that links both workspaces</li>
  <li>Docker-based deployment with a single production container</li>
</ul>

<hr>

<h2>🧠 How It Works</h2>

<h3>1. Sign → Text / Speech</h3>
<ul>
  <li>Captures webcam frames</li>
  <li>Detects hand landmarks using MediaPipe</li>
  <li>Processes data using Bi-LSTM + attention model</li>
  <li>Predicts gesture labels</li>
  <li>Builds readable sentences</li>
  <li>Outputs text and speech</li>
</ul>

<h3>2. Text → Sign</h3>
<ul>
  <li>User enters text</li>
  <li>System converts text into ASL pose sequences</li>
  <li>Known words → direct signs</li>
  <li>Unknown words → fingerspelling</li>
  <li>Rendered using 3D animation</li>
</ul>

<hr>

<h2>🧩 Application Routes</h2>

<ul>
  <li><code>/</code> — Dashboard</li>
  <li><code>/sign-translator/</code> — Sign recognition</li>
  <li><code>/text-to-sign/</code> — 3D sign studio</li>
  <li><code>/healthz</code> — Health check</li>
</ul>

<hr>

<h2>🛠 Tech Stack</h2>

<h3>Backend</h3>
<ul>
  <li>Python 3.11</li>
  <li>Flask</li>
  <li>Gunicorn</li>
  <li>PyTorch</li>
  <li>MediaPipe</li>
  <li>OpenCV</li>
</ul>

<h3>Frontend</h3>
<ul>
  <li>React</li>
  <li>Vite</li>
  <li>Tailwind CSS</li>
  <li>Three.js</li>
</ul>

<h3>Deployment</h3>
<ul>
  <li>Docker</li>
  <li>Render</li>
</ul>

<hr>

<h2>📁 Project Structure</h2>

<pre>
deploy/
├── app.py
├── Dockerfile
├── requirements.txt
├── hub/
├── sign_translator/
├── text_to_sign/
└── text_to_sign_dist/
</pre>

<hr>

<h2>🚀 Setup</h2>

<pre><code>docker build -t bidirectional-sign-system ./deploy
docker run -p 7860:7860 bidirectional-sign-system
</code></pre>

<p>Open: http://localhost:7860</p>

<hr>

<h2>🌐 Deployment Notes</h2>

<ul>
  <li>Deploy from <code>deploy/</code> folder</li>
  <li>Runs on port <code>7860</code></li>
  <li>Single container architecture</li>
</ul>

<hr>

<h2>🔮 Future Scope</h2>

<ul>
  <li>Sentence-level sign recognition</li>
  <li>Higher accuracy deep learning models</li>
  <li>Multi-language sign support (ISL/ASL)</li>
  <li>Mobile application</li>
</ul>

<hr>

<h2>⭐ Credits</h2>

<ul>
  <li>MediaPipe</li>
  <li>PyTorch</li>
  <li>Three.js</li>
  <li>Render</li>
</ul>

<hr>

<p align="center">
  ⭐ Star this repo if you found it useful!
</p>
