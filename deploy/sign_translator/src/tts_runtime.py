"""
Queue-based text-to-speech runtime for the live detector.

This module is intentionally focused on runtime reliability:
- speech requests are queued instead of dropped
- the detector loop never blocks on speech
- the UI can query whether TTS is ready, busy, or unavailable
"""

import os
import queue
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class TTSEngine:
    """A small queued wrapper around pyttsx3 for runtime use."""

    def __init__(self, rate=None, volume=None):
        self.rate = rate or config.TTS_RATE
        self.volume = volume or config.TTS_VOLUME

        self._queue = queue.Queue()
        self._shutdown_requested = False
        self._speaking = False
        self._available = True
        self._last_error = ""

        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()

        print(f"[OK] TTS engine configured (rate={self.rate}, volume={self.volume})")

    def _select_voice(self, engine):
        """Prefer a reasonable built-in Windows voice when available."""
        try:
            voices = engine.getProperty("voices")
        except Exception:
            return

        if not voices:
            return

        preferred_tokens = ("zira", "david", "female", "male")
        for token in preferred_tokens:
            for voice in voices:
                name = getattr(voice, "name", "").lower()
                if token in name:
                    engine.setProperty("voice", voice.id)
                    return

    def _speak_once(self, text):
        """Speak one utterance using a fresh pyttsx3 engine instance."""
        try:
            import pyttsx3
        except ImportError:
            raise RuntimeError("pyttsx3 is not installed. Server-side TTS is unavailable.")

        engine = pyttsx3.init()
        engine.setProperty("rate", self.rate)
        engine.setProperty("volume", self.volume)
        self._select_voice(engine)
        engine.say(text)
        engine.runAndWait()
        engine.stop()

    def _run_worker(self):
        """Consume queued utterances until shutdown."""
        while not self._shutdown_requested:
            try:
                text = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if text is None:
                break

            self._speaking = True
            try:
                self._speak_once(text)
                self._available = True
                self._last_error = ""
            except Exception as exc:
                self._available = False
                self._last_error = str(exc)
                print(f"[X] TTS worker error: {exc}")
            finally:
                self._speaking = False
                self._queue.task_done()

    def speak(self, text):
        """
        Queue text for speech.

        Returns:
            bool: True if accepted for speech, False otherwise.
        """
        if self._shutdown_requested:
            return False

        if not text or not text.strip():
            return False

        self._queue.put(text.strip())
        return True

    def is_speaking(self):
        """Return whether an utterance is currently being spoken."""
        return self._speaking

    def is_available(self):
        """Return whether the TTS backend is available."""
        return self._available

    def pending_count(self):
        """Return the number of queued utterances waiting to be spoken."""
        return self._queue.qsize()

    def get_status(self):
        """Return a short UI-friendly status string."""
        if not self._available:
            return "Unavailable"
        if self.is_speaking():
            return "Speaking"
        if self.pending_count() > 0:
            return f"Queued ({self.pending_count()})"
        return "Ready"

    def last_error(self):
        """Return the last backend error string."""
        return self._last_error

    def clear_queue(self):
        """Drop queued utterances that have not started yet."""
        cleared = 0
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break

            if item is not None:
                cleared += 1
            self._queue.task_done()
        return cleared

    def shutdown(self):
        """Gracefully stop the worker thread."""
        self._shutdown_requested = True
        self._queue.put(None)
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
        print("[OK] TTS engine shut down")


_tts_instance = None


def get_tts_engine():
    """Get or create the global TTS runtime singleton."""
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = TTSEngine()
    return _tts_instance


def shutdown_tts():
    """Shutdown the global TTS runtime."""
    global _tts_instance
    if _tts_instance:
        _tts_instance.shutdown()
        _tts_instance = None
