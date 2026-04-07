"""Wake phrase detection using whisper.cpp tiny model on rolling audio windows."""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Callable

from .audio import AudioBuffer
from .config import Config
from .transcribe import resolve_model_path, write_wav

log = logging.getLogger("cave_talk.wake")


class WakePhraseDetector:
    """Monitors audio buffer for a wake phrase using whisper.cpp tiny model.

    Every `check_interval` seconds, transcribes the last `window_seconds` of
    audio and checks for the wake phrase via fuzzy substring match.
    """

    def __init__(
        self,
        buffer: AudioBuffer,
        config: Config,
        on_wake: Callable[[], None],
        window_seconds: int = 8,
        check_interval: float = 3.0,
    ) -> None:
        self._buffer = buffer
        self._config = config
        self._on_wake = on_wake
        self._window_seconds = window_seconds
        self._check_interval = check_interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._cooldown = False

    def start(self) -> None:
        self._stop_event.clear()
        self._cooldown = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info(
            "Wake detector started: phrase=%r, model=%s, window=%ds, interval=%.1fs",
            self._config.wake_phrase, self._config.wake_model,
            self._window_seconds, self._check_interval,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        log.info("Wake detector stopped")

    def clear_cooldown(self) -> None:
        self._cooldown = False
        log.debug("Wake cooldown cleared")

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(self._check_interval)
            if self._stop_event.is_set():
                break
            if self._cooldown:
                continue
            if self._buffer.seconds_buffered < self._window_seconds:
                log.debug("Buffer too short (%ds), skipping", self._buffer.seconds_buffered)
                continue
            self._check()

    def _check(self) -> None:
        audio = self._buffer.snapshot(last_n_seconds=self._window_seconds)
        if audio is None:
            return

        text = self._transcribe_snippet(audio)
        log.debug("Wake check: %r", text[:200] if text else "(empty)")

        if not text:
            return

        matched = _fuzzy_match(text, self._config.wake_phrase)
        if matched:
            log.info("WAKE PHRASE DETECTED in: %r", text[:200])
            self._cooldown = True
            self._on_wake()

    def _transcribe_snippet(self, audio) -> str:
        """Quick transcription of a short audio clip using the wake model."""
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                wav_path = Path(tmpdir) / "wake.wav"
                output_base = Path(tmpdir) / "wake"
                write_wav(wav_path, audio, self._config.sample_rate, self._config.channels)

                model_path = resolve_model_path(self._config.wake_model)
                log.debug("Running whisper: bin=%s, model=%s", self._config.whisper_bin, model_path)

                result = subprocess.run(
                    [
                        self._config.whisper_bin,
                        "-m", model_path,
                        "-f", str(wav_path),
                        "-of", str(output_base),
                        "-oj",
                        "--no-prints",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )

                if result.returncode != 0:
                    log.warning("whisper-cli returned %d: %s", result.returncode, result.stderr[:200])

                # Read from JSON output file
                json_path = Path(str(output_base) + ".json")
                if json_path.exists():
                    data = json.loads(json_path.read_text())
                    segments = data.get("transcription", [])
                    text = " ".join(
                        seg.get("text", "").strip()
                        for seg in segments
                        if seg.get("text", "").strip()
                    )
                    return text

                # Fallback: try stdout
                log.debug("No JSON file, falling back to stdout")
                return result.stdout.strip()

        except FileNotFoundError:
            log.error("whisper-cli binary not found at %r", self._config.whisper_bin)
            return ""
        except subprocess.TimeoutExpired:
            log.warning("Wake transcription timed out")
            return ""
        except Exception as e:
            log.error("Wake transcription error: %s", e)
            return ""


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    import re
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fuzzy_match(transcript: str, phrase: str) -> bool:
    """Check if the wake phrase appears in the transcript with some tolerance.

    Uses word-level subsequence matching: all words in the phrase must appear
    in order within the transcript, but not necessarily adjacent.
    """
    t_words = _normalize(transcript).split()
    p_words = _normalize(phrase).split()

    if not p_words:
        return False

    t_idx = 0
    for pw in p_words:
        found = False
        while t_idx < len(t_words):
            if _word_similar(t_words[t_idx], pw):
                t_idx += 1
                found = True
                break
            t_idx += 1
        if not found:
            return False
    return True


def _word_similar(a: str, b: str) -> bool:
    """Check if two words are similar enough (exact match or edit distance <= 1)."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(ca != cb for ca, cb in zip(a, b)) <= 1
    short, long = (a, b) if len(a) < len(b) else (b, a)
    diffs = 0
    si = li = 0
    while si < len(short) and li < len(long):
        if short[si] != long[li]:
            diffs += 1
            li += 1
        else:
            si += 1
            li += 1
    return diffs <= 1
