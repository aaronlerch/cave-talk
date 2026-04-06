"""Audio capture and rolling buffer management."""

from __future__ import annotations

import threading
from collections import deque
from typing import Any

import numpy as np
import sounddevice as sd

from .config import Config


class AudioBuffer:
    """Rolling buffer of 1-second audio chunks stored as numpy arrays."""

    def __init__(self, duration: int, sample_rate: int, channels: int) -> None:
        self._chunks: deque[np.ndarray] = deque(maxlen=duration)
        self._sample_rate = sample_rate
        self._channels = channels
        self._lock = threading.Lock()

    @property
    def seconds_buffered(self) -> int:
        return len(self._chunks)

    def append(self, chunk: np.ndarray) -> None:
        self._chunks.append(chunk.copy())

    def snapshot(self) -> np.ndarray | None:
        """Return a copy of all buffered audio as a single array."""
        with self._lock:
            if not self._chunks:
                return None
            return np.concatenate(list(self._chunks))

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()


class AudioCapture:
    """Captures audio from a device into a rolling buffer via sounddevice."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.buffer = AudioBuffer(
            duration=config.buffer_duration,
            sample_rate=config.sample_rate,
            channels=config.channels,
        )
        self._stream: sd.InputStream | None = None
        self._chunk_samples = config.sample_rate  # 1 second per chunk
        self._accumulator = np.zeros((0, config.channels), dtype=np.float32)
        self._acc_lock = threading.Lock()

    def _callback(self, indata: np.ndarray, frames: int, time: Any, status: sd.CallbackFlags) -> None:
        if status:
            pass  # could log dropped frames, etc.
        with self._acc_lock:
            self._accumulator = np.concatenate([self._accumulator, indata.copy()])
            while len(self._accumulator) >= self._chunk_samples:
                chunk = self._accumulator[: self._chunk_samples]
                self._accumulator = self._accumulator[self._chunk_samples :]
                self.buffer.append(chunk)

    def start(self) -> None:
        device = self.config.device  # None = system default
        self._stream = sd.InputStream(
            device=device,
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


def list_devices() -> list[dict]:
    """Return a list of available audio input devices."""
    devices = sd.query_devices()
    inputs = []
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            default = i == sd.default.device[0]
            inputs.append({
                "index": i,
                "name": d["name"],
                "channels": d["max_input_channels"],
                "sample_rate": d["default_samplerate"],
                "default": default,
            })
    return inputs
