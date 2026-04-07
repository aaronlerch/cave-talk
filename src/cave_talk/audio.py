"""Audio capture and rolling buffer management."""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import threading
import time
from collections import deque
from typing import Any, Callable

import numpy as np
import sounddevice as sd

log = logging.getLogger("cave_talk.audio")

from .config import Config


# ---------------------------------------------------------------------------
# CoreAudio device-change listener (macOS only)
# ---------------------------------------------------------------------------
# Instead of polling sd.query_devices() (which segfaults during USB unplug),
# we register a CoreAudio property listener that fires when the device list
# changes.  This is notification-based — zero PortAudio calls in the hot path.

_coreaudio = None
_listener_callback_ref = None  # prevent GC of the ctypes callback

# CoreAudio constants
_kAudioObjectSystemObject = 1
_kAudioHardwarePropertyDevices = int.from_bytes(b'dev#', 'big')
_kAudioObjectPropertyScopeGlobal = int.from_bytes(b'glob', 'big')
_kAudioObjectPropertyElementMain = 0


class _AudioObjectPropertyAddress(ctypes.Structure):
    _fields_ = [
        ('mSelector', ctypes.c_uint32),
        ('mScope', ctypes.c_uint32),
        ('mElement', ctypes.c_uint32),
    ]


# Callback signature: OSStatus(AudioObjectID, UInt32, const AudioObjectPropertyAddress*, void*)
_ListenerProc = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_void_p,
    ctypes.c_void_p,
)


class DeviceChangeListener:
    """Listens for macOS CoreAudio device list changes.

    Calls ``on_change()`` from a CoreAudio thread whenever a device is
    plugged or unplugged.  The callback should be fast (just set a flag).
    """

    def __init__(self, on_change: Callable[[], None]) -> None:
        self._on_change = on_change
        self._cb_ref: Any = None  # prevent GC
        self._started = False

    def start(self) -> bool:
        """Register the listener.  Returns True on success."""
        global _coreaudio
        if _coreaudio is None:
            path = ctypes.util.find_library('CoreAudio')
            if not path:
                log.warning("CoreAudio framework not found — device change detection disabled")
                return False
            _coreaudio = ctypes.cdll.LoadLibrary(path)

        def _callback(obj_id, num_addr, addr_ptr, client_data):
            log.info("CoreAudio: device list changed")
            try:
                self._on_change()
            except Exception as e:
                log.warning("CoreAudio callback error: %s", e)
            return 0  # noErr

        self._cb_ref = _ListenerProc(_callback)

        address = _AudioObjectPropertyAddress(
            _kAudioHardwarePropertyDevices,
            _kAudioObjectPropertyScopeGlobal,
            _kAudioObjectPropertyElementMain,
        )

        status = _coreaudio.AudioObjectAddPropertyListener(
            _kAudioObjectSystemObject,
            ctypes.byref(address),
            self._cb_ref,
            None,
        )
        if status != 0:
            log.warning("AudioObjectAddPropertyListener failed: %d", status)
            return False

        self._started = True
        log.info("CoreAudio device-change listener registered")
        return True

    def stop(self) -> None:
        if not self._started or _coreaudio is None:
            return
        address = _AudioObjectPropertyAddress(
            _kAudioHardwarePropertyDevices,
            _kAudioObjectPropertyScopeGlobal,
            _kAudioObjectPropertyElementMain,
        )
        _coreaudio.AudioObjectRemovePropertyListener(
            _kAudioObjectSystemObject,
            ctypes.byref(address),
            self._cb_ref,
            None,
        )
        self._started = False
        log.info("CoreAudio device-change listener removed")


# ---------------------------------------------------------------------------
# AudioBuffer / AudioCapture
# ---------------------------------------------------------------------------

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

    def snapshot(self, last_n_seconds: int | None = None) -> np.ndarray | None:
        """Return a copy of buffered audio. Optionally only the last N seconds."""
        with self._lock:
            if not self._chunks:
                return None
            if last_n_seconds is not None and last_n_seconds < len(self._chunks):
                chunks = list(self._chunks)[-last_n_seconds:]
            else:
                chunks = list(self._chunks)
            return np.concatenate(chunks)

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
        self.last_callback_time: float = 0.0

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags) -> None:
        if status:
            log.warning("Audio callback status: %s", status)
        self.last_callback_time = time.monotonic()
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
        self.last_callback_time = time.monotonic()
        log.info("Audio stream started: device=%s, rate=%d, channels=%d, active=%s",
                 device, self.config.sample_rate, self.config.channels,
                 self._stream.active)

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception as e:
                log.warning("Error stopping stream (device may be gone): %s", e)
            try:
                self._stream.close()
            except Exception as e:
                log.warning("Error closing stream (device may be gone): %s", e)
            self._stream = None

    def abandon(self) -> None:
        """Release our reference without calling PortAudio stop/close.

        Use this when the device is gone and PA calls would deadlock.
        """
        self._stream = None


def refresh_devices() -> None:
    """Force PortAudio to re-scan hardware devices.

    IMPORTANT: Only call this when NO AudioCapture streams are active.
    Calling while a stream exists will invalidate it (crash / deadlock).
    After calling, sd.query_devices() and list_devices() return fresh data.
    """
    sd._terminate()
    sd._initialize()
    log.debug("PortAudio reinitialized")


def list_devices() -> list[dict]:
    """Return a list of available audio input devices.

    NOTE: PortAudio caches its device list.  Call refresh_devices() first
    if hardware may have changed since the last query.
    """
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
