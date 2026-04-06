"""Transcription via whisper.cpp subprocess."""

from __future__ import annotations

import json
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None


@dataclass
class Transcript:
    segments: list[Segment]
    full_text: str
    duration_seconds: float
    model: str


def write_wav(path: Path, audio: np.ndarray, sample_rate: int, channels: int = 1) -> None:
    """Write a numpy float32 array to a WAV file."""
    # Convert float32 [-1, 1] to int16
    audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    if audio_int16.ndim > 1:
        audio_int16 = audio_int16[:, 0]  # mono

    num_samples = len(audio_int16)
    data_size = num_samples * 2  # 16-bit = 2 bytes per sample
    byte_rate = sample_rate * channels * 2
    block_align = channels * 2

    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))  # chunk size
        f.write(struct.pack("<H", 1))  # PCM
        f.write(struct.pack("<H", channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", 16))  # bits per sample
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(audio_int16.tobytes())


def _parse_whisper_json(output: str) -> list[Segment]:
    """Parse whisper.cpp JSON output into segments."""
    try:
        data = json.loads(output)
        segments = []
        for seg in data.get("transcription", []):
            # whisper.cpp outputs timestamps as "HH:MM:SS.mmm" strings
            # or in some builds, offsets in the JSON
            text = seg.get("text", "").strip()
            if not text:
                continue
            start = seg.get("offsets", {}).get("from", 0) / 1000.0
            end = seg.get("offsets", {}).get("to", 0) / 1000.0
            segments.append(Segment(start=start, end=end, text=text))
        return segments
    except (json.JSONDecodeError, KeyError):
        return []


def _parse_whisper_text(output: str) -> list[Segment]:
    """Fallback: treat entire output as a single segment."""
    text = output.strip()
    if not text:
        return []
    return [Segment(start=0.0, end=0.0, text=text)]


def transcribe(audio: np.ndarray, config: Config) -> Transcript:
    """Transcribe audio using whisper.cpp."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "capture.wav"
        output_base = Path(tmpdir) / "capture"
        write_wav(wav_path, audio, config.sample_rate, config.channels)

        duration = len(audio) / config.sample_rate

        try:
            # -oj writes JSON to <output_base>.json
            result = subprocess.run(
                [
                    config.whisper_bin,
                    "-m", resolve_model_path(config.whisper_model),
                    "-f", str(wav_path),
                    "-of", str(output_base),
                    "-oj",
                    "--no-prints",
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )

            json_path = Path(str(output_base) + ".json")
            segments = []
            if json_path.exists():
                segments = _parse_whisper_json(json_path.read_text())

            if not segments:
                # Fallback: parse stdout text directly
                segments = _parse_whisper_text(result.stdout)
        except FileNotFoundError:
            raise RuntimeError(
                f"whisper.cpp binary not found at '{config.whisper_bin}'. "
                "Install it with: brew install whisper-cpp"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Transcription timed out after 5 minutes.")

    full_text = " ".join(s.text for s in segments)
    return Transcript(
        segments=segments,
        full_text=full_text,
        duration_seconds=duration,
        model=config.whisper_model,
    )


def resolve_model_path(model: str) -> str:
    """Resolve a model name to a file path. Checks common whisper.cpp model locations."""
    if Path(model).is_absolute() and Path(model).exists():
        return model

    candidates = [
        Path.home() / ".cache" / "whisper.cpp" / f"ggml-{model}.bin",
        Path(f"/opt/homebrew/share/whisper-cpp/models/ggml-{model}.bin"),
        Path(f"/usr/local/share/whisper-cpp/models/ggml-{model}.bin"),
        Path.home() / "whisper.cpp" / "models" / f"ggml-{model}.bin",
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return f"ggml-{model}.bin"
