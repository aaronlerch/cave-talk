"""Configuration management for cave-talk."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

APP_DIR = Path.home() / ".cave-talk"
CONFIG_PATH = APP_DIR / "config.json"
TRANSCRIPTS_DIR = APP_DIR / "transcripts"
LOG_DIR = APP_DIR / "logs"


@dataclass
class Config:
    device: int | None = None  # None = system default
    buffer_duration: int = 900  # seconds (15 minutes)
    sample_rate: int = 16000  # whisper expects 16kHz
    channels: int = 1
    wake_phrase: str = "hey cave talk save this"
    whisper_model: str = "small.en"  # model for transcription
    wake_model: str = "tiny.en"  # lighter model for wake phrase detection
    whisper_bin: str = "whisper-cli"  # path or command name

    @classmethod
    def load(cls) -> Config:
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text())
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        return cls()

    def save(self) -> None:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(asdict(self), indent=2) + "\n")


def ensure_dirs() -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
