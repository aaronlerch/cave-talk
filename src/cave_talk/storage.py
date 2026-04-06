"""Transcript storage and retrieval."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .config import TRANSCRIPTS_DIR, ensure_dirs
from .transcribe import Transcript, Segment


def save_transcript(transcript: Transcript, device_name: str | None = None) -> dict:
    """Save a transcript and return the stored record."""
    ensure_dirs()

    now = datetime.now(timezone.utc)
    short_id = uuid.uuid4().hex[:6]
    ts = now.strftime("%Y%m%d-%H%M%S")
    record_id = f"{ts}-{short_id}"
    filename = f"{record_id}.json"

    record = {
        "id": record_id,
        "created_at": now.isoformat(),
        "duration_seconds": transcript.duration_seconds,
        "device": device_name,
        "model": transcript.model,
        "segments": [asdict(s) for s in transcript.segments],
        "full_text": transcript.full_text,
    }

    path = TRANSCRIPTS_DIR / filename
    path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def list_transcripts() -> list[dict]:
    """Return all transcripts sorted by newest first."""
    ensure_dirs()
    transcripts = []
    for path in sorted(TRANSCRIPTS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text())
            transcripts.append(data)
        except (json.JSONDecodeError, KeyError):
            continue
    return transcripts


def get_transcript(id_or_latest: str) -> dict | None:
    """Get a transcript by ID or 'latest'."""
    if id_or_latest == "latest":
        transcripts = list_transcripts()
        return transcripts[0] if transcripts else None

    ensure_dirs()
    # Try exact match first
    path = TRANSCRIPTS_DIR / f"{id_or_latest}.json"
    if path.exists():
        return json.loads(path.read_text())

    # Try prefix match
    for p in sorted(TRANSCRIPTS_DIR.glob("*.json"), reverse=True):
        if p.stem.startswith(id_or_latest):
            return json.loads(p.read_text())

    return None


def delete_transcript(id_prefix: str) -> bool:
    """Delete a transcript by ID or prefix. Returns True if deleted."""
    ensure_dirs()
    for path in TRANSCRIPTS_DIR.glob("*.json"):
        if path.stem == id_prefix or path.stem.startswith(id_prefix):
            path.unlink()
            return True
    return False


def search_transcripts(query: str) -> list[dict]:
    """Search transcripts by text content."""
    query_lower = query.lower()
    results = []
    for t in list_transcripts():
        if query_lower in t.get("full_text", "").lower():
            results.append(t)
    return results
