---
name: CaveTalk
description: Access cave-talk conversation transcripts — list, show, search, and apply captured audio transcripts to tasks. USE WHEN cave talk, last conversation, use the transcript, captured conversation, cave-talk, what did we discuss, conversation capture.
user-invocable: true
allowed-tools:
  - Bash(cave-talk *)
  - Read
---

# /CaveTalk — Conversation Transcript Access

Provides access to transcripts captured by the cave-talk CLI tool. cave-talk continuously buffers audio from the microphone and transcribes it on demand.

Arguments passed: `$ARGUMENTS`

---

## How cave-talk works

- cave-talk runs in a terminal (`cave-talk listen`), buffering the last 15 minutes of audio
- When triggered, it transcribes the buffer and saves a JSON transcript to `~/.cave-talk/transcripts/`
- Transcripts contain timestamped segments and full text

## Dispatch on arguments

Parse `$ARGUMENTS`. If empty, show the latest transcript.

### No args or "latest" — show last transcript

1. Run `cave-talk show latest` to get the most recent transcript.
2. Display a summary: date, duration, preview.
3. Ask the user what they'd like to do with it.

### "list" — list all transcripts

1. Run `cave-talk list` to show all saved transcripts.

### "show <id>" — show a specific transcript

1. Run `cave-talk show <id>` to display the full transcript.

### "search <query>" — search transcripts

1. Run `cave-talk search <query>` to find transcripts containing the query.

### "delete <id>" — delete a transcript

1. Run `cave-talk delete <id>` to remove a transcript.

### "use <id-or-latest> to <task description>" — apply transcript to a task

This is the primary workflow. The user wants to use a conversation transcript to accomplish something.

1. Run `cave-talk show <id>` (or `cave-talk show latest` if no ID given) to get the transcript.
2. Read the full transcript text.
3. Apply the transcript content to accomplish the user's described task. Examples:
   - "use the last cave talk to update AGENTS.md with best practices" — read the transcript, extract relevant insights, update the file
   - "use the last cave talk to create a summary" — synthesize the conversation into a concise summary
   - "use the last cave talk to write a follow-up email" — draft an email based on the conversation

## Transcript storage

Transcripts are JSON files at `~/.cave-talk/transcripts/*.json` with this structure:

```json
{
  "id": "20260405-213000-a1b2",
  "created_at": "2026-04-05T21:30:00Z",
  "duration_seconds": 900,
  "device": "MacBook Pro Microphone",
  "model": "whisper-small",
  "segments": [
    {"start": 0.0, "end": 3.2, "text": "...", "speaker": null}
  ],
  "full_text": "..."
}
```

If you need the raw JSON (e.g., to inspect segments or metadata), read the file directly from `~/.cave-talk/transcripts/`.
