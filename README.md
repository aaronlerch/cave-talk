# cave-talk

Capture conversations with a rolling audio buffer and local transcription. Say a wake phrase and cave-talk saves a transcript of the last 15 minutes of audio -- entirely local, no cloud APIs.

## How it works

cave-talk runs in a terminal, continuously buffering audio from your microphone into a rolling 15-minute window. When you press Enter or say the wake phrase ("hey cave talk save this"), it transcribes the buffer using [whisper.cpp](https://github.com/ggml-org/whisper.cpp) and saves a JSON transcript locally.

Transcripts can be listed, searched, and viewed from the CLI -- or accessed from [Claude Code](https://claude.ai/claude-code) via the included `/CaveTalk` skill.

### Example use case

You're having a conversation with a colleague about system design. Midway through, you realize the discussion is valuable. You say **"hey cave talk save this"** and hear back: *"I've saved the last 12 minutes of transcript for you."*

Later, in Claude Code:

```
> use the last cave talk to update AGENTS.md with best practices from our discussion
```

## Quickstart

### Prerequisites

- macOS (Apple Silicon or Intel)
- Python 3.10+
- [Homebrew](https://brew.sh)

### 1. Install whisper.cpp

```bash
brew install whisper-cpp
```

### 2. Download a model

```bash
mkdir -p ~/.cache/whisper.cpp

# Primary model for transcription (465 MB)
curl -L -o ~/.cache/whisper.cpp/ggml-small.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin

# Lightweight model for wake phrase detection (74 MB)
curl -L -o ~/.cache/whisper.cpp/ggml-tiny.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin
```

See [all available models](https://huggingface.co/ggerganov/whisper.cpp/tree/main) for other options. The `.en` variants are English-only but more accurate for English.

### 3. Install cave-talk

```bash
# With pipx (recommended -- makes it available globally)
pipx install /path/to/cave-talk

# Or with pip in a venv
cd cave-talk
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 4. Run it

```bash
# List audio devices
cave-talk devices

# Start listening (prompts for device on first run, remembers your choice)
cave-talk listen

# Start with wake phrase detection
cave-talk listen --wake
```

On first run, cave-talk prompts you to select an audio input device. Your choice is saved to `~/.cave-talk/config.json`.

### 5. Capture a conversation

While `cave-talk listen` is running:

- **Press Enter** to capture and transcribe the buffer, or
- **Say "hey cave talk save this"** (with `--wake` enabled)

cave-talk confirms the save with macOS text-to-speech and prints a preview.

### 6. View transcripts

```bash
cave-talk list              # List all saved transcripts
cave-talk show latest       # Show the most recent transcript
cave-talk show 20260405-213000-a1b2   # Show a specific transcript
cave-talk search "database"           # Search transcript text
cave-talk delete 20260405-213000-a1b2 # Delete a transcript
```

## CLI reference

```
cave-talk devices                         List audio input devices
cave-talk listen [OPTIONS]                Start listening and buffering audio
  --device, -d INT                        Audio device index
  --duration INT                          Buffer duration in seconds (default: 900)
  --wake, -w                              Enable wake phrase detection
  --phrase, -p TEXT                        Override wake phrase
cave-talk list                            List saved transcripts
cave-talk show [ID]                       Show transcript (default: latest)
cave-talk delete ID                       Delete a transcript
cave-talk search QUERY                    Search transcripts by content
```

## Menu bar app

cave-talk also runs as a Mac menu bar app — always-on, no terminal window needed.

### Launch from terminal

```bash
cave-talk-menu
```

### Launch without a terminal (background)

```bash
nohup cave-talk-menu &>/dev/null &
```

### Launch at login automatically

Create a Launch Agent so cave-talk starts every time you log in:

```bash
cat > ~/Library/LaunchAgents/com.cave-talk.menu.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.cave-talk.menu</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/aaronlerch/.local/bin/cave-talk-menu</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <false/>
</dict>
</plist>
EOF

# Load it now (starts immediately + on future logins)
launchctl load ~/Library/LaunchAgents/com.cave-talk.menu.plist
```

To stop auto-launching:

```bash
launchctl unload ~/Library/LaunchAgents/com.cave-talk.menu.plist
rm ~/Library/LaunchAgents/com.cave-talk.menu.plist
```

### Menu bar features

- **Status** — shows current device and buffer time
- **Capture Now** — trigger a capture with one click
- **Wake phrase detection** — toggle on/off
- **Device switcher** — change audio input without restarting
- **Recent transcripts** — quick access to the last 5
- **Open Transcripts Folder** — jump to the JSON files in Finder

The app auto-starts listening on launch using your saved device from `~/.cave-talk/config.json`. Captures trigger a macOS notification with a transcript preview plus TTS confirmation.

## Configuration

Stored at `~/.cave-talk/config.json` (auto-created on first run):

```json
{
  "device": 1,
  "buffer_duration": 900,
  "sample_rate": 16000,
  "channels": 1,
  "wake_phrase": "hey cave talk save this",
  "whisper_model": "small.en",
  "wake_model": "tiny.en",
  "whisper_bin": "whisper-cli"
}
```

## Claude Code integration

cave-talk includes a Claude Code skill (`/CaveTalk`) that lets you use transcripts directly in Claude sessions:

```
/CaveTalk                           Show latest transcript
/CaveTalk list                      List all transcripts
/CaveTalk use latest to summarize   Apply transcript to a task
```

The skill is at `.claude/skills/CaveTalk/SKILL.md`. To make it available globally across all projects:

```bash
cp -r .claude/skills/CaveTalk ~/.claude/skills/
```

## File locations

| Path | Purpose |
|------|---------|
| `~/.cave-talk/config.json` | Configuration |
| `~/.cave-talk/transcripts/*.json` | Saved transcripts |
| `~/.cave-talk/logs/*.log` | Debug logs (one per day) |
| `~/.cache/whisper.cpp/` | Whisper model files |

## Development

### Setup

```bash
git clone <repo-url> cave-talk
cd cave-talk
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

The editable install (`-e`) means changes to `src/cave_talk/` take effect immediately without reinstalling.

### Project structure

```
src/cave_talk/
  cli.py          Typer CLI: listen, devices, list, show, delete, search
  menubar.py      Mac menu bar app (rumps)
  audio.py        AudioCapture + AudioBuffer (rolling deque of 1-sec numpy chunks)
  transcribe.py   whisper.cpp subprocess wrapper, WAV export, JSON parsing
  storage.py      Transcript CRUD (JSON files in ~/.cave-talk/transcripts/)
  config.py       Config dataclass, paths, directory setup
  wake.py         Wake phrase detector (whisper tiny on rolling 8-sec windows)
  log.py          File-based logging to ~/.cave-talk/logs/
```

### Architecture

Three threads during `cave-talk listen --wake`:

1. **PortAudio callback** (sounddevice) -- writes 1-second numpy chunks to a `deque(maxlen=900)`. Non-exclusive device access via CoreAudio.
2. **Main thread** -- runs the CLI event loop. On trigger (Enter or wake event), snapshots the buffer, writes a temp WAV, calls `whisper-cli`, saves the transcript JSON.
3. **Wake detector** -- every 3 seconds, grabs the last 8 seconds from the buffer, transcribes with the tiny model, checks for the wake phrase via fuzzy word-subsequence matching.

### Data flow

```
Mic -> sounddevice callback -> deque (900 x 1-sec chunks)
                                      | (on trigger)
                                np.concatenate -> WAV -> whisper-cli -> JSON transcript
```

### Key design decisions

- **Deque of 1-second chunks** rather than a circular numpy array. Simpler, auto-evicts via `maxlen`, thread-safe append under CPython GIL. ~28 MB for 15 minutes at 16kHz mono.
- **whisper.cpp as subprocess** rather than Python bindings. Simpler dependency, uses Metal acceleration on Apple Silicon automatically, easy to upgrade independently.
- **Fuzzy wake phrase matching** with edit-distance-1 tolerance per word. Whisper often slightly mishears proper nouns; the subsequence matcher handles extra words between phrase words.
- **File-based JSON transcripts** rather than a database. Simple, inspectable, easy for Claude Code to read directly.

### Debugging wake phrase detection

```bash
# Watch wake detection logs in real time
tail -f ~/.cave-talk/logs/$(date -u +%Y%m%d).log

# Logs show the transcript text from each 8-second check window
# and whether the fuzzy match hit or missed
```

### Running from source without installing

```bash
cd cave-talk
source .venv/bin/activate
python -m cave_talk.cli listen --wake
```
