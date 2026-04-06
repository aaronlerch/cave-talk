"""CLI interface for cave-talk."""

from __future__ import annotations

import sys
import threading
import time

import typer
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from .audio import AudioCapture, list_devices
from .config import Config, ensure_dirs
from .storage import (
    delete_transcript,
    get_transcript,
    list_transcripts,
    save_transcript,
    search_transcripts,
)
from .transcribe import transcribe

app = typer.Typer(
    name="cave-talk",
    help="Capture conversations with a rolling audio buffer and local transcription.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def devices():
    """List available audio input devices."""
    devs = list_devices()
    if not devs:
        console.print("[red]No audio input devices found.[/red]")
        raise typer.Exit(1)

    table = Table(title="Audio Input Devices")
    table.add_column("Index", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Channels", style="green")
    table.add_column("Sample Rate", style="yellow")
    table.add_column("Default", style="magenta")

    for d in devs:
        table.add_row(
            str(d["index"]),
            d["name"],
            str(d["channels"]),
            f"{d['sample_rate']:.0f}",
            "*" if d["default"] else "",
        )

    console.print(table)


def _pick_device(devs: list[dict]) -> dict:
    """Interactive device picker. Returns the chosen device dict."""
    console.print("\n[bold]Select an audio input device:[/bold]\n")
    for i, d in enumerate(devs):
        marker = " [magenta](default)[/magenta]" if d["default"] else ""
        console.print(f"  [cyan]{d['index']}[/cyan]  {d['name']}{marker}")
    console.print()

    while True:
        try:
            choice = input("Device index (or Enter for default): ").strip()
        except (EOFError, KeyboardInterrupt):
            raise typer.Exit(0)

        if choice == "":
            return next(d for d in devs if d["default"])

        try:
            idx = int(choice)
            match = next((d for d in devs if d["index"] == idx), None)
            if match:
                return match
        except ValueError:
            pass
        console.print("[red]Invalid choice. Try again.[/red]")


_SPINNER_FRAMES = ["   ", ".  ", ".. ", "...", " ..", "  .", "   ", "  .", " ..", "...", ".. ", ".  "]


@app.command()
def listen(
    device: int | None = typer.Option(None, "--device", "-d", help="Audio device index (use 'devices' to list)"),
    duration: int = typer.Option(900, "--duration", help="Buffer duration in seconds (default: 900 = 15 min)"),
):
    """Start listening and buffering audio. Press Enter to capture and transcribe."""
    config = Config.load()
    config.buffer_duration = duration
    ensure_dirs()

    devs = list_devices()
    if not devs:
        console.print("[red]No audio input devices found.[/red]")
        raise typer.Exit(1)

    # Device selection: explicit flag > saved config > interactive picker
    if device is not None:
        config.device = device
        dev_info = next((d for d in devs if d["index"] == device), None)
        if not dev_info:
            console.print(f"[red]Device index {device} not found. Run 'cave-talk devices' to list.[/red]")
            raise typer.Exit(1)
    elif config.device is not None:
        dev_info = next((d for d in devs if d["index"] == config.device), None)
        if not dev_info:
            console.print(f"[yellow]Saved device {config.device} not found. Pick a new one.[/yellow]")
            dev_info = _pick_device(devs)
            config.device = dev_info["index"]
    else:
        dev_info = _pick_device(devs)
        config.device = dev_info["index"]

    device_name = dev_info["name"]

    capture = AudioCapture(config)
    try:
        capture.start()
    except Exception as e:
        console.print(f"[red]Failed to start audio capture: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[green]Listening on:[/green] {device_name}")
    console.print(f"[dim]Buffer: {duration // 60}m | Press Enter to capture. Ctrl+C to quit.[/dim]\n")

    stop_event = threading.Event()
    spinner_idx = 0

    def make_status() -> Text:
        nonlocal spinner_idx
        buffered = capture.buffer.seconds_buffered
        mins, secs = divmod(buffered, 60)
        max_mins = duration // 60
        frame = _SPINNER_FRAMES[spinner_idx % len(_SPINNER_FRAMES)]
        spinner_idx += 1

        bar_width = 20
        fill = int(bar_width * buffered / duration) if duration > 0 else 0
        bar = "[green]" + "=" * fill + "[/green]" + "[dim]" + "-" * (bar_width - fill) + "[/dim]"

        text = Text.from_markup(
            f"  [cyan]{frame}[/cyan]  {bar}  [bold]{mins:02d}:{secs:02d}[/bold] / {max_mins:02d}:00"
        )
        return text

    live = Live(make_status(), console=console, refresh_per_second=4, transient=True)
    live.start()

    def status_loop():
        while not stop_event.is_set():
            live.update(make_status())
            stop_event.wait(0.25)

    status_thread = threading.Thread(target=status_loop, daemon=True)
    status_thread.start()

    try:
        while True:
            input()  # Wait for Enter
            live.stop()
            console.print("[cyan]Capturing...[/cyan]")

            audio = capture.buffer.snapshot()
            if audio is None or len(audio) == 0:
                console.print("[yellow]No audio in buffer yet.[/yellow]\n")
                live.start()
                continue

            buffered_secs = len(audio) / config.sample_rate
            mins, secs = divmod(int(buffered_secs), 60)
            console.print(f"[dim]Got {mins}m {secs}s of audio. Transcribing...[/dim]")

            try:
                transcript = transcribe(audio, config)
            except RuntimeError as e:
                console.print(f"[red]{e}[/red]\n")
                live.start()
                continue

            record = save_transcript(transcript, device_name=device_name)
            console.print(f"[green]Transcript saved: {record['id']}[/green]")
            console.print(f"[dim]Duration: {mins}m {secs}s | Segments: {len(transcript.segments)}[/dim]")

            preview = transcript.full_text[:200]
            if len(transcript.full_text) > 200:
                preview += "..."
            console.print(f"\n[white]{preview}[/white]")
            console.print(f"\n[dim]Press Enter to capture again. Ctrl+C to quit.[/dim]\n")
            live.start()

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        live.stop()
        capture.stop()
        console.print("\n[dim]Stopped.[/dim]")


@app.command(name="list")
def list_cmd():
    """List saved transcripts."""
    transcripts = list_transcripts()
    if not transcripts:
        console.print("[dim]No transcripts saved yet.[/dim]")
        return

    table = Table(title="Saved Transcripts")
    table.add_column("ID", style="cyan")
    table.add_column("Date", style="white")
    table.add_column("Duration", style="green")
    table.add_column("Preview", style="dim", max_width=50)

    for t in transcripts:
        dur = int(t.get("duration_seconds", 0))
        mins, secs = divmod(dur, 60)
        preview = t.get("full_text", "")[:50]
        if len(t.get("full_text", "")) > 50:
            preview += "..."

        table.add_row(
            t["id"],
            t.get("created_at", "")[:19],
            f"{mins}m {secs}s",
            preview,
        )

    console.print(table)


@app.command()
def show(id: str = typer.Argument("latest", help="Transcript ID or 'latest'")):
    """Show a transcript's full text."""
    record = get_transcript(id)
    if record is None:
        console.print(f"[red]Transcript not found: {id}[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Transcript: {record['id']}[/cyan]")
    console.print(f"[dim]Date: {record.get('created_at', '')} | Duration: {record.get('duration_seconds', 0):.0f}s | Model: {record.get('model', 'unknown')}[/dim]\n")
    console.print(record.get("full_text", ""))


@app.command()
def delete(id: str = typer.Argument(..., help="Transcript ID or prefix")):
    """Delete a saved transcript."""
    if delete_transcript(id):
        console.print(f"[green]Deleted transcript: {id}[/green]")
    else:
        console.print(f"[red]Transcript not found: {id}[/red]")
        raise typer.Exit(1)


@app.command()
def search(query: str = typer.Argument(..., help="Search text")):
    """Search transcripts by content."""
    results = search_transcripts(query)
    if not results:
        console.print(f"[dim]No transcripts matching '{query}'[/dim]")
        return

    console.print(f"[green]Found {len(results)} transcript(s):[/green]\n")
    for t in results:
        console.print(f"[cyan]{t['id']}[/cyan]")
        text = t.get("full_text", "")
        # Show context around the match
        idx = text.lower().find(query.lower())
        start = max(0, idx - 40)
        end = min(len(text), idx + len(query) + 40)
        snippet = ("..." if start > 0 else "") + text[start:end] + ("..." if end < len(text) else "")
        console.print(f"  {snippet}\n")
