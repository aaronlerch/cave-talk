"""Mac menu bar app for cave-talk."""

from __future__ import annotations

import subprocess
import threading
import logging
from datetime import datetime, timezone

from Foundation import NSRunLoop, NSRunLoopCommonModes
import rumps

from .audio import AudioCapture, list_devices, refresh_devices
from .config import Config, ensure_dirs
from .log import setup_logging
from .storage import list_transcripts, save_transcript
from .transcribe import transcribe
from .wake import WakePhraseDetector

log = logging.getLogger("cave_talk.menubar")

# Menu bar icon states
ICON_IDLE = "🎙️"
ICON_LISTENING = "🎙️"
ICON_CAPTURING = "⏳"


class CaveTalkApp(rumps.App):
    def __init__(self):
        super().__init__(
            "cave-talk",
            title=ICON_IDLE,
            quit_button=None,  # we'll add our own
        )

        self.config = Config.load()
        ensure_dirs()
        setup_logging()

        self.capture: AudioCapture | None = None
        self.wake_detector: WakePhraseDetector | None = None
        self._device_name: str = ""
        self._is_listening = False
        self._is_capturing = False

        # Build menu
        self._status_item = rumps.MenuItem("Not listening", callback=None)
        self._status_item.set_callback(None)
        self._buffer_item = rumps.MenuItem("Buffer: --:--", callback=None)
        self._buffer_item.set_callback(None)

        self._listen_item = rumps.MenuItem("Start Listening", callback=self.toggle_listening)
        self._wake_item = rumps.MenuItem("Wake Phrase Detection", callback=self.toggle_wake)
        self._wake_item.state = 1  # on by default
        self._wake_enabled = True

        self._capture_item = rumps.MenuItem("Capture Now", callback=self.do_capture)
        self._capture_item.set_callback(None)  # disabled until listening

        self._device_menu = rumps.MenuItem("Device")
        self._init_device_menu()

        self._transcripts_menu = rumps.MenuItem("Recent Transcripts")

        self._phrase_item = rumps.MenuItem(
            f'Phrase: "{self.config.wake_phrase}"', callback=None
        )
        self._phrase_item.set_callback(None)

        self.menu = [
            self._status_item,
            self._buffer_item,
            None,  # separator
            self._listen_item,
            self._capture_item,
            None,
            self._wake_item,
            self._phrase_item,
            None,
            self._device_menu,
            self._transcripts_menu,
            None,
            rumps.MenuItem("Quit", callback=self.quit_app),
        ]

        self._populate_transcripts()

        # Ensure timers fire even while the menu is open (macOS switches
        # the run loop to NSEventTrackingRunLoopMode during menu tracking,
        # but rumps only registers timers in NSDefaultRunLoopMode).
        rumps.events.before_start.register(self._enable_timer_during_menu)

        # Auto-start listening
        self._start_listening()

    def _enable_timer_during_menu(self):
        """Re-register all rumps timers in NSRunLoopCommonModes."""
        run_loop = NSRunLoop.currentRunLoop()
        for t in rumps.timers():
            if t.is_alive and hasattr(t, '_nstimer'):
                run_loop.addTimer_forMode_(t._nstimer, NSRunLoopCommonModes)

    def _init_device_menu(self):
        """Populate the device submenu (safe during __init__)."""
        devs = list_devices()
        for d in devs:
            label = d["name"]
            if d["default"]:
                label += " (default)"
            item = rumps.MenuItem(label, callback=self._make_device_callback(d["index"], d["name"]))
            if self.config.device == d["index"]:
                item.state = 1
            self._device_menu.add(item)

    def _rebuild_device_menu(self):
        """Rebuild device submenu after app is running.

        Does NOT call refresh_devices() here — that would invalidate any
        active PortAudio stream.  The caller is responsible for refreshing
        before the stream is opened.
        """
        self._device_menu.clear()
        self._init_device_menu()

    def _make_device_callback(self, device_index: int, device_name: str):
        def callback(sender):
            self.config.device = device_index
            self.config.save()
            log.info("Switched device to %d: %s", device_index, device_name)

            # Uncheck all, check selected
            for item in self._device_menu.values():
                item.state = 0
            sender.state = 1

            # Restart listening with new device
            if self._is_listening:
                self._stop_listening()
                self._start_listening()

        return callback

    def _start_listening(self):
        # Re-scan hardware so unplugged/replugged devices are picked up
        try:
            refresh_devices()
        except Exception as e:
            log.warning("Device refresh failed (non-fatal): %s", e)

        devs = list_devices()
        if not devs:
            rumps.notification("cave-talk", "Error", "No audio input devices found.")
            return

        # Resolve device
        if self.config.device is not None:
            dev_info = next((d for d in devs if d["index"] == self.config.device), None)
            if not dev_info:
                dev_info = next((d for d in devs if d["default"]), devs[0])
                self.config.device = dev_info["index"]
        else:
            dev_info = next((d for d in devs if d["default"]), devs[0])
            self.config.device = dev_info["index"]

        self._device_name = dev_info["name"]
        self.config.save()

        self.capture = AudioCapture(self.config)
        try:
            self.capture.start()
        except Exception as e:
            log.warning("Failed to start capture on device %d (%s): %s",
                        self.config.device, self._device_name, e)
            # Fall back to default device
            default_dev = next((d for d in devs if d["default"]), devs[0])
            if default_dev["index"] != self.config.device:
                log.info("Retrying with default device %d (%s)",
                         default_dev["index"], default_dev["name"])
                self.config.device = default_dev["index"]
                self._device_name = default_dev["name"]
                self.config.save()
                self.capture = AudioCapture(self.config)
                try:
                    self.capture.start()
                except Exception as e2:
                    log.error("Failed to start capture on default device: %s", e2)
                    rumps.notification("cave-talk", "Error", f"Failed to start: {e2}")
                    return
            else:
                log.error("Failed to start capture: %s", e)
                rumps.notification("cave-talk", "Error", f"Failed to start: {e}")
                return

        self._is_listening = True
        self._listen_item.title = "Stop Listening"
        self._capture_item.set_callback(self.do_capture)
        self._status_item.title = f"Listening: {self._device_name}"
        self.title = ICON_LISTENING
        log.info("Started listening on %s", self._device_name)

        # Start wake detection if enabled
        if self._wake_enabled:
            self._start_wake()

        # Update device menu checkmarks
        self._rebuild_device_menu()

    def _stop_listening(self):
        if self.wake_detector:
            self.wake_detector.stop()
            self.wake_detector = None

        if self.capture:
            self.capture.stop()
            self.capture = None

        self._is_listening = False
        self._listen_item.title = "Start Listening"
        self._capture_item.set_callback(None)
        self._status_item.title = "Not listening"
        self._buffer_item.title = "Buffer: --:--"
        self.title = ICON_IDLE
        log.info("Stopped listening")

    def _start_wake(self):
        if not self.capture:
            return
        self.wake_detector = WakePhraseDetector(
            buffer=self.capture.buffer,
            config=self.config,
            on_wake=self._on_wake_phrase,
        )
        self.wake_detector.start()
        log.info("Wake detection enabled: %r", self.config.wake_phrase)

    def _stop_wake(self):
        if self.wake_detector:
            self.wake_detector.stop()
            self.wake_detector = None
            log.info("Wake detection disabled")

    def _on_wake_phrase(self):
        """Called from wake detector thread when phrase is detected."""
        log.info("Wake phrase detected!")
        # Schedule capture on main thread
        rumps.Timer(0, lambda _: self._perform_capture(wake_triggered=True)).start()

    def toggle_listening(self, sender):
        if self._is_listening:
            self._stop_listening()
        else:
            self._start_listening()

    def toggle_wake(self, sender):
        self._wake_enabled = not self._wake_enabled
        sender.state = 1 if self._wake_enabled else 0

        if self._is_listening:
            if self._wake_enabled:
                self._start_wake()
            else:
                self._stop_wake()

    def do_capture(self, sender):
        if self._is_capturing:
            return
        threading.Thread(target=self._perform_capture, daemon=True).start()

    def _perform_capture(self, wake_triggered=False):
        if self._is_capturing or not self.capture:
            return
        self._is_capturing = True
        self.title = ICON_CAPTURING

        try:
            audio = self.capture.buffer.snapshot()
            if audio is None or len(audio) == 0:
                rumps.notification("cave-talk", "No audio", "Buffer is empty.")
                return

            buffered_secs = len(audio) / self.config.sample_rate
            mins, secs = divmod(int(buffered_secs), 60)
            log.info("Capturing %dm %ds of audio", mins, secs)

            transcript = transcribe(audio, self.config)
            record = save_transcript(transcript, device_name=self._device_name)

            preview = transcript.full_text[:100]
            if len(transcript.full_text) > 100:
                preview += "..."

            trigger = "Wake phrase" if wake_triggered else "Manual capture"
            rumps.notification(
                "cave-talk",
                f"Saved {mins}m {secs}s transcript",
                preview,
            )
            log.info("Transcript saved: %s (%s)", record["id"], trigger)

            # TTS feedback
            try:
                subprocess.Popen(
                    ["say", f"I've saved the last {mins} minutes of transcript for you."],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError:
                pass

            self._refresh_transcripts()

        except RuntimeError as e:
            log.error("Transcription failed: %s", e)
            rumps.notification("cave-talk", "Error", str(e))
        finally:
            self._is_capturing = False
            self.title = ICON_LISTENING if self._is_listening else ICON_IDLE
            if self.wake_detector:
                self.wake_detector.clear_cooldown()

    @staticmethod
    def _friendly_time(iso_str: str) -> str:
        """Format an ISO timestamp as a friendly relative or absolute string."""
        try:
            dt = datetime.fromisoformat(iso_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            delta = now - dt
            seconds = delta.total_seconds()

            if seconds < 60:
                return "just now"
            if seconds < 3600:
                m = int(seconds // 60)
                return f"{m}m ago"
            if seconds < 86400:
                h = int(seconds // 3600)
                return f"{h}h ago"
            if seconds < 172800:
                return "yesterday " + dt.astimezone().strftime("%-I:%M %p").lower()
            if seconds < 604800:
                return dt.astimezone().strftime("%A %-I:%M %p").lower()
            return dt.astimezone().strftime("%b %-d, %-I:%M %p").lower()
        except (ValueError, TypeError):
            return ""

    def _populate_transcripts(self):
        """Add transcript items to the submenu."""
        transcripts = list_transcripts()[:5]

        if not transcripts:
            item = rumps.MenuItem("No transcripts yet", callback=None)
            item.set_callback(None)
            self._transcripts_menu.add(item)
            return

        for t in transcripts:
            dur = int(t.get("duration_seconds", 0))
            mins, secs = divmod(dur, 60)
            when = self._friendly_time(t.get("created_at", ""))
            preview = t.get("full_text", "")[:40]
            if len(t.get("full_text", "")) > 40:
                preview += "..."
            label = f"{when} ({mins}m{secs:02d}s) {preview}"
            self._transcripts_menu.add(rumps.MenuItem(label, callback=None))

    def _refresh_transcripts(self):
        """Rebuild transcripts submenu (safe only after app is running)."""
        self._transcripts_menu.clear()
        self._populate_transcripts()

        self._transcripts_menu.add(None)  # separator
        self._transcripts_menu.add(
            rumps.MenuItem("Open Transcripts Folder", callback=self._open_transcripts_folder)
        )

    def _open_transcripts_folder(self, sender):
        from .config import TRANSCRIPTS_DIR
        subprocess.Popen(["open", str(TRANSCRIPTS_DIR)])

    @rumps.timer(1)
    def _update_buffer(self, sender):
        """Update buffer and status display."""
        if not self._is_listening or not self.capture:
            return
        buffered = self.capture.buffer.seconds_buffered
        mins, secs = divmod(buffered, 60)
        self._buffer_item.title = f"Buffer: {mins:02d}:{secs:02d}"

    def quit_app(self, sender):
        self._stop_listening()
        rumps.quit_application()


def main():
    CaveTalkApp().run()


if __name__ == "__main__":
    main()
