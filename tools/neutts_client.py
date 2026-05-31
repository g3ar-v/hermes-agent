#!/usr/bin/env python3
"""
NeuTTS client helper — manages a persistent neutts_daemon process.

Called by ``_generate_neutts()`` in ``tts_tool.py``.  The client checks
whether the daemon is already listening on the configured Unix socket;
if not, it spawns ``neutts_daemon.py``, waits for the socket to appear,
and then sends the synthesis request.

If the daemon cannot be started or is unreachable the caller should fall
back to the old one-shot ``neutts_synth.py`` subprocess.

Public API
==========
    from tools.neutts_client import synthesize_via_daemon

    ok = synthesize_via_daemon(
        text="Hello",
        out="/tmp/test.wav",
        ref_audio="/path/to/voice.wav",
        ref_text="/path/to/voice.txt",
        model="neuphonic/neutts-air-q4-gguf",
        device="mps",
        socket_path="/tmp/neutts_daemon.sock",
        daemon_idle_timeout=300,
        request_timeout=120,
    )
    # Returns True on success, False on failure.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger("neutts_client")

# ---------------------------------------------------------------------------
# Defaults — can be overridden via the public API
# ---------------------------------------------------------------------------

DEFAULT_SOCKET_PATH = "/tmp/neutts_daemon.sock"
DEFAULT_DAEMON_IDLE_TIMEOUT = 300  # seconds
DEFAULT_REQUEST_TIMEOUT = 120      # seconds per synthesize call

# How long to wait for the daemon to create its socket after spawning it
_DAEMON_STARTUP_TIMEOUT = 120  # seconds
_DAEMON_STARTUP_POLL = 0.2     # seconds between polls


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _socket_is_alive(socket_path: str) -> bool:
    """Return True if a Unix socket exists and is accepting connections."""
    if not Path(socket_path).exists():
        return False
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(socket_path)
        s.close()
        return True
    except (ConnectionRefusedError, FileNotFoundError, OSError):
        return False


def _send_request(socket_path: str, req: dict, timeout: float) -> dict:
    """Connect, send a JSON request, receive the response."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(socket_path)
        data = (json.dumps(req) + "\n").encode("utf-8")
        s.sendall(data)
        # receive response
        buf = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                raise ConnectionError("daemon closed connection")
            buf += chunk
            if b"\n" in buf:
                line, _ = buf.split(b"\n", 1)
                return json.loads(line.decode("utf-8"))
    finally:
        s.close()


def _start_daemon(
    ref_audio: str,
    ref_text: str,
    model: str,
    device: str,
    socket_path: str,
    idle_timeout: float,
) -> subprocess.Popen | None:
    """
    Spawn ``neutts_daemon.py`` and wait until the socket is listening.
    Returns the Popen object or None on failure.
    """
    synth_script = Path(__file__).parent / "neutts_daemon.py"
    if not synth_script.exists():
        log.error("neutts_daemon.py not found at %s", synth_script)
        return None

    # Remove a stale socket left behind by a crashed daemon
    sp = Path(socket_path)
    if sp.exists():
        sp.unlink()

    cmd = [
        sys.executable, str(synth_script),
        "--ref-audio", ref_audio,
        "--ref-text", ref_text,
        "--socket-path", socket_path,
        "--model", model,
        "--device", str(device),
        "--idle-timeout", str(idle_timeout),
    ]

    log.info("Starting NeuTTS daemon: %s", " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for the socket to appear
    deadline = time.monotonic() + _DAEMON_STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if _socket_is_alive(socket_path):
            log.info("Daemon ready (%.1fs).", time.monotonic() - (deadline - _DAEMON_STARTUP_TIMEOUT))
            return proc
        # Also bail if the daemon exited early
        if proc.poll() is not None:
            log.error("Daemon exited early with code %d.", proc.returncode)
            return None
        time.sleep(_DAEMON_STARTUP_POLL)

    log.error("Daemon did not start within %.0fs.", _DAEMON_STARTUP_TIMEOUT)
    proc.kill()
    proc.wait(timeout=5)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def synthesize_via_daemon(
    text: str,
    out: str,
    ref_audio: str,
    ref_text: str,
    model: str = "neuphonic/neutts-air-q4-gguf",
    device: str = "cpu",
    socket_path: str = DEFAULT_SOCKET_PATH,
    daemon_idle_timeout: float = DEFAULT_DAEMON_IDLE_TIMEOUT,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
) -> bool:
    """
    Synthesise *text* → WAV at *out* using a persistent NeuTTS daemon.

    Starts the daemon on the first call; subsequent calls reuse it.
    Returns ``True`` on success, ``False`` on failure.
    """
    socket_path = str(socket_path)

    if not _socket_is_alive(socket_path):
        log.info("NeuTTS daemon not running — starting …")
        maybe_proc = _start_daemon(
            ref_audio=ref_audio,
            ref_text=ref_text,
            model=model,
            device=device,
            socket_path=socket_path,
            idle_timeout=daemon_idle_timeout,
        )
        if maybe_proc is None:
            return False

    try:
        resp = _send_request(socket_path, {"text": text, "out": out}, timeout=float(request_timeout))
        if resp.get("ok"):
            return True
        log.error("Daemon returned error: %s", resp.get("error", "unknown"))
        return False
    except (ConnectionError, socket.timeout, OSError) as exc:
        log.error("Communication with daemon failed: %s", exc)
        return False
