#!/usr/bin/env python3
"""
NeuTTS Daemon — persistent synthesis server.

Keeps the NeuTTS model (~500MB) loaded in memory and accepts synthesis
requests over a Unix domain socket.  Exits automatically after a
configurable idle timeout with no incoming requests.

Protocol
--------
Each request is a single JSON object followed by a newline::

    {"text": "Hello world", "out": "/tmp/output.wav", "ref-audio": "/path/to/voice.wav", "ref-text": "/path/to/voice.txt"}

The daemon replies with a JSON object followed by a newline::

    {"ok": true}
    {"ok": false, "error": "..."}

Lifecycle
---------
- Model (backbone + codec) is loaded once on startup.
- An ``idle_timeout`` timer starts after the last completed request.
- If no request arrives within ``idle_timeout`` seconds, the daemon exits.
- A ``shutdown`` control message (``{"cmd": "shutdown"}``) exits immediately.

Usage
-----
.. code-block:: bash

    python -m tools.neutts_daemon \\
        --ref-audio samples/jo.wav \\
        --ref-text samples/jo.txt \\
        --socket-path /tmp/neutts_daemon.sock \\
        --idle-timeout 300

Required: ``pip install -U neutts[all]``
System:   ``brew install espeak-ng``
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import struct
import sys
import tempfile
import threading
from pathlib import Path

log = logging.getLogger("neutts_daemon")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _send_json(sock: socket.socket, obj: dict) -> None:
    data = (json.dumps(obj) + "\n").encode("utf-8")
    sock.sendall(data)



def _recv_json(sock: socket.socket) -> dict:
    """Read one newline-terminated JSON object from a socket."""
    buf = b""
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("client disconnected")
        buf += chunk
        if b"\n" in buf:
            line, _ = buf.split(b"\n", 1)
            return json.loads(line.decode("utf-8"))


def _write_wav(path: str, samples, sample_rate: int = 24000) -> None:
    import numpy as np  # noqa: F811 — imported here to avoid heavy import at module level

    if not isinstance(samples, np.ndarray):
        samples = np.array(samples, dtype=np.float32)
    samples = samples.flatten()
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767).astype(np.int16)

    import struct as _struct
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
    block_align = num_channels * (bits_per_sample // 8)
    data_size = len(pcm) * (bits_per_sample // 8)

    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(_struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(_struct.pack("<IHHIIHH", 16, 1, num_channels, sample_rate,
                             byte_rate, block_align, bits_per_sample))
        f.write(b"data")
        f.write(_struct.pack("<I", data_size))
        f.write(pcm.tobytes())

    log.info("WAV written: %s", path)


# ---------------------------------------------------------------------------
# Idle-timer thread
# ---------------------------------------------------------------------------

class IdleTimer:
    """Resets on each request; fires ``on_expire`` after ``timeout`` seconds."""

    def __init__(self, timeout: float, on_expire) -> None:
        self._timeout = timeout
        self._on_expire = on_expire
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._timeout, self._on_expire)
            self._timer.daemon = True
            self._timer.start()

    def stop(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None


# ---------------------------------------------------------------------------
# Main server loop
# ---------------------------------------------------------------------------

def serve(
    ref_audio: str,
    ref_text_path: str,
    socket_path: str = "/tmp/neutts_daemon.sock",
    backbone_repo: str = "neuphonic/neutts-air-q4-gguf",
    codec_repo: str = "neuphonic/neucodec",
    device: str = "cpu",
    idle_timeout: float = 300.0,
) -> None:
    """Load the model, listen on a Unix socket, handle requests until idle."""

    # --- remove stale socket ---
    sp = Path(socket_path)
    if sp.exists():
        sp.unlink()

    ref_text = Path(ref_text_path).expanduser().read_text(encoding="utf-8").strip()

    log.info("Loading NeuTTS model backbone=%s codec=%s device=%s …",
             backbone_repo, codec_repo, device)

    from neutts import NeuTTS  # heavy import — done once

    tts = NeuTTS(
        backbone_repo=backbone_repo,
        backbone_device=device,
        codec_repo=codec_repo,
        codec_device=device,
    )
    log.info("Encoding reference audio …")
    ref_codes = tts.encode_reference(str(Path(ref_audio).expanduser()))
    log.info("Model ready.  Listening on %s  (idle timeout %.0fs)", socket_path, idle_timeout)

    # --- socket setup ---
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(socket_path)
    srv.listen(4)
    srv.settimeout(1.0)  # allow periodic checks for shutdown flag

    stop_flag = threading.Event()

    def _on_idle_expire() -> None:
        log.info("Idle timeout (%.0fs) reached — shutting down.", idle_timeout)
        stop_flag.set()

    idle = IdleTimer(timeout=float(idle_timeout), on_expire=_on_idle_expire)

    # --- signal handlers ---
    def _handle_sigterm(*_):
        log.info("SIGTERM received — shutting down.")
        stop_flag.set()

    signal.signal(signal.SIGTERM, _handle_sigterm)

    try:
        while not stop_flag.is_set():
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue

            idle.reset()

            try:
                req = _recv_json(conn)
            except (ConnectionError, json.JSONDecodeError) as exc:
                log.warning("Bad request: %s", exc)
                conn.close()
                continue

            # --- control commands ---
            if req.get("cmd") == "shutdown":
                log.info("Shutdown requested by client.")
                _send_json(conn, {"ok": True})
                conn.close()
                stop_flag.set()
                continue

            text = req.get("text", "")
            out_path = req.get("out", "")

            if not text or not out_path:
                _send_json(conn, {"ok": False, "error": "missing 'text' or 'out'"})
                conn.close()
                continue

            try:
                log.info("Synthesising %d chars → %s", len(text), out_path)
                wav = tts.infer(text, ref_codes, ref_text)

                out = Path(out_path)
                out.parent.mkdir(parents=True, exist_ok=True)

                try:
                    import soundfile as sf
                    sf.write(str(out), wav, 24000)
                except ImportError:
                    _write_wav(str(out), wav, 24000)

                _send_json(conn, {"ok": True})
                log.info("Done: %s", out_path)
            except Exception as exc:
                log.exception("Synthesis error")
                _send_json(conn, {"ok": False, "error": str(exc)})
            finally:
                conn.close()

            idle.reset()  # start countdown after each request

    finally:
        idle.stop()
        srv.close()
        if sp.exists():
            sp.unlink()
        log.info("Daemon exited.")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="NeuTTS persistent synthesis daemon")
    parser.add_argument("--ref-audio", required=True, help="Reference voice audio path")
    parser.add_argument("--ref-text", required=True, help="Reference voice transcript path")
    parser.add_argument("--socket-path", default="/tmp/neutts_daemon.sock",
                        help="Unix domain socket path (default: /tmp/neutts_daemon.sock)")
    parser.add_argument("--model", default="neuphonic/neutts-air-q4-gguf",
                        help="Backbone model repo (default: neuphonic/neutts-air-q4-gguf)")
    parser.add_argument("--codec", default="neuphonic/neucodec",
                        help="Codec model repo (default: neuphonic/neucodec)")
    parser.add_argument("--device", default="cpu",
                        help="Device — cpu / mps / cuda  (default: cpu)")
    parser.add_argument("--idle-timeout", type=float, default=300.0,
                        help="Idle seconds before auto-exit (default: 300)")
    parser.add_argument("--log-level", default="INFO",
                        help="Python log level (default: INFO)")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    serve(
        ref_audio=args.ref_audio,
        ref_text_path=args.ref_text,
        socket_path=args.socket_path,
        backbone_repo=args.model,
        codec_repo=args.codec,
        device=args.device,
        idle_timeout=float(args.idle_timeout),
    )


if __name__ == "__main__":
    main()
