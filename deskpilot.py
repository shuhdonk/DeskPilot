#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
════════════════════════════════════════════════════════════════════════════
 Deskpilot — AI Desktop Assistant  (v1.0)
════════════════════════════════════════════════════════════════════════════
 A complete, standalone, production-ready agentic chat assistant built with
 Python + Tkinter + the OpenAI Python SDK. Works with any OpenAI-compatible
 server: Ollama, LM Studio, vLLM, llama.cpp, or the real OpenAI API.

 FEATURES
 ────────
 • Dark modern UI  (#171717 / #202123 / #40414F + blue/green/red/amber accents)
 • Customized Windows dark title bar via DwmSetWindowAttribute (ctypes)
 • Resizable split-pane layout (ttk.Panedwindow), collapsible sidebar
 • Persistent chat threads  → deskpilot_chats.json   (full OpenAI-format history)
 • Persistent settings       → deskpilot_settings.json (geometry, server URL, ...)
 • Agentic tool suite with per-tool permission dropdowns:
     searxng_search        – queries a local SearXNG instance (JSON API)
      exa_search            – Exa neural web search (api.exa.ai; key in Settings)
      firecrawl_scrape      – Firecrawl scrape to markdown (metered; key in Settings)
     fetch_url             – retrieves + cleans raw text from a webpage
     run_javascript        – Node.js via hidden subprocess, 60 s timeout
     list_directory        – folder contents (folders first, size + modified time)
     read_local_file       – text & PDF (pypdf), max FILE_READ_LIMIT chars
     write_local_file      – creates / overwrites local text files
     generate_local_image  – local FLUX/SDXL CLI (ai-imagegen/generate.py)
     capture_screen        – PIL ImageGrab, queued for vision analysis
     get_clipboard_text    – thread-safe system clipboard read
 • "Ask Permission" mode → modal, thread-safe messagebox.askyesno prompt
 • Custom streaming Markdown engine: **bold**, *italic*, `inline code`,
   fenced code blocks with interactive "📋 Copy Code" buttons, #/##/### headers
 • Collapsible accordions (Tkinter elide=True tags) for Tool Calls,
   Model Reasoning and Tool Output Results — clickable ▶ / ▼ arrows
 • Text-to-Speech (kokoro-onnx + sounddevice, local), Speech-to-Text dictation (speech_recognition)
 • File/image attachments (.png .jpg .jpeg .pdf .txt ...)
 • Live decode tok/s + TTFT speed readout (prefill excluded from the rate)
 • Live session context-usage readout (📏 used/max) in the top bar
 • All network / streaming / tool work runs in daemon threads; every UI
   mutation is marshalled back to the main thread via a thread-safe queue.

 DEPENDENCIES
 ────────────
 Required : openai              (pip install openai)
 Optional : pypdf               → PDF reading
            Pillow              → screen-capture tool
            kokoro-onnx         → text-to-speech (+ sounddevice, numpy)
            SpeechRecognition   → microphone dictation (+ PyAudio)

 RUN
 ───
 python deskpilot.py
"""

from __future__ import annotations

import atexit
import base64
import ctypes
import gzip
import io
import html as html_mod
import ipaddress
import json
import os
import queue
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

# ── OpenAI SDK (required) ────────────────────────────────────────────────────
try:
    from openai import OpenAI
except ImportError:         # pragma: no cover
    OpenAI = None           # type: ignore

# ── Optional dependencies (graceful degradation) ─────────────────────────────
try:
    from pypdf import PdfReader
except Exception:            # pragma: no cover
    PdfReader = None         # type: ignore
try:
    from PIL import ImageGrab
except Exception:            # pragma: no cover
    ImageGrab = None         # type: ignore
try:
    from PIL import Image as _PILImage      # screenshot downscale (capture + history migration)
except Exception:            # pragma: no cover
    _PILImage = None         # type: ignore
try:
    from kokoro_onnx import Kokoro      # local TTS engine (kokoro-onnx)
except Exception:            # pragma: no cover
    Kokoro = None            # type: ignore
try:
    import sounddevice       # TTS playback (+ optional mic capture)
except Exception:            # pragma: no cover
    sounddevice = None       # type: ignore
try:
    import speech_recognition as sr
except Exception:            # pragma: no cover
    sr = None                # type: ignore
try:
    from faster_whisper import WhisperModel   # local dictation - keeps audio on this machine
except Exception:            # pragma: no cover
    WhisperModel = None      # type: ignore

# ---------------------------------------------------------------------------
#  Local dictation (faster-whisper) - optional; auto-used when installed
# ---------------------------------------------------------------------------
_WHISPER_MODEL = None           # loaded singleton (preloaded at mic start, or lazily on first phrase)
_WHISPER_LOCK = threading.Lock()   # guards the load so preload + first phrase can't double-load
WHISPER_MODEL_NAME = "base.en"  # small + fast on CPU int8; tiny.en / small.en also work

def _whisper_model():
    """Load the local Whisper model once (CPU, int8). Cached under the data dir.

    Thread-safe: a background preload and the first phrase can race; the lock
    makes the losing caller wait for the winning load instead of loading twice."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        with _WHISPER_LOCK:
            if _WHISPER_MODEL is None:
                import os as _os
                cache = (MODEL_CACHE_DIR or Path(GEN_DIR).parent) / "whisper_models"   # MODEL_CACHE_DIR pins the cache in --multi mode
                try:
                    _os.makedirs(cache, exist_ok=True)
                except OSError:
                    pass
                _WHISPER_MODEL = WhisperModel(WHISPER_MODEL_NAME, device="cpu", compute_type="int8",
                                              download_root=str(cache))
    return _WHISPER_MODEL

def _whisper_preload() -> None:
    """Background preload of the Whisper model (called when dictation starts).

    Runs on its own daemon thread so the microphone opens immediately; the
    first phrase just waits for this load to finish. If the preload fails, the
    first phrase retries the load and surfaces the error as before."""
    if WhisperModel is None or _WHISPER_MODEL is not None:
        return
    threading.Thread(target=_whisper_model, daemon=True).start()

KOKORO_MODEL_NAME = "kokoro-v0_19.onnx"   # v1.0 file names also accepted (see _kokoro_model)
_KOKORO_MODEL = None           # lazy-loaded singleton (model files live in <data_dir>/kokoro_models)

def _kokoro_model():
    """Load the local Kokoro ONNX model once. Cached under the data dir.

    Model files are NOT auto-downloaded - drop them into <data_dir>/kokoro_models:
      kokoro-v0_19.onnx (or kokoro-v1.0*.onnx)  +  voices-v1.0.bin (or a voice .json)
    """
    global _KOKORO_MODEL
    if _KOKORO_MODEL is None:
        import os as _os
        cache = (MODEL_CACHE_DIR or Path(GEN_DIR).parent) / "kokoro_models"   # MODEL_CACHE_DIR pins the cache in --multi mode
        try:
            _os.makedirs(cache, exist_ok=True)
        except OSError:
            pass
        model_f = next((f for f in ("kokoro-v0_19.onnx", "kokoro-v1.0.int8.onnx", "kokoro-v1.0.onnx")
                        if (cache / f).is_file()), None)
        voices_f = next((f for f in ("voices-v1.0.bin", "af_heart.json") if (cache / f).is_file()), None)
        if model_f is None or voices_f is None:
            missing = []
            if model_f is None:
                missing.append("kokoro-v0_19.onnx (or kokoro-v1.0*.onnx)")
            if voices_f is None:
                missing.append("voices-v1.0.bin (or af_heart.json)")
            raise FileNotFoundError(
                "Kokoro model files not found in " + str(cache) + " - needed: " + ", ".join(missing))
        _KOKORO_MODEL = Kokoro(str(cache / model_f), str(cache / voices_f))
    return _KOKORO_MODEL


def wav_bytes_to_float32(wav_bytes: bytes):
    """Decode 16-bit mono WAV bytes to float32 samples @16 kHz (linear resample if needed)."""
    import io as _io
    import wave
    import numpy as np
    wf = wave.open(_io.BytesIO(wav_bytes))
    rate, nch, width = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
    raw = wf.readframes(wf.getnframes())
    if width != 2 or nch != 1:
        raise ValueError(f"unsupported WAV format (rate={rate} ch={nch} width={width})")
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if rate != 16000:
        n_out = max(1, int(len(pcm) * 16000 / rate))
        x_old = np.linspace(0.0, 1.0, len(pcm), dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, n_out, dtype=np.float32)
        pcm = np.interp(x_new, x_old, pcm).astype(np.float32)
    return pcm

def whisper_transcribe_local(wav_bytes: bytes) -> str:
    """Transcribe one captured phrase locally; '' means silence / unintelligible."""
    model = _whisper_model()
    pcm = wav_bytes_to_float32(wav_bytes)
    segments, _info = model.transcribe(pcm, language="en", beam_size=1, vad_filter=True)
    return " ".join(seg.text.strip() for seg in segments).strip()


def _open_microphone():
    """Open the system default microphone.

    Some Windows machines have no *default* capture device configured even
    though input devices exist; sr.Microphone() then raises OSError.  Fall
    back to the first real capture device, skipping Stereo Mix / loopback
    (those pick up playback instead of speech).
    """
    try:
        return sr.Microphone()
    except OSError:
        pass
    try:
        import pyaudio as _pa
    except ImportError:
        raise
    pa = _pa.PyAudio()
    try:
        for i in range(pa.get_device_count()):
            d = pa.get_device_info_by_index(i)
            if not d.get("maxInputChannels"):
                continue
            name = str(d.get("name", "")).lower()
            if "stereo mix" in name or "loopback" in name:
                continue
            try:
                return sr.Microphone(device_index=i)
            except OSError:
                continue
        raise OSError("no usable microphone found (and no default input device is set)")
    finally:
        pa.terminate()


def _safe_stdio() -> None:
    """Make print() safe under any build mode (console, windowed EXE, redirected)."""
    for name in ("stdout", "stderr"):
        try:
            st = getattr(sys, name)
            if st is None:                        # --windowed PyInstaller builds
                setattr(sys, name, io.StringIO())
            else:
                st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_safe_stdio()


# ════════════════════════════════════════════════════════════════════════════
#  MCP CLIENT (Model Context Protocol - stdio transport, hand-rolled)
# ════════════════════════════════════════════════════════════════════════════
# Deskpilot can attach external MCP servers configured in Settings -> MCP
# Servers. Each server is a local subprocess speaking newline-delimited
# JSON-RPC 2.0 on stdin/stdout (the standard MCP stdio transport). This is a
# minimal, dependency-free client: initialize -> tools/list -> tools/call.
# Every tool a server advertises shows up in the permission bar as
# mcp_<server>_<tool> and defaults to "Ask Permission" - MCP servers are
# arbitrary local programs, so they must never be silently auto-approved.

MCP_PROTOCOL_VERSION = "2025-03-26"
MCP_MAX_SERVERS = 8          # hard cap on configured servers
MCP_INIT_TIMEOUT = 15        # seconds for spawn + initialize handshake
MCP_CALL_TIMEOUT = 120       # seconds per tools/call (long-running tools)


def mcp_tool_name(server: str, raw: str) -> str:
    """Namespaced tool name: mcp_<server>_<tool>, sanitized to [A-Za-z0-9_]."""
    def _clean(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]", "_", s).strip("_") or "x"
    return f"mcp_{_clean(server)}_{_clean(raw)}"


def _parse_env_pairs(text: str) -> Dict[str, str]:
    """Parse space-separated KEY=VALUE pairs (the Add-MCP-Server env field is a
    single-line tk.Entry, so whitespace - not newlines - separates the pairs).
    Malformed tokens without "=" are ignored."""
    out: Dict[str, str] = {}
    for pair in (text or "").split():
        if "=" in pair:
            k, v = pair.split("=", 1)
            if k.strip():
                out[k.strip()] = v.strip()
    return out


def mcp_tool_schema(server: str, raw: str, tool: dict) -> dict:
    """Convert one MCP tool definition to an OpenAI function-calling schema."""
    name = mcp_tool_name(server, raw)
    desc = str(tool.get("description") or "").strip() or f"MCP tool {raw} from server {server}"
    params = tool.get("inputSchema")
    if not isinstance(params, dict):
        params = {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc[:1000],
            "parameters": params,
        },
    }


def _interruptible_wait(q, timeout: float, abort=None, out: Optional[list] = None) -> bool:
    """Wait for a queue item / Event in STOP_POLL_S slices so Stop (or app close) can cut
    the wait short. Accepts either a queue.Queue or a threading.Event.

    Returns True when the item arrived / the event was set; False on timeout OR when
    'abort' (a no-arg predicate such as 'Stop pressed') turned True. The caller decides
    what a False means - normally: drop the pending slot and raise a clean error upstream.
    For a Queue the fetched item is stored in out[0] (the wait itself consumes it, so the
    caller must NOT get() again). Plain q.get(timeout=...) cannot be interrupted, which is
    why a long MCP call or an unanswered permission modal used to hold the worker for its
    entire timeout."""
    # threading.Event and queue.Queue wait differently: Event.wait(timeout=..) returns a
    # bool, Queue.get(timeout=..) raises Empty. Detect which one we were handed so both
    # callers (MCP response queue, permission-modal Event) behave correctly.
    is_event = hasattr(q, "is_set")

    def _slice(seconds: float) -> bool:
        if is_event:
            return bool(q.wait(timeout=seconds))
        try:
            item = q.get(timeout=seconds)
            if out is not None:
                out.append(item)
            return True
        except queue.Empty:
            return False

    deadline = time.monotonic() + max(0.0, float(timeout))
    while True:
        # Compute the remaining budget BEFORE sleeping, so a timeout SHORTER than one poll
        # slice is honored exactly instead of always costing a full STOP_POLL_S (the old loop
        # slept first and only trimmed on the last pass).
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        if _slice(min(STOP_POLL_S, remaining)):
            return True
        if abort is not None and abort():
            return False


def _kill_process(proc) -> None:
    """Terminate a child process, escalating to kill(). Never raises."""
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except Exception:
                pass
        if proc.poll() is None:
            proc.kill()
    except Exception:
        pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


class MCPClient:
    """One MCP server subprocess + its JSON-RPC session.

    A dedicated reader thread parses stdout into a pending-response queue;
    call_tool() sends a request and blocks (with timeout) until the matching
    response arrives. Notifications from the server are ignored. NOT safe for
    concurrent call_tool() from multiple threads - Deskpilot executes tool
    calls sequentially in one worker thread, which is all this client needs."""

    def __init__(self, name: str, command: str, args: Optional[List[str]] = None,
                 env: Optional[Dict[str, str]] = None):
        self.name = name
        self.command = command
        self.args = list(args or [])
        self.env = dict(env or {})
        self.proc: Optional[subprocess.Popen] = None
        self.tools: List[dict] = []
        self._pending: Dict[int, "queue.Queue"] = {}
        self._next_id = 0
        self._lock = threading.Lock()
        self._closed = False

    # ── lifecycle ────────────────────────────────────────────────────────
    def connect(self, timeout: float = MCP_INIT_TIMEOUT) -> List[dict]:
        """Spawn the server, run the initialize handshake, return its tools.

        Raises RuntimeError on any failure (spawn, protocol, timeout); the
        subprocess is always cleaned up before re-raising."""
        full_env = dict(os.environ)
        full_env.setdefault("PYTHONUNBUFFERED", "1")   # piped stdout is block-buffered by default
        full_env.update(self.env)
        try:
            self.proc = subprocess.Popen(
                [self.command] + self.args,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=full_env, bufsize=1, text=True, encoding="utf-8", errors="replace")
        except Exception as e:
            raise RuntimeError(f"could not start '{self.command}': {e}")
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        try:
            init_result = self._request("initialize", {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": APP_NAME, "version": VERSION},
            }, timeout=timeout)
            if not isinstance(init_result, dict):
                raise RuntimeError(f"unexpected initialize result: {str(init_result)[:200]}")
            self._notify("notifications/initialized", {})
            tools = self._request("tools/list", {}, timeout=timeout) or {}
            self.tools = [t for t in (tools.get("tools") or []) if isinstance(t, dict)]
            return self.tools
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        """Terminate the subprocess and fail all pending requests."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            pend, self._pending = self._pending, {}
        for q in pend.values():
            try:
                q.put_nowait({"_error": "MCP server closed"})
            except Exception:
                pass
        p = self.proc
        if p is not None:
            try:
                if p.poll() is None:
                    p.terminate()
                    try:
                        p.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        p.kill()
            except Exception:
                pass
            for stream in (p.stdin, p.stdout, p.stderr):
                try:
                    if stream:
                        stream.close()
                except Exception:
                    pass

    def ready(self) -> bool:
        return (not self._closed and self.proc is not None
                and self.proc.poll() is None)

    # ── JSON-RPC core ────────────────────────────────────────────────────
    def _reader(self) -> None:
        """Parse stdout lines; route responses to waiters, ignore the rest."""
        p = self.proc
        try:
            while True:
                line = p.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except Exception:
                    continue          # non-JSON noise on stdout - skip
                mid = msg.get("id")
                if isinstance(mid, int) and ("result" in msg or "error" in msg):
                    with self._lock:
                        q = self._pending.pop(mid, None)
                    if q is not None:
                        try:
                            q.put_nowait(msg)
                        except Exception:
                            pass
        except Exception:
            pass
        # EOF: fail every still-waiting request so callers don't hang.
        with self._lock:
            pend, self._pending = self._pending, {}
        for q in pend.values():
            try:
                q.put_nowait({"_error": "MCP server exited"})
            except Exception:
                pass

    def _drain_stderr(self) -> None:
        """Drain the server's stderr so a chatty server can't block on a full pipe."""
        p = self.proc
        try:
            for _line in (p.stderr or []):
                pass
        except Exception:
            pass

    def _send(self, obj: dict) -> None:
        p = self.proc
        if p is None or p.poll() is not None or p.stdin is None:
            raise RuntimeError(f"MCP server '{self.name}' is not running")
        try:
            p.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
            p.stdin.flush()
        except Exception as e:
            raise RuntimeError(f"could not write to MCP server '{self.name}': {e}")

    def _request(self, method: str, params: dict, timeout: float, abort=None) -> Any:
        """Send one JSON-RPC request and wait for its response.

        'abort' is an optional no-arg predicate polled every STOP_POLL_S while waiting
        (Stop pressed / app closing). When it turns True the pending slot is dropped and
        RuntimeError is raised, so a long tools/call unwinds instead of holding the worker
        for the whole MCP_CALL_TIMEOUT. The server subprocess itself is NOT killed here -
        that would break keep-alive between turns; only this request is abandoned."""
        with self._lock:
            self._next_id += 1
            mid = self._next_id
            q: "queue.Queue" = queue.Queue()
            self._pending[mid] = q
        try:
            self._send({"jsonrpc": "2.0", "id": mid, "method": method, "params": params})
        except Exception:
            with self._lock:
                self._pending.pop(mid, None)
            raise
        got: list = []
        if not _interruptible_wait(q, timeout, abort, out=got):
            with self._lock:
                self._pending.pop(mid, None)
            if abort is not None and abort():
                raise RuntimeError(f"MCP server '{self.name}' call cancelled by the user")
            raise RuntimeError(f"MCP server '{self.name}' timed out after {int(timeout)}s "
                               f"responding to {method}")
        msg = got[0]
        if "_error" in msg:
            raise RuntimeError(str(msg["_error"]))
        if "error" in msg:
            err = msg.get("error") or {}
            detail = err.get("message") if isinstance(err, dict) else str(err)
            raise RuntimeError(f"MCP server '{self.name}' error on {method}: {detail}")
        return msg.get("result")

    def _notify(self, method: str, params: dict) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    # ── tools ────────────────────────────────────────────────────────────
    def call_tool(self, raw_name: str, arguments: dict, timeout: float = MCP_CALL_TIMEOUT,
                  abort=None) -> str:
        """Call one of the server's tools; returns its text output.

        Raises RuntimeError on transport/protocol errors or when the tool
        reports isError (the server's error text is included). 'abort' is polled while
        waiting so Stop can cut a long-running tool call short."""
        result = self._request("tools/call", {"name": raw_name, "arguments": arguments},
                               timeout, abort=abort)
        if not isinstance(result, dict):
            return str(result)
        parts: List[str] = []
        for item in (result.get("content") or []):
            if not isinstance(item, dict):
                continue
            itype = item.get("type")
            if itype == "text":
                parts.append(str(item.get("text", "")))
            elif itype == "image":
                data = str(item.get("data", ""))
                mime = str(item.get("mimeType", "image/png"))
                parts.append(f"[image {mime}, {len(data)} base64 chars]")
            else:
                parts.append(json.dumps(item, ensure_ascii=False)[:500])
        text = "\n".join(parts).strip() or "(tool returned no content)"
        if result.get("isError"):
            raise RuntimeError(f"MCP tool '{raw_name}' reported an error: {text[:1000]}")
        return text


# ════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

APP_NAME   = "Deskpilot"
VERSION    = "1.1.10"
BASE_DIR   = Path(__file__).resolve().parent
# Data files default to next to the script; main() relocates them to
# %LOCALAPPDATA%\Deskpilot when that is writable (see resolve_data_dir).
SETTINGS_FILE = BASE_DIR / "deskpilot_settings.json"
CHATS_FILE    = BASE_DIR / "deskpilot_chats.json"
GEN_DIR      = BASE_DIR / "generated_images"
MODEL_CACHE_DIR: Optional[Path] = None   # --multi: pin whisper/kokoro caches to the MAIN data dir (None = derive from GEN_DIR)
try:
    GEN_DIR.mkdir(exist_ok=True)
except Exception:
    pass

MAX_TOOL_STEPS   = 0        # safety cap for agentic tool loops (0 = unlimited)
FILE_READ_LIMIT  = 250000    # chars returned by read_local_file
CLIPBOARD_LIMIT  = 65536     # chars returned by get_clipboard_text (schema + code share this; user-sized for local models)
CHATS_SAVE_MIN_INTERVAL = 5.0   # min seconds between mid-turn chats.json writes (turn end always saves)
CUSTOM_PROMPT_MAX = 4000        # char cap for the custom system prompt (Settings -> Custom System Prompt)
JS_TIMEOUT       = 900        # seconds, run_javascript subprocess timeout
TURN_TIME_LIMIT_DEFAULT = 0   # default wall-clock cap per user prompt (seconds; 0 = NO LIMIT).
                              # The real value is the "Turn time limit" SETTING (Settings dialog,
                              # clamped 0..86400) - see _turn_time_limit(). It stays a backstop for
                              # the agentic loop when MAX_TOOL_STEPS == 0 so a model that keeps
                              # calling tools cannot run forever, but it is checked at step
                              # boundaries only and now defaults to unlimited: long turns are the
                              # point of the app, and an unwanted cap kept cutting sessions short.
TURN_TIME_LIMIT_MAX   = 86400 # upper bound for the setting (24 h); 0 always means "no limit".
STOP_POLL_S      = 0.25       # max latency of the Stop button while a tool is blocking
IMAGE_CLI_TIMEOUT = 900       # seconds, local FLUX/SDXL CLI subprocess
PROC_OUTPUT_CAP  = 4_000_000  # bytes per stream kept from a subprocess (deadlock-safe drain)
TOOL_OUTPUT_LIMIT = 96000   # max chars of ANY tool result sent back to the model (hard cap)
MODEL_HISTORY_MAX = 20      # previously-used model names kept for the Model Name dropdown
TEMP_VALUES       = tuple(f"{v / 10:.1f}" for v in range(21))  # per-chat temperature choices: 0.0 .. 2.0 in 0.1 steps (21 values)
TEMP_DEFAULT      = "0.7"                                    # default temperature (sent when a chat has no explicit value)
THINK_VALUES      = ("Off", "Low", "Medium", "High")          # per-chat thinking level choices (no Extra High: == High on Qwen3.8)

# Sampler parameters (Settings -> Sampling). Global for every chat session by design:
# the chat UI stays uncluttered, so these live in Settings only and apply to all chats.
#   target "top"  = a real OpenAI-schema chat.completions field (sent as a top-level kwarg)
#   target "flat" = NOT in the OpenAI schema; flattened into the request JSON by the
#                   openai client's extra_body, which is how llama.cpp/Unsloth/vLLM read them.
#   off  = value at which the key is OMITTED from the request entirely (never a placeholder:
#          repetition_penalty 0 means INFINITE repetition on llama.cpp and is invalid on
#          OpenAI; presence/frequency 0 is their real neutral; top_k/min_p 0 = disabled.
#          top_p has no in-range neutral, so it is off only when the field is blank).
SAMPLING_SCHEMA = (
    # key                  label                                                        lo     hi     int?   target off
    ("top_p",              "Top P  (nucleus; 0-1, blank = server default)",             0.0,   1.0,   False, "top",  None),
    ("top_k",              "Top K  (0 = off)",                                          0,     400,   True,  "flat", 0),
    ("min_p",              "Min P  (0 = off)",                                          0.0,   1.0,   False, "flat", 0),
    ("repetition_penalty", "Repetition Penalty  (1.0 = off; strongest of the three)",    1.0,   2.0,   False, "flat", 1.0),
    ("presence_penalty",   "Presence Penalty  (-2 to 2, 0 = off)",                     -2.0,  2.0,   False, "top",  0),
    ("frequency_penalty",  "Frequency Penalty  (-2 to 2, 0 = off)",                    -2.0,  2.0,   False, "top",  0),
)
SAMPLING_DEFAULTS = {k: "" for (k, _l, _lo, _hi, _i, _t, _o) in SAMPLING_SCHEMA}
SAMPLING_FLAT_KEYS = tuple(k for (k, _l, _lo, _hi, _i, t, _o) in SAMPLING_SCHEMA if t == "flat")
FLUSH_INTERVAL   = 0.05     # seconds between UI flushes while streaming
# Core system directories that local file tools must never touch (security guard).
FORBIDDEN_SYSTEM_DIRS_WINDOWS = ("c:\\windows", "c:\\$recycle.bin", "c:\\system volume information")
FORBIDDEN_SYSTEM_DIRS_LINUX   = ("/etc", "/bin", "/sbin", "/boot", "/dev", "/lib", "/proc", "/sys")


def _dir_prefix(parent_str: str) -> str:
    """Separator-terminated prefix of a directory path, safe for ROOT folders.

    A drive root (C: + separator) or '/' ALREADY ends with a separator, so naively
    appending os.sep produced a doubled separator - a string no real child path ever
    matches, which made every legal path look like it was OUTSIDE the folder. Strip first,
    then add exactly one separator. Used by workspace confinement and the system-dir guard."""
    return (parent_str or "").rstrip("/\\") + os.sep


def _forbidden_dir_hit(p_str: str) -> Optional[str]:
    """Return the forbidden system dir that p_str (already lowercase) equals or is inside, else None."""
    forbidden = FORBIDDEN_SYSTEM_DIRS_WINDOWS if os.name == "nt" else FORBIDDEN_SYSTEM_DIRS_LINUX
    for d in forbidden:
        dl = d.lower()
        if p_str == dl or p_str.startswith(_dir_prefix(dl)):
            return d
    return None


def _traversal_component(raw: str) -> Optional[str]:
    """Return the first path COMPONENT of raw that is a dot-only parent reference, else None.

    Component-based on purpose: the original raw substring test ('".." in raw') also
    rejected perfectly legal filenames - a..b.txt, foo...bar, report..2026.pdf. A name is
    only suspicious when a WHOLE component consists of dots/spaces with 2+ dots: that covers
    "..", "../..", and the dot/space-padded spellings (".. ", ". .", "...") that Windows
    path parsing collapses into real parent hops. Mixed names like "a.." or "..a" are legal
    (verified: Path("C:/tmp/a../b").resolve() stays inside C:/tmp, it is not a hop).
    """
    for part in re.split(r"[\\/]+", str(raw or "")):
        # A component is a parent reference only if it is made ENTIRELY of dots/spaces
        # and holds 2+ dots: "..", "../..", ".. ", ". .", "...". Names that mix letters
        # with dots ("a..b.txt", "foo...bar", "a..", "..a") are legal and never hop.
        if part.count(".") >= 2 and set(part) <= {" ", "."}:
            return part
    return None

# ── Dark modern palette ──────────────────────────────────────────────────────
COL = {
    "bg_deep":   "#171717",   # deepest surface (sidebar, code blocks)
    "bg_main":   "#202123",   # main chat workspace
    "bg_raised": "#40414F",   # raised / hover surfaces
    "accent":    "#3B82F6",   # primary blue
    "success":   "#10B981",   # green
    "danger":    "#EF4444",   # red
    "warning":   "#F59E0B",   # amber
    "text":      "#E5E7EB",
    "text_dim":  "#9CA3AF",
    "border":    "#2E2F33",
}

# Font scaling: the "UI Font Size" setting (ui_font_size, default 12) sets the base
# size; every font in the app is built through F()/FM() so one delta re-scales it.
FONT_BASE  = 12        # design baseline that ui_font_size=12 reproduces exactly
FONT_DELTA = 0         # set from settings at startup and on change

def F(size: int, weight: str = "") -> tuple:
    """Segoe UI font scaled by the user's chosen base size."""
    f = ("Segoe UI", max(7, size + FONT_DELTA))
    return f if not weight else (f[0], f[1], weight)

def FM(size: int) -> tuple:
    """Monospace font scaled the same way (code blocks / inline code)."""
    return ("Consolas", max(7, size + FONT_DELTA))
# Fixed width of the reasoning accordion title region: lets us swap
# "Model Reasoning" -> "Thought for X.Xs" in place when a thinking phase
# ends, without shifting any stored text indices (Tk Text indices are
# position-based).
REASONING_REGION_LEN = 24

# ── Tool metadata (UI order + pretty names) ──────────────────────────────────
TOOLS: List[tuple] = [
    ("searxng_search",       "🔎 Web Search"),
    ("exa_search",           "🌟 Exa Search"),
    ("firecrawl_scrape",     "🔥 Firecrawl Scrape"),    ("fetch_url",            "🌐 Fetch URL"),
    ("run_javascript",       "⚡ Run JS (Node)"),
    ("list_directory",       "📁 List Directory"),
    ("read_local_file",      "📄 Read File"),
    ("write_local_file",     "✍️ Write File"),
    ("generate_local_image", "🎨 Generate Image"),
    ("capture_screen",       "📸 Capture Screen"),
    ("get_clipboard_text",   "📋 Clipboard"),
]

# ── OpenAI-compatible tool schemas ───────────────────────────────────────────
TOOL_SCHEMAS: Dict[str, dict] = {
    "searxng_search": {
        "type": "function",
        "function": {
            "name": "searxng_search",
            "description": ("Search the web using a local SearXNG metasearch instance. "
                            "Returns top results with titles, URLs and short snippets."),
            "parameters": {
                "type": "object",
                "properties": {
                    "query":       {"type": "string",  "description": "The search query."},
                    "max_results": {"type": "integer", "description": "Maximum number of results (1-20). Default 10."}
                },
                "required": ["query"]
            }
        }
    },
    "exa_search": {
        "type": "function",
        "function": {
            "name": "exa_search",
            "description": ("Search the web with Exa (neural/semantic search). Returns titles, URLs "
                            "and relevant text excerpts. Use it when the most up-to-date information is "
                            "needed (current events, recent news, latest developments); needs an Exa API key in Settings."),
            "parameters": {
                "type": "object",
                "properties": {
                    "query":             {"type": "string",  "description": "The search query."},
                    "num_results":       {"type": "integer", "description": "Results to return (1-25). Default 10."},
                    "search_type":       {"type": "string",  "enum": ["auto", "fast", "deep"],
                                          "description": "'auto' (default); 'deep' for research-style queries."},
                    "include_full_text": {"type": "boolean",
                                          "description": "Return full page text instead of highlights. Default false."}
                },
                "required": ["query"]
            }
        }
    },
    "firecrawl_scrape": {
        "type": "function",
        "function": {
            "name": "firecrawl_scrape",
            "description": ("Scrape a webpage into clean markdown using Firecrawl. Handles JavaScript-rendered "
                            "pages and anti-bot/JS-challenge walls that plain fetch_url cannot get past. "
                            "METERED: each scrape consumes one of the user's limited monthly credits - use ONLY "
                            "when fetch_url fails or returns a bot wall, or the page is known to be "
                            "JavaScript-rendered. Never for simple static pages (fetch_url is free)."),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Absolute http(s) URL to scrape."}
                },
                "required": ["url"]
            }
        }
    },
    "fetch_url": {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": ("Fetch a webpage and return its cleaned raw text content "
                            "(HTML tags, scripts and styles stripped). Local/private network "
                            "addresses (localhost, LAN, cloud metadata) are blocked by default."),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Absolute http(s) URL to fetch."}
                },
                "required": ["url"]
            }
        }
    },
    "run_javascript": {
        "type": "function",
        "function": {
            "name": "run_javascript",
            "description": ("Execute JavaScript code locally using Node.js in a hidden subprocess. "
                            f"Hard {JS_TIMEOUT} second timeout. Use console.log() to produce output. "
                            "stdout is capped at 8000 chars (a truncation marker is appended) - "
                            "for larger results write them to a file and read it back in chunks."),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "JavaScript source code to execute."}
                },
                "required": ["code"]
            }
        }
    },
    "read_local_file": {
        "type": "function",
        "function": {
            "name": "read_local_file",
            "description": (f"Read a local file and return its text content (max {FILE_READ_LIMIT} characters). "
                            "Supports plain-text files and PDF documents."),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Filesystem path of the file to read."}
                },
                "required": ["path"]
            }
        }
    },
    "list_directory": {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": ("List a local directory's contents (folders first, then files with size and "
                            "modified time). Use it to explore a folder before reading specific files."),
            "parameters": {
                "type": "object",
                "properties": {
                    "path":        {"type": "string",  "description": "Directory path to list."},
                    "max_entries": {"type": "integer", "description": "Maximum entries to show (1-1000). Default 200."}
                },
                "required": ["path"]
            }
        }
    },
    "write_local_file": {
        "type": "function",
        "function": {
            "name": "write_local_file",
            "description": ("Create or overwrite a local file with the given text content. "
                            "Parent directories are created automatically."),
            "parameters": {
                "type": "object",
                "properties": {
                    "path":    {"type": "string", "description": "Filesystem path to write."},
                    "content": {"type": "string", "description": "Full text content to save."}
                },
                "required": ["path", "content"]
            }
        }
    },
    "generate_local_image": {
        "type": "function",
        "function": {
            "name": "generate_local_image",
            "description": ("Generate an image locally via the Deskpilot FLUX/SDXL CLI "
                            "(ai-imagegen/generate.py on this machine's RTX 5090). Takes ~2-3 min per "
                            "image (model load + diffusion). Returns the saved file path."),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt":          {"type": "string",  "description": "Text prompt describing the image."},
                    "negative_prompt": {"type": "string",  "description": "Optional negative prompt (SDXL only)."},
                    "model":           {"type": "string",  "enum": ["flux", "sdxl"],
                                        "description": ("Diffusion model. 'flux' = FLUX.1-schnell FP8, fast (~4 steps); "
                                                        "'sdxl' = SDXL bf16 (20 steps), preferred for photorealistic images.")},
                    "width":           {"type": "integer", "description": "Image width in px. Default 1600."},
                    "height":          {"type": "integer", "description": "Image height in px. Default 1200 (SDXL: keep divisible by 8)."},
                    "steps":           {"type": "integer", "description": "Optional diffusion steps override (default per model)."}
                },
                "required": ["prompt"]
            }
        }
    },
    "capture_screen": {
        "type": "function",
        "function": {
            "name": "capture_screen",
            "description": ("Capture a screenshot of the desktop. The image is saved locally and "
                            "automatically queued/attached for visual analysis on the next model call."),
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    "get_clipboard_text": {
        "type": "function",
        "function": {
            "name": "get_clipboard_text",
            "description": f"Read the text currently in the system clipboard (max {CLIPBOARD_LIMIT} characters; longer text is truncated).",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
}

# ── Permission options ───────────────────────────────────────────────────────
PERM_LABELS = {"always": "Always Allow", "ask": "Ask Permission", "off": "Off"}
PERM_VALUES = {v: k for k, v in PERM_LABELS.items()}
DEFAULT_PERMS = {
    "searxng_search":        "always",
    "exa_search":            "always",
    "firecrawl_scrape":      "always",
    "fetch_url":             "always",
    "list_directory":        "always",
    "read_local_file":       "always",
    "get_clipboard_text":    "always",
    "run_javascript":        "ask",
    "write_local_file":      "ask",
    "generate_local_image": "ask",
    "capture_screen":        "ask",
}

DEFAULT_SETTINGS: Dict[str, Any] = {
    "window_geometry":   "1280x800",
    "sidebar_width":     250,
    "sidebar_collapsed": False,
    "server_url":        "http://localhost:11434/v1",
    "api_key":           "gemma-local",
    "model_name":        "gemma3",
    "model_history":     [],   # previously used model names, newest first (Model Name dropdown)
    "searxng_url":       "",    # blank = auto-derive from server_url host, port 8080
    "exa_api_key":       "",    # Exa API key for the exa_search tool; blank = tool disabled
    "firecrawl_api_key": "",    # Firecrawl API key for firecrawl_scrape (metered); blank = tool disabled
    "ui_font_size":    12,    # base UI font size (Settings -> UI Font Size)
    "max_tokens":      0,    # per-reply token cap; 0 = server default (Settings -> Max Tokens)
    "turn_time_limit": TURN_TIME_LIMIT_DEFAULT,  # seconds per user prompt; 0 = no limit
    "file_workspace":    "",    # optional folder confining read/write_local_file; blank = unrestricted
    "handoff_threshold_pct": 75,  # auto-handoff when context usage hits this % (0 = off)
    "handoff_rearm_pct": 10,      # how much further the gauge must grow before a chat may summarize again
    "handoff_notes_dir": "",      # folder for handoff notes; blank = <this folder>/deskpilot_data/handoff_notes
    "custom_system_prompt": "",   # user instructions appended to the system prompt each request; blank = built-in only
    "sampling_defaults": dict(SAMPLING_DEFAULTS),  # sampler params applied to EVERY chat (Settings -> Sampling)
    "show_sampling_note": True,   # render one "what was actually sent" note per user prompt
    "allow_local_network": False,  # fetch_url may reach LAN/localhost/cloud-metadata addresses
    "mcp_servers":       [],   # MCP stdio servers (Settings -> MCP Servers)
    "tts_enabled":       False,
    "tool_permissions":  dict(DEFAULT_PERMS),
}

IMAGE_EXTS   = {".png", ".jpg", ".jpeg"}
TEXT_EXTS    = {".txt", ".md", ".json", ".csv", ".py", ".js", ".html", ".xml", ".log", ".yml", ".yaml", ".ini", ".toml"}

# Signatures of anti-bot / JavaScript challenge pages (DataDome, Cloudflare, ...).
# fetch_url cannot execute JS, so such sites only ever serve their "verify you are
# human" wall; we flag those responses instead of passing the wall off as content.
BOT_WALL_SIGNATURES = (
    "verifying you are human",
    "javascript is required",
    "please enable javascript",
    "enable javascript in your browser",
    "just a moment",
    "checking your browser",
    "attention required",
)


def _bot_wall_note(text: str) -> Optional[str]:
    """Advisory note if text looks like an anti-bot/JS challenge page, else None."""
    low = (text or "")[:4000].lower()
    if any(sig in low for sig in BOT_WALL_SIGNATURES):
        return ("[Note] This looks like an anti-bot / JavaScript challenge page - the site "
                "requires a real browser to verify visitors, so Deskpilot cannot fetch the "
                "actual content. Open the URL in your web browser instead.")
    return None


# ── fetch_url safety: body cap + SSRF guard ──
FETCH_BODY_LIMIT = 2 * 1024 * 1024   # max bytes downloaded from a single page (read in chunks)


# Redirect hardening for fetch_url: every 3xx hop is re-validated against the SSRF
# guard and the total hop count is capped (urllib's default opener follows redirects
# blindly, which lets a public URL bounce a fetch into the LAN / cloud metadata).
FETCH_MAX_REDIRECTS = 5       # total 3xx hops allowed per fetch_url request


class _FetchRedirectBlocked(Exception):
    """Raised mid-redirect by _ValidatingRedirectHandler; carries a user-facing reason."""


def _ip_is_blocked(ip_str: str) -> bool:
    """True for addresses that must never be fetched by default: loopback, private LAN (RFC1918), link-local (incl. the 169.254.x cloud-metadata range) and other non-global ranges; unparseable input is blocked too."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return _ip_is_blocked(str(ip.ipv4_mapped))   # ::ffff:127.0.0.1 style
    return not ip.is_global


def _fetch_url_blocked(url: str, allow_local: bool = False) -> Optional[str]:
    """SSRF guard for model-directed fetches. Returns a human-readable reason when the URL points at this machine, the local network or cloud metadata endpoints (169.254.169.254 etc.), else None (allowed). Hostnames are resolved first and EVERY returned address is checked, so a public DNS name pointing inside the LAN is blocked as well; allow_local (Settings -> Allow Local Network) disables the check."""
    if allow_local:
        return None
    try:
        host = urllib.parse.urlparse(url).hostname or ""
    except Exception:
        return "could not parse the URL"
    if not host:
        return "no hostname in URL"
    # Literal IP address in the URL
    try:
        ipaddress.ip_address(host)
        if _ip_is_blocked(host):
            return f"{host} is a local/reserved address (loopback, LAN or cloud-metadata range)"
        return None
    except ValueError:
        pass
    # Hostname: resolve and check every returned address
    try:
        infos = socket.getaddrinfo(host, None)
    except Exception as e:
        return f"could not resolve hostname {host} ({e})"
    for info in infos:
        ip = info[4][0]
        if _ip_is_blocked(ip):
            return (f"{host} resolves to a local/reserved address ({ip}) - "
                    "fetching internal network addresses is blocked by default "
                    "(enable 'Allow Local Network' in Settings to override)")
    return None


class _ValidatingRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-runs the SSRF guard on every redirect target and caps total hops.

    Used only by _FETCH_OPENER (the fetch_url tool). A blocked hop raises
    _FetchRedirectBlocked, which unwinds out of opener.open(); _tool_fetch_url
    turns it into an error string. allow_local (Settings -> Allow Local Network)
    disables the per-hop guard but the hop cap still applies."""

    _allow_local = False   # set per request by _tool_fetch_url (default: guard ON)

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        hops = getattr(req, '_dp_hops', 0) + 1
        if hops > FETCH_MAX_REDIRECTS:
            raise _FetchRedirectBlocked(
                f"stopped after {FETCH_MAX_REDIRECTS} redirects (redirect loop?)")
        blocked = _fetch_url_blocked(newurl, self._allow_local)
        if blocked:
            raise _FetchRedirectBlocked(f"{blocked} [redirect #{hops}] -> {newurl}")
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is not None:
            new_req._dp_hops = hops
        return new_req


# Module-level opener shared by all fetch_url calls (named handler instance so the
# tool can flip _allow_local without relying on opener internals ordering).
# _allow_local is re-read per request in _tool_fetch_url, so a Settings change
# applies immediately without rebuilding the opener.
_FETCH_REDIRECT_HANDLER = _ValidatingRedirectHandler()
_FETCH_OPENER = urllib.request.build_opener(_FETCH_REDIRECT_HANDLER)


# ── Screenshot handling: lightweight persistence + request-time payloads ─────
# Screenshots are stored in chat history as 'image_ref' parts (just a file path)
# instead of inline base64, so deskpilot_chats.json stays small. The actual image
# payload is attached at request time by _prepare_request_messages(), which sends
# only the most recent screenshot - re-sending every past capture would bloat the
# context window on every turn.
SCREENSHOT_MARKER = "[System] The screenshot you just captured"
SCREENSHOT_MAX_DIM = 1280     # captures are downscaled to this (keeps text legible for vision models)
THUMB_MAX_DIM    = 400        # max width/height of thumbnails embedded in the chat view


def _downscale_image_bytes(data: bytes, max_dim: int = SCREENSHOT_MAX_DIM, quality: int = 75):
    """Resize + compress image bytes (best effort). Returns (bytes, mime).

    Images without alpha become JPEG; images with transparency stay PNG so they
    don't get a black background. Falls back to the original bytes with a sniffed
    mime when PIL is missing or the data isn't a decodable image."""
    if _PILImage is not None:
        try:
            img = _PILImage.open(io.BytesIO(data))
            img.load()
            w, h = img.size
            scale = min(1.0, max_dim / max(w, h))
            if scale < 1.0:
                img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), _PILImage.LANCZOS)
            has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
            buf = io.BytesIO()
            if has_alpha:
                if img.mode not in ("RGBA", "LA"):
                    img = img.convert("RGBA")
                img.save(buf, format="PNG")
                return buf.getvalue(), "image/png"
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            img.save(buf, format="JPEG", quality=quality)
            return buf.getvalue(), "image/jpeg"
        except Exception:
            pass
    mime = "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
    return data, mime


def _is_screenshot_msg(m: dict) -> bool:
    """True for the queued-screenshot user message (keyed on its system marker)."""
    c = m.get("content") if isinstance(m, dict) else None
    if not isinstance(c, list):
        return False
    for p in c:
        if isinstance(p, dict) and p.get("type") == "text" \
                and str(p.get("text", "")).startswith(SCREENSHOT_MARKER):
            return True
    return False


def _chat_image_paths(chat: Any) -> List[Path]:
    """Absolute paths of every image file referenced by ONE chat record.

    Screenshots (screen_*), user attachments (attach_*) and migrated legacy images
    (screen_migrated_* / attach_migrated_*) are stored as lightweight "image_ref" parts -
    just a path, the bytes live in GEN_DIR - so the chat record is the only source of truth
    about which files belong to which conversation. Corrupt history entries (non-dict
    messages or parts, blank paths) are skipped, never raised: chats.json is user-editable.
    Paths are resolve()d so the same file written with different spellings compares equal."""
    found: List[Path] = []
    seen = set()
    msgs = chat.get("messages") if isinstance(chat, dict) else None
    for m in (msgs or []):
        if not isinstance(m, dict):
            continue          # corrupt history entry - skip, never raise
        c = m.get("content")
        if not isinstance(c, list):
            continue
        for p in c:
            if not isinstance(p, dict) or p.get("type") != "image_ref":
                continue
            raw = str(p.get("path", "")).strip()
            if not raw:
                continue
            try:
                fp = Path(raw)
                key = str(fp.resolve())
            except Exception:
                continue
            if key not in seen:
                seen.add(key)
                found.append(fp)
    return found


def _image_store_dirs() -> List[Path]:
    """Resolved roots that legitimately hold chat image files: the active GEN_DIR plus the
    legacy next-to-script folder (records created before persistence moved to %LOCALAPPDATA%)."""
    roots: List[Path] = []
    for d in (GEN_DIR, BASE_DIR / "generated_images"):
        try:
            rp = Path(d).resolve()
            if rp not in roots:
                roots.append(rp)
        except Exception:
            pass
    return roots


def _path_in_image_store(fp: Path) -> bool:
    """True when fp resolves inside one of the image stores.

    delete_chat() only ever unlinks files the app itself wrote. chats.json is user-editable, so a
    hand-written image_ref pointing at an unrelated file (a document, a photo in Pictures, a UNC
    share) is SKIPPED instead of deleting user data."""
    try:
        r = fp.resolve()
        parents = [r] + list(r.parents)
    except Exception:
        return False
    for root in _image_store_dirs():
        if root in parents:
            return True
    return False


def _referenced_image_keys(chats: Dict[str, Any]) -> set:
    """resolve() strings of every image file ANY chat record still points at.

    Safety net for chat deletion: a file another conversation still references must
    survive even if the deleted chat referenced it too."""
    keys = set()
    if not isinstance(chats, dict):
        return keys
    inner = chats.get("chats")
    if not isinstance(inner, dict):
        return keys
    for chat in inner.values():
        for fp in _chat_image_paths(chat):
            try:
                keys.add(str(fp.resolve()))
            except Exception:
                pass
    return keys


def _expand_image_ref(p: dict) -> dict:
    """Resolve an image_ref part to a real base64 payload (or a text placeholder
    when the stored file is gone)."""
    path = Path(str(p.get("path", "")))
    try:
        data = path.read_bytes()
        mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
        return {"type": "image_url",
                "image_url": {"url": f"data:{mime};base64," + base64.b64encode(data).decode("ascii")}}
    except Exception:
        return {"type": "text", "text": f"[attached image no longer available: {path.name or 'unknown'}]"}


def _prepare_request_messages(messages: List[dict]) -> List[dict]:
    """Outgoing copy of a message list with screenshot payloads resolved.

    Only the most recent screenshot (the last one whose image is still readable)
    is expanded into a real base64 image_url payload; older ones become text
    placeholders. User-attachment images (image_ref parts in other messages, or
    legacy inline data-URLs) always expand - they are part of the message."""
    last_shot = -1
    for i, m in enumerate(messages):
        if not _is_screenshot_msg(m):
            continue
        has_payload = any(
            (isinstance(p, dict) and p.get("type") == "image_ref"
             and Path(str(p.get("path", ""))).is_file())
            or (isinstance(p, dict) and p.get("type") == "image_url"
                and str((p.get("image_url") or {}).get("url", "")).startswith("data:image/"))
            for p in (m.get("content") or []))
        if has_payload:
            last_shot = i

    out: List[dict] = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            continue          # corrupt history entry - skip, never raise
        c = m.get("content")
        is_shot = _is_screenshot_msg(m)
        has_ref = isinstance(c, list) and any(
            isinstance(p, dict) and p.get("type") == "image_ref" for p in (c or []))
        if not is_shot and not has_ref:
            out.append(m)
            continue
        parts: List[dict] = []
        for p in (c or []):
            if not isinstance(p, dict):
                continue
            ptype = p.get("type")
            if ptype == "image_ref":
                if is_shot and i != last_shot:
                    name = Path(str(p.get("path", ""))).name or "screenshot"
                    parts.append({"type": "text", "text": f"[screenshot analyzed earlier: {name}]"})
                else:
                    parts.append(_expand_image_ref(p))   # latest screenshot / user attachment
            elif ptype == "image_url":
                url = str((p.get("image_url") or {}).get("url", ""))
                if is_shot and i != last_shot and url.startswith("data:image/"):
                    parts.append({"type": "text", "text": "[screenshot analyzed earlier]"})
                else:
                    parts.append(p)          # legacy inline payload / user attachment - keep as-is
            else:
                parts.append(p)
        out.append({"role": m.get("role", "user"),
                    "content": parts or [{"type": "text", "text": "[screenshot analyzed earlier]"}]})
    return out


# ════════════════════════════════════════════════════════════════════════════
#  PERSISTENCE HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _atomic_write(path: Path, text: str) -> None:
    """Write a file atomically (temp file + rename) to avoid corruption.

    If the target exists, the immediately-previous version is kept as
    <name>.bak first - exactly one rolling backup, overwritten on every save."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    if path.exists():
        try:
            shutil.copy2(path, path.with_name(path.name + ".bak"))
        except OSError:
            pass   # backup is best-effort; the save itself must not fail because of it
    os.replace(tmp, path)


def resolve_data_dir() -> Path:
    r"""Pick a writable directory for settings/chats/images.

    Preference order: %LOCALAPPDATA%\Deskpilot (always user-writable and it
    survives the script being copied or moved), then next to this script as a
    fallback. The first candidate that accepts a test write wins.
    """
    candidates: List[Path] = []
    local = os.environ.get("LOCALAPPDATA")
    if local:
        try:
            candidates.append(Path(local) / APP_NAME)
        except Exception:
            pass
    candidates.append(BASE_DIR)
    for cand in candidates:
        try:
            cand.mkdir(parents=True, exist_ok=True)
            probe = cand / ".deskpilot_write_test"
            _atomic_write(probe, "ok")
            probe.unlink(missing_ok=True)
            return cand
        except Exception:
            continue
    return BASE_DIR   # nothing writable found; the startup warning will fire


def _migrate_legacy_files() -> None:
    r"""One-time copy of old data files to their current names and location.

    Handles both the Gemma Assistant -> Deskpilot rename (gemma_* -> deskpilot_*)
    and the move of the data store out of the script folder into
    %LOCALAPPDATA%\Deskpilot. Copies (not moves) so nothing is lost; safe to
    call on every start. SETTINGS_FILE/CHATS_FILE must already point at the
    live store (main() relocates them before calling this).
    """
    # 1) rename copies next to the script (kept for old copies of the app)
    for old_name, new_name in (("gemma_settings.json", "deskpilot_settings.json"),
                              ("gemma_chats.json", "deskpilot_chats.json")):
        old_p, new_p = BASE_DIR / old_name, BASE_DIR / new_name
        try:
            if not new_p.exists() and old_p.exists():
                shutil.copy2(old_p, new_p)
        except OSError:
            pass

    # 2) copy any data files found next to the script into the live store
    def _copy_to_store(src: Path, dst: Path) -> None:
        try:
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
        except OSError as e:
            print(f"[{APP_NAME}] Could not migrate {src.name}: {e}")

    _copy_to_store(BASE_DIR / "deskpilot_settings.json", SETTINGS_FILE)
    if not SETTINGS_FILE.exists():
        _copy_to_store(BASE_DIR / "gemma_settings.json", SETTINGS_FILE)
    _copy_to_store(BASE_DIR / "deskpilot_chats.json", CHATS_FILE)
    if not CHATS_FILE.exists():
        _copy_to_store(BASE_DIR / "gemma_chats.json", CHATS_FILE)


def _searxng_base_url(settings: Dict[str, Any]) -> str:
    """Resolve the SearXNG base URL.

    An explicit 'searxng_url' setting wins; a blank value falls back to the host
    of the configured LLM server (server_url) on port 8080 - so a LAN setup
    works without any extra configuration.
    """
    base = (settings.get("searxng_url") or "").strip()
    if not base:
        srv = (settings.get("server_url") or "http://localhost").strip()
        try:
            host = urllib.parse.urlparse(srv).hostname or "localhost"
        except Exception:
            host = "localhost"
        base = f"http://{host}:8080"
    return base.rstrip("/")


def load_settings() -> Dict[str, Any]:
    s: Dict[str, Any] = json.loads(json.dumps(DEFAULT_SETTINGS))   # deep copy
    if SETTINGS_FILE.exists():
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                s.update(data)
        except Exception:
            pass
    tp = s.get("tool_permissions") or {}
    for name in DEFAULT_PERMS:
        tp.setdefault(name, DEFAULT_PERMS[name])
    s["tool_permissions"] = tp
    # Legacy cleanup: early versions defaulted searxng_url to localhost; blank now
    # means "use the LLM server's address on port 8080" (see _searxng_base_url).
    if s.get("searxng_url") == "http://localhost:8080":
        s["searxng_url"] = ""
    # Self-heal: model_history must be a list of non-empty strings (newest first).
    mh = s.get("model_history")
    if not isinstance(mh, list):
        mh = []
    s["model_history"] = [str(x) for x in mh if str(x).strip()][:MODEL_HISTORY_MAX]
    return s


def save_settings(settings: Dict[str, Any]) -> bool:
    try:
        _atomic_write(SETTINGS_FILE, json.dumps(settings, indent=2, ensure_ascii=False))
        return True
    except Exception as e:
        print(f"[Deskpilot] Failed to save settings: {e}")
        return False


def fetch_model_info(server_url: str, api_key: str = "", timeout: int = 8) -> List[dict]:
    """GET {server_url}/models and return the FULL list of model entries (dicts).

    The model name is NOT part of this request (it only appears in chat-completion
    bodies), so this works even when the saved model_name is stale or wrong. Each
    entry carries at least 'id'; servers like Unsloth Desktop also send
    context_length / max_context_length / native_context_length / quant / loaded /
    display_name, which the top-bar context readout uses. Returns [] on any failure
    (server down, bad URL, non-JSON reply) instead of raising."""
    url = (server_url or "").strip()
    if not url:
        return []
    try:
        req = urllib.request.Request(
            url.rstrip("/") + "/models",
            headers={
                "User-Agent": "Deskpilot/1.0",
                "Authorization": f"Bearer {(api_key or '').strip()}"
            }
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
        return [m for m in (data.get("data") or []) if isinstance(m, dict)]
    except Exception:
        return []


def fetch_model_ids(server_url: str, api_key: str = "", timeout: int = 8) -> List[str]:
    """Convenience wrapper: just the model IDs from fetch_model_info()."""
    return [str(m.get("id")).strip() for m in fetch_model_info(server_url, api_key, timeout)
            if m.get("id")]


def loaded_model_ids(entries: List[dict]) -> List[str]:
    """IDs of the entries a server reports as currently loaded (loaded=True).

    Servers that do not send a "loaded" field return [] - callers must then treat
    the loaded model as unknown rather than guessing."""
    out: List[str] = []
    for m in entries or []:
        if isinstance(m, dict) and m.get("loaded") and m.get("id"):
            mid = str(m["id"]).strip()
            if mid and mid not in out:
                out.append(mid)
    return out


def normalize_model_name(want: str, entries: List[dict]) -> tuple:
    r"""Resolve a typed/picked model name to the exact server ID that must be sent.

    Servers like Unsloth Studio expose both a namespaced "id" (the only string its
    API accepts) and a shorter "display_name". Typing the display name - or an id
    missing its namespace prefix - otherwise produces a 404 model_not_found at chat
    time, so every candidate is resolved against the live /models entries first.

    Returns (canonical_id, status):
      "exact"     -> already a valid server id (unchanged)
      "resolved"  -> matched exactly one entry by display_name / suffix / case
      "ambiguous" -> several entries match; keep the text as typed
      "unknown"   -> no entry matches; keep the text as typed (server decides)
    """
    want = str(want or "").strip()
    ids = [str(m.get("id") or "").strip() for m in entries or [] if isinstance(m, dict)]
    ids = [i for i in ids if i]
    if not want:
        return want, "unknown"
    if want in ids:
        return want, "exact"
    low = want.lower()
    ci = [i for i in ids if i.lower() == low]
    if len(ci) == 1:
        return ci[0], "resolved"
    if len(ci) > 1:
        return want, "ambiguous"
    cands: List[str] = []
    for m in entries or []:
        if not isinstance(m, dict):
            continue
        mid = str(m.get("id") or "").strip()
        if not mid:
            continue
        dname = str(m.get("display_name") or "").strip()
        dl = dname.lower()
        ml = mid.lower()
        if (dl and (dl == low or low == ml.rsplit("/", 1)[-1])) or ml.endswith("/" + low):
            if mid not in cands:
                cands.append(mid)
    if len(cands) == 1:
        return cands[0], "resolved"
    if len(cands) > 1:
        return want, "ambiguous"
    return want, "unknown"


def _fmt_ctx(n: Any) -> str:
    """Format a context length for the top bar: 262144 -> '256K', 4096 -> '4K'.
    Powers of two use binary K (n/1024); other round numbers use decimal K; anything
    else is shown as-is. Returns '' when not a positive integer."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return ""
    if n <= 0:
        return ""
    if n % 1024 == 0:
        return f"{n // 1024}K"
    if n % 1000 == 0:
        return f"{n // 1000}K"
    return str(n)


def load_chats() -> Dict[str, Any]:
    data = {"order": [], "chats": {}}
    if CHATS_FILE.exists():
        try:
            raw = json.loads(CHATS_FILE.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and isinstance(raw.get("chats"), dict):
                chats = raw["chats"]
                order = [c for c in (raw.get("order") or []) if c in chats]
                for cid in chats:                        # keep any unlisted chats too
                    if cid not in order:
                        order.append(cid)
                data = {"order": order, "chats": chats}
        except Exception:
            pass
    try:
        _migrate_inline_images(data)    # one-time shrink of legacy base64 images (no-op after run)
    except Exception as e:
        # A corrupt history must never block startup: keep the data as loaded.
        print(f"[{APP_NAME}] Inline-image migration skipped: {e}")
    return data


def save_chats(chats: Dict[str, Any]) -> bool:
    try:
        _atomic_write(CHATS_FILE, json.dumps(chats, indent=1, ensure_ascii=False))
        return True
    except Exception as e:
        print(f"[Deskpilot] Failed to save chats: {e}")
        return False


def _migrate_inline_images(chats: Dict[str, Any]) -> None:
    """One-time shrink of legacy chat history: replace inline base64 image
    payloads - auto-captured screenshots AND user attachments - with lightweight
    image_ref parts. The downscaled image is copied under GEN_DIR so the history
    survives even if the original file is deleted or moved. Pre-image_ref versions
    stored up to ~4 MB of base64 per image directly in deskpilot_chats.json,
    re-serialized on every save. Idempotent - once no data-URL parts remain it is
    a no-op."""
    changed = False
    for chat in (chats.get("chats") or {}).values():
        if not isinstance(chat, dict):
            continue
        for m in chat.get("messages") or []:
            if not isinstance(m, dict):
                continue
            c = m.get("content")
            if not isinstance(c, list):
                continue
            new_parts: List[dict] = []
            replaced = False
            for p in c:
                if isinstance(p, dict) and p.get("type") == "image_url":
                    url = str((p.get("image_url") or {}).get("url", ""))
                    if url.startswith("data:image/"):
                        try:
                            data, mime = _downscale_image_bytes(base64.b64decode(url.split(",", 1)[1]))
                            ext = ".jpg" if mime == "image/jpeg" else ".png"
                            prefix = "screen_migrated_" if _is_screenshot_msg(m) else "attach_migrated_"
                            fp = GEN_DIR / f"{prefix}{uuid.uuid4().hex[:8]}{ext}"
                            fp.write_bytes(data)
                            new_parts.append({"type": "image_ref", "path": str(fp)})
                            replaced = True
                            changed = True
                            continue
                        except Exception:
                            pass
                new_parts.append(p)
            if replaced:
                m["content"] = new_parts
    if changed:
        save_chats(chats)


# ════════════════════════════════════════════════════════════════════════════
#  MARKDOWN ENGINE
# ════════════════════════════════════════════════════════════════════════════

# Inline tokens: `code` | **bold** | *italic* / _italic_
INLINE_RE = re.compile(
    r"(`+)([^`]+?)\1"                                     # 1,2 → inline code
    r"|(\*\*)(.+?)\3"                                     # 3,4 → bold
    r"|(?<![\w*])([*_])(?!\s)([^\*_\n]+?)(?<!\s)\5(?![\w*])"   # 5,6 → italic
)

# Bare http(s) URLs in running text -> clickable "link" tag (Ctrl/middle-click).
# Greedy match; trailing sentence punctuation is stripped in _apply_urls().
URL_RE = re.compile(r"\bhttps?://[^\s<>\"'()\[\]{}]+")


def _apply_urls(widget: tk.Text, seg: str, tags: List[str], app) -> None:
    """Insert `seg` at the widget end, tagging any bare http(s) URL as a
    clickable link (registry: app._links maps (line, char offset) -> URL).

    The start position is captured via 'end-1c' BEFORE inserting anything -
    Tk's index('end') reports one line past the last real content (implicit
    trailing newline), so querying it mid-way would mis-key the registry.
    Offsets then advance by exactly the characters inserted."""
    try:
        ln0, off0 = (int(x) for x in widget.index("end-1c").split(".")[:2])
    except (tk.TclError, ValueError):
        ln0, off0 = 1, 0
    pos = 0
    for m in URL_RE.finditer(seg):
        if m.start() > pos:
            widget.insert("end", seg[pos:m.start()], tags)
            off0 += m.start() - pos
        url = m.group(0).rstrip(r".,;:!?)\]]")
        if "://" in url and len(url.split("://", 1)[1]) > 0:
            if app is not None:
                try:
                    app._links[(ln0, off0)] = url
                except Exception:
                    pass
            widget.insert("end", url, list(tags) + ["link"])
            off0 += len(url)
        else:
            widget.insert("end", m.group(0), tags)            # degenerate (scheme only) - plain text
            off0 += len(m.group(0))
        pos = m.end()
    if pos < len(seg):
        widget.insert("end", seg[pos:], tags)


def render_inline(widget: tk.Text, text: str, tags: List[str]) -> None:
    """Insert `text` at the end of a Text widget, applying bold/italic/code.

    `tags` is the list of base tag names applied to every segment (e.g. ["body"]).
    Code spans are parsed first so markers inside them are never re-interpreted.
    """
    # Drop a stray unpaired ** (odd count) so raw asterisks never leak into the
    # rendered text when a model drops or splits a bold marker mid-line. Skipped
    # for lines containing code spans, where literal ** may be legitimate.
    if "`" not in text and text.count("**") % 2 == 1:
        i = text.rfind("**")
        text = text[:i] + text[i + 2:]

    app = getattr(widget, "app", None)      # set on chat_text in _build_chat_area
    pos = 0
    for m in INLINE_RE.finditer(text):
        if m.start() > pos:
            _apply_urls(widget, text[pos:m.start()], tags, app)
        if m.group(1):                                        # inline code (never linkified)
            widget.insert("end", m.group(2), list(tags) + ["inline_code"])
        elif m.group(3):                                      # bold
            _apply_urls(widget, m.group(4), list(tags) + ["bold"], app)
        else:                                                 # italic
            _apply_urls(widget, m.group(6), list(tags) + ["italic"], app)
        pos = m.end()
    if pos < len(text):
        _apply_urls(widget, text[pos:], tags, app)


class MarkdownStream:
    """Incremental Markdown renderer for a tk.Text widget.

    Designed for token-by-token streaming: only *complete* lines are rendered,
    so partial tokens never produce broken formatting. Supports #/##/### headers,
    fenced code blocks (with an interactive "📋 Copy Code" button), lists,
    blockquotes and inline **bold** / *italic* / `code`.
    """

    def __init__(self, app: "DeskpilotApp"):
        self.app = app
        # Tag for plain paragraphs / list text. Assistant blocks set this to
        # the bold "asst_body" tag; default stays regular-weight "body".
        self.body_tag = "body"
        self.in_code = False
        self.code_lang = ""
        self.code_lines: List[str] = []
        self.pending = ""

    # ── public API ────────────────────────────────────────────────────────
    def feed(self, text: str) -> None:
        if not text:
            return
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            self._render_line(line.rstrip("\r"))

    def finish(self) -> None:
        """Flush the final partial line and close any open code block."""
        if self.pending:
            self._render_line(self.pending)
            self.pending = ""
        if self.in_code:
            self._close_code_block()

    # ── internals ─────────────────────────────────────────────────────────
    def _insert(self, text: str, tags: Optional[List[str]] = None) -> None:
        self.app.chat_text.insert("end", text, tags or [])

    def _render_line(self, line: str) -> None:
        stripped = line.strip()

        # ── inside a fenced code block ────────────────────────────────────
        if self.in_code:
            if stripped.startswith("```"):
                self._close_code_block()
                return
            self.code_lines.append(line)
            self._insert(line + "\n", ["code_block"])
            return

        # ── fence opener ──────────────────────────────────────────────────
        if stripped.startswith("```"):
            self.in_code = True
            self.code_lang = stripped[3:].strip()
            self.code_lines = []
            self._insert("\n")                        # spacer above the block
            return

        # ── headers: # / ## / ### (####+ treated as h3) ───────────────────
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            level = len(m.group(1))
            tag = "h1" if level == 1 else ("h2" if level == 2 else "h3")
            self._insert("\n")
            render_inline(self.app.chat_text, m.group(2).strip(), [tag])
            self._insert("\n")
            return

        # ── blank line ────────────────────────────────────────────────────
        if not stripped:
            self._insert("\n")
            return

        # ── table rows (| ... |): monospace; separator rows -> dim rule ──
        if stripped.startswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                self._insert("─" * 60, ["rule"])              # |---|---| separator
            else:
                render_inline(self.app.chat_text, stripped, ["table_row"])
            self._insert("\n")
            return

        # ── list items (-, *, +, 1.) ──────────────────────────────────────
        lm = re.match(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$", line)
        if lm:
            indent, marker, rest = lm.groups()
            bullet = "•  " if marker in "-*+" else f"{marker} "
            self._insert(indent + bullet)
            render_inline(self.app.chat_text, rest, [self.body_tag])
            self._insert("\n")
            return

        # ── blockquote ────────────────────────────────────────────────────
        if stripped.startswith(">"):
            render_inline(self.app.chat_text, stripped.lstrip("> "), ["quote"])
            self._insert("\n")
            return

        # ── horizontal rule (--- / *** / ___) ─────────────────────
        if re.fullmatch(r"(-{3,}|\*{3,}|_{3,})", stripped):
            self._insert("─" * 60, ["rule"])
            self._insert("\n")
            return

        # ── plain paragraph ───────────────────────────────────────────────
        render_inline(self.app.chat_text, line, [self.body_tag])
        self._insert("\n")

    def _close_code_block(self) -> None:
        code = "\n".join(self.code_lines)
        self.in_code = False
        self.code_lang = ""
        try:
            btn = tk.Button(
                self.app.chat_text,
                text="📋 Copy Code",
                command=lambda c=code: self.app.copy_to_clipboard(c),
                bg=COL["bg_raised"], fg=COL["text"],
                activebackground=COL["accent"], activeforeground="#FFFFFF",
                relief="flat", bd=0, padx=10, pady=3,
                cursor="hand2", font=F(11),
            )
            self.app._code_buttons.append(btn)          # prevent GC
            self._insert("\n")                          # spacer below the block
            idx = self.app.chat_text.index("end-1c")
            self.app.chat_text.window_create(idx, window=btn)
            self.app.chat_text.insert("end", "\n")      # terminate button line
        except Exception:                               # widget destroyed / no display
            pass


def args_preview(args: Dict[str, Any]) -> str:
    """Compact one-line preview of tool arguments for accordion headers."""
    parts = []
    for k, v in list(args.items())[:4]:
        s = str(v).replace("\n", " ")
        if len(s) > 60:
            s = s[:60] + "…"
        parts.append(f"{k}={s}")
    return ", ".join(parts) or "no args"


def chat_temperature(chat) -> float:
    """Per-chat temperature to send with the API request.

    Always returns a concrete number. A blank / legacy "Default" / unparseable
    value falls back to TEMP_DEFAULT (the standard 0.7), so the UI can always
    show exactly what will be sent."""
    tv = str((chat or {}).get("temperature", "") or "").strip()
    if not tv or tv == "Default":
        return float(TEMP_DEFAULT)
    try:
        return float(tv)
    except ValueError:
        return float(TEMP_DEFAULT)


def chat_thinking_extra_body(chat) -> Optional[dict]:
    """Per-chat thinking level -> the extra_body dict to send with the request.

    Off          -> {"enable_thinking": False} (no thinking at all)
    Low/Medium/High -> {"chat_template_kwargs": {"reasoning_effort": low|medium|xhigh}}
                       (Qwen3-style server extension; High maps to the template's
                       xhigh, its maximum - 'Extra High' would be identical)
    Blank / legacy / junk -> None: send nothing, server default applies, so old
    chats keep their existing behavior unchanged."""
    val = str((chat or {}).get("thinking", "") or "").strip().lower()
    if val == "off":
        return {"enable_thinking": False}
    effort = {"low": "low", "medium": "medium", "high": "xhigh"}.get(val)
    if effort is not None:
        return {"chat_template_kwargs": {"reasoning_effort": effort}}
    return None



def _coerce_sampling(raw, lo, hi, is_int):
    """Coerce one sampler field. Blank/garbage -> None (key omitted). Clamped to [lo, hi]."""
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    if v != v:                      # NaN guard: NaN passes both clamp comparisons
        return None
    v = max(lo, min(hi, v))
    return int(round(v)) if is_int else round(float(v), 6)


def sampling_kwargs(settings) -> dict:
    """Sampler parameters from Settings, split the way the API needs them.

    Returns {"top": {...}, "flat": {...}}: 'top' are real OpenAI chat.completions fields
    (merged straight into kwargs); 'flat' are llama.cpp/Unsloth/vLLM extensions that must
    ride in extra_body. A key is OMITTED when blank/unparseable, when it sits at its
    off-value, or when the server has rejected it. Values are clamped to their range and
    sent verbatim otherwise - no snapping, no rounding beyond float cleanup.

    Read on the worker thread at request-build time; never mutates settings."""
    top: Dict[str, Any] = {}
    flat: Dict[str, Any] = {}
    src = (settings or {}).get("sampling_defaults")
    if not isinstance(src, dict):
        src = {}
    for key, _label, lo, hi, is_int, target, off in SAMPLING_SCHEMA:
        v = _coerce_sampling(src.get(key, ""), lo, hi, is_int)
        if v is None:
            continue
        if off is not None and float(v) == float(off):
            continue
        (top if target == "top" else flat)[key] = v
    return {"top": top, "flat": flat}


def _sampling_note_enabled(settings) -> bool:
    """Whether the per-prompt "Sent: ..." note is rendered (Settings checkbox; default ON)."""
    v = (settings or {}).get("show_sampling_note", True)
    if isinstance(v, str):
        return v.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(v)

def format_sampling_note(chat, top: dict, flat: dict, pruned: dict, thinking: Optional[dict]) -> str:
    """One-line record of the sampling parameters that actually went on the wire.

    Ground truth of the REQUEST, not of the settings dialog: keys omitted because they are
    off show as 'off', and anything the server rejected is labelled 'rejected' so a silently
    ignored parameter cannot masquerade as an applied one."""
    parts = ["temp " + str(chat_temperature(chat))]
    if thinking is None:
        parts.append("thinking default")
    elif "enable_thinking" in thinking:
        parts.append("thinking off")
    else:
        eff = ((thinking.get("chat_template_kwargs") or {}).get("reasoning_effort"))
        parts.append("thinking " + str(eff))
    for key, _label, _lo, _hi, _i, _t, _o in SAMPLING_SCHEMA:
        if key in pruned:
            parts.append(key + " REJECTED")
        elif key in top or key in flat:
            parts.append(f"{key} {top.get(key, flat.get(key))}")
        else:
            parts.append(key + " off")
    return "\u2699 Sent: " + "  \u00b7  ".join(parts)

# --- Tier 3 (v1.1.10): "Verify parameters" behavioural probe (Settings -> Sampling) ----------
# v1.1.9 shipped the REQUEST-side ground truth ("Sent: ..."). This answers what a log cannot:
# does the server ACCEPT each configured key, and does its OUTPUT look like the settings are
# live? Everything here is EVIDENCE, never proof - a server can accept a key and ignore it, and
# identical output at temperature 0 is consistent with (not proof of) honouring that setting.
# Requests stay deliberately tiny: capped by VERIFY_MAX_TOKENS, ONE user message each (no chat
# history, no system prompt), so a probe run cannot inflate the context window.
VERIFY_MAX_TOKENS         = 64    # hard cap on every probe request
VERIFY_TIMEOUT            = 30    # seconds per probe request (max_retries=0: no hidden retries)
VERIFY_DETERMINISM_TRIES  = 2     # identical requests at temperature 0, compared
VERIFY_REPETITION_COPIES  = 6     # "repeat this line N times" exact-copy task
VERIFY_REPETITION_LINE    = "the quick brown fox"
VERIFY_ACCEPT_PROMPT      = "Reply with exactly this word and nothing else: ok"
VERIFY_DETERMINISM_PROMPT = "Name a number from one to nine. Answer with that single digit only."


def verify_repetition_prompt(copies: int = VERIFY_REPETITION_COPIES,
                             line: str = VERIFY_REPETITION_LINE) -> str:
    return ("Repeat the following line exactly " + str(int(copies)) + " times, one line per copy, "
            "with no numbering and no other text:\n" + line)


def verify_repeat_count(text: str, line: str = VERIFY_REPETITION_LINE,
                        copies: int = VERIFY_REPETITION_COPIES) -> int:
    """How many reply lines are EXACTLY the requested line (after strip).

    Exact copies only: a server applying repetition_penalty paraphrases some of them, which is
    the expected behaviour of a live penalty - not a failure."""
    n = 0
    for raw in str(text or "").splitlines():
        if raw.strip() == line:
            n += 1
    return n


def _probe_error_keys(err_text: str, keys) -> list:
    """Which of the given keys the server actually NAMED in its error message.

    Per-key attribution, same ladder as the turn path (v1.1.9): one rejected field must never be
    reported as "all samplers rejected"."""
    low = str(err_text or "").lower()
    return [k for k in keys if k and k.lower() in low]


def _probe_first_line(err: str) -> str:
    txt = " ".join(str(err or "").split())
    return (txt[:180] + "...") if len(txt) > 180 else txt


def _probe_reply_text(resp) -> tuple:
    """(text, usage) from a completion response - tolerant of both non-stream and stream shapes."""
    text = ""
    try:
        choices = list(getattr(resp, "choices", None) or [])
        if choices:
            ch = choices[0]
            msg = getattr(ch, "message", None)
            if msg is not None:
                text = str(getattr(msg, "content", "") or "")
            else:
                d = getattr(ch, "delta", None)
                if d is not None:
                    text = str(getattr(d, "content", "") or "")
    except Exception:
        text = ""
    return text, getattr(resp, "usage", None)


def _probe_completions(client, timeout):
    """The completions endpoint for ONE probe request, with a hard per-request timeout.

    The openai client has no per-call timeout argument; with_options() is how the SDK expresses
    one (client.timeout bounds EVERY request). Older/mocked clients without it fall back silently -
    those are tests and stubs, where the real HTTP timeout is not what is under test."""
    comp = client.chat.completions
    if not timeout:
        return comp
    try:
        return comp.with_options(timeout=int(timeout), max_retries=0)
    except Exception:
        return comp


def _probe_create(client, kw: dict, timeout=None) -> tuple:
    """Send ONE probe request; returns (text, usage, error_string) and never raises.

    A max_tokens/max_completion_tokens refusal is a server limitation, NOT a sampler verdict, so
    the request is retried once without the token cap (same rule as the handoff worker)."""
    kw = dict(kw)
    comp = _probe_completions(client, timeout)
    try:
        resp = comp.create(**kw)
    except Exception as e:
        low = str(e).lower()
        if "max_tokens" in low or "max_completion_tokens" in low:
            kw.pop("max_tokens", None)
            kw.pop("max_completion_tokens", None)
            try:
                resp = comp.create(**kw)
            except Exception as e2:
                return "", None, str(e2)
        else:
            return "", None, str(e)
    _txt, _usage = _probe_reply_text(resp)
    return _txt, _usage, None


def _probe_accept(client, model: str, top: dict, flat: dict, max_tokens: int,
              timeout=None) -> tuple:
    """Probe 1 - does the server ACCEPT the configured keys? Returns (lines, usage)."""
    configured = list(top.keys()) + list(flat.keys())
    if not configured:
        return ["ACCEPT: no sampler parameters are configured, so there is nothing to send - "
                "fill at least one Sampling field first."], None
    kw = dict(model=model, messages=[{"role": "user", "content": VERIFY_ACCEPT_PROMPT}],
             stream=False, temperature=0.0, max_tokens=int(max_tokens))
    kw.update(top)
    eb = dict(flat)
    if eb:
        kw["extra_body"] = eb
    text, usage, err = _probe_create(client, kw, timeout)
    if err is None:
        return ["ACCEPT: all configured keys accepted (" + ", ".join(sorted(configured)) + ")."], usage
    low = str(err).lower()
    named = set(_probe_error_keys(err, configured))
    if "extra_body" in low and flat:
        named |= set(flat.keys())            # whole extension block refused -> every key in it
        lines = ["ACCEPT: this server refuses the extension block (extra_body) - extensions "
                 "unsupported: " + ", ".join(sorted(flat.keys()))]
    elif named:
        lines = ["ACCEPT: rejected " + ", ".join(sorted(named)) + " -> " + _probe_first_line(err)]
    else:
        lines = ["ACCEPT: request failed for a reason unrelated to sampler keys -> " +
                 _probe_first_line(err)]
    # Second, bounded attempt WITHOUT what was refused, so the surviving (OpenAI-schema) keys get
    # their own verdict instead of being lumped in with the rejected ones.
    rest_top = {k: v for k, v in top.items() if k not in named}
    kw2 = dict(model=model, messages=[{"role": "user", "content": VERIFY_ACCEPT_PROMPT}],
              stream=False, temperature=0.0, max_tokens=int(max_tokens))
    kw2.update(rest_top)
    _t2, u2, err2 = _probe_create(client, kw2, timeout)
    if usage is None:
        usage = u2
    if err2 is None:
        lines.append("ACCEPT: accepted " + (", ".join(sorted(rest_top)) if rest_top
                                            else "(nothing left to test)"))
    else:
        named2 = set(_probe_error_keys(err2, list(rest_top.keys())))
        if named2:
            lines.append("ACCEPT: rejected " + ", ".join(sorted(named2)) + " -> " +
                         _probe_first_line(err2))
        elif rest_top:
            lines.append("ACCEPT: no verdict for " + ", ".join(sorted(rest_top)) +
                         " - request failed for another reason -> " + _probe_first_line(err2))
    return lines, usage


def _probe_determinism(client, model: str, max_tokens: int, tries: int, timeout=None) -> list:
    """Probe 2 - identical requests at temperature 0; the spread of the replies is the evidence."""
    kw = dict(model=model, messages=[{"role": "user", "content": VERIFY_DETERMINISM_PROMPT}],
             stream=False, temperature=0.0, max_tokens=int(max_tokens))
    outs, errs = [], []
    for _ in range(max(1, int(tries))):
        t, _u, err = _probe_create(client, kw, timeout)
        if err:
            errs.append(_probe_first_line(err))
        outs.append(t.strip())
    got = [o for o in outs if o]
    if not got:
        return ["DETERMINISM: no usable reply (" + (errs[0] if errs else "empty responses") + ")."]
    distinct = len(set(got))
    tail = "" if not errs else "  (" + str(len(errs)) + " attempt(s) errored)"
    if distinct == 1 and len(got) >= 2:
        return ["DETERMINISM: " + str(len(got)) + "/" + str(len(outs)) + " identical replies at "
                "temperature 0 - consistent with the server honouring temperature. NOT proof it "
                "applies the other keys." + tail]
    if distinct > 1:
        return ["DETERMINISM: " + str(distinct) + " DISTINCT replies from " + str(len(got)) +
                " IDENTICAL requests at temperature 0 -> sampling is being applied (or this server "
                "is non-deterministic). If you set temperature 0, that gap is the evidence." + tail]
    return ["DETERMINISM: only one reply survived - retry with a loaded model / check the server."]


def _probe_repetition(client, model: str, flat: dict, max_tokens: int, copies: int,
                     timeout=None) -> list:
    """Probe 3 - repetition_penalty behavioural check. Informational by design, not pass/fail."""
    rp = flat.get("repetition_penalty")
    if rp is None:
        return ["REPETITION: repetition_penalty is not configured (blank, or 1.0 = off), "
                "so there is nothing to observe."]
    kw = dict(model=model, messages=[{"role": "user",
                                      "content": verify_repetition_prompt(copies)}],
              stream=False, temperature=0.0, max_tokens=int(max_tokens),
              extra_body={"repetition_penalty": rp})
    text, _u, err = _probe_create(client, kw, timeout)
    if err:
        return ["REPETITION: request failed -> " + _probe_first_line(err)]
    n = verify_repeat_count(text, copies=copies)
    head = ("REPETITION (informational): asked for " + str(int(copies)) + " exact copies at "
            "repetition_penalty=" + str(rp) + " -> " + str(n) + " exact.")
    if n >= int(copies):
        return [head + " Verbatim copying survived the penalty: consistent with a weak/off penalty."]
    return [head + " Deviation is the EXPECTED effect of a live penalty - evidence the parameter "
            "reached the sampler, not proof of its internal value."]


def _probe_context(models_info, model: str) -> list:
    """Probe 4 - what the server reports about itself (context window)."""
    info = [m for m in (models_info or []) if isinstance(m, dict)]
    entry = None
    for m in info:
        if str(m.get("id", "")).strip() == str(model).strip():
            entry = m
            break
    if entry is None:
        loaded = [m for m in info if m.get("loaded")]
        if len(loaded) == 1:
            entry = loaded[0]
    try:
        ctx = int((entry or {}).get("context_length") or 0)
    except (TypeError, ValueError):
        ctx = 0
    if ctx > 0:
        return ["SERVER: context_length " + format(ctx, ",") + " (" +
                str((entry or {}).get("id") or model) + ") - the context gauge and the handoff "
                "threshold work on this server."]
    if not info:
        return ["SERVER: no /models reply - model name and context length are unverifiable."]
    return ["SERVER: reports NO context_length for the selected model (LM Studio/Ollama style) - "
            "the gauge and handoff threshold cannot work; probes 2-3 stay meaningful. Listed: " +
            ", ".join(str(m.get("id")) for m in info[:5])]


def _probe_usage_line(usage) -> str:
    if usage is None:
        return ("USAGE: the server echoed no usage object - token counts (and the context gauge) "
                "stay blank on this build.")
    parts = []
    for label, attr in (("prompt", "prompt_tokens"), ("completion", "completion_tokens")):
        v = getattr(usage, attr, None)
        parts.append(label + "=" + (str(v) if v is not None else "?"))
    return "USAGE: echoed " + " ".join(parts)


def run_sampling_verify(client, model: str, top: dict, flat: dict, models_info=None, *,
                        max_tokens: int = VERIFY_MAX_TOKENS,
                        tries: int = VERIFY_DETERMINISM_TRIES,
                        copies: int = VERIFY_REPETITION_COPIES, timeout: int = VERIFY_TIMEOUT,
                        report=None) -> list:
    """The bounded probe sequence. Returns report LINES; never raises (each probe reports its own
    failure). Runs on a worker thread and touches NO widget, NO chat record and NO file."""
    lines: List[str] = []
    mt = max(1, min(int(max_tokens), VERIFY_MAX_TOKENS))      # every probe request stays tiny

    def emit(group) -> None:
        """Publish one finished probe (and keep it in the returned report).

        `report` is how the UI shows progress on a slow server - the caller passes a function
        that marshals to the main thread. Report callbacks must never break the probe."""
        lines.extend(group)
        if report is not None:
            try:
                report(list(group))
            except Exception:
                pass

    try:
        _a, usage = _probe_accept(client, model, top, flat, mt, timeout)
        emit(_a)
        emit([_probe_usage_line(usage)])
        emit(_probe_determinism(client, model, mt, tries, timeout))
        emit(_probe_repetition(client, model, flat, mt, copies, timeout))
        emit(_probe_context(models_info, model))
    except Exception as e:                    # a probe must never escape its own report line
        lines.append("PROBE ABORTED: " + _probe_first_line(e))
    return lines



def build_system_message(file_workspace: str = "", exa_key: str = "", firecrawl_key: str = "",
                         custom_prompt: str = "") -> dict:
    """System prompt injected at request time (never persisted to chat history).

    The model has no other way of knowing the real current date/time; without
    this, questions like "what time is it?" get hallucinated answers. Built
    fresh for every API call so long sessions don't go stale.

    custom_prompt (Settings -> Custom System Prompt) is appended AFTER all the
    built-in context so persona/style instructions layer on top of - and can
    never replace - the operational facts above.
    """
    now = datetime.now().astimezone()
    off = now.utcoffset()
    off_s = f"UTC{off.total_seconds() // 3600:+.0f}" if off is not None else "unknown offset"
    content = (
        f"You are Deskpilot, a helpful local AI desktop assistant running on the user's computer. "
        f"Current date and time on the user's machine: "
        f"{now.strftime('%A, %B %d, %Y at %I:%M %p')} "
        f"(timezone {now.strftime('%Z') or 'local'}, {off_s}). "
        f"If the user asks about the current date or time, use this information; "
        f"for other time zones, convert from it and mention the conversion. "
        f"Keep answers concise."
    )
    if file_workspace:
        content += (f" The user has restricted the local file tools to this workspace folder: {file_workspace}. "
                    f"All read_local_file and write_local_file paths MUST be inside it; anything outside is rejected. "
                    f"Use absolute paths within that folder.")
    if exa_key:
        content += (" Web search tools: when the most up-to-date information is needed (current events, "
                    "recent news, latest developments), prefer exa_search (Exa neural/semantic search); "
                    "use searxng_search (local SearXNG) for quick lookups where freshness does not matter.")
    if firecrawl_key:
        content += (" firecrawl_scrape is a powerful scraper (handles JavaScript-rendered pages and anti-bot "
                    "walls) but it is METERED - each scrape consumes one of the user's limited monthly credits. "
                    "Use it ONLY when fetch_url fails, returns an anti-bot/JS-challenge wall, or the page is known "
                    "to be JavaScript-rendered; never for simple static pages.")
    custom = (custom_prompt or "").strip()
    if custom:
        content += "\n\nUser instructions (apply to every reply):\n" + custom
    return {"role": "system", "content": content}


def _turn_time_limit(settings: dict) -> int:
    """Wall-clock cap in seconds for ONE user prompt; 0 (or junk) means NO LIMIT.

    Read from the settings file at turn start, so it is a user choice rather than a
    constant baked into the source - and the shipped default is unlimited. It exists as
    an opt-in backstop: with MAX_TOOL_STEPS == 0 a model that keeps calling tools would
    otherwise run forever, but cutting a legitimate long turn off was never wanted."""
    try:
        v = int(float(settings.get("turn_time_limit", TURN_TIME_LIMIT_DEFAULT)))
    except (TypeError, ValueError):
        v = TURN_TIME_LIMIT_DEFAULT
    return max(0, min(TURN_TIME_LIMIT_MAX, v))


# ── Session handoff: auto-summary when context usage crosses a threshold ────────
HANDOFF_THRESHOLD_DEFAULT = 75   # % of the context window; 0 disables auto-handoff

# Handoff NOTES live in a working folder, NOT in the AppData data store (settings/chats/images stay
# where resolve_data_dir() put them - only notes moved here). Workspace convention: one "<app>_data"
# folder per app. ONE FILE PER HANDOFF - a new handoff never overwrites an earlier one.
# Resolution order for a BLANK setting: USER_WORKSPACE_ROOT/deskpilot_data/handoff_notes when that
# workspace exists on this machine (notes land where the assistant can read them), otherwise
# <script folder>/deskpilot_data/handoff_notes - so a copy of this script on any other machine still
# keeps notes in a working folder next to the app. Settings can point it anywhere absolute.
USER_WORKSPACE_ROOT = Path(r"C:\Users\Shuhdonk\Downloads\DeskPilot_Working_Directory")
DEFAULT_HANDOFF_NOTES_SUBDIR = Path("deskpilot_data") / "handoff_notes"
HANDOFF_PREV_NOTES_LISTED = 5        # how many earlier notes the header of a new note lists
HANDOFF_TITLE_SLUG_MAX = 40          # chat title chars kept in the note filename


def default_handoff_notes_dir() -> Path:
    """The notes folder used when Settings -> Handoff notes folder is blank (never AppData)."""
    try:
        if USER_WORKSPACE_ROOT.is_dir():
            return USER_WORKSPACE_ROOT / DEFAULT_HANDOFF_NOTES_SUBDIR
    except Exception:
        pass
    return BASE_DIR / DEFAULT_HANDOFF_NOTES_SUBDIR


def _handoff_threshold_pct(settings: dict) -> int:
    """The configured auto-handoff threshold as an integer percent (0-100).
    0 (or junk) disables the feature. Old settings files self-heal to the default."""
    try:
        v = int(float(settings.get("handoff_threshold_pct", HANDOFF_THRESHOLD_DEFAULT)))
    except (TypeError, ValueError):
        v = HANDOFF_THRESHOLD_DEFAULT
    return max(0, min(100, v))


def _handoff_prompt(chat_title: str) -> str:
    """The instruction sent to the model to write a session handoff summary.

    A brand-new session (empty context) reads it as its first message, so it must
    let that session continue exactly where this one left off."""
    return (
        "The full conversation is included in this message history, just above these instructions. "
        "Write a concise handoff summary of it (" + chat_title + "). "
        "A brand-new session with NO memory of this chat will read it as its first message, "
        "so it must let that session continue seamlessly. Use these exact markdown sections:\n\n"
        "## Task\nWhat we are working toward (the overall goal).\n\n"
        "## What was done\nConcrete actions completed and their outcomes.\n\n"
        "## Current state\nWhere things stand right now - what works, what is in progress, "
        "what is blocked or pending.\n\n"
        "## Key decisions & constraints\nChoices made, important facts, file paths, settings, "
        "and any rules that must carry over.\n\n"
        "## Next steps\nThe specific things to do next, in order.\n\n"
        "Be factual and specific (name files, commands, values). No preamble, no closing remarks - "
        "just the summary."
    )


def _safe_note_component(text) -> str:
    """One filesystem-safe filename component from arbitrary text.

    chats.json is USER-EDITABLE, so chat ids and titles are untrusted input here: every character
    that could build a path (separators, drive colons) is flattened to '_'. Components made only of
    dots are replaced outright - 'a..b' stays legal, '.'/'..' become 'chat', so an id can never turn
    into a parent-directory hop. Length-capped so a long chat title cannot make an unusable name."""
    s = re.sub(r"[^A-Za-z0-9_.-]", "_", str(text or ""))
    s = s.strip("._-")[:HANDOFF_TITLE_SLUG_MAX]
    return s or "chat"


def _handoff_notes_dir(settings) -> Optional[Path]:
    r"""Folder that receives handoff notes, read from Settings AT CALL TIME.

    Blank/missing/junk -> default_handoff_notes_dir() (the app working folder); any
    absolute path is honoured. Returns None when the folder cannot be created - the caller then skips
    the file write and tells the user, because a note must never block the UI and notes are never
    written to the AppData data store on the assumption that it is always writable.

    Never mutates settings (settings files stay exactly where resolve_data_dir() put them)."""
    raw = (settings or {}).get("handoff_notes_dir", "")
    if isinstance(raw, str):
        raw = raw.strip()
    else:
        raw = ""
    cand = default_handoff_notes_dir() if not raw else Path(raw).expanduser()
    try:
        cand.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None
    try:
        return cand.resolve()
    except Exception:
        return cand


def _handoff_note_path(notes_dir: Path, chat_id: str, title: str = "",
                       when=None) -> Path:
    """One file PER HANDOFF (never overwritten), inside notes_dir.

    <chatid>_<YYYYMMDD-HHMMSS-microseconds>[_<title slug>].md - id first so a folder lists grouped by
    chat and chronologically inside each chat; microsecond stamp because a marathon turn can trigger two
    summaries inside the same second (mid-turn gate, then an overflow on the next request) and a
    second-resolution name would silently overwrite the earlier one. The old design overwrote one rolling
    handoff_<chatid>.md per chat, so every older session's notes were lost."""
    stamp = (when or datetime.now()).strftime("%Y%m%d-%H%M%S-%f")
    safe = _safe_note_component(chat_id)
    slug = _safe_note_component(title)
    tail = "" if slug == "chat" else "_" + slug
    stem = f"{safe}_{stamp}{tail}"
    fp = Path(notes_dir) / (stem + ".md")
    # Windows clock resolution is ~1 ms, so datetime.now() CAN repeat its microsecond field across two
    # calls in a burst (mid-turn gate, then an overflow on the very next request). A collision would
    # silently overwrite the earlier note - exactly what per-handoff naming exists to prevent - so a
    # numeric suffix keeps every handoff's filename unique even at the same clock tick. The suffix
    # separator is '__' with a zero-padded counter on purpose: '_' (0x5F) sorts AFTER '.' (0x2E), so
    # "chat_<stamp>__02.md" ranks NEWER than "chat_<stamp>.md" in the newest-first listing, and the
    # zero padding keeps __02 < __10 ordering correct. A '-N' suffix would sort OLDER and invert it.
    n = 2
    try:
        while fp.exists() and n <= 200:
            fp = Path(notes_dir) / (stem + "__" + f"{n:02d}" + ".md")
            n += 1
    except Exception:
        pass
    return fp


def _handoff_note_files(notes_dir, chat_id: str) -> List[Path]:
    """Existing note files of ONE chat, newest first (empty list when the folder is absent).

    The prefix match must be exact: chat 'a' must not pick up chat 'ab' notes."""
    if not notes_dir:
        return []
    prefix = _safe_note_component(chat_id) + "_"
    try:
        found = [p for p in Path(notes_dir).glob(prefix + "*.md") if p.name.startswith(prefix)]
    except Exception:
        return []
    return sorted(found, key=lambda p: p.name, reverse=True)


def _handoff_previous_note_paths(notes_dir, chat_id: str, limit: int = HANDOFF_PREV_NOTES_LISTED) -> List[Path]:
    """The earlier handoff notes for this chat (newest first), for the header chain line."""
    files = _handoff_note_files(notes_dir, chat_id)
    return files[:max(0, int(limit))]


HANDOFF_NOTES_LISTED_SEEDED = 5      # how many note files the seeded handoff message lists


def _handoff_notes_context(settings, chat_id: str,
                           current_note: str = "") -> tuple:
    r"""(notes_intro, [note paths]) for a chat - pure function, NEVER touches the filesystem.

    Used by Start New Session / Auto-Continue so the continued session is told WHERE the handoff
    notes live and WHICH files belong to the chat being continued, newest first, the file written by
    this very handoff leading the list (the in-memory record kept by _show_handoff; after a restart
    that map is empty, so the newest file on disk heads the list). The model can then open them for
    the full record of every previous session and append progress. A blank setting names the default
    folder WITHOUT creating it: this text goes into an outgoing prompt, not onto the disk, so
    listing a folder that does not exist yet must be harmless (the worker creates it on first write)."""
    raw = (settings or {}).get("handoff_notes_dir", "")
    ndir = str(raw).strip() if isinstance(raw, str) and str(raw).strip() \
        else str(default_handoff_notes_dir())
    paths = []
    seen = set()
    cur = str(current_note or "")
    if cur:
        paths.append(Path(cur))
        seen.add(str(Path(cur)).lower())
    for p in _handoff_previous_note_paths(Path(ndir), chat_id,
                                         HANDOFF_NOTES_LISTED_SEEDED):
        k = str(p).lower()
        if k not in seen:
            paths.append(p)
            seen.add(k)
    if not paths:
        return "folder: " + ndir + " (no note file recorded for this chat yet)", []
    intro = "folder: " + ndir + " - newest notes for the chat being continued:\n- " + \
        "\n- ".join(str(p) for p in paths)
    return intro, paths


def _handoff_header(title: str, used_at: int, total_at: int, pct: int,
                    chat_id: str, prev_notes: List[Path]) -> str:
    """Markdown header written above every handoff summary.

    Names the previous note file(s) for this chat so a new session's notes point back at the older
    sessions they continue - the chain is explicit in the file, not only in the chat record."""
    if prev_notes:
        prev_line = "- Previous handoff notes: " + ", ".join(str(p) for p in prev_notes)
    else:
        prev_line = "- Previous handoff notes: none (first handoff for this chat)"
    return (f"# Session Handoff - {title}\n"
            f"- Chat: {title} ({chat_id})\n"
            f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} at "
            f"{_handoff_gauge_text(used_at, total_at)} ({pct}% threshold)_\n"
            f"{prev_line}\n\n")


def _handoff_rearm_pct(settings: dict) -> int:
    """How much the context gauge must grow (percent of the window) before a chat is
    allowed to produce another handoff summary. 0 = re-fire on every step boundary.
    Clamped 0..100; junk falls back to the default."""
    try:
        v = int(float(settings.get("handoff_rearm_pct", HANDOFF_REARM_PCT_DEFAULT)))
    except (TypeError, ValueError):
        v = HANDOFF_REARM_PCT_DEFAULT
    return max(0, min(100, v))


HANDOFF_TOOL_OUTPUT_CAP = 2500    # max chars of one tool result kept in a handoff request
HANDOFF_IMAGE_PLACEHOLDER = "[image attached]"
HANDOFF_SUMMARY_MAX_TOKENS = 2048   # bound on the summary itself (some servers reject it -> retried without)
HANDOFF_REARM_PCT_DEFAULT = 10      # a chat may handoff again once usage has grown by this many % of the window

# Budget used when the server reports NO context size (LM Studio / Ollama send no
# context_length, so _ctx_window() is 0). The previous flat 120000 chars (~30K tokens)
# was built from a history that had JUST overflowed - the summarization request then
# overflowed too and the turn ended with 'Handoff summary failed' and no notes at all.
HANDOFF_NO_WINDOW_BUDGET = 24000    # ~6K tokens of history; the ladder halves it on retry
HANDOFF_FALLBACK_MIN_CHARS = 3000   # below this, deterministic notes are not worth writing
HANDOFF_FALLBACK_MAX_CHARS = 20000  # hard cap on the model-free transcript fallback
HANDOFF_RETRY_OVERFLOW   = 3        # halve the history budget this many times when the
                                    # summary request itself is refused as too long


def _handoff_budget_chars(total_ctx: int) -> int:
    """Char budget for the history included in a handoff request.

    Handoff only fires when the server reports a context size (total_ctx tokens),
    so 65% of that window is a grounded figure - it leaves room for the system
    prompt, the instruction and the summary output itself. The fallback covers
    direct calls with no context info (~30K tokens of history).

    Also reserves headroom: a handoff triggered at 95% of the window has far less room
    than one triggered at 75%, so the budget is capped at (window - summary output -
    prompt overhead) instead of a flat 65% - otherwise a late trigger would build a
    summarization request that overflows too."""
    try:
        total_ctx = int(total_ctx or 0)
    except (TypeError, ValueError):
        total_ctx = 0
    if total_ctx > 0:
        room = min(int(total_ctx * 0.65), max(1024, total_ctx - HANDOFF_SUMMARY_MAX_TOKENS - 4096))
        return max(8000, room * 4)
    return HANDOFF_NO_WINDOW_BUDGET


def _flatten_handoff_content(m: dict) -> Optional[str]:
    """Message content as plain text; multimodal parts are flattened (images ->
    a placeholder) so the request works on non-vision models and stays small."""
    c = m.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: List[str] = []
        for p in c:
            if not isinstance(p, dict):
                continue
            if p.get("type") == "text":
                parts.append(str(p.get("text", "")))
            elif p.get("type") == "image_url":
                parts.append(HANDOFF_IMAGE_PLACEHOLDER)
        return "\n".join(x for x in parts if x) or None
    return c


def _sanitize_for_handoff(messages: List[dict]) -> List[dict]:
    """Copy of a chat history safe to send to the summarization model.

    - system messages are dropped (a fresh one is injected at request time)
    - image payloads (base64, up to ~4 MB each) become text placeholders
    - tool outputs are capped - they dominate long agentic chats
    - assistant tool_calls and their tool results stay intact pairs, so strict
      OpenAI-compatible servers don't reject the request"""
    out: List[dict] = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "tool":
            content = str(m.get("content") or "")
            if len(content) > HANDOFF_TOOL_OUTPUT_CAP:
                content = content[:HANDOFF_TOOL_OUTPUT_CAP] + "\n[... truncated for handoff summary]"
            out.append({"role": "tool", "tool_call_id": m.get("tool_call_id", ""),
                        "name": m.get("name", ""), "content": content})
        elif role in ("user", "assistant"):
            nm: dict = {"role": role, "content": _flatten_handoff_content(m)}
            if role == "assistant" and m.get("tool_calls"):
                nm["tool_calls"] = m["tool_calls"]
            out.append(nm)
    return out


def _trim_for_handoff(messages: List[dict], budget_chars: int) -> List[dict]:
    """Drop oldest complete turns until the history fits the char budget.

    Turns are cut at user-message boundaries, so an assistant tool_calls message
    is never orphaned from its tool results. If trimming leaves a non-user
    message first (strict servers require the conversation to open with one), a
    marker user message is prepended."""
    ms = list(messages)
    while len(ms) > 2 and estimate_prompt_tokens(ms) * 4 > budget_chars:
        users = [i for i, m in enumerate(ms) if m.get("role") == "user"]
        if len(users) < 2:
            break                      # only one user message left - can't trim further
        del ms[:users[1]]
    if ms and ms[0].get("role") != "user":
        ms.insert(0, {"role": "user", "content": "[Earlier conversation omitted for length]"})
    return ms


def _handoff_gauge_text(used: int, total: int) -> str:
    """Human-readable context gauge for handoff notes / accordion titles.

    A server that reports no context_length (LM Studio, Ollama) leaves total at 0, and an
    overflow-triggered handoff is now allowed in that state - so say 'window unknown'
    rather than printing the meaningless '800/0 tokens'."""
    u = int(used or 0)
    t = int(total or 0)
    if t <= 0:
        return f"{u} tokens (window unknown)"
    return f"{u}/{t} tokens"


def _deterministic_handoff(chat: dict, max_chars: int = HANDOFF_FALLBACK_MAX_CHARS) -> str:
    """Model-free handoff notes built straight from the chat history.

    Last-resort salvage: used when the summarization request could not be completed -
    almost always because it overflowed too (the usual case on a server that reports no
    context_length, where there is no window to size the budget from). A digest is far
    better than nothing: Start New Session / Auto-Continue feed this text into the fresh
    chat, so a marathon turn still carries its state over instead of ending with an error.

    Deliberately factual - no invented 'key decisions'. Reuses _sanitize_for_handoff so
    images become placeholders and tool outputs stay capped, then fills the budget from the
    NEWEST end (the tail is what a continuation needs most)."""
    # Snapshot too: this runs on the handoff thread while the user may already be sending.
    msgs = _sanitize_for_handoff(list((chat or {}).get("messages") or []))
    if not msgs:
        return ""
    budget = max(1000, min(int(max_chars or 0), HANDOFF_FALLBACK_MAX_CHARS))

    def _clip(s: str, n: int) -> str:
        s = str(s or "")
        return s if len(s) <= n else s[:n] + "…"

    task = ""
    for m in msgs:
        if m.get("role") == "user" and str(m.get("content") or "").strip():
            task = str(m["content"]).strip()
            break

    actions: List[str] = []
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
            actions.append(f"- ran {fn.get('name', '?')}({_clip(str(fn.get('arguments', '')), 160)})")

    tail: List[str] = []
    used = 0
    for m in reversed(msgs):
        body = _clip(str(m.get("content") or "").strip(), 500)
        if not body:
            continue
        block = f"{m.get('role', '?')}: {body}"
        if used + len(block) > budget:
            break
        tail.append(block)
        used += len(block)
    tail.reverse()

    parts = ["_Model-written summary unavailable (the summary request itself failed), so "
             "these notes are a deterministic digest of the conversation._", "",
             "## Task", _clip(task, 600) or "(no user message found)", "",
             "## What was done"]
    parts += actions[:40] or ["- (no tool calls recorded)"]
    parts += ["", "## Recent conversation (newest last)"] + tail
    return "\n".join(parts)


def is_context_overflow_error(msg: str) -> bool:
    """True when a server error means "the context window is full".

    Long agentic turns used to end with nothing saved: the request simply failed and
    the error path never reached the handoff check. Recognizing these responses (both
    OpenAI-style and llama.cpp/Unsloth wording) lets the app write a handoff summary
    instead of losing an hour of work. Deliberately narrow - a 401 or a model_not_found
    must NOT be mistaken for an overflow."""
    low = str(msg or "").lower()
    if any(k in low for k in ("context_length_exceeded", "maximum context length",
                            "max_context_length_exceeded", "context window is too large",
                            "prompt is too long", "input is too long",
                            "reduce the length of the messages",
                            "too many tokens", "exceeds the maximum context",
                            "insufficient space", "context overflow")):
        return True
    # llama.cpp / Unsloth style: "the request exceeds the available context size (N)
    # try increasing the context size" - matched as a phrase so bare numbers never trip it.
    if "exceeds the available context" in low or "available context size" in low:
        return True
    return False


def compute_speed_readout(usage: Optional[dict], full_content: str, full_reasoning: str,
                          tool_acc: dict, t0: float, first_token_t: Optional[float],
                          last_token_t: Optional[float], turn_total: int = 0) -> Optional[str]:
    """Build the top-bar speed readout for one model step.

    Decode rate = (tokens - 1) / (t_last - t_first). The first token arrives at
    t_first, i.e. AFTER prefill finished, so prefill time is excluded from the
    denominator (the old code divided by t0->last, which dragged the rate down
    as context grew). TTFT (time to first token) is reported separately.
    The "N tok" figure is CUMULATIVE for the current user prompt: turn_total
    holds the steps already completed this turn, and this step's completion
    tokens are added on top (exact usage count when known, char-based live
    estimate while streaming). It resets to 0 at the start of each new prompt.
    Returns None when there is nothing meaningful to display."""
    ttft = (first_token_t - t0) if first_token_t else 0.0
    span_ok = bool(first_token_t and last_token_t)

    if usage and usage.get("completion_tokens"):
        toks = int(usage["completion_tokens"])
        total_toks = max(0, int(turn_total or 0)) + toks
        if toks >= 2 and span_ok:
            rate = (toks - 1) / max(0.05, last_token_t - first_token_t)
            return f"⚡ {rate:.1f} tok/s · {total_toks} tok · TTFT {ttft:.1f}s"
        return f"⚡ {total_toks} tok · TTFT {ttft:.1f}s"

    if full_content or full_reasoning or tool_acc:
        # Tool-call argument characters count toward the estimate too, so a pure
        # tool-call step (no visible text) still produces a readout.
        tool_chars = sum(len(tc.get("args", "")) for tc in tool_acc.values())
        est = max(1, (len(full_content) + len(full_reasoning) + tool_chars) // 4)
        total_est = max(0, int(turn_total or 0)) + est
        if est >= 2 and span_ok:
            rate = (est - 1) / max(0.05, last_token_t - first_token_t)
            return f"⚡ ~{rate:.1f} tok/s (≈{total_est} tok) · TTFT {ttft:.1f}s"
        return f"⚡ ≈{total_est} tok · TTFT {ttft:.1f}s"

    return None


def estimate_prompt_tokens(messages: List[dict]) -> int:
    """Rough token estimate for an outgoing message list (chars ÷ 4).

    Used only when a server sends no usage block, so the context readout can
    still show *something*. Mirrors the char-based heuristic in
    compute_speed_readout. Counts text content (string or multimodal parts)
    plus tool-call argument JSON; image payloads are skipped (their base64
    would wildly overstate the estimate)."""
    chars = 0
    for m in messages:
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if isinstance(c, str):
            chars += len(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and part.get("type") == "text":
                    chars += len(str(part.get("text", "")))
        for tc in (m.get("tool_calls") or []):
            try:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                chars += len(str(fn.get("arguments", "")))
            except Exception:
                pass
    return max(0, chars // 4)


def fmt_ctx_used(used: Optional[int], total: Optional[int]) -> str:
    """Format the session context-usage readout for the top bar.

    e.g. used=98000, total=159000 -> '📏 98K/159K'. Uses the same 📏 glyph and
    _fmt_ctx() scaling as the existing max-context readout. Returns '' when
    there is no usage figure to show."""
    try:
        u = int(used) if used else 0
    except (TypeError, ValueError):
        u = 0
    if u <= 0:
        return ""
    us = _fmt_ctx(u) or str(u)
    t = _fmt_ctx(total)
    if not t:
        return f"\U0001F4CF {us}"
    return f"\U0001F4CF {us}/{t}"


def session_total_tokens(usage: Optional[dict], full_content: str, full_reasoning: str,
                         tool_acc: dict, messages: List[dict]) -> int:
    """Total session token footprint after one model step (prompt + completion).

    This is the number that actually fills the context window, so it drives the
    top-bar 📏 used/max gauge. Prefers the server's usage block: total_tokens,
    else prompt_tokens + completion_tokens (some servers omit the total). When a
    server sends no usable usage at all, estimates from the outgoing messages plus
    the characters just generated (chars ÷ 4, same heuristic as
    compute_speed_readout). The value is "as of end of last generation"; tool
    results appended after that point are picked up on the next model step."""
    u = usage or {}
    tot = u.get("total_tokens")
    if not tot and u.get("prompt_tokens") and u.get("completion_tokens"):
        tot = u["prompt_tokens"] + u["completion_tokens"]
    if tot:
        try:
            return max(0, int(tot))
        except (TypeError, ValueError):
            pass
    comp_chars = (len(full_content) + len(full_reasoning)
                  + sum(len(tc.get("args", "")) for tc in tool_acc.values()))
    return estimate_prompt_tokens(messages) + max(0, comp_chars // 4)


# Transient stream-interruption retry: a dropped connection mid-response (LM Studio
# restart, Wi-Fi blip, server OOM-kill, TCP timeout) used to end the whole turn with
# a half-finished answer. We now retry the step ONCE when the failure looks like a
# transient connection problem - but never for server-side rejections (model not
# found, context too long, bad request), which won't fix themselves on a retry.
_TRANSIENT_STREAM_KEYWORDS = (
    "connection", "reset", "timed out", "timeout", "broken pipe",
    "remote end closed", "eof occurred", "incomplete read", "temporarily",
)

def _is_transient_stream_error(e: Exception) -> bool:
    """True when a mid-stream failure looks like a transient connection problem
    worth one automatic retry (see the keyword list above)."""
    if isinstance(e, (ConnectionError, TimeoutError)):
        return True
    m = str(e).lower()
    return any(k in m for k in _TRANSIENT_STREAM_KEYWORDS)

# ════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ════════════════════════════════════════════════════════════════════════════

class DeskpilotApp:
    def __init__(self) -> None:
        # ── state ────────────────────────────────────────────────────────
        self.settings = load_settings()
        global FONT_DELTA
        try:
            FONT_DELTA = int(self.settings.get("ui_font_size", FONT_BASE)) - FONT_BASE
        except (TypeError, ValueError):
            FONT_DELTA = 0
        self.chats = load_chats()
        self.current_chat_id: Optional[str] = None

        self._busy = False
        self._closed = False
        self._stick_bottom = True
        self._run_chat_id: Optional[str] = None
        self._active_md: Optional[MarkdownStream] = None   # main thread only
        self._reasoning_sid: Optional[str] = None          # main thread only
        self._reasoning_sids: List[str] = []               # open reasoning drawers this turn (main thread)
        self._supports_usage_opts = True
        self._ctx_used = 0          # last known prompt_tokens for the active session (set on main thread)
        self._ctx_total = 0         # context window size from /models metadata (main thread)
        self._handoff_inflight: set = set()        # chats with a summary worker running RIGHT NOW
        self._handoff_lock = threading.Lock()      # guards the set above (worker + main thread)
        self._last_handoff: Dict[str, str] = {}    # chat_id -> last rendered handoff summary text
        self._last_handoff_file: Dict[str, str] = {}   # v1.1.10: chat_id -> note file that handoff wrote
        self._ctx_used_run = 0          # gauge value as of the LAST model step (worker thread copy)

        self._code_buttons: List[tk.Button] = []
        self._image_thumbs: List[tuple] = []   # (photo ref, text index, path-or-None) of embedded thumbnails
        self._accordions: Dict[str, dict] = {}
        self._pending_tool_output: Optional[str] = None   # sid of the "running" tool-output section
        self._attachments: List[Path] = []
        self._perm_combos: Dict[str, tuple] = {}
        self._mic_active = False
        self._mic_stop = threading.Event()   # set => stop continuous dictation
        self._stop_event = threading.Event()  # set => user pressed Stop; abort the in-flight turn
        self._tts_stop = threading.Event()   # set => halt TTS playback (send/close/toggle off)
        self._stt_thread: Optional[threading.Thread] = None
        self._ui_queue = queue.Queue()   # thread-safe handoff: worker threads -> UI
        self._sidebar_visible = True   # Python-side source of truth (pane.slaves() unreliable)
        self._drag_state: Optional[dict] = None   # sidebar drag-and-drop reordering state
        self._chat_visible: List[str] = []         # chat ids currently shown in the listbox (search-filtered)
        self._links: Dict[tuple, str] = {}         # (line, char offset) -> URL of clickable links in chat_text
        self._clients: Dict[tuple, Any] = {}       # (server_url, api_key) -> cached OpenAI client (keep-alive pooling)
        self._last_chats_save = 0.0                # time.monotonic() of last chats.json write (throttles mid-turn saves)
        self._pending_temperature: Optional[str] = None   # welcome-state temp change, applied to the chat the next send creates
        self._sampling_sent: Optional[dict] = None   # last request's sampling payload (worker writes, note renders)
        self._verify_running = False            # a "Verify parameters" probe is in flight (Settings -> Sampling)
        self._pending_thinking: Optional[str] = None      # welcome-state thinking change, same
        # ── MCP (Model Context Protocol) servers ────────────────────────────
        self._mcp_clients: Dict[str, MCPClient] = {}   # server name -> connected client
        self._mcp_lock = threading.Lock()              # guards _mcp_clients (iteration vs mutation across threads)
        self._mcp_connect_lock = threading.Lock()      # serializes whole connect passes (no duplicate subprocesses on rapid saves)
        self._mcp_connect_pending = threading.Event()  # set by _mcp_connect_all; consumed by the worker loop
        self._mcp_tool_map: Dict[str, tuple] = {}      # mcp_<server>_<tool> -> (server, raw tool name)
        self._mcp_perm_widgets: List[tk.Widget] = []   # MCP section widgets in the permission bar

        # ── root window ───────────────────────────────────────────────────
        self.root = tk.Tk()
        self.root.title(f"{APP_NAME} — AI Desktop Assistant")
        try:
            self.root.geometry(self.settings.get("window_geometry", "1280x800"))
        except tk.TclError:
            self.root.geometry("1280x800")
        self.root.minsize(960, 620)
        self.root.configure(bg=COL["bg_main"])

        # ── build UI ──────────────────────────────────────────────────────
        self._configure_styles()
        self._build_ui()
        self._configure_text_tags()
        self._update_topbar_labels()

        # ── restore chats / sidebar state ─────────────────────────────────
        if not self.chats["order"]:
            self.new_chat()          # first run: creates + loads the first conversation
        else:
            # Always launch with a cleared chat box (welcome screen) instead of
            # re-opening the previous session. Saved threads are listed in the
            # sidebar and load on click; sending from this fresh state starts a
            # brand-new conversation (see send_message fallback).
            try:
                self.chat_text.delete("1.0", "end")
            except tk.TclError:
                pass
            self._render_welcome()
            self._refresh_chat_list()   # restore saved threads in the sidebar (none selected)
        try:
            self.sidebar.configure(width=int(self.settings.get("sidebar_width", 250)))
        except (tk.TclError, ValueError):
            pass
        if self.settings.get("sidebar_collapsed"):
            self._set_sidebar_visible(False, save=False)

        # ── bindings / lifecycle ──────────────────────────────────────────
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<Control-n>", lambda e: self.new_chat())
        self.root.bind("<Control-f>", lambda e: (self.chat_search.focus_set(), "break")[1])
        self.root.after(300, self.apply_dark_titlebar)      # HWND must exist first
        self._poll_ui_queue()             # start draining worker-thread UI callbacks
        self._refresh_ctx_topbar()        # populate the top-bar context readout (background)
        self._mcp_connect_all()           # start configured MCP servers (background thread)

        print(f"[Deskpilot] v{VERSION} ready — "
              f"pypdf={'✓' if PdfReader else '✗'}  Pillow={'✓' if ImageGrab else '✗'}  "
              f"TTS={'✓' if (Kokoro and sounddevice) else '✗'}  SpeechRecognition={'✓' if sr else '✗'}  Whisper={'✓' if WhisperModel else '✗'}")

        print(f"[{APP_NAME}] Data files: {SETTINGS_FILE}  |  {CHATS_FILE}")

    def run(self) -> None:
        self.root.mainloop()

    # ════════════════════════════════════════════════════════════════════
    #  UI CONSTRUCTION
    # ════════════════════════════════════════════════════════════════════

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")                        # most customizable theme
        except tk.TclError:
            pass
        style.configure(".", background=COL["bg_main"], foreground=COL["text"])
        style.configure("TFrame", background=COL["bg_main"])
        style.configure("TPanedwindow", background=COL["border"], relief="flat", borderwidth=0)
        style.configure("Tool.TCombobox",
                        fieldbackground=COL["bg_raised"], background=COL["bg_deep"],
                        foreground=COL["text"], arrowcolor=COL["text_dim"],
                        lightcolor=COL["bg_raised"], darkcolor=COL["bg_raised"],
                        bordercolor=COL["border"])
        style.map("Tool.TCombobox",
                  fieldbackground=[("readonly", COL["bg_raised"]), ("disabled", COL["bg_deep"])],
                  foreground=[("readonly", COL["text"])])

    def _pane_add(self, widget, minsize=None):
        """ttk.Panedwindow.add with graceful fallback: the -minsize option is
        missing on some Tk builds (e.g. < 8.6 and Tk 9), so retry without it."""
        try:
            if minsize is not None:
                self.pane.add(widget, minsize=minsize)
            else:
                self.pane.add(widget)
        except tk.TclError:
            self.pane.add(widget)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # ── resizable split pane (sidebar | chat workspace) ───────────────
        self.pane = ttk.Panedwindow(self.root, orient="horizontal")
        self.pane.grid(row=0, column=0, sticky="nsew")
        self._build_sidebar()
        self._build_chat_area()

        # ── tool permissions bar (bottom) ─────────────────────────────────
        self._build_permissions_bar()

    def _build_sidebar(self) -> None:
        side = tk.Frame(self.pane, bg=COL["bg_deep"], width=250)
        self._pane_add(side, 170)
        self.sidebar = side

        tk.Label(side, text="💬 Conversations", bg=COL["bg_deep"], fg=COL["text"],
                 font=F(13, "bold")).pack(anchor="w", padx=12, pady=(14, 8))

        # Search box: filters the list below as you type (title match).
        self.chat_search = tk.Entry(
            side, bg=COL["bg_main"], fg=COL["text_dim"], insertbackground=COL["text"],
            relief="flat", bd=0, highlightthickness=1,
            highlightbackground=COL["border"], highlightcolor=COL["accent"], font=F(11))
        self.chat_search.insert(0, "🔎 Search chats…")
        self.chat_search.pack(fill="x", padx=12, pady=(0, 8))
        self.chat_search.bind("<KeyRelease>", lambda e: self._on_chat_search())
        self.chat_search.bind("<FocusIn>", self._chat_search_focus_in)
        self.chat_search.bind("<FocusOut>", self._chat_search_focus_out)
        self.chat_search.bind("<Return>", lambda e: self._chat_search_enter())
        self.chat_search.bind("<Escape>", lambda e: self._clear_chat_search())
        self.chat_search_results = tk.Label(side, text="", bg=COL["bg_deep"],
                                            fg=COL["text_dim"], font=F(10))
        self.chat_search_results.pack(anchor="w", padx=14)

        tk.Button(side, text="+ New Chat", command=self.new_chat,
                  bg=COL["accent"], fg="#FFFFFF", activebackground="#2563EB",
                  relief="flat", bd=0, padx=10, pady=7, cursor="hand2",
                  font=F(12, "bold")).pack(fill="x", padx=12, pady=(0, 8))

        row = tk.Frame(side, bg=COL["bg_deep"])
        row.pack(fill="x", padx=12, pady=(0, 10))
        tk.Button(row, text="✏️ Rename Chat", command=self.rename_chat,
                  bg=COL["bg_raised"], fg=COL["text"], activebackground="#4B5563",
                  relief="flat", bd=0, padx=8, pady=5, cursor="hand2",
                  font=F(11)).pack(side="left", fill="x", expand=True, padx=(0, 4))
        tk.Button(row, text="🗑️ Delete Chat", command=self.delete_chat,
                  bg=COL["bg_raised"], fg=COL["danger"], activebackground="#5B2323",
                  relief="flat", bd=0, padx=8, pady=5, cursor="hand2",
                  font=F(11)).pack(side="left", fill="x", expand=True, padx=(4, 0))

        tk.Button(side, text="📥 Export Chat", command=self.export_chat,
                  bg=COL["bg_raised"], fg=COL["text"], activebackground="#4B5563",
                  relief="flat", bd=0, padx=8, pady=5, cursor="hand2",
                  font=F(11)).pack(fill="x", padx=12)

        list_frame = tk.Frame(side, bg=COL["bg_deep"])
        list_frame.pack(fill="both", expand=True, padx=12)
        self.chat_list = tk.Listbox(
            list_frame, bg=COL["bg_main"], fg=COL["text"],
            selectbackground=COL["accent"], selectforeground="#FFFFFF",
            highlightthickness=0, activestyle="none", relief="flat",
            font=F(11))
        sb = tk.Scrollbar(list_frame, orient="vertical", command=self.chat_list.yview,
                          bg=COL["bg_raised"], troughcolor=COL["bg_deep"],
                          highlightthickness=0, borderwidth=0)
        self.chat_list.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.chat_list.pack(side="left", fill="both", expand=True)
        self.chat_list.bind("<<ListboxSelect>>", lambda e: self._on_chat_selected())
        self.chat_list.bind("<Double-Button-1>", lambda e: self.rename_chat())
        # Drag-and-drop reordering (a press that moves >6px becomes a drag;
        # plain clicks still select, double-click still renames).
        self.chat_list.bind("<ButtonPress-1>", self._chat_list_press)
        self.chat_list.bind("<B1-Motion>", self._chat_list_drag)
        self.chat_list.bind("<ButtonRelease-1>", self._chat_list_release)

        tk.Frame(side, bg=COL["border"], height=1).pack(fill="x", padx=12, pady=8)
        tk.Button(side, text="⚙ Settings", command=self.open_settings,
                  bg=COL["bg_raised"], fg=COL["text"], activebackground="#4B5563",
                  relief="flat", bd=0, padx=10, pady=6, cursor="hand2",
                  font=F(11)).pack(fill="x", padx=12)
        tk.Label(side, text=f"{APP_NAME} v{VERSION}", bg=COL["bg_deep"], fg=COL["text_dim"],
                 font=F(10)).pack(pady=(6, 10))

    def _build_chat_area(self) -> None:
        area = tk.Frame(self.pane, bg=COL["bg_main"])
        self._pane_add(area, 420)
        self.chat_area = area

        # ── top bar ───────────────────────────────────────────────────────
        top = tk.Frame(area, bg=COL["bg_main"])
        top.pack(fill="x")
        tk.Button(top, text="☰", command=self.toggle_sidebar,
                  bg=COL["bg_raised"], fg=COL["text"], activebackground="#4B5563",
                  relief="flat", bd=0, width=3, cursor="hand2",
                  font=F(13)).pack(side="left", padx=(10, 4), pady=8)
        tk.Label(top, text=f"◆ {APP_NAME}", bg=COL["bg_main"], fg=COL["accent"],
                 font=F(15, "bold")).pack(side="left")
        tk.Label(top, text="AI Desktop Assistant", bg=COL["bg_main"], fg=COL["text_dim"],
                 font=F(11)).pack(side="left", padx=(8, 0), pady=(6, 0))

        self.status_label = tk.Label(top, text="Ready", bg=COL["bg_main"],
                                     fg=COL["warning"], font=F(11))
        self.speed_label = tk.Label(top, text="⚡ –", bg=COL["bg_main"],
                                    fg=COL["success"], font=F(11, "bold"))
        self.ctx_top_label = tk.Label(top, text="", bg=COL["bg_main"],
                                      fg=COL["text_dim"], font=F(11))
        self.temp_top_label = tk.Label(top, text="", bg=COL["bg_main"],
                                       fg=COL["text_dim"], font=F(11))
        self.top_model_label = tk.Label(top, text="", bg=COL["bg_main"],
                                        fg=COL["text_dim"], font=F(11))
        # pack right->left: status | speed | context | temperature | model (model at far right)
        for w in (self.status_label, self.speed_label, self.ctx_top_label,
                  self.temp_top_label, self.top_model_label):
            w.pack(side="right", padx=(0, 12 if w is not self.top_model_label else 14), pady=10)

        tk.Frame(area, bg=COL["border"], height=1).pack(fill="x")

        # ── chat display (scrollable Text with markdown tags) ─────────────
        mid = tk.Frame(area, bg=COL["bg_main"])
        mid.pack(fill="both", expand=True)
        self.chat_text = tk.Text(
            mid, wrap="word", bg=COL["bg_main"], fg=COL["text"],
            insertbackground=COL["text"], relief="flat", bd=0,
            font=F(12), padx=14, pady=10, spacing1=2, spacing3=2,
            selectbackground=COL["bg_raised"], selectforeground=COL["text"])
        csb = tk.Scrollbar(mid, orient="vertical", command=self.chat_text.yview,
                           bg=COL["bg_raised"], troughcolor=COL["bg_main"],
                           highlightthickness=0, borderwidth=0)
        self.chat_text.configure(yscrollcommand=csb.set)
        csb.pack(side="right", fill="y")
        self.chat_text.pack(side="left", fill="both", expand=True)
        self.chat_text.app = self      # render_inline() resolves the app for the link registry
        for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.chat_text.bind(ev, lambda e: setattr(self, "_stick_bottom", False))

        # ── attachment chips + input row ──────────────────────────────────
        self.chips_frame = tk.Frame(area, bg=COL["bg_main"])
        self.chips_frame.pack(fill="x", padx=12)

        inp = tk.Frame(area, bg=COL["bg_main"])
        inp.pack(fill="x", padx=12, pady=(4, 8))

        tk.Button(inp, text="📎", command=self.attach_files,
                  bg=COL["bg_raised"], fg=COL["text"], activebackground="#4B5563",
                  relief="flat", bd=0, width=3, cursor="hand2",
                  font=F(13)).pack(side="left", padx=(0, 6))

        # The input box gets its OWN vertical scrollbar (user request): long prompts and
        # Shift+Enter multi-line text were invisible with no way to scroll. It lives in a
        # thin frame packed exactly where the Text used to be, so every button on the row
        # keeps its original side="left" order (mic / TTS / temp+thinking / Send).
        # Styling mirrors the chat pane scrollbar; Tk 9-safe (explicit bd/highlightthickness).
        box = tk.Frame(inp, bg=COL["bg_deep"])
        box.pack(side="left", fill="both", expand=True)
        self.input_text = tk.Text(
            box, height=3, wrap="word", bg=COL["bg_deep"], fg=COL["text"],
            insertbackground=COL["text"], relief="flat", bd=0,
            font=F(12), padx=10, pady=8)
        self.input_scroll = tk.Scrollbar(box, orient="vertical", command=self.input_text.yview,
                                         bg=COL["bg_raised"], troughcolor=COL["bg_deep"],
                                         highlightthickness=0, borderwidth=0)
        self.input_text.configure(yscrollcommand=self.input_scroll.set)
        self.input_scroll.pack(side="right", fill="y")
        self.input_text.pack(side="left", fill="both", expand=True)
        self.input_text.bind("<Return>", self._on_input_return)
        # Right-click context menu (Cut/Copy/Paste/Select All) on both text widgets;
        # chat_text additionally gets click handlers for embedded image thumbnails.
        self.chat_text.bind("<Button-1>", self._on_chat_click)
        self.chat_text.bind("<Control-Button-1>", lambda e: self._open_link_at(e))
        self.chat_text.bind("<Button-2>", lambda e: self._open_link_at(e))
        self.chat_text.bind("<Motion>", self._on_chat_motion)
        self.chat_text.bind("<Button-3>", self._on_chat_right_click)
        self.input_text.bind("<Button-3>", self._show_context_menu)

        self.mic_btn = tk.Button(inp, text="🎤", command=self.toggle_mic,
                                 bg=COL["bg_raised"], fg=COL["text"], activebackground="#4B5563",
                                 relief="flat", bd=0, width=3, cursor="hand2",
                                 font=F(13))
        self.mic_btn.pack(side="left", padx=(6, 2))

        self.tts_btn = tk.Button(inp, text="🔊", command=self.toggle_tts,
                                 bg=COL["bg_raised"], fg=COL["text_dim"], activebackground="#4B5563",
                                 relief="flat", bd=0, width=3, cursor="hand2",
                                 font=F(13))
        self.tts_btn.pack(side="left", padx=(2, 6))

        # Per-chat temperature + thinking selectors stacked in a narrow column:
        # temperature on top, Thinking On/Off directly below it (user-requested layout).
        sel_col = tk.Frame(inp, bg=COL["bg_main"])
        sel_col.pack(side="left", padx=(6, 2))
        self.temp_combo = ttk.Combobox(sel_col, values=list(TEMP_VALUES), state="readonly",
                                       width=8, style="Tool.TCombobox", font=F(11))
        self.temp_combo.set(TEMP_DEFAULT)
        self.temp_combo.bind("<<ComboboxSelected>>", lambda e: self._on_temp_selected())
        self.temp_combo.pack(side="top")
        self.think_combo = ttk.Combobox(sel_col, values=list(THINK_VALUES), state="readonly",
                                        width=8, style="Tool.TCombobox", font=F(11))
        self.think_combo.set("High")
        self.think_combo.bind("<<ComboboxSelected>>", lambda e: self._on_thinking_selected())
        self.think_combo.pack(side="top", pady=(2, 0))

        self.send_btn = tk.Button(inp, text="➤ Send", command=self.send_message,
                                  bg=COL["accent"], fg="#FFFFFF", activebackground="#2563EB",
                                  relief="flat", bd=0, padx=18, pady=10, cursor="hand2",
                                  font=F(12, "bold"))
        self.send_btn.pack(side="left")

    def _build_permissions_bar(self) -> None:
        """Dynamic per-tool permission bar at the bottom of the window."""
        bar = tk.Frame(self.root, bg=COL["bg_deep"])
        bar.grid(row=1, column=0, sticky="ew")
        tk.Label(bar, text="🔐 Tool Permissions", bg=COL["bg_deep"], fg=COL["text_dim"],
                 font=F(11, "bold")).pack(side="left", padx=(12, 8), pady=7)

        canvas = tk.Canvas(bar, bg=COL["bg_deep"], height=46, highlightthickness=0)
        hsb = ttk.Scrollbar(bar, orient="horizontal", command=canvas.xview)
        canvas.configure(xscrollcommand=hsb.set)
        hsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self._perm_inner = tk.Frame(canvas, bg=COL["bg_deep"])
        inner = self._perm_inner
        canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        for name, pretty in TOOLS:
            cell = tk.Frame(inner, bg=COL["bg_deep"])
            cell.pack(side="left", padx=7, pady=4)
            lab = tk.Label(cell, text=pretty, bg=COL["bg_deep"],
                           fg=COL["text_dim"], font=F(10))
            lab.pack(anchor="w")
            cb = ttk.Combobox(cell, values=list(PERM_LABELS.values()), state="readonly",
                            width=13, style="Tool.TCombobox", font=F(11))
            perm = self.settings["tool_permissions"].get(name, "ask")
            cb.set(PERM_LABELS[perm])
            cb.pack(anchor="w")

            def on_change(e=None, name=name, cb=cb, lab=lab):
                label = cb.get()
                new_perm = PERM_VALUES.get(label, "ask")
                self.settings["tool_permissions"][name] = new_perm
                lab.configure(fg={"always": COL["success"],
                                  "ask":    COL["warning"],
                                  "off":    COL["danger"]}[new_perm])
                save_settings(self.settings)

            cb.bind("<<ComboboxSelected>>", on_change)
            self._perm_combos[name] = (cb, lab)

        # color the initial state too
        for name, (_, lab) in self._perm_combos.items():
            perm = self.settings["tool_permissions"].get(name, "ask")
            lab.configure(fg={"always": COL["success"], "ask": COL["warning"], "off": COL["danger"]}[perm])

        # Fit the canvas to its content so labels + dropdowns are never clipped at
        # the bottom of the window, regardless of the user's chosen UI font size.
        # MCP tools (discovered from connected servers) are appended after the
        # built-in tools; _mcp_rebuild_permissions_bar() owns that section.
        self._mcp_rebuild_permissions_bar()

        self._perm_canvas = canvas
        try:
            inner.update_idletasks()
            need = int(inner.winfo_reqheight()) + 8
            if need > int(canvas.cget("height")):
                canvas.configure(height=need)
        except tk.TclError:
            pass

    def _configure_text_tags(self) -> None:
        """Configure all Text-widget tags (order matters for tag priority)."""
        t = self.chat_text
        t.tag_configure("body", font=F(12))
        # Assistant response paragraphs: bold, like LM Studio's chat view.
        # (User messages and the welcome text keep the regular "body" tag.)
        t.tag_configure("asst_body", font=F(12, "bold"), foreground=COL["text"])
        t.tag_configure("user_hdr", font=F(12, "bold"), foreground="#60A5FA")
        t.tag_configure("asst_hdr", font=F(12, "bold"), foreground=COL["success"])
        t.tag_configure("h1", font=F(17, "bold"), foreground="#F9FAFB")
        t.tag_configure("h2", font=F(15, "bold"), foreground="#F3F4F6")
        t.tag_configure("h3", font=F(13, "bold"), foreground=COL["text"])
        # NOTE: no explicit `background` on these tags - a tag-level background overrides the
        # widget's selectbackground and makes text selection invisible inside code regions.
        t.tag_configure("inline_code", font=FM(11), foreground="#7DD3FC")
        t.tag_configure("code_block", font=FM(11),
                        foreground="#D1D5DB", lmargin1=16, lmargin2=16)
        t.tag_configure("bold",   font=F(12, "bold"))
        t.tag_configure("italic", font=F(12, "italic"))
        t.tag_configure("quote", foreground=COL["text_dim"], lmargin1=14, lmargin2=14)
        t.tag_configure("link",      foreground="#60A5FA", underline=True)   # no font: inherit surrounding weight
        t.tag_configure("table_row", font=FM(11), foreground=COL["text"])
        t.tag_configure("rule",      foreground=COL["border"], font=F(11))
        t.tag_configure("dim",   foreground=COL["text_dim"], font=F(11))
        t.tag_configure("error", foreground=COL["danger"], font=F(12, "bold"))

    def _update_topbar_labels(self) -> None:
        url = self.settings.get("server_url", "")
        model = self.settings.get("model_name", "")
        self.top_model_label.configure(text=f"{url} · {model}")
        self._update_temp_topbar()

    def _set_actual_model(self, model: str) -> None:
        """Topbar shows the model ID the server actually used (from stream chunks),
        not just the configured name - so it always matches reality."""
        if self._closed or not model:
            return
        try:
            url = self.settings.get("server_url", "")
            self.top_model_label.configure(text=f"{url} · {model}")
        except tk.TclError:
            pass

    def _update_temp_topbar(self) -> None:
        """Topbar shows the ACTIVE chat's temperature. No server round-trip is
        needed - we send this value with every request, so it is always known.
        Always a concrete number (blank/legacy -> TEMP_DEFAULT). In the welcome
        state (no active chat yet) it shows the pending value the user just set,
        so the topbar never snaps back to the default under the combobox."""
        try:
            chat = self.current_chat()
            if chat is None and getattr(self, "_pending_temperature", None) is not None:
                t = chat_temperature({"temperature": self._pending_temperature})
            else:
                t = chat_temperature(chat)
            # t is now always a float; show exactly what will be sent, never "Default"
            self.temp_top_label.configure(text=f"\U0001F321 {t}")
        except tk.TclError:
            pass

    def _update_ctx_topbar(self, entries: List[dict]) -> None:
        """Top-bar context readout (e.g. 'RULER 256K') from /models metadata.
        Prefers the configured model_name; falls back to the single loaded model.
        Servers that don't report context_length (LM Studio, Ollama) leave it blank."""
        try:
            want = (self.settings.get("model_name") or "").strip()
            entry = None
            for m in entries:
                if want and str(m.get("id", "")).strip() == want:
                    entry = m
                    break
            if entry is None:
                loaded = [m for m in entries if m.get("loaded")]
                if len(loaded) == 1:
                    entry = loaded[0]
            raw_total = entry.get("context_length") if entry else ""
            try:
                self._ctx_total = int(raw_total) if raw_total else 0
            except (TypeError, ValueError):
                self._ctx_total = 0
            self._update_ctx_usage_label()
        except tk.TclError:
            pass

    def _update_ctx_usage_label(self) -> None:
        """Render the top-bar context-usage readout from stored used/total.
        Shows "used/max" once a step has reported prompt_tokens; until then
        (or when no usage is available) it falls back to just the max window."""
        try:
            text = fmt_ctx_used(self._ctx_used, self._ctx_total)
            if not text and self._ctx_total:
                text = f"\U0001F4CF {_fmt_ctx(self._ctx_total)}"
            self.ctx_top_label.configure(text=text)
        except tk.TclError:
            pass

    def _set_ctx_used(self, used: int) -> None:
        """Main-thread setter for the session's current prompt_tokens.
        Called via self._post from the worker after each model step."""
        try:
            u = int(used) if used else 0
        except (TypeError, ValueError):
            u = 0
        self._ctx_used = u
        self._update_ctx_usage_label()

    # ── Session handoff (auto-summary at context threshold) ────────────────────

    def _handoff_rearm_needed(self, chat: dict, used: int, total: int) -> bool:
        """May this chat produce (another) handoff summary at "used" tokens?

        A chat that has never handed off fires on the first crossing. One that already
        has a summary re-fires only once the gauge has grown by handoff_rearm_pct percent
        of the window since that summary was written - so a long agentic turn can refresh
        the notes several times without a summary request after literally every step.
        The old one-shot guard (handoff_summary present -> never again) is what made a
        second overflow in the same chat unsalvageable."""
        if total <= 0:
            # No measurable window (server reports no context_length). The re-arm margin is
            # a percentage OF the window, so with no window there is no defensible trigger -
            # refuse rather than summarize at every step boundary. Both callers already gate
            # on total > 0; this exists so a future caller cannot divide by zero.
            return False
        prev = int(chat.get("handoff_used") or 0)
        if prev <= 0:
            return True                     # no summary for this chat yet
        rearm = _handoff_rearm_pct(self.settings)
        if rearm <= 0:
            return True
        return (max(0, used - prev) / float(total)) * 100.0 >= float(rearm)

    def _maybe_trigger_handoff(self) -> None:
        """After a completed turn: summarize if this chat crossed the configured
        context-usage threshold (re-arming as the gauge keeps growing). Inert when the
        server reports no context size."""
        if self._closed or self._busy:
            return
        total = int(self._ctx_total or 0)
        used = int(self._ctx_used or 0)
        pct_set = _handoff_threshold_pct(self.settings)
        if pct_set <= 0 or total <= 0 or used <= 0:
            return
        if (used / total) * 100.0 < float(pct_set):
            return
        cid = self.current_chat_id
        if not cid:
            return
        chat = self.chats["chats"].get(cid)
        if not chat or not chat.get("messages"):
            return
        # Persistent guard (restart case): a saved summary is not regenerated until the
        # gauge has grown by the re-arm margin - see _handoff_rearm_needed(). This is the ONLY
        # repeat-suppression rule. The old in-memory "already handoffed this run" set defeated
        # it: one mid-turn handoff silenced the end-of-turn auto-summary for that chat until
        # restart - precisely when a marathon turn needs refreshed notes.
        if not self._handoff_rearm_needed(chat, used, total):
            return
        self._handoff_start(cid, int(self._ctx_used or 0), int(self._ctx_total or 0))

    def _ctx_window(self) -> int:
        """Context window size as seen from a worker thread.

        _ctx_total is main-thread state; read through getattr so half-built instances
        (tests that skip __init__) never raise AttributeError inside the agentic loop."""
        try:
            return int(getattr(self, "_ctx_total", 0) or 0)
        except (TypeError, ValueError):
            return 0

    def _begin_midturn_handoff(self, chat_id: str, used: int, total: int, why: str,
                               force: bool = False) -> bool:
        """Called from the worker at a step boundary: cut this turn short and write
        handoff notes instead of pushing another request into an almost-full window.

        Runs on the worker thread but only reads plain ints / settings (never a widget),
        so the decision itself needs no _post; everything that touches the UI is queued.
        Returns True when the caller must end the turn. Deliberately inert - no note, no
        summary - when the feature is off or the chat has nothing to summarize, in which
        case the turn continues exactly as before.

        force=True (server itself refused the request because the window is full) skips the
        gauge comparison and the re-arm margin: there is no point estimating how full a
        window the server has already declared full, and LM Studio / Ollama report no
        context_length at all, so total is 0 and the gauge-based gate could never open.
        Without this, an overflow on those servers ended the turn with a bare error and
        nothing saved - exactly the failure this feature exists to prevent. A threshold of 0
        still means OFF (the user's explicit switch is never overridden), and the chat must
        still have messages to summarize; _handoff_worker handles a zero total."""
        if self._closed or self._stop_event.is_set():
            return False
        pct_set = _handoff_threshold_pct(self.settings)
        if pct_set <= 0:
            return False          # feature explicitly off (Settings -> Handoff threshold % = 0)
        if not force:
            # Gauge-based trigger: needs a measurable window and a real usage figure.
            if total <= 0 or used <= 0:
                return False
            if (used / total) * 100.0 < float(pct_set):
                return False
        chat = self.chats["chats"].get(chat_id)
        if not chat or not chat.get("messages"):
            return False
        if not force and not self._handoff_rearm_needed(chat, used, total):
            return False
        # No "already done" bookkeeping here on purpose: a mid-turn handoff must not silence
        # the end-of-turn auto-handoff (see _maybe_trigger_handoff). Duplicate suppression is
        # purely temporal - _handoff_start() refuses a SECOND concurrent worker for the same
        # chat; re-arming after that is governed by the gauge plus the re-arm margin.
        pend = {"chat_id": chat_id, "used": int(used), "total": int(total), "why": why}
        self._post(lambda p=dict(pend): self._ui_finish_for_handoff(p))
        return True

    def _ui_finish_for_handoff(self, info: dict) -> None:
        """Main thread: end a turn that was stopped on purpose to write handoff notes,
        then start the summary request. _finish_turn_ui() is what releases _busy
        (invariant 10), so it runs before anything that could go wrong."""
        if self._closed:
            return
        cid = info.get("chat_id")
        used = int(info.get("used") or 0)
        total = int(info.get("total") or 0)
        self._ctx_used = used
        self._update_ctx_usage_label()
        pct = _handoff_threshold_pct(self.settings)
        msg = (f"\U0001F4CF Context {_handoff_gauge_text(used, total)} ({pct}% threshold) - " +
               str(info.get("why") or "") + ". Stopping to write handoff notes\u2026")
        self.render_note(msg)
        # _finish_turn_ui() is what releases _busy (invariant 10), so it runs before anything
        # that could go wrong - and it now carries the informative text through instead of
        # stamping "Ready" over the reason the turn was cut short.
        self._finish_turn_ui(status=msg)
        if not cid:
            return
        self._handoff_start(cid, used, total)

    def _handoff_start(self, chat_id: str, used_at: int = 0, total_at: int = 0) -> bool:
        """Start ONE summary worker per chat; returns False when one is already running.

        Without this guard two threads could summarize the same chat at once (a mid-turn
        summary still in flight when the next turn ends) and both would write
        handoff_<chat_id>.md - last writer wins, with a .bak that looks intentional but is
        just a race. Only ONE suppression rule is temporal; whether a chat MAY summarize
        again is decided by _handoff_rearm_needed()."""
        lock = getattr(self, "_handoff_lock", None)
        if lock is None:                      # half-built instances (tests that skip __init__)
            lock = threading.Lock()
            self._handoff_lock = lock
        inflight = getattr(self, "_handoff_inflight", None)
        if inflight is None:
            inflight = set()
            self._handoff_inflight = inflight
        with lock:
            if chat_id in inflight:
                return False
            inflight.add(chat_id)
        threading.Thread(target=self._handoff_run,
                         args=(chat_id, int(used_at or 0), int(total_at or 0)), daemon=True).start()
        return True

    def _handoff_run(self, chat_id: str, used_at: int = 0, total_at: int = 0) -> None:
        """Thread body: run the summary worker, then ALWAYS release the in-flight slot.
        The release lives here (not inside _handoff_worker) so every exit path - including its
        early returns and any unexpected exception - frees the chat, and so a test that swaps
        in a fake worker is released exactly like the real one."""
        try:
            self._handoff_worker(chat_id, used_at, total_at)
        except Exception:
            # Same policy as the chat worker's safety net: report it, never let a summary
            # thread die with an unhandled exception (the notes file is best-effort).
            traceback.print_exc()
        finally:
            inflight = getattr(self, "_handoff_inflight", None)
            lock = getattr(self, "_handoff_lock", None)
            if inflight is not None and lock is not None:
                with lock:
                    inflight.discard(chat_id)

    def _handoff_worker(self, chat_id: str, used_at: int = 0, total_at: int = 0) -> None:
        """Background: ask the model to summarize this chat, save it to a file, and
        surface it in an accordion with Copy / Start-New-Session actions.
        The chat's own history is included (sanitized + trimmed) so the summary
        reflects what actually happened. used_at/total_at are the gauge values
        captured at trigger time (main thread)."""
        client = self._get_client()
        if client is None:
            return
        chat = self.chats["chats"].get(chat_id)
        if not chat:
            return
        title = chat.get("title") or "New Chat"
        model = (self.settings.get("model_name") or "gpt-4o-mini").strip()
        # The summary must reflect what actually happened: include the chat's own
        # history, sanitized so it stays small and works on non-vision models
        # (images -> text placeholders, tool outputs capped), and trimmed to fit
        # the context window alongside the prompt + summary output.
        # Snapshot the history (shallow copy) instead of iterating the live list:
        # _finish_turn_ui() released _busy before this thread started, so the user can
        # already be typing/sending. A CPython list does not raise on a concurrent append,
        # it would just silently pull the brand-new message into the summary.
        sanitized = _sanitize_for_handoff(list(chat.get("messages") or []))
        if not sanitized:
            return
        # Budget ladder. Start from the configured budget; if the SUMMARY request itself is
        # refused as too long, halve it and try again. Without this, the first attempt -
        # built from a history that just overflowed, and on a no-context_length server sized
        # only by the fallback constant - fails and the turn ends with 'Handoff summary
        # failed' and no notes at all.
        budget = _handoff_budget_chars(int(total_at or 0))
        attempts = 0
        summary = ""
        last_err = ""
        while True:
            history = _trim_for_handoff(sanitized, budget)
            kwargs: Dict[str, Any] = dict(
                model=model,
                messages=[
                    build_system_message(str(self.settings.get("file_workspace") or ""),
                                         str(self.settings.get("exa_api_key") or ""),
                                         str(self.settings.get("firecrawl_api_key") or ""),
                                         str(self.settings.get("custom_system_prompt") or "")),
                    *history,
                    {"role": "user", "content": _handoff_prompt(title)},
                ],
                stream=False,
                temperature=0,      # factual summary - no sampling noise
                max_tokens=HANDOFF_SUMMARY_MAX_TOKENS,   # some servers reject it -> retried without
            )
            try:
                try:
                    resp = client.chat.completions.create(**kwargs)
                except Exception as e:
                    emsg = str(e).lower()
                    if "max_tokens" in emsg or "max_completion_tokens" in emsg:
                        kwargs.pop("max_tokens", None)
                        resp = client.chat.completions.create(**kwargs)
                    else:
                        raise
                summary = (resp.choices[0].message.content or "").strip()
                break
            except Exception as e:
                last_err = str(e)
                # Overflow AND room left to shrink -> retry smaller. Any other failure (auth,
                # connection, dead server) is not fixed by a shorter history: stop trying.
                if (is_context_overflow_error(last_err)
                        and attempts < HANDOFF_RETRY_OVERFLOW
                        and budget // 2 >= HANDOFF_FALLBACK_MIN_CHARS):
                    attempts += 1
                    budget //= 2
                    self._post(lambda b=budget: self.render_note(
                        f"\U0001F4E6 Handoff summary too large - retrying with a smaller "
                        f"history budget ({b} chars)…"))
                    continue
                break

        if not summary:
            # Model-written summary unavailable: salvage the turn with a deterministic digest
            # (no model call) so Start New Session / Auto-Continue still have something to carry
            # over. An overflow is EXPECTED here (that is why this path exists) so it gets a note;
            # any other final failure - including an auth/connection error reached after a
            # successful shrink - is reported as an error too, because something is genuinely wrong.
            if not is_context_overflow_error(last_err):
                self._post(lambda m=last_err: self.render_error(f"Handoff summary failed:\n{m}"))
            fallback = _deterministic_handoff(chat, max(
                HANDOFF_FALLBACK_MIN_CHARS, min(budget, HANDOFF_FALLBACK_MAX_CHARS)))
            if not fallback:
                return
            summary = fallback
            self._post(lambda m=last_err: self.render_note(
                f"\U0001F4E6 Wrote deterministic handoff notes instead "
                f"(model summary unavailable: {m[:200]})"))

        # Write the notes to the working folder (best-effort; never blocks the UI). ONE FILE PER
        # HANDOFF - a new handoff never overwrites an earlier one, and the header lists this chat's
        # previous note files so a new session can reference the sessions it continues. Notes go to
        # Settings -> Handoff notes folder (blank = <script folder>/deskpilot_data/handoff_notes);
        # the AppData data store is NOT a notes location. _atomic_write() still keeps the immediately
        # previous content of THIS file as .bak (it exists for single-shot files too).
        file_path = ""
        try:
            notes_dir = _handoff_notes_dir(self.settings)
            if notes_dir is None:
                self._post(lambda d=str((self.settings or {}).get("handoff_notes_dir")
                                         or default_handoff_notes_dir()):
                           self.render_note(f"\U0001F4E6 Handoff summary was NOT written - the notes "
                                            f"folder is not usable: {d}"))
            else:
                prev = _handoff_previous_note_paths(notes_dir, chat_id)
                fp = _handoff_note_path(notes_dir, chat_id, title)
                header = _handoff_header(title, used_at, total_at,
                                         _handoff_threshold_pct(self.settings), chat_id, prev)
                _atomic_write(fp, header + summary + "\n")
                file_path = str(fp)
        except Exception:
            pass

        self._post(lambda s=summary, f=file_path, c=chat_id, u=used_at, t=total_at:
                   self._show_handoff(s, f, c, u, t))

    def _render_handoff_body(self, sid: str, summary: str) -> None:
        """Render a markdown-ish handoff summary into an accordion body (elidable).
        Every line carries the accordion body tag so collapsing hides it; no tag
        here sets a background (that would hide text selection - see invariants)."""
        body_tag = sid + "_body"
        for raw in summary.splitlines():
            s = raw.strip()
            if not s:
                self.chat_text.insert("end", "\n", [body_tag])
                continue
            m = re.match(r"^#{1,6}\s+(.*)$", s)
            if m:
                self.chat_text.insert("end", m.group(1).strip() + "\n", [body_tag, "h3"])
                continue
            lm = re.match(r"^([-*+]|\d+[.)])\s+(.*)$", s)
            if lm:
                marker, rest = lm.groups()
                bullet = "\u2022  " if marker in "-*+" else f"{marker} "
                self.chat_text.insert("end", bullet, [body_tag, "dim"])
                render_inline(self.chat_text, rest, [body_tag, "dim"])
                self.chat_text.insert("end", "\n", [body_tag])
                continue
            render_inline(self.chat_text, s, [body_tag, "dim"])
            self.chat_text.insert("end", "\n", [body_tag])

    def _show_handoff(self, summary: str, file_path: str, chat_id: str,
                      used_at: int = 0, total_at: int = 0) -> None:
        """Render the handoff accordion (summary body + Copy / Start-New-Session).

        Persists the summary to the chat record FIRST so it survives a restart or a
        switch away from this chat: load_chat() re-renders it, and
        _maybe_trigger_handoff() will not regenerate one that is already saved."""
        chat = self.chats["chats"].get(chat_id)
        if chat:
            chat["handoff_summary"] = summary
            if used_at:
                chat["handoff_used"] = int(used_at)
            if total_at:
                chat["handoff_total"] = int(total_at)
            save_chats(self.chats)
        self._last_handoff[chat_id] = summary
        self._last_handoff_file[chat_id] = file_path   # v1.1.10: the note THIS handoff wrote
        if self._closed or self.current_chat_id != chat_id:
            return
        pct = _handoff_threshold_pct(self.settings)
        # Live gauge values when available; fall back to the persisted ones on a
        # re-render (e.g. after a restart, where the gauge has been reset).
        used = int(self._ctx_used or 0) or int((chat or {}).get("handoff_used") or 0)
        total = int(self._ctx_total or 0) or int((chat or {}).get("handoff_total") or 0)
        title = (f"\U0001F4E6 Session Handoff \u00b7 {_handoff_gauge_text(used, total)} "
                 f"({pct}% threshold)")
        sid = self._add_accordion(title, COL["warning"], expanded=True)
        self._render_handoff_body(sid, summary)
        # Action buttons on their own line (NOT elided - stay available when the
        # summary body is collapsed).
        try:
            idx = self.chat_text.index("end-1c")
            self.chat_text.insert(idx, "\n")
            b0 = self.chat_text.index("end-1c")
            copy_btn = tk.Button(
                self.chat_text, text="\U0001F4CB Copy Summary",
                command=lambda s=summary: self.copy_to_clipboard(s),
                bg=COL["bg_raised"], fg=COL["text"], activebackground=COL["accent"],
                activeforeground="#FFFFFF", relief="flat", bd=0, padx=10, pady=3,
                cursor="hand2", font=F(11))
            self._code_buttons.append(copy_btn)
            self.chat_text.window_create(b0, window=copy_btn)
            self.chat_text.insert("end", "   ")
            b1 = self.chat_text.index("end-1c")
            new_btn = tk.Button(
                self.chat_text, text="\u2795 Start New Session",
                command=lambda c=chat_id: self._start_new_session(c),
                bg=COL["accent"], fg="#FFFFFF", activebackground="#2563EB",
                relief="flat", bd=0, padx=10, pady=3, cursor="hand2", font=F(11))
            self._code_buttons.append(new_btn)
            self.chat_text.window_create(b1, window=new_btn)
            self.chat_text.insert("end", "   ")
            b2 = self.chat_text.index("end-1c")
            go_btn = tk.Button(
                self.chat_text, text="\u2795\u26a1 Auto-Continue",
                command=lambda c=chat_id: self._auto_continue_session(c),
                bg=COL["success"], fg="#FFFFFF", activebackground="#059669",
                relief="flat", bd=0, padx=10, pady=3, cursor="hand2", font=F(11))
            self._code_buttons.append(go_btn)
            self.chat_text.window_create(b2, window=go_btn)
            self.chat_text.insert("end", "\n")
        except Exception:
            pass
        if file_path:
            self.render_note(f"Handoff saved to: {file_path}")
        self._autoscroll()

    def _auto_continue_session(self, source_chat_id: str) -> None:
        """Handoff accordion's Auto-Continue: start the fresh session AND immediately ask
        the model to resume the task, so a long job survives its own context limit without
        the user re-typing anything. Refuses while a turn is running (_busy) and when no
        summary exists - it never silently starts an empty session."""
        if self._closed or self._busy:
            self._set_status("Wait for the current turn to finish before auto-continuing")
            return
        src = self.chats["chats"].get(source_chat_id)
        if not src:
            messagebox.showinfo(APP_NAME, "The source chat of this handoff no longer exists.")
            return
        # A missing summary is NOT checked here: _start_new_session validates BOTH sources (the
        # persisted / in-memory summary and the chat note files), so a notes-only handoff is not a
        # dead end and the manual Start New Session button keeps exactly the same contract.
        self._start_new_session(source_chat_id, auto_send=True)

    def _start_new_session(self, source_chat_id: str, auto_send: bool = False) -> None:
        """Open a fresh chat seeded with the handoff summary AND the handoff NOTES of
        `source_chat_id` (notes folder + its newest note files), so the model is instantly up to
        speed, the token count resets, and the full written record stays reachable. Refuses only
        when BOTH sources are empty."""
        src = self.chats["chats"].get(source_chat_id)
        if not src:
            return
        # Persistent chat record first (survives restarts), in-memory cache as fallback.
        summary = src.get("handoff_summary") or self._last_handoff.get(source_chat_id, "")
        notes_intro, note_paths = _handoff_notes_context(
            self.settings, source_chat_id, self._last_handoff_file.get(source_chat_id, ""))
        if not summary and not note_paths:
            messagebox.showinfo(
                APP_NAME,
                "No handoff summary is available to carry over, and this chat has no handoff "
                "notes (its notes folder appears to be empty).")
            return
        self.new_chat()
        fresh = self.current_chat()
        if fresh is None:
            return
        if summary:
            intro = ("[Session Handoff] The previous session reached its context limit. "
                     "Here is a summary of the work so far - continue from here:\n\n" + summary)
        else:
            # Notes-only handoff (the condensed summary is gone, e.g. a chat record written before
            # summaries were persisted): announce that plainly instead of promising a summary that is
            # not there, and point at the notes as the only record.
            intro = ("[Session Handoff] The previous session reached its context limit. Its condensed"
                     " summary is unavailable, so the handoff notes below are the ONLY record of the"
                     " work done so far - read them before continuing.")
        # v1.1.10: the notes FOLDER and this chat's newest note files travel WITH the summary, so
        # the continued session (and any assistant with file tools there) can open them and read the
        # FULL record of every earlier session - the accordion summary is only what the model condensed
        # at handoff time, not everything that was done. The folder is named even when this chat has no
        # note file yet (a legacy summary-only handoff), so the append instruction below always names a
        # concrete destination and never dangles on a file that was not listed.
        intro += "\n\nHandoff notes on disk: " + notes_intro + (
            "\nIf this folder is readable from your tools, open these files for the full record of"
            " past sessions; if it is not reachable, say so once and continue with the summary above."
            if note_paths else
            "\nIf this folder is readable from your tools, list it and read any note file belonging to"
            " the chat being continued; if it is not reachable, say so once.")
        if auto_send:
            # One user message only (summary + notes references + the resume instruction together),
            # so the new chat never opens with two consecutive user turns.
            intro += ("\n\nResume the task described above now: pick up at 'Next steps', keep to"
                      " the constraints already established, do not repeat work listed under 'What"
                      " was done', and append a dated section (date + what changed + next steps) to"
                      " the newest note file named above" +
                      ("" if note_paths else " (create one in that folder if none exists)") +
                      " if that folder is writable.")
        fresh["messages"].append({"role": "user", "content": intro})
        base = (src.get("title") or "Session").strip()[:24]
        fresh["title"] = f"Continued: {base}"
        save_chats(self.chats)
        self._refresh_chat_list()
        self.render_user_message(intro)
        # Files that travel with the handoff are counted; the folder is named in EVERY seeded intro
        # (a chat with no note file yet still needs its destination spelled out), so say "folder only".
        self._set_status("\u2795 New session started from handoff summary - notes " + (
            "folder + %d note file(s) referenced" % len(note_paths) if note_paths
            else "folder referenced (no note file for this chat yet)"))
        if auto_send and not self._busy:
            # Deliberately NOT send_message(): the input box is empty and that path would
            # append a second user message. Start the worker on the seeded history instead
            # - same button/busy bookkeeping as any turn, so _finish_turn_ui still owns it.
            self._busy = True
            self._stop_event.clear()
            self.send_btn.configure(text="\u23f9 Stop", bg=COL["danger"], activebackground="#B91C1C",
                                    command=self.stop_generation)
            self.speed_label.configure(text="\u26a1 \u2026")
            cid = self.current_chat_id
            self._run_chat_id = cid
            threading.Thread(target=self._chat_worker,
                             args=(cid, list(fresh["messages"])), daemon=True).start()

    def _refresh_ctx_topbar(self) -> None:
        """Background-fetch {server_url}/models and refresh the top-bar context
        readout. Non-blocking; called on startup, after settings changes, and at the
        start of every send (so a server-side context change is picked up without a
        Deskpilot restart)."""
        url = (self.settings.get("server_url") or "").strip()
        if not url:
            return
        key = (self.settings.get("api_key") or "").strip()

        def _worker():
            info = fetch_model_info(url, key)
            self._post(lambda: self._update_ctx_topbar(info))

        threading.Thread(target=_worker, daemon=True).start()

    # ════════════════════════════════════════════════════════════════════
    #  WINDOWS DARK TITLE BAR (DwmSetWindowAttribute)
    # ════════════════════════════════════════════════════════════════════

    def apply_dark_titlebar(self) -> None:
        if sys.platform != "win32":
            return
        try:
            import ctypes
            self.root.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            if not hwnd:
                return
            value = ctypes.c_int(1)
            dwmapi = ctypes.windll.dwmapi
            # DWMWA_USE_IMMERSIVE_DARK_MODE = 20 (Win11 / late Win10), 19 on early builds
            for attr in (20, 19):
                try:
                    dwmapi.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(value),
                                                  ctypes.sizeof(value))
                except Exception:
                    pass
        except Exception:
            pass

    # ════════════════════════════════════════════════════════════════════
    #  CHAT MANAGEMENT (threads / persistence)
    # ════════════════════════════════════════════════════════════════════

    def current_chat(self) -> Optional[dict]:
        if self.current_chat_id:
            return self.chats["chats"].get(self.current_chat_id)
        return None

    def selected_chat_id(self) -> Optional[str]:
        sel = self.chat_list.curselection()
        if sel and 0 <= sel[0] < len(self._chat_visible):
            return self._chat_visible[sel[0]]
        return self.current_chat_id

    def new_chat(self, create_only: bool = False) -> None:
        cid = uuid.uuid4().hex[:10]
        self.chats["chats"][cid] = {
            "title":   "New Chat",
            "created": datetime.now().isoformat(timespec="seconds"),
            "temperature": "",     # per-chat; blank/legacy -> TEMP_DEFAULT (always a real number, always sent)
            "thinking":    "High",  # per-chat level; Off/Low/Medium/High -> enable_thinking / reasoning_effort
            "messages": [],
        }
        # Consume any welcome-state pending values: they describe the
        # conversation the user is about to start, so the new chat inherits them.
        if self._pending_temperature is not None:
            self.chats["chats"][cid]["temperature"] = self._pending_temperature
            self._pending_temperature = None
        if self._pending_thinking is not None:
            self.chats["chats"][cid]["thinking"] = self._pending_thinking
            self._pending_thinking = None
        self.chats["order"].append(cid)
        save_chats(self.chats)
        self._refresh_chat_list()
        if not create_only:
            self.load_chat(cid, force=True)

    #  Per-chat temperature -------------------------------------------------

    def _on_temp_selected(self) -> None:
        """User picked a temperature for the active chat; persist immediately.

        In the fresh-launch welcome state there is no active chat yet, so the
        value is held as _pending_temperature and applied to the brand-new
        conversation that the next send starts (see send_message) - otherwise
        the first request would go out at the default (the "set it twice" bug)."""
        chat = self.current_chat()
        if chat is not None:
            chat["temperature"] = str(self.temp_combo.get())
            save_chats(self.chats)
        else:
            self._pending_temperature = str(self.temp_combo.get())
        self._update_temp_topbar()

    def _sync_temp_combo(self, chat: Optional[dict]) -> None:
        """Point the combobox at a chat's saved temperature (blank/legacy -> TEMP_DEFAULT)."""
        val = (chat or {}).get("temperature", "") or ""
        self.temp_combo.set(val if val in TEMP_VALUES else TEMP_DEFAULT)

    #  Per-chat thinking toggle ------------------------------------------------

    def _on_thinking_selected(self) -> None:
        """User picked a thinking level for the active chat; persist immediately.

        In the fresh-launch welcome state there is no active chat yet, so the
        value is held as _pending_thinking and applied to the brand-new
        conversation that the next send starts (see send_message)."""
        chat = self.current_chat()
        if chat is not None:
            chat["thinking"] = str(self.think_combo.get())
            save_chats(self.chats)
        else:
            self._pending_thinking = str(self.think_combo.get())

    def _sync_thinking_combo(self, chat: Optional[dict]) -> None:
        """Point the combobox at a chat's saved thinking level (blank/legacy -> High)."""
        val = str((chat or {}).get("thinking", "") or "").strip()
        self.think_combo.set(val if val in THINK_VALUES else "High")

    def rename_chat(self) -> None:
        cid = self.selected_chat_id()
        chat = self.chats["chats"].get(cid) if cid else None
        if not chat:
            messagebox.showinfo(APP_NAME, "Select a conversation to rename.")
            return
        new_title = simpledialog.askstring(
            "Rename Chat", "New name:", initialvalue=chat.get("title", ""), parent=self.root)
        if new_title and new_title.strip():
            chat["title"] = new_title.strip()[:60]
            save_chats(self.chats)
            self._refresh_chat_list()

    def delete_chat(self) -> None:
        cid = self.selected_chat_id()
        chat = self.chats["chats"].get(cid) if cid else None
        if not chat:
            messagebox.showinfo(APP_NAME, "Select a conversation to delete.")
            return
        if not messagebox.askyesno(
                APP_NAME, f"Delete “{chat.get('title')}” and its full history? Its images/attachments will be deleted too.", parent=self.root):
            return
        # If this chat owns the in-flight turn, abort it BEFORE removing the
        # record. Otherwise the worker keeps streaming and burning tokens on a
        # conversation that no longer exists (its _ui_sync_messages() writes are
        # silently dropped by the deleted-chat guard, so nothing is saved).
        # stop_generation() only checks _busy, which can belong to a DIFFERENT
        # chat - so set the event directly here.
        stopping_run = bool(self._busy) and self._run_chat_id == cid
        if stopping_run:
            self._stop_event.set()
            self._set_status("\u23f9 Stopping deleted chat\u2026")
        # ── Reclaim this chat's image files (backlog #4) ─────────────────────
        # Screenshots and attachments live in GEN_DIR as orphan-proof "image_ref"
        # paths. Nothing else can render them once the record is gone, so they are
        # deleted here instead of piling up forever. The in-flight worker was already
        # stopped above (and tools now honour Stop within STOP_POLL_S), and
        # _ui_sync_messages() drops writes to a missing record, so no new reference
        # can appear after the pop. Files are unlinked AFTER save_chats(): if
        # something dies mid-deletion the persisted history never points at a
        # missing file (a lost image degrades to a text placeholder, not a crash).

        self.chats["chats"].pop(cid, None)
        if cid in self.chats["order"]:
            self.chats["order"].remove(cid)
        save_chats(self.chats)

        # Now that the record is gone from disk too, remove the files it owned.
        # Guard: skip any path a REMAINING chat still references (should not happen -
        # every capture/attach writes a unique name - but never delete data another
        # conversation can still render).
        try:
            keep = _referenced_image_keys(self.chats)
        except Exception:
            keep = set()
        removed = skipped = 0
        for fp in _chat_image_paths(chat):
            try:
                if str(fp.resolve()) in keep:
                    continue          # another conversation can still render it
                if not _path_in_image_store(fp):
                    skipped += 1      # outside the image stores - never touch user files
                    continue
                if fp.is_file():
                    fp.unlink()
                    removed += 1
            except Exception:
                pass      # a locked/missing file must not abort the deletion itself
        if removed or skipped:
            _txt = "Deleted " + str(removed) + " image file(s) with this chat"
            if skipped:
                _txt += " (" + str(skipped) + " outside the image store left alone)"
            self._set_status(_txt)

        if not self.chats["order"]:
            self.new_chat()
        else:
            # NOTE: _run_chat_id deliberately keeps pointing at the deleted chat.
            # _viewing_run_chat() returns True whenever it is None, so clearing it
            # would let the dying worker stream into whatever chat loads next;
            # leaving it set makes every render guard return False instead.
            was_current = (cid == self.current_chat_id)
            self._refresh_chat_list()
            if was_current or cid == self.current_chat_id:
                self.load_chat(self.chats["order"][0], force=True)

    # ── sidebar drag-and-drop reordering ─────────────────────────────

    def _chat_list_press(self, event) -> None:
        """Record where the press started; it becomes a drag only after
        the pointer moves more than 6px (plain clicks keep working).""" 
        try:
            idx = self.chat_list.nearest(event.y)
        except tk.TclError:
            self._drag_state = None
            return
        if self._chat_search_query():
            self._drag_state = None      # reordering is disabled while a search filter is active
            return
        if not (0 <= idx < len(self._chat_visible)):
            self._drag_state = None
            return
        self._drag_state = {"cid": self._chat_visible[idx], "y0": event.y, "active": False}

    def _chat_list_drag(self, event) -> None:
        """While dragging, move the grabbed chat to the row under the pointer."""
        st = self._drag_state
        if not st:
            return
        if not st["active"]:
            if abs(event.y - st["y0"]) < 6:
                return                      # still a plain click
            st["active"] = True
            try:
                self.chat_list.configure(cursor="fleur")
            except tk.TclError:
                pass
        try:
            idx = self.chat_list.nearest(event.y)
        except tk.TclError:
            return
        if not (0 <= idx < len(self._chat_visible)):
            return
        order = self.chats["order"]
        cur = order.index(st["cid"])
        if idx != cur:
            order.insert(idx, order.pop(cur))
            self._refresh_chat_list(select_cid=st["cid"])   # keep the dragged row selected

    def _chat_list_release(self, event) -> None:
        """End of press/drag: persist the new order if a drag actually happened."""
        st = self._drag_state
        self._drag_state = None
        if not st:
            return
        try:
            self.chat_list.configure(cursor="")
        except tk.TclError:
            pass
        if st["active"]:
            save_chats(self.chats)          # persist the new order
            self._set_status("\U0001F4AC Chat order updated")

    def _on_chat_selected(self) -> None:
        if self._drag_state and self._drag_state.get("active"):
            return                          # don't reload chats mid-drag
        cid = self.selected_chat_id()
        if cid and cid != self.current_chat_id:
            self.load_chat(cid)

    def load_chat(self, cid: str, force: bool = False) -> None:
        chat = self.chats["chats"].get(cid)
        if not chat or (not force and cid == self.current_chat_id):
            return
        self.current_chat_id = cid
        # clear display + embedded widgets
        try:
            self.chat_text.delete("1.0", "end")
        except tk.TclError:
            pass
        self._code_buttons.clear()
        self._image_thumbs.clear()
        self._links.clear()
        self._accordions.clear()
        self._active_md = None
        self._reasoning_sid = None
        self._pending_tool_output = None
        self._reasoning_sids.clear()
        self._ctx_used = 0          # fresh view of this chat: no usage known yet
        self._update_ctx_usage_label()
        msgs = chat.get("messages", [])
        if not msgs:
            self._render_welcome()
        else:
            for m in msgs:
                if not isinstance(m, dict):
                    continue      # corrupt entry - skip (rendering never raises)
                role = m.get("role")
                if role == "user":
                    self.render_user_message(m.get("content"))
                elif role == "assistant":
                    content = m.get("content")
                    tcs = m.get("tool_calls") or []
                    if content:
                        self.render_markdown_full(content)
                    for tc in tcs:
                        try:
                            a = json.loads(tc["function"].get("arguments", "{}") or "{}")
                        except Exception:
                            a = {}
                        self._ui_tool_call_accordion(tc["function"]["name"], a)
                elif role == "tool":
                    self._ui_tool_output_accordion(m.get("name", "tool"), m.get("content", ""))
        self._stick_bottom = True
        self.chat_text.see("end")
        self._sync_temp_combo(chat)
        self._sync_thinking_combo(chat)
        self._update_temp_topbar()
        # A real conversation is now active: welcome-state pending values are stale.
        self._pending_temperature = None
        self._pending_thinking = None
        # Restore a previously generated handoff accordion (survives chat switches;
        # _show_handoff re-persists harmlessly and refreshes the in-memory cache).
        if chat.get("handoff_summary"):
            self._show_handoff(chat["handoff_summary"], "", cid,
                               int(chat.get("handoff_used") or 0),
                               int(chat.get("handoff_total") or 0))

    def _refresh_chat_list(self, select_cid: Optional[str] = None) -> None:
        lb = self.chat_list
        lb.delete(0, "end")
        q = self._chat_search_query().lower()
        self._chat_visible = [cid for cid in self.chats["order"]
                              if not q or q in (self.chats["chats"][cid].get("title", "Untitled") or "Untitled").lower()]
        for cid in self._chat_visible:
            title = self.chats["chats"][cid].get("title", "Untitled") or "Untitled"
            lb.insert("end", f"  {title}")   # leading spaces = item padding
        want = select_cid or self.current_chat_id
        # Selection indices are LISTBOX rows = positions in _chat_visible, never
        # chats["order"].index(): while a search filter is active the two disagree
        # and the wrong row (or none) gets highlighted. A chat hidden by the
        # filter simply leaves the listbox with no selection.
        if want and want in self._chat_visible:
            idx = self._chat_visible.index(want)
            lb.selection_clear(0, "end")
            lb.selection_set(idx)
            lb.see(idx)

    # ── chat search (sidebar filter) ─────────────────────────────

    def _chat_search_query(self) -> str:
        """The active search text; the placeholder counts as empty."""
        try:
            q = self.chat_search.get().strip()
        except tk.TclError:
            return ""
        if not q or q == "🔎 Search chats…":
            return ""
        return q

    def _on_chat_search(self, _event=None) -> None:
        """Re-filter the sidebar list as the user types (title match)."""
        try:
            q = self._chat_search_query().lower()
            self._refresh_chat_list()
            if q:
                self.chat_search_results.configure(
                    text=f"{len(self._chat_visible)} of {len(self.chats['order'])} chats")
            else:
                self.chat_search_results.configure(text="")
        except tk.TclError:
            pass

    def _chat_search_enter(self) -> None:
        """Enter in the search box: open the selected (or first) match, then clear."""
        try:
            sel = self.chat_list.curselection()
            idx = sel[0] if sel else 0
            cid = self._chat_visible[idx]
        except (tk.TclError, IndexError):
            return
        self._clear_chat_search()
        self.load_chat(cid, force=True)

    def _clear_chat_search(self) -> None:
        """Restore the placeholder + full list."""
        try:
            self.chat_search.delete(0, "end")
            self.chat_search.insert(0, "🔎 Search chats…")
            self.chat_search.configure(fg=COL["text_dim"])
        except tk.TclError:
            return
        self._on_chat_search()

    def _chat_search_focus_in(self, _event=None) -> None:
        """Drop the placeholder on focus so typing starts a fresh query."""
        try:
            if str(self.chat_search.get()).strip() in ("", "🔎 Search chats…"):
                self.chat_search.delete(0, "end")
                self.chat_search.configure(fg=COL["text"])
            else:
                self.chat_search.selection_range(0, "end")
        except tk.TclError:
            pass

    def _chat_search_focus_out(self, _event=None) -> None:
        """Show the placeholder again once the box is empty and unfocused."""
        try:
            if self._chat_search_query() == "":
                self.chat_search.delete(0, "end")
                self.chat_search.insert(0, "🔎 Search chats…")
                self.chat_search.configure(fg=COL["text_dim"])
        except tk.TclError:
            pass

    # ── chat export (markdown) ─────────────────────────────────────

    def _chat_to_markdown(self, chat: dict) -> str:
        """Render one chat as a standalone markdown document."""
        title = chat.get("title") or "Untitled"
        created = str(chat.get("created") or "")[:10]
        out = [f"# {title}", ""]
        if created:
            out += [f"_Exported from {APP_NAME} · started {created}_", ""]
        for m in chat.get("messages") or []:
            role = m.get("role")
            if role == "user":
                out.append("## You")
                c = m.get("content")
                if isinstance(c, str):
                    out += [c, ""]
                elif isinstance(c, list):
                    for p in c:
                        if not isinstance(p, dict):
                            continue
                        pt = p.get("type")
                        if pt == "text":
                            out += [str(p.get("text", "")), ""]
                        elif pt in ("image_url", "image_ref"):
                            name = str(p.get("path", "")).replace("\\", "/").rsplit("/", 1)[-1]
                            out += [f"[image: {name or 'attached'}]", ""]
            elif role == "assistant":
                content = m.get("content")
                if content:
                    out += ["## Deskpilot", str(content), ""]
                for tc in (m.get("tool_calls") or []):
                    try:
                        fn = tc.get("function", {}) or {}
                        name = fn.get("name", "tool")
                        args = str(fn.get("arguments") or "{}").strip() or "{}"
                        try:
                            args = json.dumps(json.loads(args), indent=2, ensure_ascii=False)
                        except Exception:
                            pass
                    except Exception:
                        name, args = "tool", "{}"
                    out += [f"### ⚙️ Tool call: `{name}`", "",
                            "```json", args, "```", ""]
            elif role == "tool":
                out += [f"### 📦 Result: {m.get('name', 'tool')}", "",
                        "```", str(m.get("content") or ""), "```", ""]
        return "\n".join(out).rstrip() + "\n"

    def export_chat(self) -> None:
        """Save the selected chat as a markdown file (asks where to put it)."""
        cid = self.selected_chat_id()
        chat = self.chats["chats"].get(cid) if cid else None
        if not chat:
            messagebox.showinfo(APP_NAME, "Select a conversation to export.")
            return
        title = (chat.get("title") or "chat").strip().replace("/", "-")[:60] or "chat"
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        path = filedialog.asksaveasfilename(
            parent=self.root, title="Export Chat",
            initialfile=f"{title}_{stamp}.md", defaultextension=".md",
            filetypes=[("Markdown", "*.md"), ("Text", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            Path(path).write_text(self._chat_to_markdown(chat), encoding="utf-8")
        except OSError as e:
            messagebox.showerror(APP_NAME, f"Export failed:\n{e}")
            return
        self._set_status(f"📥 Exported to {Path(path).name}")

    def _render_welcome(self) -> None:
        self.chat_text.insert("end", "\n")
        self.chat_text.insert("end", f"◆ {APP_NAME}\n", ["asst_hdr"])
        render_inline(
            self.chat_text,
            "Hi! I'm **Deskpilot**, your local AI assistant. Ask me anything — I can search the web, "
            "fetch pages, run JavaScript, read & write files, generate images and capture your screen.",
            ["body"])
        self.chat_text.insert("end", "\nTip: configure each tool in the 🔐 permissions bar below.\n", ["dim"])

    # ════════════════════════════════════════════════════════════════════
    #  RENDERING (main thread only)
    # ════════════════════════════════════════════════════════════════════

    def render_user_message(self, content: Any) -> None:
        self.chat_text.insert("end", "\n")
        self.chat_text.insert("end", "You\n", ["user_hdr"])
        if isinstance(content, str):
            render_inline(self.chat_text, content, ["body"])
            self.chat_text.insert("end", "\n")
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    render_inline(self.chat_text, part.get("text", ""), ["body"])
                    self.chat_text.insert("end", "\n")
                elif ptype == "image_url":
                    url = str((part.get("image_url") or {}).get("url", ""))
                    if url.startswith("data:image/") and _PILImage is not None:
                        try:
                            img = _PILImage.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
                            img.load()
                            if not self._embed_pil_image(img):
                                raise ValueError("embed failed")
                        except Exception:
                            self.chat_text.insert("end", "🖼 Image attached\n", ["dim"])
                    else:
                        self.chat_text.insert("end", "🖼 Image attached\n", ["dim"])
                elif ptype == "image_ref":
                    self._insert_image_thumb(str(part.get("path", "")))

        self._autoscroll()

    def render_markdown_full(self, text: str) -> None:
        """Render a complete markdown document (used when replaying history)."""
        self.chat_text.insert("end", "\n")
        self.chat_text.insert("end", f"{APP_NAME}\n", ["asst_hdr"])
        md = MarkdownStream(self)
        md.body_tag = "asst_body"                # history replay matches live rendering
        md.feed(text)
        md.finish()

    def _begin_assistant_block(self) -> None:
        # The thinking phase is over: collapse its drawer(s) and make
        # sure any later step starts a fresh reasoning section.
        self._collapse_reasoning_sections()
        self._reasoning_sid = None
        self.chat_text.insert("end", "\n")
        self.chat_text.insert("end", f"{APP_NAME}\n", ["asst_hdr"])
        self._active_md = MarkdownStream(self)
        self._active_md.body_tag = "asst_body"   # responses render bold (LM Studio style)

    def render_error(self, text: str) -> None:
        if self._closed:
            return
        self.chat_text.insert("end", "\n⚠ " + text.replace("\n", " ")[:600] + "\n", ["error"])
        self._autoscroll()

    def render_note(self, text: str) -> None:
        if self._closed:
            return
        self.chat_text.insert("end", "\n" + text + "\n", ["dim"])
        self._autoscroll()

    # ── accordions (elide-based collapsible drawers) ───────────────────────

    def _add_accordion(self, title: str, color: str, expanded: bool, font=None) -> str:
        """Create a clickable accordion header. Body lines tagged with the
        returned section's body tag are hidden via elide=True when collapsed."""
        sid = "acc_" + uuid.uuid4().hex[:8]
        body_tag = sid + "_body"
        hdr_tag  = sid + "_hdr"
        self.chat_text.tag_configure(body_tag, elide=not expanded)
        # Tk 9 text tags reject -cursor; degrade gracefully (cosmetic only).
        try:
            self.chat_text.tag_configure(hdr_tag, cursor="hand2", foreground=color,
                                         font=font or F(12, "bold"))
        except tk.TclError:
            self.chat_text.tag_configure(hdr_tag, foreground=color,
                                         font=font or F(12, "bold"))

        idx = self.chat_text.index("end-1c")
        arrow = "▼ " if expanded else "▶ "
        a0 = idx
        self.chat_text.insert(idx, arrow, hdr_tag)
        a1 = f"{a0}+2c"
        self.chat_text.insert(a1, title + "    ", hdr_tag)
        self.chat_text.insert("end", "\n", hdr_tag)

        handler = lambda e, s=sid: self._toggle_accordion(s)
        self.chat_text.tag_bind(hdr_tag, "<Button-1>", handler)
        self._accordions[sid] = {"body_tag": body_tag, "a0": a0, "a1": a1,
                                 "expanded": expanded}
        return sid

    def _toggle_accordion(self, sid: str) -> None:
        info = self._accordions.get(sid)
        if not info:
            return
        expanded = not info["expanded"]
        info["expanded"] = expanded
        # elide=True hides the body text while keeping the layout stable
        self.chat_text.tag_configure(info["body_tag"], elide=not expanded)
        # swap ▶ / ▼ (both exactly 2 chars → no index shift anywhere)
        try:
            self.chat_text.delete(info["a0"], info["a1"])
            self.chat_text.insert(info["a0"], "▼ " if expanded else "▶ ")
        except tk.TclError:
            pass

    def _ensure_reasoning_section(self) -> str:
        if self._reasoning_sid and self._reasoning_sid in self._accordions:
            return self._reasoning_sid
        # LM Studio style: dim, non-bold header; the body stays visible while
        # the model is thinking. The title region has a FIXED width so it can
        # be swapped for "Thought for X.Xs" in place when the phase ends,
        # without shifting any stored text indices (Tk Text indices are
        # position-based).
        base = "Model Reasoning".ljust(REASONING_REGION_LEN)
        sid = self._add_accordion(base, COL["text_dim"], expanded=True,
                                  font=F(11))
        info = self._accordions[sid]
        a0 = info["a0"]
        info["r0"] = f"{a0}+2c"
        info["r1"] = f"{a0}+{2 + REASONING_REGION_LEN}c"
        info["started_at"] = time.time()
        self._reasoning_sids.append(sid)
        self._reasoning_sid = sid
        return sid

    def _ui_tool_call_accordion(self, name: str, args: dict) -> None:
        if not self._viewing_run_chat():
            return
        title = f"🛠 Tool Call · {name}({args_preview(args)})"
        sid = self._add_accordion(title, COL["accent"], expanded=False)
        body = json.dumps(args, ensure_ascii=False, indent=2)[:4000]
        lines = body.splitlines()[:12]
        if len(body.splitlines()) > 12:
            lines.append("… [truncated]")
        self.chat_text.insert("end", "\n".join(lines) + "\n", [sid + "_body", "code_block"])
        self._autoscroll()

    def _ui_tool_output_start(self, name: str) -> None:
        """Expanded 'running' section for a tool that is about to execute.

        It stays open while the tool works (live status glyph in the header);
        _ui_tool_output_accordion() finishes it in place and collapses it.
        """
        if not self._viewing_run_chat():
            return
        title = f"📤 Tool Output · {name}  🛠"
        sid = self._add_accordion(title, COL["warning"], expanded=True)
        info = self._accordions[sid]
        # NOTE (Tk 9): index("end") includes a phantom final newline - a captured end-index
        # string resolves one line early for delete(). Append at literal "end" and capture the
        # true placeholder boundaries via tag_ranges() instead.
        self.chat_text.insert("end", "Running…\n", [sid + "_body", "code_block"])
        r = self.chat_text.tag_ranges(sid + "_body")
        info["b0"], info["b1"] = str(r[0]), str(r[1])
        # Canonical position of the running glyph (🛠) for an in-place swap later.
        off = len("📤 Tool Output · ") + len(name) + 2
        try:
            base = self.chat_text.index(info["a1"])
            info["g0"] = self.chat_text.index(f"{base}+{off}c")
        except tk.TclError:
            info["g0"] = None
        self._pending_tool_output = sid
        self._autoscroll()
        return sid

    def _ui_tool_output_accordion(self, name: str, result: str) -> None:
        """Finish the running tool-output section (or create one if none exists)."""
        if not self._viewing_run_chat():
            return
        ok = not result.startswith("ERROR")
        lines = (result or "").splitlines()[:15]
        total = len((result or "").splitlines())
        body = "\n".join(lines)
        if total > 15:
            body += f"\n… [{total - 15} more lines truncated]"

        sid = self._pending_tool_output
        self._pending_tool_output = None
        info = self._accordions.get(sid or "")
        if not info or "b0" not in info:      # defensive fallback (no start section)
            title = f"📤 Tool Output · {name}" + ("  ✓" if ok else "  ✗")
            sid2 = self._add_accordion(title, COL["success"] if ok else COL["danger"], expanded=False)
            self.chat_text.insert("end", (body or "(empty)") + "\n", [sid2 + "_body", "code_block"])
            self._autoscroll()
            return

        # Replace the 'Running…' placeholder in place with the real result. This section is at
        # the tail of the widget, so nothing after it exists yet and no other stored index shifts.
        self.chat_text.delete(info["b0"], info["b1"])
        self.chat_text.insert(info["b0"], (body or "(empty)") + "\n", [info["body_tag"], "code_block"])
        # Swap the running glyph in place: 🛠 -> ✓/✗ (both 1 char - no index shift) and recolor.
        if info.get("g0"):
            try:
                self.chat_text.delete(info["g0"], f"{info['g0']}+1c")
                self.chat_text.insert(info["g0"], "✓" if ok else "✗")
            except tk.TclError:
                pass
        try:
            self.chat_text.tag_configure(sid + "_hdr",
                                         foreground=COL["success"] if ok else COL["danger"])
        except tk.TclError:
            pass
        self._toggle_accordion(sid)       # collapse (it was expanded while running)
        self._autoscroll()

    def _viewing_run_chat(self) -> bool:
        """True if the chat currently on screen is the one being streamed."""
        return (self._run_chat_id is None) or (self.current_chat_id == self._run_chat_id)

    # ── clipboard / scrolling helpers ───────────────────────────────────────

    # ── embedded image thumbnails (generated / captured / attached) ──

    def _embed_pil_image(self, img, path: Optional[str] = None) -> bool:
        """Scale a PIL image down and embed it at the end of the chat text.
        Returns False on any failure (caller falls back to a text line)."""
        try:
            img.thumbnail((THUMB_MAX_DIM, THUMB_MAX_DIM))
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            photo = tk.PhotoImage(data=buf.getvalue())   # Tk 8.6+ reads PNG natively
            idx = self.chat_text.index("end-1c")
            self.chat_text.image_create(idx, image=photo)
            self._image_thumbs.append((photo, idx, path))   # keep the ref alive (Tk drops it otherwise)
            self.chat_text.insert("end", "\n")
            return True
        except Exception:
            return False

    def _insert_image_thumb(self, path_str: str) -> None:
        """Embed a clickable thumbnail of a local image file (dim text fallback)."""
        if _PILImage is not None and Path(path_str).is_file():
            try:
                img = _PILImage.open(path_str)
                img.load()
                if self._embed_pil_image(img, path=path_str):
                    return
            except Exception:
                pass
        self.chat_text.insert("end", "🖼 " + (Path(path_str).name or 'image') + "\n", ["dim"])

    def _open_image_file(self, path: str) -> None:
        try:
            if sys.platform == "win32":
                os.startfile(path)                       # type: ignore[attr-defined]
            else:
                webbrowser.open(Path(path).as_uri())
        except Exception as e:
            self.render_note(f"Could not open {path}: {e}")

    def _image_at(self, x: int, y: int) -> Optional[str]:
        """Path of the embedded thumbnail under (x, y), or None.
        An embedded image occupies exactly one character position in the Text
        widget; chat content is append-only while a view is displayed and all
        accordion edits are same-length, so the index strings recorded at
        creation time stay valid."""
        try:
            idx = self.chat_text.index(f"@{x},{y}")
        except tk.TclError:
            return None
        for _photo, rec_idx, path in self._image_thumbs:
            if path and rec_idx == idx:
                return path
        return None

    # ── clickable links (Ctrl+click / middle-click / hover) ─────────

    def _link_at(self, index: str) -> Optional[str]:
        """URL whose tagged range contains the given Text index, else None.
        The registry is keyed by (line, char offset) so lines never collide."""
        try:
            parts = str(index).split(".")
            line, pos = int(parts[0]), int(parts[1])         # 1-based line, 0-based col
        except (ValueError, IndexError):
            return None
        best = None
        for (ln, off), url in self._links.items():
            if ln == line and off <= pos < off + len(url):
                if best is None or off > best[0]:
                    best = (off, url)
        return best[1] if best else None

    def _open_link_at(self, event) -> Optional[str]:
        """Ctrl+click / middle-click on a link: open it in the default browser."""
        try:
            idx = self.chat_text.index(f"@{event.x},{event.y}")
        except tk.TclError:
            return None
        url = self._link_at(idx)
        if not url:
            return None
        try:
            webbrowser.open(url)
        except Exception as e:
            self.render_error(f"Could not open link: {e}")
            return "break"
        self._set_status("\U0001F517 Opened " + url[:80])
        return "break"

    def _on_chat_motion(self, event) -> None:
        """Hover: hand cursor over links + URL preview in the status bar.
        The preview (a "🔗 http…" string) only replaces "Ready" or a previous
        preview - real status messages (speed readout etc.) are never clobbered."""
        try:
            idx = self.chat_text.index(f"@{event.x},{event.y}")
        except tk.TclError:
            return
        url = self._link_at(idx)
        if url:
            try:
                self.chat_text.configure(cursor="hand2")
            except tk.TclError:
                pass
            cur = str(self.status_label.cget("text"))
            if not (cur.startswith("\U0001F517 http") or cur == "Ready"):
                return                       # don't clobber real status messages
            self._set_status("\U0001F517 " + url[:90])
        else:
            try:
                self.chat_text.configure(cursor="")
            except tk.TclError:
                pass
            if str(self.status_label.cget("text")).startswith("\U0001F517 http"):
                self._set_status("Ready")    # restore after leaving a link preview

    def _on_chat_click(self, event) -> Optional[str]:
        """Left-click: open the image under the cursor; otherwise normal text behavior."""
        path = self._image_at(event.x, event.y)
        if not path:
            return None
        self._open_image_file(path)
        self._set_status("🖼 Opened " + Path(path).name)
        return "break"

    def _on_chat_right_click(self, event) -> Optional[str]:
        """Right-click: image menu (Open / Copy Path) over a thumbnail, else the
        usual Cut/Copy/Paste menu."""
        path = self._image_at(event.x, event.y)
        if not path:
            self._show_context_menu(event)
            return "break"
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="📂 Open Image",
                         command=lambda p=path: self._open_image_file(p))
        menu.add_command(label="📋 Copy Path",
                         command=lambda p=path: self.copy_to_clipboard(p))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        except tk.TclError:
            pass
        return "break"

    def copy_to_clipboard(self, text: str) -> None:
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self._set_status("📋 Copied to clipboard")
        except tk.TclError:
            pass

    def _build_ctx_menu(self, w):
        """Cut/Copy/Paste/Select All popup for a Text widget (right-click)."""
        try:
            has_sel = bool(w.tag_ranges(tk.SEL))
        except tk.TclError:
            return None
        menu = tk.Menu(self.root, tearoff=0)

        def _sel_text():
            r = w.tag_ranges(tk.SEL)
            return w.get(r[0], r[1]) if r else ""

        def _cut():
            txt = _sel_text()
            if not txt:
                return
            self.copy_to_clipboard(txt)
            r = w.tag_ranges(tk.SEL)
            w.delete(r[0], r[1])

        def _copy():
            txt = _sel_text()
            if txt:
                self.copy_to_clipboard(txt)

        def _paste():
            try:
                clip = self.root.clipboard_get()
            except tk.TclError:
                return
            if not clip:
                return
            r = w.tag_ranges(tk.SEL)          # standard paste semantics: replace selection
            if r:
                w.delete(r[0], r[1])
            w.insert("insert", clip)
            if w is self.input_text:
                self._see_input("insert")

        menu.add_command(label="Cut", state=("normal" if has_sel else "disabled"), command=_cut)
        menu.add_command(label="Copy", state=("normal" if has_sel else "disabled"), command=_copy)
        menu.add_command(label="Paste", command=_paste)
        menu.add_separator()
        menu.add_command(label="Select All", command=lambda: w.tag_add("sel", "1.0", "end"))
        return menu

    def _show_context_menu(self, event):
        """Right-click popup for the chat text / input box."""
        menu = self._build_ctx_menu(event.widget)
        if menu is None:
            return
        try:
            menu.tk_popup(event.x_root, event.y_root)
        except tk.TclError:
            pass

    def _see_input(self, where: str = "insert") -> None:
        """Scroll the INPUT box so `where` is visible (main thread only).

        Tk auto-scrolls on typed keys, but programmatic inserts (Shift+Enter newline,
        dictation append, context-menu paste) do not - without this a long prompt ends up
        scrolled off the top of the 3-line box. TclError-guarded for Tk 9 / half-built apps.
        """
        try:
            self.input_text.see(where)
        except (tk.TclError, AttributeError):
            pass

    def _at_bottom(self) -> bool:
        """True when the chat view sits at (or within a hair of) the bottom."""
        try:
            top, bot = self.chat_text.yview()
            return float(bot) >= 0.995
        except tk.TclError:
            return True

    def _autoscroll(self) -> None:
        if self._closed:
            return
        # Re-engage stick-to-bottom once the user has manually returned to the
        # bottom (scrollbar drag / wheel down); a wheel-up opts out until then.
        if not self._stick_bottom and self._at_bottom():
            self._stick_bottom = True
        if self._stick_bottom:
            try:
                self.chat_text.see("end")
            except tk.TclError:
                pass

    # ════════════════════════════════════════════════════════════════════
    #  THREAD-SAFE UI MARSHALLING
    # ════════════════════════════════════════════════════════════════════

    def _post(self, fn) -> None:
        """Hand `fn` to the Tk main thread (the only safe place for UI).

        Worker threads must never call Tcl directly - cross-thread after()
        calls are unreliable. Callbacks go through a queue that a periodic
        main-thread timer drains.
        """
        try:
            self._ui_queue.put(fn)
        except Exception:
            pass

    def _poll_ui_queue(self) -> None:
        """Drain queued UI callbacks on the main thread; reschedules itself."""
        if self._closed:
            return
        while True:
            try:
                fn = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                fn()
            except Exception:
                import traceback
                traceback.print_exc()
        if not self._closed:
            try:
                self.root.after(50, self._poll_ui_queue)
            except tk.TclError:
                pass

    def _set_status(self, text: str) -> None:
        if not self._closed:
            self.status_label.configure(text=text)

    # ── main-thread callbacks driven by the worker thread ───────────────────

    def _collapse_reasoning_sections(self) -> None:
        """Close every open reasoning drawer when its thinking phase ends.

        LM Studio style: the body is hidden with elide=True (which removes all
        display space while keeping text indices valid), the arrow flips in
        place, and the fixed-width title region is stamped with the elapsed
        thinking time. Every edit is same-length, so no stored index anywhere
        shifts.
        """
        for sid in list(self._reasoning_sids):
            info = self._accordions.get(sid)
            if not info or not info["expanded"]:
                continue
            try:
                self.chat_text.tag_configure(info["body_tag"], elide=True)
            except tk.TclError:
                pass
            info["expanded"] = False
            try:
                self.chat_text.delete(info["a0"], info["a1"])
                self.chat_text.insert(info["a0"], "\u25b6 ")   # collapsed arrow (2 chars)
            except tk.TclError:
                pass
            started = info.get("started_at")
            if started is not None and info.get("r0"):
                elapsed = max(0.0, time.time() - started)
                label = f"Thought for {elapsed:.1f}s"[:REASONING_REGION_LEN]
                label = label.ljust(REASONING_REGION_LEN)
                try:
                    self.chat_text.delete(info["r0"], info["r1"])
                    self.chat_text.insert(info["r0"], label, sid + "_hdr")
                except tk.TclError:
                    pass
        self._reasoning_sids.clear()

    def _ui_discard_partial_render(self) -> None:
        """Clear the partially-rendered reply so an automatic retry can re-render
        it cleanly (no duplicated text / code buttons / links). Reuses load_chat's
        full reset; safe mid-turn because it doesn't touch _run_chat_id or _busy.
        Only acts when the streaming chat is actually on screen."""
        if self._closed or not self._viewing_run_chat():
            return
        cid = self.current_chat_id
        if cid is None:
            return
        self.load_chat(cid, force=True)

    def _ui_reset_turn_state(self) -> None:
        # A new model step begins: any earlier thinking phase is finished, so
        # collapse its drawer(s) now (LM Studio shows each "Thought for Xs"
        # block collapsed as soon as that phase ends).
        self._collapse_reasoning_sections()
        self._active_md = None
        self._reasoning_sid = None
        self._pending_tool_output = None

    def _ui_stream_chunk(self, content: str, reasoning: str) -> None:
        if not self._viewing_run_chat():
            return
        if reasoning:
            sid = self._ensure_reasoning_section()
            info = self._accordions.get(sid)
            if info and not info["expanded"]:
                # User collapsed the drawer mid-stream; re-open it so live
                # thinking stays visible ("open while it is happening").
                try:
                    self.chat_text.tag_configure(info["body_tag"], elide=False)
                    self.chat_text.delete(info["a0"], info["a1"])
                    self.chat_text.insert(info["a0"], "\u25bc ")
                except tk.TclError:
                    pass
                info["expanded"] = True
            self.chat_text.insert("end", reasoning, [sid + "_body", "dim"])
        if content and self._active_md is None:
            self._begin_assistant_block()
        if content and self._active_md is not None:
            self._active_md.feed(content)
        self._autoscroll()

    def _ui_sync_messages(self, chat_id: str, messages: List[dict]) -> None:
        """Persist the (possibly extended) message list for a running chat.

        The in-memory list is always updated; the disk write is throttled to at
        most once per CHATS_SAVE_MIN_INTERVAL seconds so a long tool phase does
        not rewrite the whole chats JSON after every step. _finish_turn_ui()
        guarantees a final save at turn end."""
        chat = self.chats["chats"].get(chat_id)
        if chat is None:
            return
        chat["messages"] = messages
        now = time.monotonic()
        if now - self._last_chats_save >= CHATS_SAVE_MIN_INTERVAL:
            save_chats(self.chats)
            self._last_chats_save = now

    def _ui_turn_complete(self, full_content: str) -> None:
        if self._active_md is not None:
            try:
                self._active_md.finish()
            except tk.TclError:
                pass
            self._active_md = None
        self._finish_turn_ui()
        self._maybe_trigger_handoff()   # auto-summary once context usage crosses the threshold
        if self.settings.get("tts_enabled"):
            self.speak(full_content)

    def _ui_turn_stopped(self) -> None:
        """Finish the UI after a user-initiated stop.

        Like _ui_turn_complete() but without TTS (no point reading an aborted
        reply aloud) and without the handoff check (the context did not grow)."""
        if self._active_md is not None:
            try:
                self._active_md.finish()
            except tk.TclError:
                pass
            self._active_md = None
        # Only paint into the chat that actually ran the turn: a stop triggered by
        # deleting that chat must not write into whatever chat is on screen now.
        if self._viewing_run_chat():
            self.render_note("\u23f9 Stopped by user")
        self._finish_turn_ui()

    def _finish_turn_ui(self, status: Optional[str] = None) -> None:
        """End a turn: persist history, release _busy, restore the Send button.

        status= keeps an informative message visible (e.g. why a turn was cut short for a
        context handoff); without it the bar reads "Ready"."""
        if self._closed:
            return
        # Mid-turn saves are throttled; make sure the final state of this turn
        # (last assistant reply / tool results) always reaches disk.
        save_chats(self.chats)
        self._last_chats_save = time.monotonic()
        self._busy = False
        self._stop_event.clear()
        try:
            self.send_btn.configure(text="➤ Send", bg=COL["accent"], activebackground="#2563EB",
                                    command=self.send_message)
        except tk.TclError:
            pass
        self._set_status(status or "Ready")
        self._collapse_reasoning_sections()   # close any still-open reasoning drawer(s)
        self._reasoning_sid = None             # next turn starts a fresh section

    # ════════════════════════════════════════════════════════════════════
    #  SENDING + AGENTIC WORKER (background daemon thread)
    # ════════════════════════════════════════════════════════════════════

    def _on_input_return(self, event) -> Optional[str]:
        if event.state & 0x0001:             # Shift+Enter → newline
            self.input_text.insert("insert", "\n")
            self._see_input("insert")   # programmatic insert does NOT auto-scroll the caret
            return "break"
        self.send_message()
        return "break"

    def send_message(self) -> None:
        # If dictation is running, stop it and wait briefly for the last captured
        # phrase to land in the input box before reading + sending.
        if self._mic_active:
            self._stop_mic("Dictation stopped")
            t = self._stt_thread
            if t is not None and t.is_alive():
                t.join(timeout=3)
        # Interrupt any in-flight TTS playback before starting a new turn.
        self.stop_tts()
        if self._busy or OpenAI is None:
            return
        raw = self.input_text.get("1.0", "end-1c").strip()
        atts = list(self._attachments)
        if not raw and not atts:
            return

        # build the user message content (string, or multimodal parts list)
        if atts:
            parts: List[dict] = []
            if raw:
                parts.append({"type": "text", "text": raw})
            for p in atts:
                part = self._attachment_to_part(p)
                if isinstance(part, list):
                    parts.extend(part)
                else:
                    parts.append(part)
            content: Any = parts
        else:
            content = raw

        chat = self.current_chat()
        if chat is None:                        # fresh launch state: no thread selected yet
            self.new_chat()                    # start a brand-new conversation for this message
            chat = self.current_chat()
        chat["messages"].append({"role": "user", "content": content})

        # auto-title from the first user message
        if chat.get("title") in (None, "", "New Chat"):
            base = raw.strip() or "Attachment"
            chat["title"] = base[:32] + ("…" if len(base) > 32 else "")
            self._refresh_chat_list()

        # clear input area
        self.input_text.delete("1.0", "end")
        self._attachments.clear()
        self._render_chips()

        save_chats(self.chats)
        self._stick_bottom = True   # sending re-engages follow-to-bottom first
        self.render_user_message(content)

        self._busy = True
        self._stop_event.clear()
        # The Send button becomes the Stop button for the duration of the turn:
        # it must stay ENABLED so the user can click it to abort.
        self.send_btn.configure(text="⏹ Stop", bg=COL["danger"], activebackground="#B91C1C",
                                command=self.stop_generation)
        self.speed_label.configure(text="⚡ …")
        cid = self.current_chat_id
        self._run_chat_id = cid
        messages = list(chat["messages"])
        self._refresh_ctx_topbar()   # pick up server-side context changes (e.g. after a model restart)
        threading.Thread(target=self._chat_worker, args=(cid, messages), daemon=True).start()

    def stop_generation(self) -> None:
        """User pressed Stop: abort the in-flight turn at the next check point.

        The worker checks this event between stream chunks, between tool calls
        and between model steps. Whatever was generated so far is kept (a
        partial reply is persisted); the button flips back to Send when the
        worker finishes winding down."""
        if not self._busy:
            return
        self._stop_event.set()
        # Tools now poll this event, so the turn unwinds within STOP_POLL_S; say what is
        # happening while they do (a killed node/CLI subprocess takes a moment to die).
        self._set_status("⏹ Stopping… (aborting any running tool)")

    def _get_client(self) -> Optional[Any]:
        """Return a cached OpenAI client for the current server/key.

        Reusing one client across turns keeps its HTTP connection pool warm
        (keep-alive), so each request skips the TCP+TLS handshake."""
        if OpenAI is None:
            return None
        url = (self.settings.get("server_url") or "").strip()
        key = (self.settings.get("api_key") or "").strip() or "sk-local"
        cached = self._clients.get((url, key))
        if cached is not None:
            return cached
        try:
            client = OpenAI(base_url=url or None, api_key=key)
        except Exception as e:
            self._post(lambda m=str(e): self.render_error(f"Could not create API client:\n{m}"))
            return None
        self._clients[(url, key)] = client
        return client

    def _invalidate_clients(self) -> None:
        """Drop cached OpenAI clients (called when server URL / API key change)."""
        for c in self._clients.values():
            try:
                c.close()
            except Exception:
                pass
        self._clients.clear()

    def _chat_worker(self, chat_id: str, messages: List[dict]) -> None:
        """Thread entry point: run the agentic loop, and NEVER let an exception
        escape without handing the UI back to the user.

        _busy is cleared only by _finish_turn_ui(), so an escaping exception
        (corrupt history, a tool-schema/MCP hiccup while building the request,
        any bug at all in the ~300 lines below) would kill this thread with
        _busy stuck True: Send stays "Stop" and every later send_message()
        returns silently until the app is restarted. The inner loop's own
        error paths post _finish_turn_ui themselves; this net catches what
        they cannot, on the abnormal-exit path only (no double-finish)."""
        try:
            self._chat_worker_inner(chat_id, messages)
        except Exception as e:
            traceback.print_exc()
            self._post(lambda m=str(e): self.render_error(
                f"Internal error during generation:\n{m}"))
            self._post(self._finish_turn_ui)

    def _chat_worker_inner(self, chat_id: str, messages: List[dict]) -> None:
        """Background daemon thread: streams model responses and executes tools.
        All UI access is marshalled through self._post (root.after)."""
        client = self._get_client()
        if client is None:
            self._post(lambda: self.render_error(
                "The 'openai' package is not installed. Run:  pip install openai"))
            self._post(self._finish_turn_ui)
            return

        model = (self.settings.get("model_name") or "gpt-4o-mini").strip()
        step = 0
        usage_opts_ok = True
        thinking_ok = True          # flipped off if the server rejects enable_thinking
        sampling_ok = True        # flipped off if the server rejects the sampler extension keys
        pruned_keys = set()       # individual sampler keys a strict server rejected (kept out of later steps)
        sampling_noted = False    # the "what was sent" note fires once per user prompt
        turn_total = 0           # completion tokens accumulated this user prompt (resets per prompt)
        stream_retry_ok = True   # one automatic retry per step on a transient mid-stream drop
        model_fix_ok = True      # one auto-correction of a stale model name per turn
        self._ctx_used_run = 0   # gauge as of the last completed step of THIS prompt (worker-side copy;
                                 # _ctx_used is main-thread state and must not be written from here)
        # Wall-clock backstop for this prompt, from the "Turn time limit" setting
        # (0 = NO LIMIT, which is the default). MAX_TOOL_STEPS == 0 means the loop has no
        # step cap, so a deadline is the only thing that can end a model that keeps calling
        # tools - but it is opt-in now: a long turn must not be cut short unless the user
        # asked for it. Checked at each step boundary; the per-tool abort path (Stop /
        # _run_cancellable) is what stops a single long call mid-flight either way.
        turn_limit = _turn_time_limit(self.settings)
        turn_deadline = (time.monotonic() + turn_limit) if turn_limit else None

        while (MAX_TOOL_STEPS == 0 or step < MAX_TOOL_STEPS) and not self._closed \
                and not self._stop_event.is_set():
            # ── build request (only tools whose permission ≠ Off) ─────────
            enabled_tools = [s for n, s in TOOL_SCHEMAS.items()
                             if self.settings.get("tool_permissions", {}).get(n, "ask") != "off"]
            # MCP tools from connected servers (namespaced mcp_<server>_<tool>).
            enabled_tools += [s for s in self._mcp_schemas()
                              if self.settings.get("tool_permissions", {}).get(
                                  s["function"]["name"], "ask") != "off"]
            enabled_tools = enabled_tools or None
            kwargs: Dict[str, Any] = dict(model=model,
                                messages=[build_system_message(str(self.settings.get("file_workspace") or ""),
                                   str(self.settings.get("exa_api_key") or ""),
                                   str(self.settings.get("firecrawl_api_key") or ""),
                                   str(self.settings.get("custom_system_prompt") or ""))]
                                  + _prepare_request_messages(list(messages)),
                                stream=True)
            if enabled_tools:
                kwargs["tools"] = enabled_tools
                kwargs["tool_choice"] = "auto"
            if usage_opts_ok:
                kwargs["stream_options"] = {"include_usage": True}
            try:
                mt = int(self.settings.get("max_tokens", 0) or 0)
            except (TypeError, ValueError):
                mt = 0
            if mt > 0:
                kwargs["max_tokens"] = mt        # per-reply token cap (Settings -> Max Tokens)
            # Per-chat temperature: always sent now (blank/legacy -> TEMP_DEFAULT).
            t = chat_temperature(self.chats["chats"].get(chat_id))
            if t is not None:
                kwargs["temperature"] = t
            # Per-chat thinking level: Off/Low/Medium/High map to Qwen3-style server
            # extensions (enable_thinking / chat_template_kwargs.reasoning_effort - not in
            # the OpenAI schema), sent via extra_body. Blank/legacy chats send nothing
            # (server default). Servers that reject it are retried without it (thinking_ok).
            # Sampler parameters (Settings -> Sampling) ride in the SAME extra_body dict, so
            # build them together: an assignment here would overwrite the other one.
            _tb = None                # thinking extra_body for THIS step (None = server default)
            eb: Dict[str, Any] = {}
            if thinking_ok:
                _tb = chat_thinking_extra_body(self.chats["chats"].get(chat_id))
                if _tb is not None:
                    eb.update(_tb)
            _sw = sampling_kwargs(self.settings)
            kwargs.update(_sw["top"])
            if sampling_ok:
                eb.update({k: v for k, v in _sw["flat"].items() if k not in pruned_keys})
            self._sampling_sent = dict(top=_sw["top"], flat=dict(eb), thinking=_tb)
            kwargs["extra_body"] = eb

            if turn_deadline is not None and time.monotonic() > turn_deadline:
                # Wall-clock backstop hit (only reachable when MAX_TOOL_STEPS == 0).
                # If the context ALSO crossed the handoff threshold, prefer the handoff
                # path - it saves the work AND explains itself. Otherwise render a plain
                # note in the chat (not just the status bar), so a long aborted turn does
                # not look like the app silently froze.
                if self._begin_midturn_handoff(chat_id, int(self._ctx_used_run or 0),
                                               self._ctx_window(),
                                               "the " + str(turn_limit) + " s time limit was reached"):
                    return
                _msg = ("\u23f1 Turn stopped after the " + str(turn_limit) +
                        " s time limit (" + str(step) + " tool steps).")
                self._post(lambda m=_msg: self.render_note(m))
                self._post(lambda m=_msg: self._finish_turn_ui(status=m))
                return

            _cap = "" if MAX_TOOL_STEPS == 0 else f"/{MAX_TOOL_STEPS}"
            self._post(lambda s=f"🤖 Thinking… (step {step + 1}{_cap})": self._set_status(s))
            try:
                stream = client.chat.completions.create(**kwargs)
                # One note per user prompt recording the sampling parameters that actually left the
                # app for THIS request (post-rejection), so "are my settings in effect?" is answerable
                # from the chat log itself. Emitted after create() so a rejected-and-retried parameter
                # is reported truthfully; the retry loop re-enters with sampling_noted already True.
                if _sampling_note_enabled(self.settings) and not sampling_noted:
                    sampling_noted = True
                    _sn = self._sampling_sent or {}
                    self._post(lambda c=self.chats["chats"].get(chat_id), t=_sn.get("top") or {},
                               f=_sn.get("flat") or {}, p=set(pruned_keys), th=_sn.get("thinking"):
                               self.render_note(format_sampling_note(c, t, f, p, th)))
            except Exception as e:
                msg = str(e)
                low_msg = msg.lower()
                if usage_opts_ok and "stream_options" in msg.lower():
                    usage_opts_ok = False         # server rejected the option → retry without it
                    continue
                if thinking_ok and any(k in low_msg for k in
                                       ("enable_thinking", "chat_template_kwargs", "reasoning_effort")):
                    thinking_ok = False           # server rejects the extension -> retry without it
                    kwargs.pop("extra_body", None)
                    continue
                if sampling_ok and any(k in low_msg for k in SAMPLING_FLAT_KEYS):
                    # Strict server (real OpenAI, many proxies): drop ONLY the offending sampler
                    # extension key and retry, so one rejected field cannot silently kill the rest.
                    hit = [k for k in SAMPLING_FLAT_KEYS if k in low_msg]
                    dropped = False
                    for k in hit:
                        if k not in pruned_keys:
                            pruned_keys.add(k); dropped = True
                    if dropped:
                        self._post(lambda ks=hit: self._set_status(
                            "\u2699 Sampler parameter(s) rejected by the server, dropped: " + ", ".join(ks)))
                        continue
                if sampling_ok and "extra_body" in low_msg:
                    sampling_ok = False       # whole extra_body block refused -> builtins only
                    kwargs["extra_body"] = {k: v for k, v in (kwargs.get("extra_body") or {}).items()
                                            if k not in SAMPLING_FLAT_KEYS}
                    continue
                # Stale model name (404 model_not_found): re-read /models and, when one
                # model is loaded, retry this step with its exact server ID. One attempt
                # per turn, so a genuinely wrong server can never loop here.
                if (model_fix_ok and ("model_not_found" in low_msg
                                      or "is downloaded but not loaded" in low_msg
                                      or "model not found" in low_msg)):
                    model_fix_ok = False
                    info = fetch_model_info((self.settings.get("server_url") or "").strip(),
                                            (self.settings.get("api_key") or "").strip())
                    loaded = loaded_model_ids(info)
                    canon, status = normalize_model_name(model, info) if info else ("", "unknown")
                    new_id = canon if status == "resolved" else ""
                    if not new_id and len(loaded) == 1:
                        new_id = loaded[0]
                    if new_id and new_id != model:
                        model = new_id
                        kwargs["model"] = new_id
                        self.settings["model_name"] = new_id
                        hist = [h for h in (self.settings.get("model_history") or [])
                                if isinstance(h, str) and h != new_id]
                        hist.insert(0, new_id)
                        self.settings["model_history"] = hist[:MODEL_HISTORY_MAX]
                        save_settings(self.settings)
                        self._post(lambda n=new_id: (
                            self.render_note(f"🔧 Model Name corrected to the loaded model: {n}"),
                            self._set_actual_model(n), self._update_topbar_labels())[0])
                        continue
                    opts = ", ".join(loaded) if loaded else "(none reported loaded)"
                    self._post(lambda m=msg, o=opts: self.render_error(
                        f"Model request failed:\n{m}\n\nModels the server reports as loaded: {o}\nSettings → Model Name must use one of those exact IDs."))
                    self._post(self._finish_turn_ui)
                    return
                # Context full: the request itself was refused. Save what this turn has
                # produced by writing handoff notes (the summary request is trimmed to fit,
                # so it can succeed when the real request cannot) instead of ending with a
                # bare error and nothing carried over.
                # The server itself says the window is full, so trust that over the
                # char-based estimate (which lags behind tool output it has not seen).
                _ctx_win = self._ctx_window()
                _used_est = max(int(self._ctx_used_run or 0),
                                estimate_prompt_tokens(kwargs["messages"]), _ctx_win)
                if is_context_overflow_error(msg) and self._begin_midturn_handoff(
                        chat_id, _used_est, _ctx_win,
                        "the server refused the request (context full)", force=True):
                    return
                self._post(lambda m=msg: self.render_error(f"Model request failed:\n{m}"))
                self._post(self._finish_turn_ui)
                return

            # ── consume the stream (batched UI flushes every ~50 ms) ──────
            self._post(self._ui_reset_turn_state)
            buf_c = ""
            buf_r = ""
            last_flush = time.time()
            full_content = ""
            full_reasoning = ""          # thinking tokens (for the fallback estimate)
            tool_acc: Dict[int, dict] = {}
            usage = None
            actual_model = ""
            t0 = time.time()             # stream start (prefill still pending)
            first_token_t = None         # first generated token = end of TTFT
            last_token_t = None          # last generated token

            def _mark_token() -> None:
                nonlocal first_token_t, last_token_t
                now = time.time()
                if first_token_t is None:
                    first_token_t = now
                last_token_t = now

            def flush(force: bool = False) -> None:
                nonlocal buf_c, buf_r, last_flush
                now = time.time()
                # Throttle by timer only: a pure tool-call stream keeps both text
                # buffers empty, and the live speed readout must still animate.
                if not force and now - last_flush < FLUSH_INTERVAL:
                    return
                c, r = buf_c, buf_r
                buf_c, buf_r = "", ""
                last_flush = now
                if c or r:   # skip no-op UI posts while only tool JSON is streaming
                    self._post(lambda cc=c, rr=r: self._ui_stream_chunk(cc, rr))
                # LIVE readout while streaming: usage is only known in the final chunk,
                # so mid-stream this uses the character-based estimate.
                live = compute_speed_readout(None, full_content, full_reasoning, tool_acc,
                                             t0, first_token_t, last_token_t, turn_total)
                if live:
                    self._post(lambda l=live: self.speed_label.configure(text=l))

            try:
                for chunk in stream:
                    if self._closed or self._stop_event.is_set():
                        break
                    u = getattr(chunk, "usage", None)
                    if u is not None:
                        try:
                            usage = u.model_dump()
                        except Exception:
                            pass
                    # Show the model ID the server actually used (from stream chunks).
                    if not actual_model:
                        m = getattr(chunk, "model", None)
                        if m:
                            actual_model = str(m)
                            self._post(lambda mm=actual_model: self._set_actual_model(mm))
                    choices = getattr(chunk, "choices", None) or []
                    if not choices:
                        continue
                    delta = choices[0].delta
                    if delta is None:
                        continue

                    # reasoning / thoughts (DeepSeek-R1 style `reasoning_content`)
                    extra = getattr(delta, "model_extra", None) or {}
                    rc = getattr(delta, "reasoning_content", None) or extra.get("reasoning_content")
                    if rc:
                        buf_r += rc
                        full_reasoning += rc
                        _mark_token()

                    c = delta.content
                    if c:
                        full_content += c
                        buf_c += c
                        _mark_token()

                    # accumulate streamed tool calls by index
                    for tc in (delta.tool_calls or []):
                        i = getattr(tc, "index", 0) or 0
                        slot = tool_acc.setdefault(i, {"id": "", "name": "", "args": ""})
                        if getattr(tc, "id", None):
                            slot["id"] = tc.id
                        fn = getattr(tc, "function", None)
                        if fn is not None:
                            if getattr(fn, "name", None):
                                slot["name"] += fn.name
                            if getattr(fn, "arguments", None):
                                slot["args"] += fn.arguments
                        # A tool-call delta is generated output too: mark it so a pure
                        # tool-call step still has first/last token timestamps.
                        _mark_token()
                    flush()
            except Exception as e:
                if (stream_retry_ok and not self._closed and not self._stop_event.is_set()
                        and _is_transient_stream_error(e)):
                    # Transient connection drop mid-stream: discard the partial
                    # render and re-issue the SAME request once. The loop top
                    # resets every per-step accumulator, so the retry starts clean.
                    stream_retry_ok = False
                    buf_c = ""      # drop unflushed tokens so finally's flush posts nothing
                    buf_r = ""
                    self._post(self._ui_discard_partial_render)
                    self._post(lambda m=str(e): self.render_note(
                        f"\u26a1 Connection interrupted ({m[:80]}) - retrying this step once\u2026"))
                    continue
                self._post(lambda m=str(e): self.render_error(f"Stream interrupted:\n{m}"))
            finally:
                flush(force=True)

            if self._stop_event.is_set():
                # Stopped mid-stream: keep whatever was generated so far (the
                # partial reply is persisted as a normal assistant message, so
                # the next turn can continue from it), skip tool execution.
                if full_content.strip():
                    messages.append({"role": "assistant", "content": full_content})
                self._post(lambda cid=chat_id, m=list(messages): self._ui_sync_messages(cid, m))
                self._post(self._ui_turn_stopped)
                return

            # ── speed metrics (decode rate excludes prefill; TTFT shown separately) ──
            readout = compute_speed_readout(usage, full_content, full_reasoning, tool_acc,
                                            t0, first_token_t, last_token_t, turn_total)
            if readout:
                self._post(lambda l=readout: self.speed_label.configure(text=l))
            # cumulative turn total: add this step's completion tokens
            if usage and usage.get("completion_tokens"):
                turn_total += int(usage["completion_tokens"])
            else:
                _cc = (len(full_content) + len(full_reasoning)
                       + sum(len(tc.get("args", "")) for tc in tool_acc.values()))
                turn_total += max(0, _cc // 4)   # char-based estimate (no usage block)

            # ── context usage: total session tokens (prompt + completion) ──
            tot = session_total_tokens(usage, full_content, full_reasoning,
                                       tool_acc, kwargs["messages"])
            self._ctx_used_run = int(tot or 0)     # worker-side copy for the mid-turn gate
            self._post(lambda u=tot: self._set_ctx_used(u))

            # ── finalize tool calls accumulated from the stream ───────────
            tool_calls: List[dict] = []
            for i in sorted(tool_acc):
                s = tool_acc[i]
                if not s["name"]:
                    continue
                args_raw = s["args"] or "{}"
                try:
                    json.loads(args_raw)
                except Exception:
                    args_raw = json.dumps({"_raw_arguments": args_raw})
                tool_calls.append({
                    "id": s["id"] or f"call_{uuid.uuid4().hex[:10]}",
                    "type": "function",
                    "function": {"name": s["name"], "arguments": args_raw},
                })

            if not tool_calls:
                # ── final answer → persist + finish ───────────────────────
                messages.append({"role": "assistant", "content": full_content})
                self._post(lambda cid=chat_id, m=list(messages): self._ui_sync_messages(cid, m))
                if not full_content.strip():
                    self._post(lambda: self.render_note("(model returned an empty response)"))
                self._post(lambda c=full_content: self._ui_turn_complete(c))
                return

            # ── tool phase: record assistant msg, execute each call ───────
            tool_phase_start = len(messages)   # rollback point if the user stops mid-phase
            messages.append({"role": "assistant", "content": full_content or None,
                             "tool_calls": tool_calls})

            for tc in tool_calls:
                if self._closed or self._stop_event.is_set():
                    break
                name = tc["function"]["name"]
                args_raw_dispatch = tc["function"].get("arguments") or "{}"
                try:
                    args = json.loads(args_raw_dispatch)
                    # The finalizer above masks malformed model JSON as
                    # {"_raw_arguments": ...} so history stays sendable to strict
                    # servers - unmask it here, otherwise the call would reach the
                    # handler and die with a confusing TypeError instead of the
                    # clear "re-issue valid JSON" error below.
                    if isinstance(args, dict) and "_raw_arguments" in args:
                        orig_raw = str(args["_raw_arguments"])
                        if orig_raw.strip() not in ("", "{}"):
                            raise ValueError("masked malformed JSON")
                        args = {}
                except Exception:
                    # Malformed tool-call arguments (common with local models on
                    # large payloads): never silently run the tool with {} - that
                    # executes a no-op and looks like "the tool produced no output".
                    self._post(lambda n=name: self._ui_tool_call_accordion(n, {"_malformed_arguments": True}))
                    err = (f"ERROR: The {name} tool call had malformed JSON arguments and "
                           f"could not be parsed - it was NOT executed. Re-issue the call "
                           f"with valid JSON (keep large payloads like code concise).")
                    messages.append({"role": "tool", "tool_call_id": tc["id"],
                                     "name": name, "content": err})
                    self._post(lambda n=name, r=err: self._ui_tool_output_accordion(n, err))
                    continue
                self._post(lambda n=name, a=args: self._ui_tool_call_accordion(n, a))
                self._post(lambda s=f"🛠 Running {name}…": self._set_status(s))
                self._post(lambda n=name: self._ui_tool_output_start(n))

                result = self._execute_tool(name, args)      # may block on permission modal
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "name": name, "content": result})
                self._post(lambda n=name, r=result: self._ui_tool_output_accordion(n, r))

                if name == "capture_screen" and result.startswith("OK"):
                    vision = self._build_vision_message(result)
                    if vision is not None:
                        messages.append(vision)
                        self._post(lambda: self.render_note("📸 Screenshot queued for vision analysis"))

            if self._stop_event.is_set():
                # Stopped mid-tool-phase: roll back the assistant tool_calls
                # message and any tool results added this step (a tool_call
                # without its result would be rejected by strict servers), then
                # end the turn with what was already persisted.
                del messages[tool_phase_start:]
                self._post(lambda cid=chat_id, m=list(messages): self._ui_sync_messages(cid, m))
                self._post(self._ui_turn_stopped)
                return

            self._post(lambda cid=chat_id, m=list(messages): self._ui_sync_messages(cid, m))
            stream_retry_ok = True   # a completed step restores the one-retry budget

            # ── mid-turn context gate ────────────────────────────────────────
            # Deliberately checked AFTER the tool phase: the tool results just appended
            # are exactly what a handoff summary needs. Continuing to the next request
            # would push an almost-full window past the server's limit, and the turn used
            # to die there with an error - hours of work and no notes. Here the loop stops
            # on purpose and _ui_finish_for_handoff() writes the summary instead.
            # Gate on the LARGER of the server's figure and a char estimate of what the NEXT
            # request would really carry: 'tot' was measured when that request was made, so
            # this step's tool results are NOT in it. A large tool output could push the next
            # request past the window while the gauge still looked safe - the gate firing one
            # step late is exactly the failure this feature exists to prevent. The estimate is
            # only ever a lower bound here, and _begin_midturn_handoff() still refuses to act
            # when the server reports no context size (no guessing).
            _gate_used = max(int(tot or 0), estimate_prompt_tokens(messages))
            if self._begin_midturn_handoff(chat_id, _gate_used, self._ctx_window(),
                                           "the agentic loop is still running"):
                return

            step += 1

        if self._closed:
            return
        if self._stop_event.is_set():
            # Stop landed exactly on a step boundary (or before the very first
            # iteration): the while-condition exited on its own, so no turn-ending
            # path ran. With MAX_TOOL_STEPS == 0 the cap branch below is dead code,
            # which left _busy stuck True forever - Send stayed "Stop" and every
            # later send was silently ignored until restart. Finish the turn like
            # any other user stop (partial reply already persisted).
            self._post(self._ui_turn_stopped)
        elif MAX_TOOL_STEPS != 0:
            # Only reachable by exhausting the step cap (the deadline path returns
            # inside the loop), so the "safety limit" wording is now accurate too.
            self._post(lambda: self.render_note(
                f"⚠ Stopped after {MAX_TOOL_STEPS} tool iterations (safety limit)."))
            self._post(self._finish_turn_ui)

    def _build_vision_message(self, result_text: str) -> Optional[dict]:
        """Turn a successful capture_screen result into a queued vision message.

        The image is stored as a lightweight 'image_ref' part (just the file
        path) - NOT inline base64 - so it doesn't bloat deskpilot_chats.json.
        _prepare_request_messages() attaches the actual payload at request time
        (most recent screenshot only)."""
        m = re.search(r"Path:\s*(\S+)\s*$", result_text.strip())
        if not m:
            return None
        path = Path(m.group(1))
        if not path.is_file():
            return None
        return {
            "role": "user",
            "content": [
                {"type": "text",
                 "text": SCREENSHOT_MARKER + " is attached below for your visual analysis."},
                {"type": "image_ref", "path": str(path)},
            ],
        }

    # ── permission gate + tool dispatch (worker thread) ─────────────────────

    def _ask_permission_modal(self, name: str, args: dict) -> bool:
        """Modal askyesno prompt, marshalled to the main thread; blocks this
        worker until the user answers (or 5 minutes pass)."""
        preview = json.dumps(args, ensure_ascii=False)
        if len(preview) > 600:
            preview = preview[:600] + "…"
        ans: Dict[str, bool] = {"ok": False}
        ev = threading.Event()

        def show():
            # The post is queued on the main thread; if the user stopped the turn (or the
            # app is closing) while it sat in the queue, do NOT open a dialog nobody asked
            # for any more - just release the worker's wait with a "No".
            if self._closed or self._stop_event.is_set():
                ans["ok"] = False
                ev.set()
                return
            try:
                ans["ok"] = messagebox.askyesno(
                    f"{APP_NAME} — Tool Permission",
                    f"Deskpilot wants to run the tool:\n\n  {name}\n\nArguments:\n{preview}\n\nAllow this action?",
                    parent=self.root)
            except Exception:
                ans["ok"] = False
            finally:
                ev.set()

        self._post(show)
        # Sliced wait: a Stop press (or closing the app) must not leave the worker
        # parked behind an unanswered permission dialog for 5 minutes. A cancelled
        # wait answers "No" - the tool is NOT run, which is the safe default.
        if not _interruptible_wait(ev, 300, lambda: self._closed or self._stop_event.is_set()):
            return False
        return bool(ans["ok"])

    def _abort_requested(self) -> bool:
        """True when the user pressed Stop or the app is closing. Polled by every
        blocking tool so a long-running call unwinds within STOP_POLL_S.

        Reads via getattr defaults so half-built instances (tests that skip __init__)
        never raise AttributeError inside a tool handler."""
        if getattr(self, "_closed", False):
            return True
        ev = getattr(self, "_stop_event", None)
        return ev is not None and ev.is_set()

    def _run_cancellable(self, cmd: List[str], timeout: float,
                         abort=None, cwd: Optional[str] = None
                         ) -> Optional[subprocess.CompletedProcess]:
        """Run a subprocess so Stop / app close can KILL it within STOP_POLL_S.

        subprocess.run(timeout=...) cannot be interrupted: the worker sat inside a 900 s
        node run (or a 15 min image generation) doing nothing while the user hammered
        Stop, because the stop event was only examined between model steps. Here the
        child is spawned with Popen and polled in STOP_POLL_S slices; on timeout OR when
        'abort' turns True it is terminated (escalating to kill) and None is returned.

        stdout/stderr are drained by reader threads into capped byte lists, so a chatty
        child can never deadlock the poll loop on a full OS pipe buffer (the trap that
        makes naive Popen+wait() worse than subprocess.run). Returns a CompletedProcess
        on normal exit; None when cancelled or timed out."""
        flags = 0x08000000 if os.name == "nt" else 0      # CREATE_NO_WINDOW (hidden)
        # A spawn failure propagates; every caller wraps this call and turns it into
        # an ERROR string (invariant: tool handlers never raise to the UI).
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                stdin=subprocess.DEVNULL, creationflags=flags,
                                cwd=cwd or tempfile.gettempdir())

        sink: Dict[str, list] = {"out": [], "err": []}
        used: Dict[str, int] = {"out": 0, "err": 0}

        def _drain(pipe, key: str) -> None:
            try:
                while True:
                    # read1(): returns whatever one raw read yields instead of blocking
                    # until the full chunk size is available.
                    b = pipe.read1(65536)
                    if not b:
                        break
                    if used[key] < PROC_OUTPUT_CAP:
                        sink[key].append(b)
                        used[key] += len(b)
            except Exception:
                pass

        threads = [threading.Thread(target=_drain, args=(proc.stdout, "out"), daemon=True),
                   threading.Thread(target=_drain, args=(proc.stderr, "err"), daemon=True)]
        for t in threads:
            t.start()

        cancelled = False
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            if abort is not None and abort():
                cancelled = True
                _kill_process(proc)
                break
            if time.monotonic() >= deadline:
                cancelled = True
                _kill_process(proc)
                break
            time.sleep(STOP_POLL_S)

        for t in threads:
            t.join(timeout=2)          # pipes are closed at exit, so drains finish fast
        for pipe in (proc.stdout, proc.stderr):
            try:
                if pipe:
                    pipe.close()
            except Exception:
                pass
        if cancelled:
            return None
        return subprocess.CompletedProcess(
            args=cmd, returncode=proc.returncode,
            stdout=b"".join(sink["out"]).decode("utf-8", "replace"),
            stderr=b"".join(sink["err"]).decode("utf-8", "replace"))

    def _execute_tool(self, name: str, args: dict) -> str:
        perm = self.settings.get("tool_permissions", {}).get(name, "ask")
        if perm == "off":
            return f"ERROR: Tool '{name}' is disabled by the user (permission set to Off)."
        if perm == "ask" and not self._ask_permission_modal(name, args):
            return f"ERROR: User denied permission to run '{name}'."
        if name.startswith("mcp_"):
            result = self._mcp_dispatch(name, args)
        else:
            handler = getattr(self, f"_tool_{name}", None)
            if handler is None:
                return f"ERROR: Unknown tool '{name}'."
            try:
                result = handler(**args) if args else handler()
            except TypeError as e:
                return f"ERROR: Invalid arguments for {name}: {e}"
            except Exception as e:
                return f"ERROR: {name} failed: {e}"
        if not isinstance(result, str):
            result = str(result)
        # Hard cap on tool output so a huge read/fetch can't blow the context window.
        # The marker tells the model (and the user in the accordion) content was cut.
        if len(result) > TOOL_OUTPUT_LIMIT:
            return result[:TOOL_OUTPUT_LIMIT] + f"\n[... output truncated at {TOOL_OUTPUT_LIMIT} chars]"
        return result

    # ════════════════════════════════════════════════════════════════════
    #  TOOL HANDLERS (executed in the worker thread)
    # ════════════════════════════════════════════════════════════════════

    def _tool_searxng_search(self, query: str = "", max_results: int = 10) -> str:
        base = _searxng_base_url(self.settings)
        url = f"{base}/search?{urllib.parse.urlencode({'q': query, 'format': 'json'})}"
        req = urllib.request.Request(url, headers={"User-Agent": "Deskpilot/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw = resp.read().decode("utf-8", "replace")
        except Exception as e:
            return (f"ERROR: Could not reach SearXNG at {base} ({e}). "
                    f"Is the instance running? In Settings, set 'SearXNG URL' "
                    f"(blank = LLM server's address on port 8080).")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return ("ERROR: SearXNG did not return JSON. Enable the JSON format in "
                    "searxng/settings.yml  →  search.formats: [html, json]")
        results = []
        for item in (data.get("results") or [])[:int(max_results)]:
            title = str(item.get("title", "(no title)").strip())
            u = str(item.get("url", "")).strip()
            snippet = re.sub(r"\s+", " ", str(item.get("content", ""))).strip()[:300]
            results.append(f"• {title}\n  {u}\n  {snippet}")
        if not results:
            return f"No SearXNG results found for “{query}”."
        return (f"SearXNG results for “{query}”:\n\n" + "\n\n".join(results))[:8000]

    def _tool_exa_search(self, query: str = "", num_results: int = 10,
                         search_type: str = "auto", include_full_text: bool = False) -> str:
        """Exa neural web search (POST https://api.exa.ai/search).

        Request shape follows Exa's official guidance: query + highlights,
        nothing else unless the model explicitly asks for more.
        """
        key = (self.settings.get("exa_api_key") or "").strip()
        if not key:
            return ("ERROR: No Exa API key set. Get one at exa.ai/dashboard and paste it "
                    "into 'Exa API Key' in Settings.")
        try:
            n = max(1, min(int(num_results), 25))
        except (TypeError, ValueError):
            n = 10
        stype = search_type if search_type in ("auto", "fast", "deep") else "auto"
        body = {
            "query": query,
            "numResults": n,
            "type": stype,
            "contents": {"highlights": True},
        }
        if include_full_text:
            body["contents"]["text"] = True
            body["contents"]["maxCharacters"] = 3000
        req = urllib.request.Request(
            "https://api.exa.ai/search",
            data=json.dumps(body).encode("utf-8"),
            headers={"x-api-key": key, "Content-Type": "application/json",
                     "User-Agent": "Deskpilot/1.0"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8", "replace"))
        except Exception as e:
            code = getattr(e, "code", None)
            if code == 401:
                return ("ERROR: Exa rejected the API key (HTTP 401). "
                        "Check 'Exa API Key' in Settings.")
            if code is not None:
                try:
                    detail = json.loads(e.read().decode("utf-8", "replace")).get("error", "")
                except Exception:
                    detail = ""
                return f"ERROR: Exa search failed (HTTP {code}){': ' + str(detail) if detail else ''}."
            return (f"ERROR: Could not reach api.exa.ai ({e}). "
                    "Is this machine connected to the internet?")
        results = []
        for item in (data.get("results") or []):
            title = str(item.get("title", "(no title)")).strip()
            u = str(item.get("url", "")).strip()
            if include_full_text:
                snippet = re.sub(r"\s+", " ", str(item.get("text", ""))).strip()[:2500]
            else:
                snippet = re.sub(r"\s+", " ",
                                 " ".join(str(h) for h in (item.get("highlights") or []))).strip()[:800]
            results.append(f"• {title}\n  {u}\n  {snippet}")
        if not results:
            return f"No Exa results found for “{query}”."
        return (f"Exa results for “{query}”:\n\n" + "\n\n".join(results))[:8000]

    def _tool_firecrawl_scrape(self, url: str = "") -> str:
        """Firecrawl scrape (POST https://api.firecrawl.dev/v1/scrape).

        Metered service: each scrape consumes one of the user's limited monthly
        credits. The model-facing description + system prompt steer it to be used
        only when plain fetch_url cannot get the content (JS-rendered pages,
        anti-bot walls) - never as a first choice for static pages.
        """
        key = (self.settings.get("firecrawl_api_key") or "").strip()
        if not key:
            return ("ERROR: No Firecrawl API key set. Get one at firecrawl.dev and paste it "
                    "into 'Firecrawl API Key' in Settings.")
        if not re.match(r"^https?://", url or ""):
            return "ERROR: URL must start with http:// or https://"
        body = {"url": url, "formats": ["markdown"]}
        req = urllib.request.Request(
            "https://api.firecrawl.dev/v1/scrape",
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json",
                     "User-Agent": "Deskpilot/1.0"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8", "replace"))
        except Exception as e:
            code = getattr(e, "code", None)
            if code == 401:
                return ("ERROR: Firecrawl rejected the API key (HTTP 401). "
                        "Check 'Firecrawl API Key' in Settings.")
            if code is not None:
                try:
                    detail = json.loads(e.read().decode("utf-8", "replace")).get("error", "")
                except Exception:
                    detail = ""
                return f"ERROR: Firecrawl scrape failed (HTTP {code}){': ' + str(detail) if detail else ''}."
            return (f"ERROR: Could not reach api.firecrawl.dev ({e}). "
                    "Is this machine connected to the internet?")
        if data.get("success") is False or not data.get("data"):
            err = str(data.get("error") or "unknown error")
            return f"ERROR: Firecrawl could not scrape {url}: {err}"
        md = str((data.get("data") or {}).get("markdown", "")).strip()
        if not md:
            return f"Firecrawl scraped {url} but no readable text content was found."
        out = f"Content from {url} (via Firecrawl):\n\n{md[:6000]}"
        if len(md) > 6000:
            out += f"\n[... truncated, {len(md)} chars total]"
        return out

    def _tool_fetch_url(self, url: str = "") -> str:
        if not re.match(r"^https?://", url or ""):
            return "ERROR: URL must start with http:// or https://"
        # SSRF guard: block loopback / LAN / cloud-metadata addresses unless the
        # user enabled 'Allow Local Network' in Settings.
        blocked = _fetch_url_blocked(url, bool(self.settings.get("allow_local_network")))
        if blocked:
            return (f"ERROR: Refusing to fetch {url}: {blocked}. "
                    "If this is a machine on your own network that the app genuinely "
                    "needs, the user can enable 'Allow Local Network' in Settings.")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Deskpilot)"})
        capped = False
        _FETCH_REDIRECT_HANDLER._allow_local = bool(self.settings.get("allow_local_network"))
        try:
            with _FETCH_OPENER.open(req, timeout=20) as resp:
                chunks: List[bytes] = []
                total = 0
                while True:
                    if self._abort_requested():
                        return ("ERROR: Fetch was stopped by the user before it finished "
                                f"(downloaded {total} bytes of {url}).")
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > FETCH_BODY_LIMIT:
                        capped = True      # stop downloading; use what we have
                        break
                    chunks.append(chunk)
                data = b"".join(chunks)
                content_encoding = (resp.headers.get("Content-Encoding") or "").lower()
        except _FetchRedirectBlocked as e:
            return (f"ERROR: Refusing to fetch {url}: {e}. "
                    "If this is a machine on your own network that the app genuinely "
                    "needs, the user can enable 'Allow Local Network' in Settings.")
        except urllib.error.HTTPError as e:
            return f"ERROR: Failed to fetch {url}: HTTP {e.code} {e.reason}"
        except Exception as e:
            return f"ERROR: Failed to fetch {url}: {e}"
        if content_encoding == "gzip" or data[:2] == b"\x1f\x8b":   # gzip payload
            try:
                data = gzip.decompress(data)
            except Exception:
                pass
            if len(data) > FETCH_BODY_LIMIT:      # cap DEcompressed size too (zip-bomb guard)
                data = data[:FETCH_BODY_LIMIT]
                capped = True
        raw = data.decode("utf-8", "replace")
        text = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", raw)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        text = html_mod.unescape(text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
        if not text:
            return f"Fetched {url} but no readable text content was found."
        out = f"Content from {url}:\n\n{text[:6000]}"
        if len(text) > 6000:
            out += f"\n[... truncated, {len(text)} chars total]"
        if capped:
            out += f"\n[... download stopped at {FETCH_BODY_LIMIT // (1024 * 1024)} MB]"
        note = _bot_wall_note(text)
        if note:
            out += "\n\n" + note
            if (self.settings.get("firecrawl_api_key") or "").strip():
                out += ("\n[Tip] This wall often blocks plain HTTP clients. firecrawl_scrape can usually get "
                        "past it - but each scrape uses one of the user's limited monthly credits, so only "
                        "escalate when the content is actually needed.")
        return out

    def _find_node(self) -> Optional[str]:
        exe = shutil.which("node") or shutil.which("node.exe")
        if exe:
            return exe
        if sys.platform == "win32":
            for cand in (r"C:\Program Files\nodejs\node.exe",
                         r"C:\Program Files (x86)\nodejs\node.exe"):
                if os.path.exists(cand):
                    return cand
        return None

    def _tool_run_javascript(self, code: str = "") -> str:
        # Fail LOUDLY when no code arrived: a malformed tool-call JSON used to
        # silently dispatch this handler with empty args (an empty .js file runs
        # fine and prints nothing), which looked like "the loop produced no output".
        if not (code or "").strip():
            return ("ERROR: No JavaScript code was received. The tool-call arguments were "
                    "empty or malformed - re-issue the run_javascript call with the full "
                    "'code' argument as valid JSON.")
        node = self._find_node()
        if not node:
            return "ERROR: Node.js ('node') was not found on PATH. Install it from https://nodejs.org"
        tmp = tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8")
        try:
            tmp.write(code)
            tmp.close()
            if os.name != "nt":
                try:
                    os.chmod(tmp.name, 0o600)
                except OSError:
                    pass
            p = self._run_cancellable([node, tmp.name], JS_TIMEOUT,
                                      abort=self._abort_requested)
            if p is None:
                if self._abort_requested():
                    return ("ERROR: JavaScript execution was stopped by the user before it "
                            f"finished (the node process was killed after running under "
                            f"{JS_TIMEOUT} s). Re-issue the call with shorter/cheaper code.")
                return f"ERROR: JavaScript execution timed out after {JS_TIMEOUT} seconds."
            out = (p.stdout or "")
            # Mark the cap so a cut-off result is never mistaken for "no output".
            if len(out) > 8000:
                out = out[:8000] + "\n[... stdout truncated at 8000 chars - print less per call or write to a file and read it back]"
            if p.stderr:
                out += ("\n[stderr]\n" + p.stderr) if out else ("[stderr]\n" + p.stderr)
            return (out or "(no output)")
        except Exception as e:
            return f"ERROR: Failed to execute JavaScript: {e}"
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    def _safe_path(self, raw_path: str) -> Path:
        """Resolve a tool-supplied path with traversal + system-dir protection.

        Raises PermissionError if any path COMPONENT of the (user-expanded) raw
        string is a parent reference ('..', including dot/space-padded forms like
        '.. ' or '. .') - see _traversal_component, which deliberately inspects
        components rather than the raw substring so legal names such as a..b.txt or
        foo...bar are NOT rejected. Also raises if the resolved path equals or sits
        inside a forbidden core system directory for this OS.
        """
        raw = str(raw_path or "")
        expanded = os.path.expanduser(raw)
        bad = _traversal_component(expanded)
        if bad is not None:
            raise PermissionError(f"path traversal blocked ('..' not allowed): {raw!r}")
        pth = Path(expanded).resolve()
        p_str = str(pth).lower()
        hit = _forbidden_dir_hit(p_str)
        if hit is not None:
            raise PermissionError(f"path is inside a protected system directory ({hit}): {pth}")
        # Optional user-chosen workspace (Settings -> File Workspace): when set, every path must
        # also live inside it. Blank = unrestricted (the system-dir guard above still applies).
        _st = getattr(self, "settings", None) or {}
        ws_raw = str(_st.get("file_workspace") or "").strip()
        if ws_raw:
            try:
                ws = Path(os.path.expanduser(ws_raw)).resolve()
            except Exception as e:
                raise PermissionError(f"invalid file workspace configured ({ws_raw!r}): {e}")
            ws_str = str(ws).lower()
            wsh = _forbidden_dir_hit(ws_str)
            if wsh is not None:
                raise PermissionError(
                    f"file workspace itself is inside a protected system directory ({wsh}): {ws}")
            # _dir_prefix(), NOT ws_str + os.sep: when File Workspace is a ROOT folder
            # (a drive root like D: + separator, or '/' on Linux) it already ends with a
            # separator, and appending another one matches NO child path - every legal file
            # would then be refused as "outside the workspace". Lowercase string compare is
            # kept deliberately: Path.is_relative_to() would break case-insensitive Windows
            # matching after resolve() lowercases both sides.
            if p_str != ws_str and not p_str.startswith(_dir_prefix(ws_str)):
                raise PermissionError(f"path is outside the allowed file workspace ({ws}): {pth}")
        return pth

    def _tool_list_directory(self, path: str = "", max_entries: int = 200) -> str:
        """List a directory's contents: folders first, then files with size + mtime.
        Read-only; goes through _safe_path so the system-dir guard and the
        optional File Workspace restriction apply exactly as for read_local_file."""
        try:
            p = self._safe_path(path or ".")
        except PermissionError as e:
            return f"ERROR: Access denied (blocked path): {e}"
        if not p.exists():
            return f"ERROR: Directory not found: {p}"
        if not p.is_dir():
            return f"ERROR: Not a directory: {p} (it is a file - use read_local_file)"
        try:
            entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        except Exception as e:
            return f"ERROR: Could not list {p}: {e}"
        try:
            n = max(1, min(int(max_entries or 200), 1000))
        except (TypeError, ValueError):
            n = 200
        lines: List[str] = []
        dirs = files = 0
        for e in entries[:n]:
            try:
                if e.is_dir():
                    dirs += 1
                    lines.append(f"📁 {e.name}/")
                else:
                    files += 1
                    st = e.stat()
                    size = f"{st.st_size / 1024:.1f} KB" if st.st_size >= 1024 else f"{st.st_size} B"
                    mtime = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
                    lines.append(f"📄 {e.name}  ({size}, {mtime})")
            except OSError:
                lines.append(f"❓ {e.name}  (unreadable)")
        out = (f"Contents of {p} ({dirs} folders, {files} files shown):\n\n"
               + "\n".join(lines) if lines else f"Empty directory: {p}")
        if len(entries) > n:
            out += f"\n[... {len(entries) - n} more entries not shown]"
        return out[:TOOL_OUTPUT_LIMIT]

    def _tool_read_local_file(self, path: str = "") -> str:
        try:
            p = self._safe_path(path)
            if not p.is_file():
                return f"ERROR: File not found: {p}"
            if p.suffix.lower() == ".pdf":
                if PdfReader is None:
                    return "ERROR: 'pypdf' is not installed. Run:  pip install pypdf"
                reader = PdfReader(str(p))
                parts, total = [], 0
                for page in reader.pages:
                    t = page.extract_text() or ""
                    parts.append(t)
                    total += len(t)
                    if total >= FILE_READ_LIMIT:
                        break
                text = "\n".join(parts)[:FILE_READ_LIMIT]
            else:
                data = p.read_bytes()
                if b"\x00" in data[:1024]:
                    return (f"ERROR: {p.name} appears to be a binary file. "
                            f"Only text files and PDFs are supported.")
                text = data.decode("utf-8", "replace")[:FILE_READ_LIMIT]
        except PermissionError as e:
            return f"ERROR: Access denied (blocked path): {e}"
        except Exception as e:
            return f"ERROR: Failed to read file: {e}"
        note = f"\n[... truncated at {FILE_READ_LIMIT} chars]" if len(text) >= FILE_READ_LIMIT else ""
        return f"Contents of {p.name}:\n\n{text}{note}"

    def _tool_write_local_file(self, path: str = "", content: str = "") -> str:
        if not (path or "").strip():
            return "ERROR: No target path provided."
        try:
            p = self._safe_path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        except PermissionError as e:
            return f"ERROR: Access denied (blocked path): {e}"
        except Exception as e:
            return f"ERROR: Failed to write file: {e}"
        return f"OK: wrote {len(content)} characters to {p}"

    def _image_cli_dir(self):
        """Locate the local FLUX/SDXL CLI (ai-imagegen/generate.py + venv Python).

        In a PyInstaller EXE, BASE_DIR is the temp extraction folder, so also look
        next to the running executable and in the current working directory.
        Returns Path or None if not found.
        """
        cands = [BASE_DIR / "ai-imagegen"]
        try:
            cands.append(Path(sys.executable).resolve().parent / "ai-imagegen")
        except Exception:
            pass
        try:
            cands.append(Path.cwd() / "ai-imagegen")
        except Exception:
            pass
        for c in cands:
            if (c / "generate.py").is_file():
                return c
        return None

    def _tool_generate_local_image(self, prompt: str = "", negative_prompt: str = "",
                                   model: str = "flux", width: int = 1600, height: int = 1200,
                                   steps: int = 0) -> str:
        """Generate an image with the local FLUX/SDXL CLI (ai-imagegen/generate.py).

        Runs in a subprocess using the ai-imagegen venv's Python. Takes ~2-3 min per
        image (model load + one-time FP8 quantize, then diffusion), so the timeout is
        generous. Output lands in ai-imagegen/generated/ (+ copy into GEN_DIR).
        """
        cli_dir = self._image_cli_dir()
        if cli_dir is None:
            return ("ERROR: Image CLI not found. Put the 'ai-imagegen' folder (with generate.py "
                    "and its .venv) in the same folder as this app/EXE, or next to the script.")
        py = cli_dir / ".venv" / "Scripts" / "python.exe"
        if not py.is_file():
            return (f"ERROR: venv Python not found at {py}. Re-run setup.bat in ai-imagegen.")

        model = str(model or "flux").lower()
        if model not in ("flux", "sdxl"):
            model = "flux"
        cmd = [str(py), str(cli_dir / "generate.py"), str(prompt or ""), "--model", model,
               "--width", str(int(width or 1600)), "--height", str(int(height or 1200))]
        if negative_prompt:
            cmd += ["--negative", str(negative_prompt)]
        if int(steps or 0) > 0:
            cmd += ["--steps", str(int(steps))]

        self._post(lambda: self._set_status("🎨 Generating image (local FLUX/SDXL, ~2-3 min)…"))
        try:
            # cwd = the CLI folder. generate.py resolves everything from its own __file__, so
            # this changes nothing today; it is defensive (a child that DID rely on a relative
            # path would then find ai-imagegen/, not %TEMP%).
            proc = self._run_cancellable(cmd, IMAGE_CLI_TIMEOUT, abort=self._abort_requested,
                                         cwd=str(cli_dir))
        except Exception as e:
            return f"ERROR: Failed to run image CLI: {e}"
        if proc is None:
            if self._abort_requested():
                return ("ERROR: Image generation was stopped by the user before it finished "
                        "(the CLI process was killed).")
            return f"ERROR: Image generation timed out after {IMAGE_CLI_TIMEOUT // 60} minutes."

        tail = (proc.stdout or "").strip().splitlines()[-6:]
        if proc.returncode != 0:
            err_tail = (((proc.stderr or "") + "\n".join(tail)).strip())[-800:]
            return f"ERROR: Image CLI failed (exit {proc.returncode}):\n{err_tail}"

        m = re.search(r"SAVED:\s*(\S+)", proc.stdout or "")
        if not m:
            return "ERROR: Image CLI finished but reported no saved file.\n" + "\n".join(tail)
        out_path = Path(m.group(1))
        try:
            shutil.copy2(out_path, GEN_DIR / out_path.name)   # keep a copy in the app's image folder
        except Exception:
            pass
        return f"OK: image generated and saved to {out_path}"

    def _tool_capture_screen(self) -> str:
        if ImageGrab is None:
            return "ERROR: Pillow is not installed (pip install Pillow) — screen capture unavailable."
        try:
            try:
                img = ImageGrab.grab(all_screens=True)
            except TypeError:
                img = ImageGrab.grab()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Downscale + JPEG-compress before saving: a full-res multi-monitor
            # PNG can be several MB, and the payload rides along with every
            # request while it is the active screenshot. ~1280px JPEG q75 keeps
            # on-screen text legible for vision models at a fraction of the size.
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            data, mime = _downscale_image_bytes(buf.getvalue())
            out_path = GEN_DIR / (f"screen_{ts}.jpg" if mime == "image/jpeg" else f"screen_{ts}.png")
            out_path.write_bytes(data)
        except Exception as e:
            return f"ERROR: Screenshot failed: {e}"
        return (f"OK: screenshot captured, queued for visual analysis on the next model call. "
                f"Path: {out_path}")

    def _tool_get_clipboard_text(self) -> str:
        """Reads the clipboard safely: Tk access must happen on the main thread,
        so this marshals a callback and blocks until it completes."""
        result: Dict[str, Any] = {}
        ev = threading.Event()

        def do():
            try:
                t = self.root.clipboard_get()
                if not t.strip():
                    result["error"] = "Clipboard is empty."
                else:
                    if len(t) > CLIPBOARD_LIMIT:
                        result["text"] = (t[:CLIPBOARD_LIMIT] +
                            f"\n[truncated: showing {CLIPBOARD_LIMIT} of {len(t)} chars - "
                            "ask the user to copy less if you need the rest]")
                    else:
                        result["text"] = t
            except Exception as e:
                result["error"] = f"Clipboard does not contain plain text ({e})"
            finally:
                ev.set()

        self._post(do)
        ev.wait(timeout=10)
        return str(result.get("text") or f"ERROR: {result.get('error', 'no clipboard access')}")

    # ════════════════════════════════════════════════════════════════════
    #  ATTACHMENTS
    # ════════════════════════════════════════════════════════════════════

    def attach_files(self) -> None:
        paths = filedialog.askopenfilenames(
            parent=self.root, title="Attach files",
            filetypes=[("Supported files", "*.png *.jpg *.jpeg *.pdf *.txt *.md *.json *.csv "
                                            "*.py *.js *.html *.xml *.log"),
                       ("All files", "*.*")])
        for p in paths:
            self._attachments.append(Path(p))
        self._render_chips()

    def _remove_attachment(self, p: Path) -> None:
        if p in self._attachments:
            self._attachments.remove(p)
        self._render_chips()

    def _render_chips(self) -> None:
        for w in self.chips_frame.winfo_children():
            w.destroy()
        for p in self._attachments:
            chip = tk.Frame(self.chips_frame, bg=COL["bg_raised"])
            chip.pack(side="left", padx=(0, 6), pady=3)
            icon = "🖼" if p.suffix.lower() in IMAGE_EXTS else ("📕" if p.suffix.lower() == ".pdf" else "📄")
            tk.Label(chip, text=f"{icon} {p.name}", bg=COL["bg_raised"], fg=COL["text"],
                     font=F(11)).pack(side="left", padx=(8, 2), pady=3)
            tk.Button(chip, text="✖", command=lambda p=p: self._remove_attachment(p),
                      bg=COL["bg_raised"], fg=COL["danger"], activebackground="#5B2323",
                      relief="flat", bd=0, width=2, cursor="hand2",
                      font=F(11)).pack(side="left", padx=(0, 6))

    def _attachment_to_part(self, p: Path) -> List[dict]:
        ext = p.suffix.lower()
        try:
            if ext in IMAGE_EXTS:
                # Copy (downscaled) into the app's image store and reference it -
                # keeps deskpilot_chats.json small; _prepare_request_messages()
                # attaches the payload at request time. Falls back to inline
                # base64 if the copy fails so the attachment still works.
                try:
                    data, mime = _downscale_image_bytes(p.read_bytes())
                    ext2 = ".jpg" if mime == "image/jpeg" else ".png"
                    fp = GEN_DIR / f"attach_{uuid.uuid4().hex[:8]}{ext2}"
                    fp.write_bytes(data)
                    return [
                        {"type": "text", "text": f"[Attached image: {p.name}]"},
                        {"type": "image_ref", "path": str(fp)},
                    ]
                except Exception:
                    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
                    mime = "image/png" if ext == ".png" else "image/jpeg"
                    return [
                        {"type": "text", "text": f"[Attached image: {p.name}]"},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    ]
            if ext == ".pdf":
                if PdfReader is None:
                    return [{"type": "text",
                             "text": f"[PDF attached but pypdf is not installed — could not extract text from {p.name}]"}]
                reader = PdfReader(str(p))
                txt = "\n".join((pg.extract_text() or "") for pg in reader.pages)[:FILE_READ_LIMIT]
                return [{"type": "text", "text": f"--- Contents of attached PDF '{p.name}' ---\n{txt}"}]
            if ext in TEXT_EXTS:
                txt = p.read_text("utf-8", "replace")[:FILE_READ_LIMIT]
                return [{"type": "text", "text": f"--- Contents of attached file '{p.name}' ---\n{txt}"}]
        except Exception as e:
            return [{"type": "text", "text": f"[Failed to read attachment {p.name}: {e}]"}]
        return [{"type": "text", "text": f"[Unsupported attachment type skipped: {p.name}]"}]

    # ════════════════════════════════════════════════════════════════════
    #  AUDIO — TTS (kokoro-onnx + sounddevice) & STT dictation (speech_recognition)
    # ════════════════════════════════════════════════════════════════════

    def _tts_available(self) -> bool:
        return Kokoro is not None and sounddevice is not None

    def stop_tts(self) -> None:
        """Halt any in-flight TTS playback (safe to call when idle)."""
        self._tts_stop.set()
        if sounddevice is not None:
            try:
                sounddevice.stop()
            except Exception:
                pass

    def toggle_tts(self) -> None:
        if not self._tts_available():
            self._set_status("\U0001F50A TTS unavailable — pip install kokoro-onnx sounddevice")
            try:
                self.tts_btn.configure(state="disabled", fg=COL["text_dim"])
            except tk.TclError:
                pass
            return
        if self.settings.get("tts_enabled"):
            # turning OFF also interrupts whatever is playing right now
            self.stop_tts()
        else:
            self._tts_stop.clear()
        self.settings["tts_enabled"] = not self.settings.get("tts_enabled", False)
        save_settings(self.settings)
        on = self.settings["tts_enabled"]
        try:
            self.tts_btn.configure(fg=COL["success"] if on else COL["text_dim"])
        except tk.TclError:
            pass
        self._set_status("\U0001F50A Text-to-speech " + ("ON" if on else "OFF"))

    def speak(self, text: str) -> None:
        """Speak a finished assistant message (markdown stripped), off-thread."""
        if not self.settings.get("tts_enabled") or not self._tts_available():
            return
        clean = re.sub(r"```.*?```", " Code block. ", text, flags=re.S)
        clean = re.sub(r"[#*_`>]+", " ", clean)
        clean = re.sub(r"\s+", " ", clean).strip()[:2000]
        if not clean:
            return
        # chunk by sentences so playback can start (and be interrupted) sooner
        chunks = [c.strip() for c in re.split(r"(?<=[.!?])\s+", clean) if c.strip()]
        self._tts_stop.clear()
        threading.Thread(target=self._tts_worker, args=(chunks,), daemon=True).start()

    def _tts_worker(self, chunks: list) -> None:
        """Speak sentence chunks using a 1-step look-ahead pipeline.

        A producer thread synthesizes each sentence while the previous one is
        still playing (double buffering), so there is no dead air between
        sentences on CPU. The stop flag + sounddevice.stop() interrupt playback
        promptly; when idle-waiting for audio we poll every 250 ms so a stop
        request is honored even before any audio exists yet.
        """
        import numpy as _np
        try:
            model = _kokoro_model()
        except Exception as e:
            self._post(lambda e=e: self._set_status("\U0001F50A TTS error: " + str(e)[:200]))
            return

        q = queue.Queue(maxsize=2)   # bounded; generation is slower than playback anyway
        sentinel = object()

        def _producer() -> None:
            try:
                for chunk in chunks:
                    if self._tts_stop.is_set():
                        break
                    try:
                        audio, rate = model.create(chunk, voice="af_heart", speed=1.0, lang="en-us")
                    except Exception as e:
                        self._post(lambda e=e: self._set_status("\U0001F50A TTS error: " + str(e)[:200]))
                        break
                    q.put((_np.asarray(audio, dtype=_np.float32), rate))
            finally:
                q.put(sentinel)      # always unblock the player, even on error/stop

        threading.Thread(target=_producer, daemon=True).start()

        while True:
            if self._tts_stop.is_set():
                break
            try:
                item = q.get(timeout=0.25)
            except queue.Empty:
                continue
            if item is sentinel:
                break
            audio, rate = item
            if self._tts_stop.is_set():
                break
            try:
                sounddevice.play(audio, rate)
                sounddevice.wait()
            except Exception:
                break  # playback failed or was interrupted - stop quietly

    def toggle_mic(self) -> None:
        """Toggle continuous dictation.

        While active the microphone stays open and every completed phrase is
        transcribed live into the input box; it stops when Send is pressed or
        the mic button is clicked again, so multi-sentence speech is never cut
        off mid-thought.
        """
        if sr is None:
            self._set_status("🎤 Dictation unavailable — pip install SpeechRecognition PyAudio")
            return
        if self._mic_active:
            self._stop_mic("Dictation stopped")
            return
        self._mic_stop.clear()
        self._mic_active = True
        try:
            self.mic_btn.configure(bg=COL["accent"], fg="#FFFFFF")
        except tk.TclError:
            pass
        engine = "local Whisper" if WhisperModel is not None else "Google (cloud)"
        self._set_status(f"🎤 Listening via {engine}… speak freely; hit Send (or click the mic) to stop")
        if WhisperModel is not None and _WHISPER_MODEL is None:
            # Preload the local model in the background so the first phrase isn't
            # delayed by the one-time load; the worker waits for it to finish.
            _whisper_preload()
            self._set_status("🎤 Loading local Whisper model…")
        self._stt_thread = threading.Thread(target=self._stt_worker, daemon=True)
        self._stt_thread.start()

    def _stop_mic(self, status_msg: str) -> None:
        """Stop continuous dictation; transcribed text stays in the input box."""
        self._mic_active = False
        self._mic_stop.set()
        try:
            self.mic_btn.configure(bg=COL["bg_raised"], fg=COL["text"])
        except tk.TclError:
            pass
        self._set_status(f"🎤 {status_msg}")

    def _stt_insert(self, text: str) -> None:
        """Append a transcribed phrase to the input box (UI thread)."""
        try:
            existing = self.input_text.get("1.0", "end-1c")
            prefix = "" if not existing else (" " if not existing.endswith((" ", "\n")) else "")
            self.input_text.insert("end-1c", prefix + text)
            self._see_input("end")      # keep the newest dictated phrase visible
            self._set_status(f"🎤 Heard: {text[:60]}")
        except tk.TclError:
            pass

    def _stt_worker(self) -> None:
        """Continuous dictation loop (background thread).

        Keeps the mic open until stopped. Each time a pause ends a phrase it is
        transcribed and appended to the input box, so long speech keeps going;
        Recognition is local-first: faster-whisper when installed (audio stays on
        this machine), otherwise Google's cloud STT.
        silence alone never closes the microphone.
        """
        try:
            rec = sr.Recognizer()
            mic = _open_microphone()   # falls back to first real capture device
            mic.__enter__()
            if getattr(mic, "stream", None) is None:
                raise OSError("could not open the microphone stream (no usable audio input in this session)")
            src = mic
            if WhisperModel is not None:
                # Block until the background preload (started in toggle_mic) has
                # finished, so the first phrase isn't delayed by the model load.
                # If the preload failed, this performs the load here instead and
                # any error surfaces on the first phrase as before.
                _whisper_model()
                if self._mic_active:
                    # Preload done - restore the listening status (the bar showed
                    # "Loading local Whisper model…" while we waited).
                    self._post(lambda: self._set_status(
                        "\U0001F3A4 Listening via local Whisper\u2026 speak freely; hit Send (or click the mic) to stop"))
            try:
                    while not self._mic_stop.is_set():
                        try:
                            audio = rec.listen(src, timeout=5, phrase_time_limit=30)
                        except Exception as e:
                            if "WaitTimeout" in type(e).__name__:
                                continue        # silence — keep listening until stopped
                            raise
                        if self._mic_stop.is_set():
                            break
                        try:
                            if WhisperModel is not None:
                                text = whisper_transcribe_local(audio.get_wav_data())
                                if not text:
                                    continue    # silence / unintelligible - keep listening
                            else:
                                text = rec.recognize_google(audio)
                        except sr.UnknownValueError:
                            continue            # couldn't understand it — keep listening
                        except sr.RequestError as e:
                            self._post(lambda m=str(e): self._set_status(f"🎤 Dictation error (network?): {m}"))
                            return              # Google unreachable — no point continuing
                        except Exception as e:   # local engine failure (model download/load/decode)
                            self._post(lambda m=str(e): self._set_status(f"Dictation error: {m}"))
                            return              # retrying won't help; user can restart the mic
                        if not self._mic_active:
                            break               # stopped while transcribing; discard tail
                        self._post(lambda t=text: self._stt_insert(t))
            finally:
                try:
                    mic.__exit__(None, None, None)
                except Exception:
                    pass   # SR's __exit__ crashes if the stream never opened; harmless here
        except Exception as e:
            if "WaitTimeout" not in type(e).__name__:
                self._post(lambda m=str(e): self._set_status(f"🎤 Dictation error: {m}"))
        finally:
            self._mic_active = False

            def _reset_btn():
                try:
                    self.mic_btn.configure(bg=COL["bg_raised"], fg=COL["text"])
                except tk.TclError:
                    pass

            self._post(_reset_btn)   # Tk calls must happen on the UI thread

    # ════════════════════════════════════════════════════════════════════
    #  SETTINGS DIALOG
    # ════════════════════════════════════════════════════════════════════

    def _apply_ui_font(self) -> None:
        """Re-scale every visible font to the user's chosen UI font size.

        Chat content tags are rebuilt via _configure_text_tags(); everything else
        is found by walking the widget tree and bumping any Segoe UI font by the
        delta change (ttk widgets that got an explicit font= option are covered
        too; style-driven fonts carry no per-widget value and are left alone).
        """
        global FONT_DELTA
        try:
            new_delta = int(self.settings.get("ui_font_size", FONT_BASE)) - FONT_BASE
        except (TypeError, ValueError):
            return
        old_delta = FONT_DELTA
        if new_delta == old_delta:
            return
        FONT_DELTA = new_delta
        self._configure_text_tags()

        def bump(w) -> None:
            try:
                cur = w.cget("font")
            except tk.TclError:
                return
            if not isinstance(cur, str) or "Segoe" not in cur:
                return
            toks = cur.split()
            for i, t in enumerate(toks):
                if t.isdigit():
                    toks[i] = str(max(7, int(t) + (new_delta - old_delta)))
                    w.configure(font=" ".join(toks))
                    break

        def walk(w) -> None:
            bump(w)
            try:
                kids = w.winfo_children()
            except tk.TclError:
                return
            for c in kids:
                walk(c)

        walk(self.root)

    def open_settings(self) -> None:
        top = tk.Toplevel(self.root)
        top.title(f"{APP_NAME} — Settings")
        top.configure(bg=COL["bg_main"])
        top.transient(self.root)
        # Map the window BEFORE grabbing: on some Tk builds (e.g. Tk 9) a grab
        # taken while the toplevel is still unmapped leaves keyboard focus on
        # the main window, so typing in the fields below silently goes nowhere.
        try:
            top.update_idletasks()
        except tk.TclError:
            pass
        try:
            top.grab_set()
        except tk.TclError:
            pass

        rows = [
            ("Server URL (OpenAI-compatible base)", "server_url"),
            ("API Key",                             "api_key"),
            ("Model Name",                          "model_name"),
            ("SearXNG URL (blank = LLM server IP :8080)", "searxng_url"),
            ("UI Font Size (8-20, default 12)",     "ui_font_size"),
            ("Max Tokens per reply (0 = server default)", "max_tokens"),
            ("Turn time limit seconds (0 = NO LIMIT; cap a runaway tool loop)", "turn_time_limit"),
            ("Sampling",  "sampling_defaults"),
            ("File Workspace (confines Read/Write File tools; blank = unrestricted)", "file_workspace"),
            ("Handoff threshold % (auto-summary at this context usage; 0 = off)", "handoff_threshold_pct"),
            ("Handoff re-arm % (grow this much more before summarizing again; 0 = every step)", "handoff_rearm_pct"),
            ("Handoff notes folder (blank = app working folder; one file per handoff)", "handoff_notes_dir"),
            ("Custom System Prompt (appended to the built-in system prompt on every request; blank = none)", "custom_system_prompt"),
            ("Exa API Key (blank = Exa Search disabled)", "exa_api_key"),
            ("Firecrawl API Key (metered; blank = Firecrawl Scrape disabled)", "firecrawl_api_key"),
            ("Allow Local Network (fetch_url)", "allow_local_network"),
            ("MCP Servers (stdio subprocesses; tools appear in the permission bar)", "mcp_servers"),
        ]
        entries: Dict[str, tk.Entry] = {}
        model_combo: Optional[ttk.Combobox] = None
        local_net_var: Optional[tk.BooleanVar] = None
        _note_var: Optional[tk.BooleanVar] = None
        custom_prompt_text: Optional[tk.Text] = None
        for i, (label, key) in enumerate(rows):
            if key == "mcp_servers":
                # Managed list of stdio MCP servers (Add/Remove); connecting is
                # live - discovered tools appear in the permission bar.
                tk.Label(top, text=label, bg=COL["bg_main"], fg=COL["text_dim"],
                         font=F(11)).grid(row=i, column=0, sticky="w", padx=(16, 8), pady=6)
                mcp_frame = tk.Frame(top, bg=COL["bg_main"])
                mcp_frame.grid(row=i, column=1, columnspan=2, sticky="ew", pady=6, padx=(0, 8))
                self._draw_mcp_server_rows(mcp_frame, top)
                continue
            tk.Label(top, text=label, bg=COL["bg_main"], fg=COL["text_dim"],
                     font=F(11)).grid(row=i, column=0, sticky="w", padx=(16, 8), pady=6)
            if key == "sampling_defaults":
                # Sampler parameters (Top P / Top K / Min P / Repetition / Presence / Frequency),
                # GLOBAL for every chat session by design - the chat UI stays uncluttered.
                # Built in a sub-frame so these six rows never appear among the dialog's own
                # children (the smoke suite counts plain Entries there positionally).
                tk.Label(top, text=label, bg=COL["bg_main"], fg=COL["text_dim"],
                         font=F(11)).grid(row=i, column=0, sticky="w", padx=(16, 8), pady=6)
                samp_frame = tk.Frame(top, bg=COL["bg_main"])
                samp_frame.grid(row=i, column=1, columnspan=2, sticky="ew", pady=6, padx=(0, 8))
                for _r, (_sk, _slabel, _lo, _hi, _si, _st, _so) in enumerate(SAMPLING_SCHEMA):
                    tk.Label(samp_frame, text=_slabel, bg=COL["bg_main"], fg=COL["text_dim"],
                             font=F(10)).grid(row=_r, column=0, sticky="w", padx=(0, 8))
                    _se = tk.Entry(samp_frame, width=10, bg=COL["bg_deep"], fg=COL["text"],
                                   insertbackground=COL["text"], relief="flat", font=F(10))
                    _sv = (self.settings.get("sampling_defaults") or {}).get(_sk, "")
                    _se.insert(0, "" if _sv is None else str(_sv))
                    _se.grid(row=_r, column=1, sticky="w", pady=1)
                    entries[_sk] = _se
                _note_var = tk.BooleanVar(value=bool(self.settings.get("show_sampling_note", True)))
                tk.Checkbutton(samp_frame, text="Show a \"Sent: ...\" note in chat after each reply, "
                               "listing the sampling parameters that were actually sent",
                               variable=_note_var, bg=COL["bg_main"], fg=COL["text_dim"],
                               activebackground=COL["bg_main"], activeforeground=COL["text"],
                               selectcolor=COL["bg_deep"], font=F(9)) \
                    .grid(row=len(SAMPLING_SCHEMA), column=0, columnspan=2, sticky="w", pady=(4, 0))
                # Tier 3 (v1.1.10): behavioural probe for the six keys above. It reports EVIDENCE from tiny
                # bounded requests into a read-only box - it writes NOTHING to chat history, chats.json or
                # settings, and lives inside this sub-frame, so the dialog's own plain-Entry count is unchanged.
                verify_btn = tk.Button(samp_frame, text="Verify parameters", font=F(10), bg=COL["bg_raised"],
                   fg=COL["text"], activebackground="#4B5563", relief="flat", bd=0, padx=10, pady=2,
                   cursor="hand2", command=lambda: self._verify_sampling(verify_btn, verify_out))
                verify_out = tk.Text(samp_frame, width=58, height=6, wrap="word", state="disabled",
                   bg=COL["bg_deep"], fg=COL["text_dim"], insertbackground=COL["text"], relief="flat", font=F(9))
                verify_btn.grid(row=len(SAMPLING_SCHEMA) + 1, column=0, sticky="w", pady=(6, 0))
                verify_out.grid(row=len(SAMPLING_SCHEMA) + 2, column=0, columnspan=2, sticky="ew", pady=(4, 0))
                continue
            if key == "model_name":
                # Editable combobox: dropdown of live server models + previously
                # used names, but free typing still works (exact IDs matter for
                # strict servers like Unsloth Desktop).
                model_combo = ttk.Combobox(top, values=[], width=44,
                                           style="Tool.TCombobox", font=F(11))
                model_combo.set(self.settings.get("model_name", ""))
                model_combo.grid(row=i, column=1, sticky="ew", pady=6, padx=(0, 8))
                continue
            if key == "custom_system_prompt":
                # Multi-line: persona/style instructions can run to several lines.
                custom_prompt_text = tk.Text(top, width=44, height=5, wrap="word",
                                             bg=COL["bg_deep"], fg=COL["text"],
                                             insertbackground=COL["text"], relief="flat",
                                             font=F(10))
                custom_prompt_text.insert("1.0", self.settings.get("custom_system_prompt", ""))
                custom_prompt_text.grid(row=i, column=1, sticky="ew", pady=6, padx=(0, 16))
                continue
            if key == "allow_local_network":
                local_net_var = tk.BooleanVar(value=bool(self.settings.get("allow_local_network", False)))
                tk.Checkbutton(top, text="Allow fetch_url to reach local network addresses "
                               "(localhost, LAN, cloud metadata) - off by default",
                               variable=local_net_var, bg=COL["bg_main"], fg=COL["text"],
                               activebackground=COL["bg_main"], activeforeground=COL["text"],
                               selectcolor=COL["bg_deep"], font=F(11)) \
                    .grid(row=i, column=1, sticky="w", pady=6, padx=(0, 8))
                continue
            e = tk.Entry(top, width=44, bg=COL["bg_deep"], fg=COL["text"],
                         insertbackground=COL["text"], relief="flat",
                         show="*" if key in ("api_key", "exa_api_key", "firecrawl_api_key") else "")
            e.insert(0, self.settings.get(key, ""))
            e.grid(row=i, column=1, sticky="ew", pady=6, padx=(0, 16))
            entries[key] = e

        # File Workspace row gets a folder picker next to its entry.
        ws_row = next(i for i, (_, k) in enumerate(rows) if k == "file_workspace")

        def _browse_ws() -> None:
            cur = entries["file_workspace"].get().strip()
            d = filedialog.askdirectory(parent=top, title="Choose the File Workspace folder",
                                        initialdir=(cur or str(Path.home())))
            if d:
                entries["file_workspace"].delete(0, "end")
                entries["file_workspace"].insert(0, d)

        tk.Button(top, text="Browse...", command=_browse_ws, bg=COL["bg_raised"], fg=COL["text"],
                  activebackground="#4B5563", relief="flat", bd=0, padx=10, pady=2,
                  cursor="hand2", font=F(10)).grid(row=ws_row, column=2, sticky="e", padx=(0, 16), pady=6)

        # Handoff notes folder row: picker + the EFFECTIVE folder shown live, so it is always
        # obvious where notes are being written (blank field = the app default working folder).
        nd_row = next(i for i, (_, k) in enumerate(rows) if k == "handoff_notes_dir")

        def _browse_notes() -> None:
            cur = entries["handoff_notes_dir"].get().strip()
            d = filedialog.askdirectory(parent=top, title="Choose the handoff notes folder",
                                        initialdir=(cur or str(default_handoff_notes_dir())))
            if d:
                entries["handoff_notes_dir"].delete(0, "end")
                entries["handoff_notes_dir"].insert(0, d)
                _update_notes_note()

        def _effective_notes_dir() -> str:
            try:
                raw = entries["handoff_notes_dir"].get().strip()
            except tk.TclError:
                return ""
            cand = default_handoff_notes_dir() if not raw else Path(raw).expanduser()
            return str(cand)

        def _update_notes_note() -> None:
            try:
                notes_dir_lbl.configure(text="Notes will be written to:\n" + _effective_notes_dir())
            except tk.TclError:
                pass

        nd_btns = tk.Frame(top, bg=COL["bg_main"])
        nd_btns.grid(row=nd_row, column=2, sticky="e", padx=(0, 16), pady=6)
        tk.Button(nd_btns, text="Browse...", command=_browse_notes, bg=COL["bg_raised"], fg=COL["text"],
                  activebackground="#4B5563", relief="flat", bd=0, padx=10, pady=2,
                  cursor="hand2", font=F(10)).pack(side="left")
        notes_dir_lbl = tk.Label(nd_btns, text="", bg=COL["bg_main"], fg=COL["text_dim"],
                                 font=F(9), justify="left", wraplength=260)
        notes_dir_lbl.pack(side="left", padx=(8, 0))
        _update_notes_note()
        entries["handoff_notes_dir"].bind("<KeyRelease>", lambda _e: _update_notes_note())

        # Model Name row: Refresh button re-fetches {server_url}/models in a
        # background thread (same pattern as Test Connection). The request carries
        # no model name, so it works even with a stale/wrong saved model_name.
        m_row = next(i for i, (_, k) in enumerate(rows) if k == "model_name")
        latest_entries: List[dict] = []      # last /models reply (used by Save to normalize)
        loaded_ids: List[str] = []           # models the server reports as loaded right now

        def _refresh_models(silent: bool = False, auto_fix: bool = True) -> None:
            url = entries["server_url"].get().strip()
            key = entries["api_key"].get().strip()
            if not url:
                return
            try:
                refresh_btn.configure(text="Loading...")
                model_combo.configure(state="disabled")
            except tk.TclError:
                return

            def _worker():
                info = fetch_model_info(url, key)
                ids = [str(m.get("id")).strip() for m in info if m.get("id")]

                def _done():
                    try:
                        if not top.winfo_exists():
                            return
                        latest_entries.clear()
                        latest_entries.extend(info)
                        loaded_ids.clear()
                        loaded_ids.extend(loaded_model_ids(info))
                        hist = [h for h in (self.settings.get("model_history") or [])
                                if isinstance(h, str)]
                        merged: List[str] = []
                        for m in list(ids) + hist:          # live list first, then history
                            if m and m not in merged:
                                merged.append(m)
                        model_combo.configure(values=merged[:MODEL_HISTORY_MAX], state="normal")
                        refresh_btn.configure(text="Refresh models")
                        # Loaded-model row: the exact IDs chat requests must use.
                        if loaded_ids:
                            note = "Loaded on server (use these exact IDs): " + ", ".join(loaded_ids)
                        elif info:
                            note = "Server lists models but reports none loaded - pick one to load."
                        else:
                            note = "No model list from the server - saved history shown below."
                        loaded_lbl.configure(text=note[:200])
                        use_btn.configure(state=("normal" if len(loaded_ids) == 1 else "disabled"))
                        # Auto-fill: a stale/unknown name + exactly one loaded model -> adopt it.
                        # Only on the silent open-time fetch; a manual Refresh never overwrites
                        # what the user just typed or picked.
                        if auto_fix and len(loaded_ids) == 1:
                            cur = str(model_combo.get()).strip()
                            if cur != loaded_ids[0] and cur not in ids:
                                model_combo.set(loaded_ids[0])
                                self._set_status(
                                    f"Model Name auto-corrected to the loaded model: {loaded_ids[0]}")
                        elif auto_fix and len(loaded_ids) > 1:
                            cur = str(model_combo.get()).strip()
                            if cur not in ids:
                                self._set_status("Several models are loaded - pick one from the dropdown")
                        self._update_ctx_topbar(info)       # top-bar context readout
                        if ids:
                            self._set_status(f"Model list updated: {len(ids)} model(s) from server")
                        elif not silent:
                            self._set_status("Model list unavailable - showing saved history")
                    except tk.TclError:
                        pass

                self._post(_done)

            threading.Thread(target=_worker, daemon=True).start()

        refresh_btn = tk.Button(top, text="Refresh models",
                                command=lambda: _refresh_models(False, auto_fix=False),
                                bg=COL["bg_raised"], fg=COL["text"], activebackground="#4B5563",
                                relief="flat", bd=0, padx=10, pady=2, cursor="hand2", font=F(10))
        refresh_btn.grid(row=m_row, column=2, sticky="e", padx=(0, 16), pady=6)

        # "Use loaded model": copies the single loaded model's exact server ID into
        # Model Name - the reliable fix when you do not know the name yourself.
        use_btn = tk.Button(top, text="Use loaded model", state="disabled",
                            bg=COL["bg_raised"], fg=COL["text"], activebackground="#4B5563",
                            relief="flat", bd=0, padx=10, pady=2, cursor="hand2", font=F(10))
        use_btn.grid(row=m_row, column=3, sticky="e", padx=(0, 16), pady=6)

        def _use_loaded() -> None:
            try:
                if len(loaded_ids) == 1 and top.winfo_exists():
                    model_combo.set(loaded_ids[0])
                    self._set_status(f"Model Name set to loaded model: {loaded_ids[0]}")
            except tk.TclError:
                pass

        use_btn.configure(command=_use_loaded)
        # Loaded-model note: created here (the refresh callback references it), packed
        # into the footer frame further down so it never collides with a grid row.
        loaded_lbl = tk.Label(top, text="", bg=COL["bg_main"], fg=COL["text_dim"],
                              font=F(9), justify="left")

        # Force keyboard focus into the dialog and its first field -- then keep
        # re-asserting it for a few seconds. On Windows, the window manager often
        # hands keyboard focus back to the main window right after a Toplevel is
        # opened; that happens asynchronously, i.e. AFTER any focus_force() done
        # during the click handler has run, so user typing would silently land in
        # the chat box instead of these entries (a documented Tk-on-Windows issue).
        # A short self-healing poller beats it: whenever Tk's internal focus
        # wanders outside this dialog while it is open, pull it back within 150 ms.
        def _ensure_dialog_focus() -> None:
            try:
                if not top.winfo_exists():
                    return
                fg = self.root.focus_get()
                inside = bool(fg) and str(fg).startswith(str(top))
                # Don't fight a child dialog (e.g. the Test Connection messagebox):
                # while one is open, let it keep the focus.
                child_open = any(
                    isinstance(c, tk.Toplevel) and c.winfo_exists()
                    for c in top.winfo_children())
                if not inside and not child_open:
                    top.lift()
                    top.focus_force()
                    entries["server_url"].focus_set()
            except tk.TclError:
                return

        def _keep_dialog_focus(n_left: int) -> None:
            try:
                if not top.winfo_exists():
                    return
            except tk.TclError:
                return
            _ensure_dialog_focus()
            if n_left > 0:
                try:
                    top.after(150, _keep_dialog_focus, n_left - 1)
                except tk.TclError:
                    pass

        try:
            top.lift()
            top.focus_force()
            entries["server_url"].focus_set()
        except tk.TclError:
            pass
        try:
            top.after(150, _keep_dialog_focus, 40)   # ~6 s of self-healing focus
        except tk.TclError:
            pass

        # Populate the Model Name dropdown now (live server list + saved history;
        # silent = no status-bar complaint if the server is just down).
        _refresh_models(True)

        def save():
            # Validate File Workspace FIRST so a bad folder can't leave half-saved settings.
            ws_raw = entries["file_workspace"].get().strip()
            if ws_raw:
                try:
                    ws_p = Path(os.path.expanduser(ws_raw)).resolve()
                    hit = _forbidden_dir_hit(str(ws_p).lower())
                    if hit is not None:
                        raise PermissionError(f"that folder is inside a protected system directory ({hit})")
                    ws_p.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    messagebox.showwarning(
                        f"{APP_NAME} - File Workspace",
                        "That folder cannot be used as the File Workspace:\n\n"
                        f"{e}\n\nLeave the field blank to keep the file tools unrestricted.")
                    return
            # Handoff notes folder: blank = default_handoff_notes_dir(); anything else must be an absolute
            # folder this app can create. Validated BEFORE mutating self.settings (same reason as File
            # Workspace: a bad value must not leak the other edited fields into live memory). Notes are
            # working files - never store them inside a protected system directory.
            nd_raw = entries["handoff_notes_dir"].get().strip() if "handoff_notes_dir" in entries else ""
            nd_p = None
            if nd_raw:
                try:
                    nd_p = Path(os.path.expanduser(nd_raw)).resolve()
                    hit = _forbidden_dir_hit(str(nd_p).lower())
                    if hit is not None:
                        raise PermissionError(f"that folder is inside a protected system directory ({hit})")
                    nd_p.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    messagebox.showwarning(
                        f"{APP_NAME} - Handoff notes folder",
                        "That folder cannot be used for handoff notes:\n\n"
                        f"{e}\n\nLeave the field blank to use the app working folder: "
                        f"{default_handoff_notes_dir()}")
                    return
            # Custom System Prompt: multi-line text box (not a plain entry).
            # Validate BEFORE mutating self.settings so an over-limit prompt's early
            # return can't leak the other edited fields into live memory.
            cp = custom_prompt_text.get("1.0", "end-1c").strip() if custom_prompt_text is not None else ""
            if len(cp) > CUSTOM_PROMPT_MAX:
                messagebox.showwarning(
                    f"{APP_NAME} - Custom System Prompt too long",
                    f"That text is {len(cp)} characters; the limit is {CUSTOM_PROMPT_MAX}.\n"
                    "Shorten it and save again (nothing was saved this time).")
                return
            for k, e in entries.items():
                self.settings[k] = e.get().strip()
            self.settings["custom_system_prompt"] = cp
            if local_net_var is not None:
                self.settings["allow_local_network"] = bool(local_net_var.get())
            # Model Name comes from the combobox (editable - free typing still works),
            # then resolved to the exact server ID: a display name or an id missing its
            # namespace prefix would otherwise 404 at chat time.
            model_val = str(model_combo.get()).strip()
            if model_val and latest_entries:
                canon, status = normalize_model_name(model_val, latest_entries)
                if status == "resolved" and canon != model_val:
                    self._set_status(f"Model Name corrected to the server ID: {canon}")
                    model_val = canon
                elif status == "ambiguous":
                    self._set_status("Model Name matches several server models - used as typed")
            self.settings["model_name"] = model_val
            if model_val:
                hist = [h for h in (self.settings.get("model_history") or [])
                        if isinstance(h, str) and h != model_val]
                hist.insert(0, model_val)               # newest first
                self.settings["model_history"] = hist[:MODEL_HISTORY_MAX]
            if ws_raw:
                self.settings["file_workspace"] = str(ws_p)   # normalized absolute path
            if nd_p is not None:
                self.settings["handoff_notes_dir"] = str(nd_p)   # normalized absolute path
            # UI Font Size: numeric, clamped to 8..20 (invalid input -> default)
            try:
                fs = int(float(self.settings.get("ui_font_size", FONT_BASE)))
            except (TypeError, ValueError):
                fs = FONT_BASE
            self.settings["ui_font_size"] = max(8, min(20, fs))
            # Max Tokens: numeric, clamped to 0..262144 (invalid input -> server default)
            try:
                mt = int(float(self.settings.get("max_tokens", 0)))
            except (TypeError, ValueError):
                mt = 0
            self.settings["max_tokens"] = max(0, min(262144, mt))
            # Turn time limit: seconds, clamped 0..TURN_TIME_LIMIT_MAX. 0 (the default) means
            # NO LIMIT - a turn runs until it finishes or the user presses Stop.
            try:
                tl = int(float(self.settings.get("turn_time_limit", TURN_TIME_LIMIT_DEFAULT)))
            except (TypeError, ValueError):
                tl = TURN_TIME_LIMIT_DEFAULT
            self.settings["turn_time_limit"] = max(0, min(TURN_TIME_LIMIT_MAX, tl))
            # Handoff threshold: numeric percent, clamped to 0..100 (invalid -> default).
            try:
                hp = int(float(self.settings.get("handoff_threshold_pct", HANDOFF_THRESHOLD_DEFAULT)))
            except (TypeError, ValueError):
                hp = HANDOFF_THRESHOLD_DEFAULT
            self.settings["handoff_threshold_pct"] = max(0, min(100, hp))
            # Handoff re-arm margin: same clamping rule (invalid -> default).
            try:
                hr = int(float(self.settings.get("handoff_rearm_pct", HANDOFF_REARM_PCT_DEFAULT)))
            except (TypeError, ValueError):
                hr = HANDOFF_REARM_PCT_DEFAULT
            self.settings["handoff_rearm_pct"] = max(0, min(100, hr))
            # Sampling: coerce + clamp each field; blank/garbage -> "" (key omitted from requests).
            _samp: Dict[str, Any] = {}
            for _sk, _sl, _lo, _hi, _si, _st, _so in SAMPLING_SCHEMA:
                _sv = _coerce_sampling(entries.get(_sk).get() if entries.get(_sk) else "", _lo, _hi, _si)
                _samp[_sk] = "" if _sv is None else _sv
            self.settings["sampling_defaults"] = _samp
            # The six fields also land in `entries`, so the generic loop above wrote them as
            # top-level settings keys too. Drop that duplication - they live in ONE place.
            for _sk, *_rest in SAMPLING_SCHEMA:
                self.settings.pop(_sk, None)
            if _note_var is not None:
                self.settings["show_sampling_note"] = bool(_note_var.get())
            # MCP servers may have been added/removed in the dialog: reconnect
            # everything so the permission bar matches the saved configuration.
            self._mcp_connect_all()
            if not save_settings(self.settings):
                messagebox.showwarning(
                    f"{APP_NAME} - Could not save settings",
                    "Settings could not be written to:\n\n"
                    f"{SETTINGS_FILE}\n\nThe folder may be read-only. Saving is\n"
                    "retried automatically when the app closes.")
            self._invalidate_clients()   # server URL / API key may have changed
            self._update_topbar_labels()
            self._refresh_ctx_topbar()   # server/model may have changed - refresh context readout
            top.destroy()
            self._apply_ui_font()   # live re-scale of all fonts (no restart needed)
            self._set_status("⚙ Settings saved")

        btns = tk.Frame(top, bg=COL["bg_main"])
        btns.grid(row=len(rows), column=0, columnspan=4, pady=14)   # spans the model row (cols 0-3)
        loaded_lbl.pack(in_=btns, side="top", pady=(0, 4))   # "Loaded on server: <exact id>"

        tk.Label(btns, text=f"Data files saved in: {SETTINGS_FILE.parent}", bg=COL["bg_main"],
                 fg=COL["text_dim"], font=F(9)).pack(side="top")

        def test():
            url = (entries["server_url"].get().strip() or "http://localhost:11434/v1")
            threading.Thread(target=self._test_connection, args=(url, top), daemon=True).start()

        tk.Button(btns, text="Save", command=save, bg=COL["accent"], fg="#FFFFFF",
                  activebackground="#2563EB", relief="flat", bd=0, padx=18, pady=6,
                  cursor="hand2", font=F(11, "bold")).pack(side="left", padx=6)
        tk.Button(btns, text="Test Connection", command=test, bg=COL["bg_raised"], fg=COL["text"],
                  activebackground="#4B5563", relief="flat", bd=0, padx=12, pady=6,
                  cursor="hand2", font=F(11)).pack(side="left", padx=6)
        tk.Button(btns, text="Cancel", command=top.destroy, bg=COL["bg_raised"], fg=COL["text"],
                  activebackground="#4B5563", relief="flat", bd=0, padx=12, pady=6,
                  cursor="hand2", font=F(11)).pack(side="left", padx=6)

    # ════════════════════════════════════════════════════════════════════
    #  MCP SERVER MANAGEMENT (Settings + permission bar + dispatch)
    # ════════════════════════════════════════════════════════════════════

    def _mcp_servers_cfg(self) -> List[dict]:
        """Sanitized list of configured MCP servers from settings."""
        raw = self.settings.get("mcp_servers") or []
        out: List[dict] = []
        if isinstance(raw, list):
            for s in raw[:MCP_MAX_SERVERS]:
                if not isinstance(s, dict):
                    continue
                name = str(s.get("name", "")).strip()
                command = str(s.get("command", "")).strip()
                if not name or not command:
                    continue
                args = [str(a) for a in (s.get("args") or [])
                        if isinstance(a, (str, int, float))][:32]
                env_raw = s.get("env")
                env = ({str(k): str(v) for k, v in env_raw.items() if isinstance(k, str)}
                       if isinstance(env_raw, dict) else {})
                out.append({"name": name, "command": command, "args": args, "env": env})
        return out

    def _mcp_tool_names(self) -> List[str]:
        """Namespaced names of all currently-discovered MCP tools (stable order)."""
        return list(self._mcp_tool_map.keys())

    def _mcp_schemas(self) -> List[dict]:
        """OpenAI tool schemas for every tool advertised by a connected server.

        Also refreshes self._mcp_tool_map (namespaced name -> (server, raw))
        which the dispatcher and permission bar use."""
        out: List[dict] = []
        new_map: Dict[str, tuple] = {}
        # Snapshot under the lock: the Settings worker may add/remove servers
        # while this runs on the chat worker thread, and mutating a dict
        # mid-iteration raises RuntimeError (verified empirically).
        with self._mcp_lock:
            clients = list(self._mcp_clients.values())
        for client in clients:
            if not client.ready():
                continue
            for t in client.tools:
                raw = str(t.get("name", "")).strip()
                if not raw:
                    continue
                ns = mcp_tool_name(client.name, raw)
                new_map[ns] = (client.name, raw)
                out.append(mcp_tool_schema(client.name, raw, t))
        self._mcp_tool_map = new_map   # full rebuild: stale entries drop out
        return out

    def _mcp_dispatch(self, name: str, args: dict) -> str:
        """Execute a namespaced MCP tool call; returns its text output."""
        entry = self._mcp_tool_map.get(name)
        if entry is None:
            return f"ERROR: MCP tool '{name}' is not available (server not connected)."
        server_name, raw = entry
        client = self._mcp_clients.get(server_name)
        if client is None or not client.ready():
            return (f"ERROR: MCP server '{server_name}' is not running. "
                    f"Reconnect it in Settings -> MCP Servers.")
        try:
            return client.call_tool(raw, args, abort=self._abort_requested)
        except Exception as e:
            return f"ERROR: {name} failed: {e}"

    def _mcp_connect_all(self) -> None:
        """(Re)connect every configured MCP server on a background thread.

        A second call while a connect is already running just marks the work
        pending; the in-flight worker re-runs once when it finishes, so rapid
        double-Saves cannot spawn duplicate (leaked) server subprocesses."""
        self._mcp_connect_pending.set()
        threading.Thread(target=self._mcp_connect_worker_loop, daemon=True).start()

    def _mcp_connect_worker_loop(self) -> None:
        """Run connect passes until no newer save is pending (serialized)."""
        while True:
            with self._mcp_connect_lock:
                try:
                    self._mcp_connect_worker()
                except Exception as e:
                    print(f"[{APP_NAME}] MCP connect pass failed: {e}")
            if not self._mcp_connect_pending.is_set():
                return
            self._mcp_connect_pending.clear()

    def _mcp_connect_worker(self) -> None:
        wanted = self._mcp_servers_cfg()
        # Close servers that are no longer configured. Pop under the lock,
        # close OUTSIDE it (close() can block ~3 s on a stuck subprocess).
        with self._mcp_lock:
            stale = [(n, c) for n, c in self._mcp_clients.items()
                     if not any(w.get("name") == n for w in wanted)]
            for n, _c in stale:
                self._mcp_clients.pop(n, None)
        for _n, c in stale:
            try:
                c.close()
            except Exception:
                pass
        results: List[str] = []
        for cfg in wanted:
            name, command = cfg["name"], cfg["command"]
            # Keep a healthy server whose config is unchanged ALIVE - saving
            # unrelated settings (font size, temperature, ...) must not wipe its
            # state. Restart only on config change or if the process died.
            with self._mcp_lock:
                old = self._mcp_clients.get(name)
            if old is not None:
                if (old.ready() and old.command == command
                        and old.args == cfg["args"] and old.env == cfg["env"]):
                    results.append(f"{name}: {len(old.tools)} tool(s) (kept alive)")
                    continue
                with self._mcp_lock:
                    self._mcp_clients.pop(name, None)
                try:
                    old.close()
                except Exception:
                    pass
            client = MCPClient(name, command, cfg["args"], cfg["env"])
            try:
                tools = client.connect()
                with self._mcp_lock:
                    self._mcp_clients[name] = client
                results.append(f"{name}: {len(tools)} tool(s)")
            except Exception as e:
                results.append(f"{name}: FAILED ({str(e)[:120]})")
        self._post(lambda rs=results: self._mcp_connect_done(rs))

    def _mcp_connect_done(self, results: List[str]) -> None:
        """Main-thread: refresh the permission bar's MCP section + status."""
        self._mcp_schemas()   # refresh the namespaced tool map from live clients
        self._mcp_rebuild_permissions_bar()
        if results:
            self._set_status("🔌 MCP " + "; ".join(results)[:200])

    def _close_mcp_servers(self) -> None:
        with self._mcp_lock:
            clients = list(self._mcp_clients.values())
            self._mcp_clients.clear()
        for client in clients:
            try:
                client.close()
            except Exception:
                pass

    def _mcp_rebuild_permissions_bar(self) -> None:
        """(Re)draw the MCP section of the permission bar: one dropdown per
        discovered tool, defaulting to Ask Permission (MCP servers are
        arbitrary local programs - never silently auto-approved)."""
        for w in getattr(self, "_mcp_perm_widgets", []):
            try:
                if w.winfo_exists():
                    w.destroy()
            except tk.TclError:
                pass
        self._mcp_perm_widgets = []
        for n in [k for k in list(self._perm_combos) if k.startswith("mcp_")]:
            del self._perm_combos[n]
        inner = getattr(self, "_perm_inner", None)
        if inner is None:
            return
        names = self._mcp_tool_names()
        if not names:
            return
        hdr = tk.Label(inner, text="🔌 MCP Tools", bg=COL["bg_deep"], fg=COL["accent"],
                       font=F(10, "bold"))
        hdr.pack(side="left", padx=(14, 7), pady=4)
        self._mcp_perm_widgets.append(hdr)
        for name in names:
            server_name, raw = self._mcp_tool_map[name]
            cell = tk.Frame(inner, bg=COL["bg_deep"])
            cell.pack(side="left", padx=7, pady=4)
            lab = tk.Label(cell, text=f"{raw} ({server_name})", bg=COL["bg_deep"],
                           fg=COL["text_dim"], font=F(10))
            lab.pack(anchor="w")
            cb = ttk.Combobox(cell, values=list(PERM_LABELS.values()), state="readonly",
                              width=13, style="Tool.TCombobox", font=F(11))
            perm = self.settings.get("tool_permissions", {}).get(name, "ask")
            cb.set(PERM_LABELS.get(perm, PERM_LABELS["ask"]))
            cb.pack(anchor="w")

            def on_change(e=None, name=name, cb=cb, lab=lab):
                label = cb.get()
                new_perm = PERM_VALUES.get(label, "ask")
                self.settings["tool_permissions"][name] = new_perm
                lab.configure(fg={"always": COL["success"],
                                  "ask":    COL["warning"],
                                  "off":    COL["danger"]}[new_perm])
                save_settings(self.settings)

            cb.bind("<<ComboboxSelected>>", on_change)
            self._perm_combos[name] = (cb, lab)
            self._mcp_perm_widgets.append(cell)
            # color the initial state too
            perm = self.settings.get("tool_permissions", {}).get(name, "ask")
            lab.configure(fg={"always": COL["success"], "ask": COL["warning"],
                              "off": COL["danger"]}[perm])
        # re-fit the canvas to the (possibly larger) content
        pc = getattr(self, "_perm_canvas", None)
        if pc is not None:
            try:
                inner.update_idletasks()
                need = int(inner.winfo_reqheight()) + 8
                if need > int(pc.cget("height")):
                    pc.configure(height=need)
            except tk.TclError:
                pass

    def _draw_mcp_server_rows(self, frame: tk.Frame, parent_top: tk.Toplevel) -> None:
        """Draw the configured MCP servers (name — command + Remove) and the
        Add button inside the Settings dialog."""
        for w in frame.winfo_children():
            w.destroy()
        for cfg in self._mcp_servers_cfg():
            rowf = tk.Frame(frame, bg=COL["bg_main"])
            rowf.pack(fill="x", pady=2)
            desc = f"{cfg['name']}  —  {cfg['command']}"
            if cfg["args"]:
                desc += "  " + " ".join(cfg["args"])[:60]
            tk.Label(rowf, text=desc, bg=COL["bg_main"], fg=COL["text"], font=F(10),
                     anchor="w").pack(side="left", fill="x", expand=True)

            def _remove(n=cfg["name"]):
                self._mcp_remove_server(n)
                self._draw_mcp_server_rows(frame, parent_top)

            tk.Button(rowf, text="Remove", command=_remove, bg=COL["bg_raised"],
                      fg=COL["danger"], activebackground="#5B2323", relief="flat", bd=0,
                      padx=8, pady=1, cursor="hand2", font=F(9)).pack(side="right")
        tk.Button(frame, text="+ Add MCP Server",
                  command=lambda: self._mcp_add_server_dialog(parent_top),
                  bg=COL["bg_raised"], fg=COL["text"], activebackground="#4B5563",
                  relief="flat", bd=0, padx=8, pady=2, cursor="hand2", font=F(10)) \
            .pack(anchor="w", pady=(4, 0))

    def _mcp_remove_server(self, name: str) -> None:
        servers = self._mcp_servers_cfg()
        self.settings["mcp_servers"] = [s for s in servers if s.get("name") != name]
        save_settings(self.settings)
        with self._mcp_lock:
            client = self._mcp_clients.pop(name, None)
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        self._mcp_schemas()   # drop the removed server's tools from the map
        self._mcp_rebuild_permissions_bar()

    def _mcp_add_server_dialog(self, parent: tk.Toplevel) -> None:
        d = tk.Toplevel(parent)
        d.title(f"{APP_NAME} — Add MCP Server")
        d.configure(bg=COL["bg_main"])
        d.transient(parent)
        try:
            d.update_idletasks()
        except tk.TclError:
            pass
        try:
            d.grab_set()
        except tk.TclError:
            pass
        fields = [("Name (e.g. filesystem)", "name"),
                  ("Command (e.g. npx, or full path to an executable)", "command"),
                  ("Args (space-separated, e.g. -m  mcp_server_fs  C:\\data)", "args"),
                  ("Env vars (space-separated KEY=VALUE pairs; optional)", "env")]
        ents: Dict[str, tk.Entry] = {}
        for i, (label, key) in enumerate(fields):
            tk.Label(d, text=label, bg=COL["bg_main"], fg=COL["text_dim"], font=F(10)) \
                .grid(row=i, column=0, sticky="w", padx=(12, 8), pady=5)
            e = tk.Entry(d, width=46, bg=COL["bg_deep"], fg=COL["text"],
                         insertbackground=COL["text"], relief="flat", font=F(10))
            e.grid(row=i, column=1, sticky="ew", padx=(0, 12), pady=5)
            ents[key] = e
        tk.Label(d, text=("Stdio MCP servers only (v1). The server runs as a local "
                          "subprocess with your user privileges; its tools appear in the "
                          "permission bar defaulting to Ask Permission."),
                 bg=COL["bg_main"], fg=COL["text_dim"], font=F(9), wraplength=400,
                 justify="left") \
            .grid(row=len(fields), column=0, columnspan=2, sticky="w", padx=(12, 12), pady=(4, 8))

        def ok():
            name = ents["name"].get().strip()
            command = ents["command"].get().strip()
            if not name or not command:
                messagebox.showwarning(f"{APP_NAME} - MCP Server",
                                       "Name and Command are both required.", parent=d)
                return
            existing = {str(s.get("name")) for s in (self.settings.get("mcp_servers") or [])
                        if isinstance(s, dict)}
            if name in existing:
                messagebox.showwarning(f"{APP_NAME} - MCP Server",
                                       f"A server named '{name}' already exists.", parent=d)
                return
            servers = [s for s in (self.settings.get("mcp_servers") or []) if isinstance(s, dict)]
            if len(servers) >= MCP_MAX_SERVERS:
                messagebox.showwarning(f"{APP_NAME} - MCP Server",
                                       f"Maximum of {MCP_MAX_SERVERS} servers.", parent=d)
                return
            args = ents["args"].get().strip().split()
            env = _parse_env_pairs(ents["env"].get())
            servers.append({"name": name, "command": command, "args": args, "env": env})
            self.settings["mcp_servers"] = servers
            save_settings(self.settings)
            d.destroy()
            self._mcp_connect_all()

        btns = tk.Frame(d, bg=COL["bg_main"])
        btns.grid(row=len(fields) + 1, column=0, columnspan=2, pady=(0, 12))
        tk.Button(btns, text="Add & Connect", command=ok, bg=COL["accent"], fg="#FFFFFF",
                  activebackground="#2563EB", relief="flat", bd=0, padx=14, pady=5,
                  cursor="hand2", font=F(11)).pack(side="left", padx=6)
        tk.Button(btns, text="Cancel", command=d.destroy, bg=COL["bg_raised"], fg=COL["text"],
                  activebackground="#4B5563", relief="flat", bd=0, padx=12, pady=5,
                  cursor="hand2", font=F(11)).pack(side="left", padx=6)
        try:
            d.lift()
            d.focus_force()
            ents["name"].focus_set()
        except tk.TclError:
            pass


    # ------------------------------------------------------------------------------
    #  SAMPLER VERIFICATION (Settings -> Sampling -> "Verify parameters", tier 3)
    # ------------------------------------------------------------------------------

    def _verify_report(self, out_box, lines=(), status: str = "", clear: bool = False) -> None:
        """Write probe output into the read-only results box - ALWAYS via _post (invariant 4).

        `out_box` belongs to a Settings dialog the user can close mid-run, so every write checks
        winfo_exists() and swallows TclError; a vanished widget must never kill the probe thread."""
        def apply() -> None:
            if self._closed:
                return
            try:
                if out_box is not None and out_box.winfo_exists():
                    if clear:
                        out_box.configure(state="normal")
                        out_box.delete("1.0", "end")
                    else:
                        out_box.configure(state="normal")
                    for ln in list(lines):
                        out_box.insert("end", str(ln) + "\n")
                    out_box.see("end")
                    out_box.configure(state="disabled")
            except tk.TclError:
                pass
            if status:
                self._set_status(status)
        self._post(apply)

    def _verify_set_button(self, btn, state: str) -> None:
        """Enable/disable the button on EVERY exit path (mirrors the _chat_worker wrapper)."""
        if btn is None:
            return
        self._post(lambda: self._enable_verify_button(btn, state))

    def _enable_verify_button(self, btn, state: str) -> None:
        try:
            if btn.winfo_exists():
                btn.configure(state=state)
        except tk.TclError:
            pass

    def _verify_client(self):
        """A PRIVATE client for probes only - never the pooled chat client.

        timeout + max_retries=0 are the point: a chat turn wants keep-alive and SDK retries, a
        probe wants exactly one bounded attempt per request. Keeping them separate means this
        feature cannot perturb the turn path's samplers/thinking/handoff behaviour at all."""
        if OpenAI is None:
            return None
        url = (self.settings.get("server_url") or "").strip()
        key = (self.settings.get("api_key") or "").strip() or "sk-local"
        try:
            return OpenAI(base_url=url or None, api_key=key,
                          timeout=VERIFY_TIMEOUT, max_retries=0)
        except Exception:
            return None

    def _verify_sampling(self, btn=None, out_box=None) -> None:
        """Settings -> Sampling -> "Verify parameters". Main-thread entry point.

        Refuses while a turn is live (_busy): the probe must not stack requests onto a running
        conversation. One run at a time; results go ONLY into the read-only box (never into chat
        history, chats.json, settings, or a file)."""
        if self._closed:
            return
        if getattr(self, "_verify_running", False):
            self._verify_report(out_box, ["(verification already running - please wait)"],
                               "Sampling verification in progress&")
            return
        model = (self.settings.get("model_name") or "").strip()
        if self._busy:
            self._verify_report(
                out_box,
                [f"REFUSED: a chat turn is running. Verification waits for it to finish - "
                 f"two concurrent request streams make the verdict meaningless."],
                "Verification refused while a turn is running")
            return
        if not model:
            self._verify_report(out_box, ["REFUSED: Model Name is empty - set it first."],
                               "Verification refused: no model name")
            return
        if OpenAI is None:
            self._verify_report(out_box, ["REFUSED: the 'openai' package is not installed."],
                                "Verification refused: openai not installed")
            return
        self._verify_running = True
        self._verify_set_button(btn, "disabled")
        threading.Thread(target=self._verify_worker, args=(btn, out_box, model),
                         daemon=True).start()

    def _verify_worker(self, btn, out_box, model: str) -> None:
        """Daemon thread: run the bounded probe sequence. All UI via _verify_report/_post."""
        try:
            self._verify_report(out_box, [], "Verifying sampling parameters&", clear=True)
            client = self._verify_client()
            if client is None:
                self._verify_report(out_box, ["REFUSED: could not create an API client "
                                              "(check Server URL / API Key)."],
                                    "Verification failed: no API client")
                return
            url = (self.settings.get("server_url") or "").strip()
            key = (self.settings.get("api_key") or "").strip()
            info = fetch_model_info(url, key) if url else []
            _sw = sampling_kwargs(self.settings)
            lines = run_sampling_verify(client, model, _sw["top"], _sw["flat"], info,
                                        timeout=VERIFY_TIMEOUT,
                                        report=lambda group: self._verify_report(out_box, group))
            self._verify_report(out_box, [" evidence only: a server can accept a key and ignore "
                                          "it "])
            self._verify_report(out_box, [], "Sampling verification finished")
        except Exception as e:
            traceback.print_exc()
            self._verify_report(out_box, ["Verification aborted: " + _probe_first_line(e)],
                                "Sampling verification aborted")
        finally:
            self._verify_running = False
            self._verify_set_button(btn, "normal")

    def _test_connection(self, url: str, top: tk.Toplevel) -> None:
        ok = False
        detail = ""
        try:
            key = (self.settings.get("api_key") or "").strip()
            req = urllib.request.Request(
                url.rstrip("/") + "/models",
                headers={
                    "User-Agent": "Deskpilot/1.0",
                    "Authorization": f"Bearer {key}"
                }
            )
            with urllib.request.urlopen(req, timeout=8) as r:
                data = json.loads(r.read().decode("utf-8", "replace"))
            models = [m.get("id") for m in (data.get("data") or [])][:5]
            ok = True
            detail = "Connected! Models: " + (", ".join(models) if models else "(none listed)")
        except Exception as e:
            detail = f"Connection failed:\n{e}"

        def show():
            try:
                (messagebox.showinfo if ok else messagebox.showerror)(
                    f"{APP_NAME} — Connection Test", detail, parent=top)
            except Exception:
                pass

        self._post(show)

    # ════════════════════════════════════════════════════════════════════
    #  SIDEBAR COLLAPSE + WINDOW LIFECYCLE
    # ════════════════════════════════════════════════════════════════════

    def _set_sidebar_visible(self, visible: bool, save: bool = True) -> None:
        # Visibility is tracked in Python (self._sidebar_visible): on some Tk
        # builds pane.slaves() is unreliable, and tkinter does not expose the
        # 'move' subcommand as a method. Widths use the frame's requested size
        # because sashpos() corrupts child mapping on some Tk 9 builds.
        if visible == self._sidebar_visible:
            return
        try:
            if not visible:
                self.pane.forget(self.sidebar)
            else:
                # Restore the saved width via the frame's requested size.
                # (sashpos() corrupts child mapping on some Tk 9 builds.)
                self.sidebar.configure(width=int(self.settings.get("sidebar_width", 250)))
                self._pane_add(self.sidebar, 170)
                # Keep the sidebar leftmost: prefer the 'move' subcommand
                # (Tk >= 8.6); otherwise rebuild child order without slaves().
                try:
                    self.pane.tk.call((str(self.pane), "move", str(self.sidebar), 0))
                except tk.TclError:
                    try:
                        self.pane.forget(self.chat_area)
                        self._pane_add(self.chat_area, 420)
                    except tk.TclError:
                        pass
        except tk.TclError:
            pass
        self._sidebar_visible = visible
        if save:
            self.settings["sidebar_collapsed"] = not visible
            try:
                self.settings["sidebar_width"] = int(self.sidebar.winfo_width())
            except (tk.TclError, ValueError):
                pass
            save_settings(self.settings)

    def toggle_sidebar(self) -> None:
        self._set_sidebar_visible(not self._sidebar_visible)

    def on_close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.settings["window_geometry"] = self.root.geometry()
            if self._sidebar_visible:
                try:
                    self.settings["sidebar_width"] = int(self.sidebar.winfo_width())
                except (tk.TclError, ValueError):
                    pass
        except tk.TclError:
            pass
        self._mic_stop.set()          # release the microphone if dictation was running
        self.stop_tts()               # halt any in-flight TTS playback
        self._close_mcp_servers()     # terminate MCP server subprocesses
        _release_instance_lock(SETTINGS_FILE.parent)   # drop our lock (no-op in --multi / tests)
        if not (save_settings(self.settings) and save_chats(self.chats)):
            try:
                messagebox.showwarning(
                    f"{APP_NAME} - Could not save data",
                    "Settings/chat history could not be written to:\n\n"
                    f"{SETTINGS_FILE.parent}\n\nThey will be lost when the app closes.")
            except Exception:
                pass
        try:
            self.root.destroy()
        except tk.TclError:
            pass


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def _check_data_dir_writable() -> bool:
    """Probe that the live data directory accepts writes (see resolve_data_dir).

    A read-only or protected location would otherwise fail silently (errors only
    go to the console, which pythonw users never see).
    """
    try:
        probe = SETTINGS_FILE.parent / ".deskpilot_write_test"
        _atomic_write(probe, "ok")
        probe.unlink(missing_ok=True)
        return True
    except Exception as e:
        print(f"[{APP_NAME}] Data directory is not writable: {e}")
        return False


# Single-instance lock + --multi ephemeral profile (Change #14)
LOCK_FILE_NAME = ".deskpilot.lock"

_OWN_IMAGE = None   # lowercased image path of this process (cached; same-exe match for frozen builds)


def _own_image() -> str:
    """Full image path of the current process ("" if unreadable)."""
    global _OWN_IMAGE
    if _OWN_IMAGE is None:
        try:
            from ctypes import POINTER as _POINTER
            k32 = ctypes.windll.kernel32
            buf = ctypes.create_unicode_buffer(32768)
            size = ctypes.c_uint32(len(buf))
            k32.QueryFullProcessImageNameW.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                                       ctypes.c_wchar_p, _POINTER(ctypes.c_uint32)]
            k32.QueryFullProcessImageNameW.restype = ctypes.c_bool
            if k32.QueryFullProcessImageNameW(k32.GetCurrentProcess(), 0, buf, ctypes.byref(size)):
                _OWN_IMAGE = buf.value.lower()
        except Exception:
            _OWN_IMAGE = ""
    return _OWN_IMAGE


def _pid_alive(pid: int) -> bool:
    """True if a process with this PID is running.

    Windows: OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION), then
    GetExitCodeProcess (STILL_ACTIVE = genuinely running; a terminated zombie
    that still has open handles reports a real exit code -> dead), plus an
    image-name check (python / deskpilot / same exe name) guarding against
    PID reuse by an unrelated process.
    Non-Windows or any failure -> False, so stale locks are always taken over
    and the user can never be locked out."""
    if pid <= 0:
        return False
    try:
        k32 = ctypes.windll.kernel32
        k32.OpenProcess.restype = ctypes.c_void_p   # 64-bit handles must not truncate
        h = k32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        from ctypes import POINTER
        try:
            # A terminated process can still be OpenProcess-able while some
            # handle to it is open (zombie); GetExitCodeProcess distinguishes
            # the two - only STILL_ACTIVE means genuinely running.
            code = ctypes.c_uint32()
            k32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, POINTER(ctypes.c_uint32)]
            if not k32.GetExitCodeProcess(h, ctypes.byref(code)) or code.value == 259:   # STILL_ACTIVE
                buf = ctypes.create_unicode_buffer(32768)
                size = ctypes.c_uint32(len(buf))
                k32.QueryFullProcessImageNameW.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                                           ctypes.c_wchar_p, POINTER(ctypes.c_uint32)]
                k32.QueryFullProcessImageNameW.restype = ctypes.c_bool
                if k32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
                    img = buf.value.lower()
                    own = _own_image()
                    same_exe = bool(own) and Path(img).name == Path(own).name   # PyInstaller: instances may run from different paths (onefile temp dir)
                    return "python" in img or "deskpilot" in img or same_exe
                return True   # running but image name unreadable -> assume live
            return False      # real exit code -> the process is dead (stale lock)
        finally:
            k32.CloseHandle.argtypes = [ctypes.c_void_p]
            k32.CloseHandle(h)
    except Exception:
        return False


def _acquire_instance_lock(data_dir: Path) -> bool:
    """Take <data_dir>/.deskpilot.lock for this PID (stores str(os.getpid())).

    Returns True when the lock is ours now: fresh, or a stale takeover (dead /
    unreadable holder). A live foreign holder wins -> False; the caller must
    then refuse to start a second instance on the same profile. A failed lock
    write (unwritable dir) degrades to best-effort: warn and continue, since
    saves would fail anyway in that case."""
    lock = data_dir / LOCK_FILE_NAME
    try:
        if lock.exists():
            try:
                pid = int(lock.read_text(encoding="utf-8").strip())
            except (ValueError, OSError):
                pid = 0
            if pid != os.getpid() and _pid_alive(pid):
                return False
        lock.write_text(str(os.getpid()), encoding="utf-8")
        return True
    except OSError as e:
        print(f"[{APP_NAME}] Could not write instance lock {lock}: {e} (continuing without the guard)")
        return True


def _release_instance_lock(data_dir: Path) -> None:
    """Delete the lock file if it still holds OUR PID (never another instance's)."""
    try:
        lock = data_dir / LOCK_FILE_NAME
        if lock.exists() and lock.read_text(encoding="utf-8").strip() == str(os.getpid()):
            lock.unlink()
    except OSError:
        pass


def _setup_multi_profile(data_dir: Path) -> None:
    """--multi mode: relocate persistence to an ephemeral per-PID scratch dir.

    scratch_<pid> is seeded from the main profile (settings + chats, if present)
    so window 2 opens with the current settings/history; every save then lands
    in scratch, so the two instances never touch the same JSON files (no
    clobbering by construction). MODEL_CACHE_DIR is pinned to the MAIN data dir
    so whisper/kokoro models are reused instead of re-downloaded. The scratch
    dir is deleted on clean exit via atexit."""
    global SETTINGS_FILE, CHATS_FILE, GEN_DIR, MODEL_CACHE_DIR
    scratch = data_dir / f"scratch_{os.getpid()}"
    try:
        scratch.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"[{APP_NAME}] --multi: cannot create {scratch}: {e}")
        return
    for name in ("deskpilot_settings.json", "deskpilot_chats.json"):
        src = data_dir / name
        try:
            if src.exists() and not (scratch / name).exists():
                shutil.copy2(src, scratch / name)
        except OSError as e:
            print(f"[{APP_NAME}] --multi: could not seed {name}: {e}")
    SETTINGS_FILE = scratch / "deskpilot_settings.json"
    CHATS_FILE = scratch / "deskpilot_chats.json"
    GEN_DIR = scratch / "generated_images"
    try:
        GEN_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    MODEL_CACHE_DIR = data_dir   # dictation/TTS keep the main profile's model caches
    atexit.register(_cleanup_multi_profile, scratch)


def _cleanup_multi_profile(scratch: Path) -> None:
    """Delete an ephemeral --multi scratch dir on clean exit (best-effort)."""
    try:
        shutil.rmtree(scratch, ignore_errors=True)
    except Exception:
        pass


def main() -> int:
    if OpenAI is None:
        print("ERROR: the 'openai' package is required.  Run:  pip install openai")
        try:
            r = tk.Tk()
            r.withdraw()
            messagebox.showerror(
                f"{APP_NAME} — Missing dependency",
                "The 'openai' Python package is required.\n\nRun:\n    pip install openai")
            r.destroy()
        except Exception:
            pass
        return 1
    global SETTINGS_FILE, CHATS_FILE, GEN_DIR
    multi = "--multi" in sys.argv[1:]
    data_dir = resolve_data_dir()
    if data_dir != BASE_DIR:
        SETTINGS_FILE = data_dir / "deskpilot_settings.json"
        CHATS_FILE = data_dir / "deskpilot_chats.json"
        GEN_DIR = data_dir / "generated_images"
        try:
            GEN_DIR.mkdir(parents=True, exist_ok=True)   # module-level mkdir only covered the script folder
        except Exception:
            pass
    if multi:
        _setup_multi_profile(data_dir)   # ephemeral scratch_<pid> seeded from main; no lock
    elif not _acquire_instance_lock(data_dir):
        try:
            r = tk.Tk(); r.withdraw()
            cmd = sys.executable if getattr(sys, "frozen", False) else f'python "{Path(__file__).resolve()}"'
            messagebox.showwarning(
                f"{APP_NAME} - Already running",
                "Deskpilot is already running (same data profile).\n\n"
                "Close the existing window first, or start a second isolated copy with:\n\n"
                f"    {cmd} --multi")
            r.destroy()
        except Exception:
            pass
        return 1
    else:
        atexit.register(_release_instance_lock, data_dir)   # safety net; on_close() releases first
    _migrate_legacy_files()
    if not _check_data_dir_writable():
        try:
            r = tk.Tk(); r.withdraw()
            messagebox.showwarning(
                f"{APP_NAME} - Cannot save data",
                "Settings and chat history are saved in:\n\n"
                f"{SETTINGS_FILE.parent}\n\nThat folder is not writable, so nothing will be remembered\n"
                "between runs. Check your user profile permissions (or run the app\n"
                "from a normal folder such as Documents) and start it again.")
            r.destroy()
        except Exception:
            pass
    app = DeskpilotApp()
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())