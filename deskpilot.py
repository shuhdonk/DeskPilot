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
      firecrawl_scrape      – Firecrawl scrape to markdown (key in Settings)
     fetch_url             – retrieves + cleans raw text from a webpage
                         (Reddit URLs are read via Reddit's public Atom feed)
     run_javascript        – Node.js via hidden subprocess, 1800 s timeout
     run_shell             – ONE unsandboxed shell command per call (Ask-gated)
     list_directory        – folder contents (folders first, size + modified time)
     search_files          – bounded grep over files/dirs (locate before reading)
     read_local_file       – text & PDF (pypdf); offset/limit chunked reads
     write_local_file      – creates / overwrites local text files
     edit_local_file       – transactional in-place replacements (surgical edits)
     generate_local_image  – local FLUX/SDXL CLI (ai-imagegen/generate.py)
     capture_screen        – PIL ImageGrab, queued for vision analysis
     get_clipboard_text    – thread-safe system clipboard read
     list_skills           – Agent Skills found (folders with a SKILL.md)
     load_skill            – read one skill's instructions + bundled files
     search_chats          – FTS5 full-text search over every saved chat
     set_clipboard         – thread-safe system clipboard write
     transcribe_audio      – local Whisper over any PyAV-decodable audio file
     list_mcp_resources / read_mcp_resource  – MCP resources/list + resources/read
     list_mcp_prompts / get_mcp_prompt       – MCP prompts/list + prompts/get
                         (plus every tool a connected MCP server advertises,
                          namespaced mcp_<server>_<tool>)
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
            faster-whisper      → local (on-machine) dictation; uses the GPU
                                  when nvidia-cublas-cu12 is present, else CPU

 RUN
 ───
 python deskpilot.py
"""

from __future__ import annotations

import atexit
import base64
import ctypes
import difflib
import fnmatch
import gzip
import hashlib
import io
import html as html_mod
import ipaddress
import json
import os
import queue
import re
import shutil
import sqlite3
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
import xml.etree.ElementTree as ET
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
WHISPER_MODEL_NAME = "large-v3-turbo"   # GPU model: ~3x more accurate than base.en AND faster on a GPU
WHISPER_CPU_MODEL_NAME = "base.en"      # CPU model: large-v3-turbo is far too heavy to load/run on CPU
_WHISPER_DEVICE_LABEL = "local"         # set once loaded: "GPU" / "CPU" (shown in the mic status)

def _ensure_cublas_on_path() -> bool:
    """Make cuBLAS findable by ctranslate2; True only if it was located.

    ctranslate2 needs cublas (cublas64_12.dll on Windows, libcublas.so.12 on
    Linux; cublas also needs cublasLt) at model-creation time, but does not know
    that `pip install nvidia-cublas-cu12` put them under
    site-packages/nvidia/cublas/{bin,lib}. Without this the model loads on
    "cuda" and then fails on the first phrase.

    Probe order (first hit wins):
      1. site-packages            - running from Python, or a build that collected them
      2. sys._MEIPASS / BASE_DIR  - DLLs bundled inside a onefile exe
      3. next to the exe          - DLLs copied into <exe dir>/nvidia/cublas/bin
                                    (keeps the onefile exe small: cuBLAS is ~736 MB)
      4. (Windows) the system Python's site-packages, discovered via
         %LOCALAPPDATA%/Programs/Python/* - lets a frozen exe reuse a
         machine-wide cuBLAS install
      5. (Linux) the system loader - a CUDA toolkit installed distro-wide is
         already resolvable by dlopen, so no path fix is needed

    Windows and Linux resolve libraries differently, so the mechanism differs:
    Windows uses add_dll_directory + PATH; Linux ignores both for dlopen and
    needs LD_LIBRARY_PATH plus an explicit pre-load (setting the env var alone
    is too late for an already-running process, hence the CDLL call).
    """
    import os as _os
    import glob as _glob
    is_win = sys.platform == "win32"
    try:
        import site
        roots = list(site.getsitepackages()) + [site.getusersitepackages()]
    except Exception:
        roots = []
    for extra in (getattr(sys, "_MEIPASS", None), str(BASE_DIR), str(Path(sys.executable).resolve().parent)):
        if extra:
            roots.append(extra)
    # frozen builds have no site-packages of their own; fall back to a system install
    if is_win:
        try:
            roots += _glob.glob(str(Path(_os.environ.get("LOCALAPPDATA", "")) /
                                  "Programs" / "Python" / "*" / "Lib" / "site-packages"))
        except Exception:
            pass
    # the nvidia wheels ship DLLs under .../bin on Windows and .so files under .../lib
    sub = "bin" if is_win else "lib"
    found = False
    for r in roots:
        for pkg in ("cublas", "cudnn", "cuda_nvrtc"):
            d = _os.path.join(r, "nvidia", pkg, sub)
            if not _os.path.isdir(d):
                continue
            if is_win:
                try:
                    _os.add_dll_directory(d)          # Windows 3.8+
                except (AttributeError, OSError):
                    pass
                _os.environ["PATH"] = d + _os.pathsep + _os.environ.get("PATH", "")
            else:
                lp = [p for p in _os.environ.get("LD_LIBRARY_PATH", "").split(_os.pathsep) if p]
                if d not in lp:
                    _os.environ["LD_LIBRARY_PATH"] = _os.pathsep.join([d] + lp)
                # glibc snapshots LD_LIBRARY_PATH at process start, so changing it
                # here does NOT affect later dlopen() calls. Pre-loading by full
                # path is what actually makes the libs resolvable; cublasLt must
                # go first because libcublas.so.12 depends on it, and cuDNN is
                # needed by Whisper's convolutional layers.
                for so in ("libcublasLt.so.12", "libcublas.so.12",
                           "libcudnn.so.9", "libcudnn_cnn.so.9", "libcudnn_ops.so.9"):
                    p = _os.path.join(d, so)
                    if _os.path.exists(p):
                        try:
                            ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
                        except OSError:
                            pass        # missing/partial dir: the probe below decides
            if pkg == "cublas":
                found = True
    if not found and not is_win:
        # CUDA installed via the distro package manager: already dlopen-able.
        try:
            ctypes.CDLL("libcublas.so.12", mode=ctypes.RTLD_GLOBAL)
            found = True
        except OSError:
            pass
    return found

def _ensure_bundled_node_on_path() -> bool:
    """Put an app-bundled Node.js on PATH so MCP `npx` servers work out of the box.

    The installer ships a private Node under <install>\\node\\ next to the exe.
    MCPClient resolves commands with shutil.which against PATH (shell=False does
    not append .CMD, which is why `npx` must be resolved, not passed through), so
    making it findable is the whole integration - no MCP code has to know about it.

    Probe order mirrors _ensure_cublas_on_path: a Node already on PATH (the user
    installed it themselves) always wins, and we only append ours as a fallback.
    Prepending instead would shadow the system install and surprise people who
    manage Node with nvm. Returns True if node is resolvable either way.
    """
    if shutil.which("node"):
        return True
    exe_dir = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) \
        else Path(__file__).resolve().parent
    for cand in (exe_dir / "node", BASE_DIR / "node"):
        if (cand / "node.exe").is_file() or (cand / "node").is_file():
            d = str(cand)
            if d not in os.environ.get("PATH", "").split(os.pathsep):
                # Append, never prepend: see the docstring.
                os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + d
            return True
    return False


def _whisper_device():
    """Pick (device, compute_type): CUDA float16 only when a GPU AND its cuBLAS are usable.

    A GPU is not enough: ctranslate2 needs cublas64_12.dll at model-creation time.
    Picking CUDA without it makes the load throw, so we check for the DLLs first
    and fall back to CPU int8 (which also selects the smaller CPU model).
    """
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() < 1:
            return "cpu", "int8"
    except Exception:
        return "cpu", "int8"
    if not _ensure_cublas_on_path():
        return "cpu", "int8"
    return "cuda", "float16"

def _whisper_model():
    """Load the local Whisper model once (GPU float16 when possible, else CPU int8).

    Thread-safe: a background preload and the first phrase can race; the lock
    makes the losing caller wait for the winning load instead of loading twice."""
    global _WHISPER_MODEL, _WHISPER_DEVICE_LABEL
    if _WHISPER_MODEL is None:
        with _WHISPER_LOCK:
            if _WHISPER_MODEL is None:
                import os as _os
                cache = (MODEL_CACHE_DIR or Path(GEN_DIR).parent) / "whisper_models"   # MODEL_CACHE_DIR pins the cache in --multi mode
                try:
                    _os.makedirs(cache, exist_ok=True)
                except OSError:
                    pass
                device, compute = _whisper_device()
                name = WHISPER_MODEL_NAME if device == "cuda" else WHISPER_CPU_MODEL_NAME
                try:
                    _WHISPER_MODEL = WhisperModel(name, device=device, compute_type=compute,
                                                  download_root=str(cache))
                except Exception:
                    if device == "cpu":
                        raise
                    # GPU unusable (missing cuDNN/cuBLAS, unsupported arch) - keep dictation working
                    _WHISPER_MODEL = WhisperModel(WHISPER_CPU_MODEL_NAME, device="cpu", compute_type="int8",
                                                  download_root=str(cache))
                    device = "cpu"
                _WHISPER_DEVICE_LABEL = "GPU" if device == "cuda" else "CPU"
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
        model_f = next((f for f in (KOKORO_MODEL_NAME, "kokoro-v1.0.int8.onnx", "kokoro-v1.0.onnx")
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
    return _transcribe_pcm(wav_bytes_to_float32(wav_bytes), language="en")


_WHISPER_INFER_LOCK = threading.Lock()   # one transcription at a time on the GPU


def _transcribe_pcm(pcm, language: str = "") -> str:
    """Run Whisper over float32 mono @16 kHz samples. '' = silence / nothing heard.

    language='' lets Whisper auto-detect (right for arbitrary files); dictation keeps
    passing "en". Serialized on purpose: dictation and a file-transcription tool call
    can otherwise contend on the GPU, and a single-slot GPU server does worse than
    finishing them one after the other.
    """
    model = _whisper_model()
    kwargs = dict(beam_size=1, vad_filter=True)
    if language:
        kwargs["language"] = language
    with _WHISPER_INFER_LOCK:
        segments, _info = model.transcribe(pcm, **kwargs)
        return " ".join(seg.text.strip() for seg in segments).strip()


# ── Audio file decoding (for the transcribe_audio tool) ──────────────────────
# whisper_transcribe_local only accepts 16-bit mono WAV, which is what the
# microphone produces - but real recordings are mp3/m4a/flac/ogg/24-bit/stereo.
# PyAV is already a hard dependency of faster-whisper (and pinned av<19), so it
# decodes those here at no extra install cost. Verified on av 18.1.0: mp3, flac,
# ogg and m4a all decode through AudioResampler(format='s16', layout='mono',
# rate=16000) to (1, N) int16 arrays.
AUDIO_FILE_MAX_BYTES = 64 * 1024 * 1024   # largest audio file the tool will read
AUDIO_MAX_SECONDS    = 5400               # decoded audio longer than this is cut
AUDIO_SAMPLE_RATE    = 16000              # what Whisper wants


def _audio_error_text(exc) -> str:
    """A readable reason from a decode failure.

    PyAV raises av.error.InvalidDataError, which IS a ValueError subclass, so it can
    escape the conversion below and reach the model as
    "[Errno 1094995529] Invalid data found when processing input: '<none>'". Its
    strerror is the only part worth showing.
    """
    msg = str(getattr(exc, "strerror", "") or "").strip() or str(exc).strip()
    msg = re.sub(r"^\[Errno \d+\]\s*", "", msg)
    msg = re.sub(r":\s*'<none>'\s*$", "", msg)
    return msg or type(exc).__name__


def decode_audio_bytes(data: bytes) -> tuple:
    """Any decodable audio -> (float32 mono @16 kHz, seconds). Raises ValueError.

    Tries PyAV first (every container/codec it supports), then falls back to the
    stdlib wave path used by dictation so the tool still works in a build without
    av. Every failure surfaces as ValueError with a readable reason - callers show
    that text to the model instead of getting a traceback. Returns plain samples,
    never a file path: the caller already did the path-safety work.
    """
    if not data:
        raise ValueError("file is empty")
    try:
        import av
    except Exception:
        av = None
    if av is not None:
        container = None
        try:
            container = av.open(io.BytesIO(data))
            astream = next((s for s in container.streams if s.type == "audio"), None)
            if astream is None:
                raise ValueError("file has no audio track")
            resampler = av.AudioResampler(format="s16", layout="mono",
                                          rate=AUDIO_SAMPLE_RATE)
            chunks = []
            for frame in container.decode(astream):
                for out in resampler.resample(frame):
                    chunks.append(out.to_ndarray())
            try:                       # flush the resampler's tail
                for out in (resampler.resample(None) or []):
                    chunks.append(out.to_ndarray())
            except Exception:
                pass                   # older av: losing one tail frame is fine
            if not chunks:
                raise ValueError("decoded to no audio samples")
            import numpy as np
            flat = np.concatenate([np.asarray(c, dtype=np.int16).reshape(-1)
                                   for c in chunks])
            return flat.astype(np.float32) / 32768.0, len(flat) / float(AUDIO_SAMPLE_RATE)
        except ValueError as e:
            # av.error.InvalidDataError is a ValueError SUBCLASS, so it arrives here
            # too. Ours (no audio track / no samples) pass through verbatim; av's
            # carry a strerror and get their errno noise stripped.
            if getattr(e, "strerror", None):
                raise ValueError(f"not readable as audio ({_audio_error_text(e)})")
            raise
        except Exception as e:         # truncated/corrupt/unsupported -> one contract
            raise ValueError(f"not readable as audio ({_audio_error_text(e)})")
        finally:
            if container is not None:
                try:
                    container.close()
                except Exception:
                    pass
    # no av: stdlib WAV path (same decoder the microphone feed uses)
    try:
        return wav_bytes_to_float32(data), 0.0
    except Exception as e:
        raise ValueError(f"not readable as audio ({e})")


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
MCP_MAX_IMAGES_PER_CALL = 8  # images taken from ONE tools/call result (a chatty server
                             # must not be able to flood the context with dozens of shots)
MCP_MAX_IMAGE_BYTES = 12_000_000   # decoded size cap per image (a server must not be
                                   # able to hand us a 500 MB "image" to decode)
MCP_LIST_CAP = 60                  # resources/prompts listed per tool call (servers can
                                   # expose thousands; the model needs a digest, not a dump)

# MCP servers return image content (browser/computer-use servers live by it). Like
# screenshots they are stored as lightweight 'image_ref' parts and expanded into a
# real payload at request time - never inline base64 in chats.json.
MCP_IMAGE_MARKER = "[System] An image returned by the MCP tool that just ran"


def mcp_split_content(result: Any) -> tuple:
    r"""(text, images) from one tools/call result - pure, no I/O.

    MCP content parts are typed: text, image (base64 + mimeType), resource, audio.
    Before this, image parts were flattened to f"[image {mime}, {n} base64 chars]",
    so a screenshot-returning server told the model an image EXISTED and threw the
    pixels away - the model could never actually see it. The image list is returned
    separately so the caller can save it and hand it to the vision path.

    Unknown part types keep the old JSON-preview behaviour (useful for debugging a
    server that speaks something we do not model yet). Never raises: a malformed
    result from a third-party server must degrade, not crash the tool loop."""
    if not isinstance(result, dict):
        return (str(result), [])
    text_parts: List[str] = []
    images: List[dict] = []
    for item in (result.get("content") or []):
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "text":
            text_parts.append(str(item.get("text", "")))
        elif itype == "image":
            data = str(item.get("data", ""))
            if data.strip():
                images.append({"mime": str(item.get("mimeType") or "image/png"),
                               "data": data})
        elif itype == "resource":
            # Embedded resource: text resources are useful verbatim; blobs are
            # reported, not silently dropped, so the model knows something exists.
            res = item.get("resource") or {}
            if isinstance(res, dict):
                if res.get("text") is not None:
                    text_parts.append(str(res.get("text")))
                elif res.get("blob"):
                    uri = str(res.get("uri") or "(no uri)")
                    text_parts.append(f"[binary resource {uri}, "
                                      f"{len(str(res.get('blob')))} base64 chars]")
        else:
            text_parts.append(json.dumps(item, ensure_ascii=False)[:500])
    # structuredContent (newer servers) carries a machine-readable twin of the result.
    sc = result.get("structuredContent")
    if sc is not None and not text_parts:
        text_parts.append(json.dumps(sc, ensure_ascii=False)[:2000])
    return ("\n".join(text_parts).strip(), images)


def mcp_tool_name(server: str, raw: str) -> str:
    """Namespaced tool name: mcp_<server>_<tool>, sanitized to [A-Za-z0-9_]."""
    def _clean(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]", "_", s).strip("_") or "x"
    return f"mcp_{_clean(server)}_{_clean(raw)}"


# v1.1.56 (#4 of the external review): the old implementation split on
# whitespace, so a value containing a space was torn in half and the tail
# silently vanished: 'PATH="C:\Program Files\bin"' became two junk tokens.
# shlex was considered and REJECTED: in POSIX mode it treats '\\' as an escape
# and deletes the backslashes from unquoted Windows paths ('C:\Program Files\bin'
# -> 'C:Program Filesbin'), turning a visible mis-split into silent corruption.
# A quoted value keeps its spaces; a bare value stops at the first space, which
# is the same rule as a shell and is what the docstring tells the user to do.
_ENV_PAIR_RE = re.compile(r"([^\s=]+)=(\"[^\"]*\"|'[^']*'|\S*)")


def _parse_env_pairs(text: str) -> Dict[str, str]:
    """Parse space-separated KEY=VALUE pairs (the Add-MCP-Server env field is a
    single-line tk.Entry, so whitespace - not newlines - separates the pairs).
    Malformed tokens without "=" are ignored.

    A VALUE may be quoted to include spaces: PATH="C:\\Program Files\\bin".
    Surrounding quotes are stripped; the empty value KEY= is preserved."""
    out: Dict[str, str] = {}
    for m in _ENV_PAIR_RE.finditer(text or ""):
        v = m.group(2)
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        k = m.group(1)
        if k:
            out[k] = v
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


# ── Tool descriptions for the permission-bar hover tooltip (v1.1.61) ─────────
# The tip shows what a tool DOES as well as its permission state. The text comes
# from the SAME description the model is given (TOOL_SCHEMAS for built-ins, the
# server's own tool definition for MCP), so there is no second copy to maintain
# and the tooltip cannot drift out of date with the tool.
# Descriptions run 97-948 chars (run_shell is the longest), so they are collapsed
# to one line and trimmed before a tooltip shows them.
TOOL_TIP_DESC_MAX = 320


def _one_line_desc(text: str) -> str:
    """Collapse whitespace and trim a tool description to tooltip length.

    Preference order for where to cut, so the tip never ends mid-word:
      1. the last sentence end inside the budget;
      2. failing that, the last space inside the budget (a sentence boundary that
         falls in the first half would waste most of the room);
      3. only if there is no space at all (one enormous token), a hard cut.
    """
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(s) <= TOOL_TIP_DESC_MAX:
        return s
    room = TOOL_TIP_DESC_MAX
    cut = s.rfind(". ", 0, room)
    if cut <= room // 2:
        sp = s.rfind(" ", 0, room)
        cut = sp if sp > 0 else room
    else:
        cut += 1                      # keep the period: rfind gives its index
    return s[:cut].rstrip() + "\u2026"


def tool_description(name: str) -> str:
    """The model-facing description of a BUILT-IN tool, tooltip-sized ('' if none)."""
    fn = (TOOL_SCHEMAS.get(name) or {}).get("function") or {}
    return _one_line_desc(fn.get("description") or "")


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
    """Terminate a child process, escalating to kill(). Never raises.

    On Windows, terminating the shell does NOT terminate what the shell started:
    `Popen(["cmd","/c","longjob.exe"])` leaves longjob.exe running as an orphan
    when the cmd.exe is killed. run_shell would hit that constantly, so the kill
    is escalated to the whole process tree via taskkill /T. On POSIX the same
    effect comes from killing the process group (setpgrp at spawn is not done
    here, so this degrades to the old single-process behaviour).

    ⚠️ This is best-effort, not a guarantee: a child that has already daemonised
    away from the tree survives. Documented in the run_shell tool description."""
    if os.name == "nt":
        try:
            if proc.poll() is None:
                # /T kills the tree, /F forces; run from a hidden window and
                # never let a failure here abort the plain terminate path below.
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                               creationflags=0x08000000,      # CREATE_NO_WINDOW
                               timeout=5, capture_output=True)
        except Exception:
            pass
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
        self.resources: List[dict] = []      # resources/list, when the server has them
        self.prompts: List[dict] = []        # prompts/list, when the server has them
        self.server_caps: Dict[str, Any] = {}  # capabilities reported by initialize
        self.on_notification = None          # optional fn(method: str, params: dict)
        self.on_progress = None              # optional fn(progress, total, message)
        self.on_elicitation = None           # optional fn(server, params) -> ElicitResult
        self._pending: Dict[int, "queue.Queue"] = {}
        self._next_id = 0
        self._progress_seq = 0               # progressTokens we issued
        self._progress_msgs: Dict[int, List[str]] = {}   # token -> reported messages
        self._lock = threading.Lock()
        # stdin is now written from TWO threads: the worker (requests) and the reader
        # (replies to server-initiated requests). Without this a response line could
        # interleave with a request line and corrupt the JSON stream.
        self._write_lock = threading.Lock()
        self._closed = False

    # ── lifecycle ────────────────────────────────────────────────────────
    def connect(self, timeout: float = MCP_INIT_TIMEOUT) -> List[dict]:
        """Spawn the server, run the initialize handshake, return its tools.

        Raises RuntimeError on any failure (spawn, protocol, timeout); the
        subprocess is always cleaned up before re-raising."""
        full_env = dict(os.environ)
        full_env.setdefault("PYTHONUNBUFFERED", "1")   # piped stdout is block-buffered by default
        full_env.update(self.env)
        # Resolve the command through PATH first. With shell=False, Windows will NOT
        # append .CMD/.BAT, so the single most common MCP launch (`npx -y ...`) dies
        # with FileNotFoundError because the real file on disk is npx.CMD. shutil.which
        # applies PATHEXT and is a no-op for an absolute path or on POSIX.
        resolved = shutil.which(self.command) or self.command
        # ⚠️ creationflags is NOT optional on Windows. `npx` is npx.CMD, and running a
        # .CMD/.BAT with shell=False spawns a visible cmd.exe console window that must
        # stay open for the server's lifetime - the user sees a black console sitting on
        # the desktop for every MCP server. Every other subprocess spawn in this app
        # already passes CREATE_NO_WINDOW (_run_cancellable); this one was missed.
        try:
            self.proc = subprocess.Popen(
                [resolved] + self.args,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=full_env, bufsize=1, text=True, encoding="utf-8", errors="replace",
                creationflags=(0x08000000 if os.name == "nt" else 0))  # CREATE_NO_WINDOW
        except Exception as e:
            hint = ""
            if resolved != self.command:
                hint = f" (resolved '{self.command}' to '{resolved}')"
            elif not Path(str(self.command)).is_absolute():
                hint = (" - not found on PATH; give the full path to the executable "
                        "or install it first")
            raise RuntimeError(f"could not start '{self.command}': {e}{hint}")
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        try:
            init_result = self._request("initialize", {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                # Declaring a capability is a PROMISE: per spec a client that declares
                # `elicitation` MUST be able to answer elicitation/create, so this list
                # has to stay in step with what _handle_server_request actually serves.
                "capabilities": {
                    "tools": {},
                    "resources": {"listChanged": True},
                    "prompts": {"listChanged": True},
                    "elicitation": {},
                },
                "clientInfo": {"name": APP_NAME, "version": VERSION},
            }, timeout=timeout)
            if not isinstance(init_result, dict):
                raise RuntimeError(f"unexpected initialize result: {str(init_result)[:200]}")
            self.server_caps = init_result.get("capabilities") or {}
            if not isinstance(self.server_caps, dict):
                self.server_caps = {}
            self._notify("notifications/initialized", {})
            tools = self._request("tools/list", {}, timeout=timeout) or {}
            self.tools = [t for t in (tools.get("tools") or []) if isinstance(t, dict)]
            # resources/prompts are OPTIONAL server features. Ask only when initialize
            # says the server has them - calling an unsupported method makes the server
            # return an error, which would fail the whole connection over a feature the
            # model may never use.
            if "resources" in self.server_caps:
                try:
                    rl = self._request("resources/list", {}, timeout=timeout) or {}
                    self.resources = [r for r in (rl.get("resources") or []) if isinstance(r, dict)]
                except Exception:
                    self.resources = []
            if "prompts" in self.server_caps:
                try:
                    pl = self._request("prompts/list", {}, timeout=timeout) or {}
                    self.prompts = [p for p in (pl.get("prompts") or []) if isinstance(p, dict)]
                except Exception:
                    self.prompts = []
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
        """Parse stdout lines; route responses to waiters, handle the rest.

        Three kinds of message arrive on stdout:
          1. a response to one of OUR requests  -> hand it to the waiting caller
          2. a NOTIFICATION (method, no id)     -> progress / list-changed, forwarded
          3. a REQUEST from the SERVER (method + id) -> we must reply, or the server
             waits forever. This is how elicitation works.
        Case 3 did not exist before: every non-response line was silently skipped, so
        a server asking the user something hung its own tools/call until the client
        timeout fired."""
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
                if not isinstance(msg, dict):
                    continue
                mid = msg.get("id")
                if isinstance(mid, int) and ("result" in msg or "error" in msg):
                    with self._lock:
                        q = self._pending.pop(mid, None)
                    if q is not None:
                        try:
                            q.put_nowait(msg)
                        except Exception:
                            pass
                    continue
                if "method" in msg:
                    try:
                        self._handle_incoming(msg)
                    except Exception:
                        pass          # a bad notification must not kill the session
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

    def _handle_incoming(self, msg: dict) -> None:
        """Handle a server-initiated notification or request (reader thread)."""
        method = str(msg.get("method") or "")
        params = msg.get("params") if isinstance(msg.get("params"), dict) else {}
        mid = msg.get("id")
        is_request = mid is not None          # a notification has no id

        # progress: rides on any request/notification carrying a progressToken.
        tok = params.get("progressToken")
        if tok is not None and method == "notifications/progress":
            msg = str(params.get("message") or "")
            with self._lock:
                bucket = self._progress_msgs.get(tok)
                if bucket is not None and len(bucket) < 20:
                    bucket.append(msg or f"{params.get('progress')}/{params.get('total') or '?'}")
            if self.on_progress is not None:
                try:
                    self.on_progress(params.get("progress"), params.get("total"), msg)
                except Exception:
                    pass
            return

        if not is_request:
            # Plain notification (list_changed, logging, ...): forward if anyone cares.
            if self.on_notification is not None:
                try:
                    self.on_notification(method, params)
                except Exception:
                    pass
            return

        # A REQUEST: we owe the server a response, always - silence deadlocks it.
        if method == "elicitation/create":
            result = self._handle_elicitation(params)
        elif method == "ping":
            result = {}
        elif method == "sampling/createMessage":
            # The client would have to run a model for the server. Not supported, and
            # refusing is a legitimate answer (spec: method not found).
            self._respond_error(mid, -32601, "client does not support sampling")
            return
        else:
            self._respond_error(mid, -32601, f"client does not support '{method}'")
            return
        self._respond(mid, result)

    def _respond(self, mid: Any, result: Any) -> None:
        try:
            self._send({"jsonrpc": "2.0", "id": mid, "result": result})
        except Exception:
            pass

    def _respond_error(self, mid: Any, code: int, message: str) -> None:
        try:
            self._send({"jsonrpc": "2.0", "id": mid,
                        "error": {"code": code, "message": message}})
        except Exception:
            pass

    def _handle_elicitation(self, params: dict) -> dict:
        """Answer a server's elicitation/create by asking the user.

        Returns the ElicitResult: {"action": "accept", "content": {...}} or
        {"action": "decline"} / {"action": "cancel"}. The callback is supplied by the
        app (it owns the dialogs); if none is wired, decline - never invent an answer.
        """
        if self.on_elicitation is None:
            return {"action": "decline"}
        try:
            res = self.on_elicitation(self.name, params)
        except Exception:
            return {"action": "decline"}
        if not isinstance(res, dict):
            return {"action": "decline"}
        action = str(res.get("action") or "decline")
        if action not in ("accept", "decline", "cancel"):
            return {"action": "decline"}
        if action == "accept":
            content = res.get("content")
            # Per spec, "accept" MUST carry content matching the requested schema.
            # Forwarding accept with no content makes the server proceed on data no
            # user ever supplied (it reads missing fields as null/None) - strictly
            # worse than declining. An EMPTY dict is fine: that is a valid answer to a
            # schema with no properties (confirmation-only).
            if not isinstance(content, dict):
                return {"action": "decline"}
            return {"action": "accept", "content": content}
        return {"action": action}

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
            # Serialised: the reader thread replies to server requests while the worker
            # thread may be sending a request of its own. One whole line per write,
            # under the lock, so the two can never interleave mid-object.
            with self._write_lock:
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
    def call_tool_parts(self, raw_name: str, arguments: dict,
                        timeout: float = MCP_CALL_TIMEOUT, abort=None) -> tuple:
        """Call one of the server's tools; returns (text, image_parts).

        image_parts is a list of {"mime": str, "data": base64str} exactly as the
        server sent them. Returning them SEPARATELY is the whole point: an MCP image
        is pixels, not text, and must reach the model through the multimodal path
        rather than being described in a sentence.

        A progressToken is always offered: a server that wants to report progress
        needs the client to name the token, and notifications/progress then arrives
        on the reader thread while this call is still outstanding.

        Raises RuntimeError on transport/protocol errors or when the tool reports
        isError (the server's error text is included). 'abort' is polled while
        waiting so Stop can cut a long-running tool call short."""
        tok = self._next_progress_token()
        params = {"name": raw_name, "arguments": arguments,
                  "_meta": {"progressToken": tok}}
        with self._lock:
            self._progress_msgs[tok] = []
        try:
            result = self._request("tools/call", params, timeout, abort=abort)
        finally:
            with self._lock:
                got = self._progress_msgs.pop(tok, [])
        text, images = mcp_split_content(result)
        if isinstance(result, dict) and result.get("isError"):
            raise RuntimeError(f"MCP tool '{raw_name}' reported an error: {(text or '(no content)')[:1000]}")
        note = ""
        if got:
            # Progress lines are surfaced rather than dropped: a long call that said
            # "3 of 10 done" is far more useful to the model than silence.
            note = "\n".join(f"[progress] {x}" for x in got[:8])
        return ((text or "(tool returned no content)")
                + (("\n" + note) if note else ""), images)

    def _next_progress_token(self) -> int:
        with self._lock:
            self._progress_seq += 1
            return self._progress_seq

    # ── resources ────────────────────────────────────────────────────────
    def list_resources(self, cursor: str = "", timeout: float = MCP_INIT_TIMEOUT) -> dict:
        """resources/list. Returns the raw result dict (resources + nextCursor)."""
        params: Dict[str, Any] = {}
        if cursor:
            params["cursor"] = cursor
        res = self._request("resources/list", params, timeout=timeout) or {}
        return res if isinstance(res, dict) else {}

    def read_resource(self, uri: str, timeout: float = MCP_CALL_TIMEOUT) -> tuple:
        """resources/read. Returns (text, images) like call_tool_parts.

        Contents are text or blob; a blob whose mime is an image becomes an image
        part so reading a screenshot resource behaves like a tool returning one."""
        res = self._request("resources/read", {"uri": uri}, timeout=timeout) or {}
        texts: List[str] = []
        images: List[dict] = []
        for c in (res.get("contents") or [] if isinstance(res, dict) else []):
            if not isinstance(c, dict):
                continue
            if c.get("text") is not None:
                texts.append(str(c.get("text")))
            elif c.get("blob"):
                mime = str(c.get("mimeType") or "")
                if mime.startswith("image/"):
                    images.append({"mime": mime, "data": str(c.get("blob"))})
                else:
                    texts.append(f"[binary resource {uri}, "
                                 f"{len(str(c.get('blob')))} base64 chars]")
        return ("\n".join(texts).strip(), images)

    # ── prompts ──────────────────────────────────────────────────────────
    def list_prompts(self, cursor: str = "", timeout: float = MCP_INIT_TIMEOUT) -> dict:
        """prompts/list. Returns the raw result dict (prompts + nextCursor)."""
        params: Dict[str, Any] = {}
        if cursor:
            params["cursor"] = cursor
        res = self._request("prompts/list", params, timeout=timeout) or {}
        return res if isinstance(res, dict) else {}

    def get_prompt(self, name: str, arguments: Optional[dict] = None,
                   timeout: float = MCP_CALL_TIMEOUT) -> tuple:
        """prompts/get. Returns (description, messages) where messages are
        {role, text} - prompt messages are text or image; images are reported, not
        inlined, because a prompt is injected as conversation text."""
        res = self._request("prompts/get",
                            {"name": name, "arguments": arguments or {}},
                            timeout=timeout) or {}
        if not isinstance(res, dict):
            return ("", [])
        out: List[dict] = []
        for m in (res.get("messages") or []):
            if not isinstance(m, dict):
                continue
            content = m.get("content") or {}
            text = ""
            if isinstance(content, dict):
                if content.get("type") == "text":
                    text = str(content.get("text") or "")
                elif content.get("type") == "image":
                    mime = str(content.get("mimeType") or "image")
                    text = f"[image {mime} omitted from prompt]"
                else:
                    text = json.dumps(content, ensure_ascii=False)[:500]
            out.append({"role": str(m.get("role") or "user"), "text": text})
        return (str(res.get("description") or ""), out)

    def call_tool(self, raw_name: str, arguments: dict, timeout: float = MCP_CALL_TIMEOUT,
                  abort=None) -> str:
        """Text-only convenience wrapper over call_tool_parts (images discarded)."""
        return self.call_tool_parts(raw_name, arguments, timeout, abort=abort)[0]


# ════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

APP_NAME   = "Deskpilot"
VERSION    = "1.1.70"
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
MAX_STEER_QUEUE  = 8        # v1.1.52 #10: mid-run steering messages held for the next
                            # step boundary (bounded, so repeated sends cannot grow it)
FILE_READ_LIMIT  = 500000    # chars returned by read_local_file
CLIPBOARD_LIMIT  = 65536     # chars for get_clipboard_text / set_clipboard (schema + code share this; user-sized for local models)
CHATS_SAVE_MIN_INTERVAL = 5.0   # min seconds between mid-turn chats.json writes (turn end always saves)
CUSTOM_PROMPT_MAX = 4000        # char cap for the custom system prompt (Settings -> Custom System Prompt)
JS_TIMEOUT       = 1800        # seconds, run_javascript subprocess timeout
TURN_TIME_LIMIT_DEFAULT = 0   # default wall-clock cap per user prompt (seconds; 0 = NO LIMIT).
                              # The real value is the "Turn time limit" SETTING (Settings dialog,
                              # clamped 0..86400) - see _turn_time_limit(). It stays a backstop for
                              # the agentic loop when MAX_TOOL_STEPS == 0 so a model that keeps
                              # calling tools cannot run forever, but it is checked at step
                              # boundaries only and now defaults to unlimited: long turns are the
                              # point of the app, and an unwanted cap kept cutting sessions short.
TURN_TIME_LIMIT_MAX   = 86400 # upper bound for the setting (24 h); 0 always means "no limit".
COMPACTION_THRESHOLD_DEFAULT = 70   # % of the context window; 0 disables auto-compaction (C1).
                                    # Kept BELOW the default handoff threshold (75) so an in-place
                                    # compaction fires before a fork would - see SOW C5 note.
COMPACTION_KEEP_RECENT_DEFAULT = 10 # most-recent message entries kept verbatim when compacting (C1)
CHAT_RENDER_WINDOW_DEFAULT = 120   # message entries rendered per chat view (0 = render all); v1.1.36
STOP_POLL_S      = 0.25       # max latency of the Stop button while a tool is blocking
IMAGE_CLI_TIMEOUT = 900       # seconds, local FLUX/SDXL CLI subprocess
PROC_OUTPUT_CAP  = 4_000_000  # bytes per stream kept from a subprocess (deadlock-safe drain)
TOOL_OUTPUT_LIMIT = 192000   # max chars of ANY tool result sent back to the model (hard cap)
JS_STDOUT_CAP    = 48_000   # chars of run_javascript stdout returned to the model (schema text shares this)
# ── run_shell (SOW #6) ───────────────────────────────────────────────────────
# Deliberately NOT sandboxed: it grants exactly what run_javascript already
# grants (unsandboxed node with full fs + child_process), in one line instead of
# a JS blob. The protection is the Ask permission + per-turn checkpoints, NOT a
# command denylist - a denylist is trivially bypassable and would sell false
# confidence while blocking legitimate use.
SHELL_TIMEOUT     = 600      # seconds; shorter than JS_TIMEOUT on purpose (a shell
                             # command is normally quick; long jobs must be explicit)
SHELL_TIMEOUT_MAX = 3600     # ceiling on a model-supplied timeout
SHELL_STDOUT_CAP  = 48_000   # chars per stream returned to the model (schema shares this)
EDIT_FILE_MAX    = 2_000_000  # max bytes of a file edit_local_file will load and patch
SEARCH_PER_FILE_CAP        = 20    # search_files: max matches reported per file
SEARCH_DEFAULT_MAX_MATCHES = 200   # search_files: default global match cap
SEARCH_MAX_MATCHES_CAP     = 500   # search_files: hard upper bound for max_matches
SEARCH_CONTEXT_MAX         = 5     # search_files: max context lines per match
SEARCH_SKIP_DIRS = frozenset({     # search_files: noise dirs never descended into
    "__pycache__", ".git", "node_modules", ".venv", "venv",
    "site-packages", "dist", "build", ".cache"})
MODEL_HISTORY_MAX = 20      # previously-used model names kept for the Model Name dropdown

# ── ask_expert (v1.1.63): a frontier model as a CONSULTANT, not the agent ────
# The primary model stays local and stays in charge; it may ask one outside
# model for a narrow opinion. The whole point of these constants is the user's
# constraint: "I only ever want it to see what it is being explicitly asked
# about." So the payload is built from scratch (never build_system_message, and
# never the chat history) and the file reads are jailed to the workspace even
# when the optional File Workspace setting is blank.
EXPERT_MAX_CALLS_PER_TURN = 5     # a small model will over-delegate; this is a
                                  # sanity/latency guard, not a budget (the user's
                                  # plan is a fixed monthly allowance that degrades
                                  # to flash-lite when spent, so cost is not the reason)
EXPERT_MAX_TOKENS = 32768         # generous on purpose. Gemini 3 CANNOT disable
                                  # thinking, and a thinking model spends the whole
                                  # cap on reasoning_content and returns content=''
                                  # (reproduced in v1.1.58 at max_tokens=1024). A
                                  # tight cap here would silently blank the answer
                                  # to exactly the hard questions worth asking.
EXPERT_TIMEOUT_S = 300            # one call; a thinking model can take a while
EXPERT_FILE_CHARS = 60000         # per file handed to the expert
EXPERT_TOTAL_CHARS = 200000       # all files + context combined
EXPERT_MIN_CAP = 4096             # below this, retrying without a cap is pointless

# ── Full-text chat search (v1.1.42) ───────────────────────────────────────────
# The sidebar search box only ever matched chat TITLES, so a 20 MB history was
# effectively unsearchable. This adds a SQLite FTS5 index over every message body.
# sqlite3 + FTS5 are STDLIB (verified: sqlite 3.49.1, FTS5 and the trigram tokenizer
# both present, and the frozen exe already bundles _sqlite3.pyd + sqlite3.dll, so no
# build flags change). Measured on the real 20.7 MB / 44-chat / 6,590-message file:
# full rebuild 0.13 s, every query sub-millisecond - so a rebuild is cheap enough to
# do on demand rather than hooking it into every save.
SEARCH_DB_NAME          = "deskpilot_search.db"  # next to CHATS_FILE; disposable cache
SEARCH_INDEX_MIN_INTERVAL = 3.0   # s between rebuilds (stops a storm while streaming)
SEARCH_BODY_DEBOUNCE_MS   = 250   # idle before body search runs (title filter stays instant)
SEARCH_BODY_CAP         = 8000    # chars of one message fed to the index
SEARCH_MAX_SNIPPETS     = 12      # hits returned to the model / shown per query
SEARCH_SNIPPET_TOKENS   = 22      # FTS5 snippet() window, in tokens
SEARCH_SIDEBAR_HITS     = 4       # chats listed in the sidebar (the Listbox must survive)
SEARCH_SIDEBAR_SNIPPET  = 90      # chars of snippet shown per sidebar row
# v1.1.54 Tier 2: the FTS5 index now also covers chat["compaction_archive"], which holds
# every message a compaction folded away (6,212 messages / 12 MB in the live store). They
# were readable in-app but INVISIBLE to the agent: search_chats could not reach them and
# no tool could read them, so "preserved" meant preserved for the user, lost to the model.
# Bump when the indexed row shape changes - search_index_is_stale() then forces a rebuild
# instead of letting old code read a new table (or vice versa) and silently degrade.
SEARCH_SCHEMA_VERSION   = "2"
ARCHIVE_PAGE_MSGS       = 20      # read_archive: messages returned per call (default)
ARCHIVE_PAGE_MSGS_MAX   = 60      # read_archive: hard ceiling on one call
ARCHIVE_MSG_CAP         = 4000    # read_archive: chars of one archived message

# ── v1.1.54 Tier 3: the ledger ───────────────────────────────────────────────
# An APPEND-ONLY store of things worth keeping, with a BOUNDED view.
#
# Why this shape: every other record of a long chat is lossy or replaceable. The
# live message list is rewritten by compaction; the compaction summary is a model
# guess made after the fact; chat["compaction_archive"] survives but is deleted with
# the chat. A ledger that is only ever appended to cannot silently lose an earlier
# entry, and it lives in its own file so deleting the conversation does not take the
# memory with it. The store is unbounded; only what gets injected into the prompt is
# capped - an uncapped "keep everything" view would just hit the same wall later.
LEDGER_DIR_NAME         = "ledger"
LEDGER_GLOBAL_NAME      = "_global.jsonl"   # standing facts, shared by every chat
LEDGER_ENTRY_MAX_CHARS  = 4000    # one entry's text (caps a single remember() call)
LEDGER_ENTRY_HARD_MAX   = 20000   # beyond this the call is refused, not truncated
LEDGER_TAGS_MAX         = 8
LEDGER_PAGE_DEFAULT     = 25      # read_ledger: entries per call
LEDGER_PAGE_MAX         = 100
LEDGER_INJECT_DEFAULT   = 4000    # chars of ledger tail in the system prompt (0 = off)
LEDGER_INJECT_MAX       = 20000   # ceiling on that setting
LEDGER_INJECT_ENTRIES   = 60      # never inject more than this many entries
LEDGER_SCAN_MAX_BYTES   = 4_000_000  # a runaway ledger is read up to this far

# ── v1.1.66: project memory index ────────────────────────────────────────────
# The workspace-root MEMORY.md is an INDEX (what projects exist, where each
# project's own MEMORY.md is), not an archive. Injecting it is what makes the
# "read the project's memory file first" convention automatic: a session cannot
# miss a file it has never been shown. Same shape as the ledger block - bounded
# VIEW, unbounded store on disk, and the text names the tool that reaches the
# rest. Kept small by convention (~6 KB); PROJECT_MEMORY_MAX is the hard ceiling
# so a bloated index cannot silently inflate every request.
PROJECT_MEMORY_NAME     = "MEMORY.md"
PROJECT_MEMORY_SUBDIR   = "projects"      # where per-project memory files live
PROJECT_MEMORY_DEFAULT  = 8000            # chars injected (0 = off)
PROJECT_MEMORY_MAX      = 24000           # ceiling on that setting
PROJECT_MEMORY_READ_MAX = 64000           # bytes read from disk before capping
_LEDGER_LOCK = threading.Lock()   # serialises appends (worker + main thread)
# Serialises index rebuilds: _ensure_search_index() is called from the main thread
# (sidebar search) and the chat worker (search_chats tool), and the rebuild does
# DROP/CREATE on a shared scratch table inside one transaction. A lock, not just the
# time-based rate limiter, is what makes check-and-build atomic.
_SEARCH_BUILD_LOCK = threading.Lock()
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


_SYS_DIR_NAMES_NT = ("$recycle.bin", "system volume information")


def _system_dir_roots(system_drive: str, drive_types: Dict[str, int]) -> List[str]:
    r"""System-critical Windows roots, independent of the boot-drive letter (v1.1.12).

    FORBIDDEN_SYSTEM_DIRS_WINDOWS pins C:\, so a machine booted from D:\ - and every
    secondary/removable drive's $Recycle.Bin / System Volume Information - was not
    covered. This returns:
      * <SystemDrive>:\windows        (e.g. d:\windows when SystemDrive=D:)
      * $recycle.bin + system volume information on each drive whose GetDriveTypeW is
        REMOVABLE(2) or FIXED(3).
    Pure function (drive letters + types injected) so the headless suite can pin the
    logic without real drives; this machine has no D:/E:. Callers pass raw strings."""
    roots: List[str] = []
    sd = (system_drive or "").rstrip("\\/").lower()
    if re.fullmatch(r"[a-z]:", sd):
        roots.append(sd + "\\windows")
    for letter, dtype in drive_types.items():
        if dtype in (2, 3):
            base = str(letter).lower() + ":\\"
            for name in _SYS_DIR_NAMES_NT:
                roots.append(base + name)
    return roots


def _nt_drive_types() -> Dict[str, int]:
    """GetDriveTypeW for A-Z (unreadable letters raise -> skipped)."""
    out: Dict[str, int] = {}
    try:
        k32 = ctypes.windll.kernel32
        for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            try:
                out[ch] = int(k32.GetDriveTypeW(ch + ":\\"))
            except Exception:
                pass
    except Exception:
        pass
    return out


def _forbidden_dir_hit(p_str: str) -> Optional[str]:
    """Return the forbidden system dir that p_str (already lowercase) equals or is inside, else None."""
    if os.name == "nt":
        forbidden = list(FORBIDDEN_SYSTEM_DIRS_WINDOWS) + _system_dir_roots(
            os.environ.get("SystemDrive", ""), _nt_drive_types())
    else:
        forbidden = list(FORBIDDEN_SYSTEM_DIRS_LINUX)
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


DIFF_MAX_CHARS = 6000      # per-diff cap: one huge diff must not crowd out the turn


def _unified_diff_text(before: str, after: str, context: int = 3) -> str:
    """A unified diff between two texts ('' when identical).

    Stdlib difflib, no dependency. Lines are taken with splitlines(), which strips
    the line terminator - so a CRLF-preserving edit does NOT show every line as
    changed and the diff reports what was actually altered rather than the file's
    EOL convention. (split("\n") would leave a trailing '\r' on every line of a CRLF
    file and make a one-line edit look like a whole-file rewrite - negative control
    #3 pins that.)
    """
    if before == after:
        return ""
    a = (before or "").splitlines()
    b = (after or "").splitlines()
    try:
        ctx = max(0, int(context))
    except (TypeError, ValueError):
        ctx = 3
    out = list(difflib.unified_diff(a, b, fromfile="before", tofile="after",
                                    lineterm="", n=ctx))
    if not out:
        return ""
    # A file read with errors='replace' can differ only in replacement chars; still
    # worth showing, but never let the diff itself become the biggest thing in the
    # message.
    if len(out) > 400:
        kept = out[:200]
        kept.append(f"[... diff truncated: {len(out) - 400} of {len(out)} diff lines omitted "
                    f"- the edit still applied in full]")
        out = kept
    text = "\n".join(out)
    if len(text) > DIFF_MAX_CHARS:
        text = text[:DIFF_MAX_CHARS] + f"\n[... diff truncated at {DIFF_MAX_CHARS} chars - " \
                                       f"the edit still applied in full]"
    return text


def _diff_block(diff: str) -> str:
    """Wrap a diff in a labelled fence for the tool receipt."""
    if not diff:
        return "(no textual change detected)"
    return f"--- Diff ---\n{diff}"


def _as_bool(value) -> bool:
    """Coerce a tool argument to a real bool.

    LLMs frequently send booleans as strings ("false", "0", "true"); plain
    bool("false") is True, which would silently invert the intent. Only
    explicit truthy spellings count as True; anything else is False.
    """
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "y", "on")
    return bool(value)


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
    ("searxng_search",       "🔎 Web"),
    ("exa_search",           "🌟 Exa"),
    ("firecrawl_scrape",     "🔥 Firecrawl"),
    ("fetch_url",            "🌐 Fetch"),
    ("run_javascript",       "⚡ JS"),
    ("run_shell",            "🖥 Shell"),
    ("list_directory",       "📁 List"),
    ("search_files",         "🔎 Files"),
    ("read_local_file",      "📄 Read"),
    ("write_local_file",     "✍️ Write"),
    ("edit_local_file",      "✂️ Edit"),
    ("generate_local_image", "🎨 Image"),
    ("capture_screen",       "📸 Screen"),
    ("get_clipboard_text",   "📋 Paste"),
    ("list_skills",          "📚 Skills"),
    ("load_skill",           "📖 Load"),
    ("search_chats",         "🔎 Chats"),
    ("read_archive",         "📦 Archive"),
    ("remember",             "🧷 Remember"),
    ("read_ledger",          "📓 Ledger"),
    ("set_clipboard",        "📋 Set"),
    ("transcribe_audio",     "🎧 Audio"),
    # Consultant. Must be listed here too or it gets no permission chip, and the
    # bar is the only place the user can see it is set to Ask or switch it Off.
    ("ask_expert",           "🧠 Expert"),
    ("list_mcp_resources",   "🔌 Res"),
    ("read_mcp_resource",    "📥 Res"),
    ("list_mcp_prompts",     "📝 Prompts"),
    ("get_mcp_prompt",       "📜 Prompt"),
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
                            "Use it whenever a page needs a browser engine or resists plain HTTP fetching - "
                            "the user's plan carries a generous monthly allowance, so there is no need to "
                            "hoard calls. fetch_url is still the faster first choice for simple static "
                            "pages. Firecrawl REFUSES reddit.com outright - do not call it on a Reddit "
                            "URL; fetch_url handles Reddit natively."),
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
                            "addresses (localhost, LAN, cloud metadata) are blocked by default. "
                            "Reddit works: pass the normal reddit.com thread / subreddit / search "
                            "URL and Deskpilot reads it through Reddit's public Atom feed, "
                            "returning the post plus its comments (no vote counts, roughly a "
                            "100-comment cap)."),
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
                            f"stdout is capped at {JS_STDOUT_CAP} chars (a truncation marker is appended) - "
                            f"for larger results write them to a file and read it back in chunks. The "
                            "process runs in the configured File Workspace - the SAME root the "
                            "read/write/edit file tools use - so prefer paths relative to it "
                            "over long absolute path literals; a relative path you passed to "
                            "write_local_file will be found here."),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "JavaScript source code to execute."}
                },
                "required": ["code"]
            }
        }
    },
    "run_shell": {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": ("Run ONE shell command line and return stdout/stderr. "
                            "On Windows this is cmd.exe /c; on POSIX /bin/sh -c. "
                            "⚠️ UNSANDBOXED: the command runs with your full user "
                            "privileges, with NO path jail - it can read, modify or "
                            "delete anything your account can, anywhere on the "
                            "machine. It is gated by the Ask permission for that "
                            "reason. Files it changes are NOT covered by the "
                            "per-turn checkpoints (those only snapshot write_local_file "
                            "and edit_local_file), so \u21ba Revert turn will NOT undo "
                            "shell-driven edits. "
                            f"Stateless: each call is a fresh process, so cd and "
                            "environment variables do NOT persist between calls - "
                            "chain them in one command line (e.g. "
                            "`cd dir && cmd`). Default timeout "
                            f"{SHELL_TIMEOUT} s (overridable up to {SHELL_TIMEOUT_MAX}); "
                            "Stop kills the whole process tree. stdout and stderr are "
                            f"each capped at {SHELL_STDOUT_CAP} chars - redirect to a "
                            "file and read it back with read_local_file for larger "
                            "output. The command starts in the configured File "
                            "Workspace, so prefer paths relative to it."),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string",
                                "description": "The command line to execute."},
                    "timeout": {"type": "integer",
                                "description": (f"Optional seconds before the command "
                                                f"is killed (default {SHELL_TIMEOUT}, "
                                                f"max {SHELL_TIMEOUT_MAX}).")},
                },
                "required": ["command"]
            }
        }
    },
    "read_local_file": {
        "type": "function",
        "function": {
            "name": "read_local_file",
            "description": (f"Read a local file and return its text content (max {FILE_READ_LIMIT} characters). "
                            "Supports plain-text files and PDF documents. For files longer than the "
                            "limit, read again with offset set to the previous end position; the "
                            "footer always reports the total length."),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Filesystem path of the file to read (absolute, or relative to the current working directory)."},
                    "offset": {"type": "integer",
                               "description": "0-based character index to start reading from. Default 0."},
                    "limit": {"type": "integer",
                               "description": f"Max characters to return (1-{FILE_READ_LIMIT}). Default {FILE_READ_LIMIT}."}
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
                    "path":        {"type": "string",  "description": "Directory path to list (absolute, or relative to the current working directory)."},
                    "max_entries": {"type": "integer", "description": "Maximum entries to show (1-1000). Default 200."}
                },
                "required": ["path"]
            }
        }
    },
    "search_files": {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": ("Search the local filesystem for a text pattern and return matching "
                            "file:line locations with trimmed line text. Use this to LOCATE code "
                            "before reading it - far cheaper than reading whole files. Binary "
                            "files and noise dirs (__pycache__, .git, node_modules, ...) are "
                            "skipped."),
            "parameters": {
                "type": "object",
                "properties": {
                    "path":        {"type": "string",  "description": "File or directory to search. Default = current working directory; prefer paths relative to it."},
                    "pattern":     {"type": "string",  "description": "Text to find."},
                    "regex":       {"type": "boolean", "description": "Treat pattern as a Python regex. Default false (literal)."},
                    "ignore_case": {"type": "boolean", "description": "Case-insensitive matching. Default false."},
                    "glob":        {"type": "string",  "description": "Filename filter, e.g. \"*.py\". Default: all text files."},
                    "context":     {"type": "integer", "description": f"Lines of context per match (0-{SEARCH_CONTEXT_MAX}). Default 0."},
                    "max_matches": {"type": "integer", "description": f"Stop after N matches (1-{SEARCH_MAX_MATCHES_CAP}). Default {SEARCH_DEFAULT_MAX_MATCHES}."}
                },
                "required": ["pattern"]
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
                    "path":    {"type": "string", "description": "Filesystem path to write (absolute, or relative to the current working directory)."},
                    "content": {"type": "string", "description": "Full text content to save."}
                },
                "required": ["path", "content"]
            }
        }
    },
    "edit_local_file": {
        "type": "function",
        "function": {
            "name": "edit_local_file",
            "description": ("Perform surgical in-place text replacements in an existing file. "
                            "Each edit's old_text must match EXACTLY once in the file (whitespace- "
                            "and newline-exact). All edits are applied as ONE transaction: if any "
                            "old_text is not found exactly once, NOTHING is written and every "
                            "mismatch is reported. Strongly preferred over write_local_file for "
                            "changing existing files - only the changed regions are sent. Edits "
                            "are applied sequentially in listed order: later edits see the output "
                            "of earlier ones (dependent edits are allowed)."),
            "parameters": {
                "type": "object",
                "properties": {
                    "path":  {"type": "string",
                              "description": "File to edit (must exist; absolute, or relative to the current working directory)."},
                    "edits": {
                        "type": "array",
                        "description": "One or more replacements, applied in order.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "old_text": {"type": "string",
                                             "description": "Exact text to find (whitespace/newline-exact)."},
                                "new_text": {"type": "string",
                                             "description": "Replacement text."},
                                "allow_multiple": {"type": "boolean",
                                                   "description": "Replace every occurrence instead of requiring exactly one. Default false."}
                            },
                            "required": ["old_text", "new_text"]
                        }
                    },
                    "dry_run": {"type": "boolean",
                                "description": "Validate and report counts without writing. Default false."}
                },
                "required": ["path", "edits"]
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
    "set_clipboard": {
        "type": "function",
        "function": {
            "name": "set_clipboard",
            "description": ("Replace the system clipboard contents with the given text. "
                            f"Capped at {CLIPBOARD_LIMIT} characters. Pair it with "
                            "get_clipboard_text to move text between apps: put the text here, "
                            "then tell the user to paste it (Ctrl+V) into the target app."),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The text to place on the clipboard."}
                },
                "required": ["text"]
            }
        }
    },
    "transcribe_audio": {
        "type": "function",
        "function": {
            "name": "transcribe_audio",
            "description": ("Transcribe an audio file on this machine using the same local "
                            "Whisper model as the microphone - the audio never leaves the "
                            "computer. Accepts anything PyAV can decode (wav, mp3, m4a, flac, "
                            "ogg, ...), up to 64 MB. Returns the transcript text. Use it for "
                            "voice memos, meeting recordings and podcasts; it is slow on long "
                            "files, so say so before starting one."),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string",
                             "description": "Filesystem path of the audio file to transcribe."},
                    "language": {"type": "string",
                                 "description": ("ISO language code, e.g. 'en' or 'de'. "
                                                 "Blank = auto-detect. Default blank.")}
                },
                "required": ["path"]
            }
        }
    },
    "list_mcp_resources": {
        "type": "function",
        "function": {
            "name": "list_mcp_resources",
            "description": ("List the RESOURCES a connected MCP server exposes "
                            "(files, database rows, app state). Resources are the "
                            "server's way of offering context it thinks you need - "
                            "check this before asking the user for something that is "
                            "already available. Returns URIs to pass to "
                            "read_mcp_resource. Empty if no server advertises resources."),
            "parameters": {
                "type": "object",
                "properties": {
                    "server": {"type": "string",
                               "description": ("MCP server name. Optional: required only "
                                               "when several servers expose resources.")}
                }
            }
        }
    },
    "read_mcp_resource": {
        "type": "function",
        "function": {
            "name": "read_mcp_resource",
            "description": ("Read one MCP resource by URI (from list_mcp_resources). "
                            "Returns its text content; image resources are attached for "
                            "visual analysis exactly like a tool image would be."),
            "parameters": {
                "type": "object",
                "properties": {
                    "uri": {"type": "string",
                            "description": "Resource URI exactly as list_mcp_resources reported it."},
                    "server": {"type": "string",
                               "description": ("MCP server name. Optional: required only "
                                               "when several servers expose resources.")}
                },
                "required": ["uri"]
            }
        }
    },
    "list_mcp_prompts": {
        "type": "function",
        "function": {
            "name": "list_mcp_prompts",
            "description": ("List the PROMPT TEMPLATES a connected MCP server offers, "
                            "with the arguments each one takes. Servers use these to "
                            "package a task its authors know how to do well - worth "
                            "checking before writing a prompt from scratch. "
                            "Fetch one with get_mcp_prompt."),
            "parameters": {
                "type": "object",
                "properties": {
                    "server": {"type": "string",
                               "description": ("MCP server name. Optional: required only "
                                               "when several servers expose prompts.")}
                }
            }
        }
    },
    "get_mcp_prompt": {
        "type": "function",
        "function": {
            "name": "get_mcp_prompt",
            "description": ("Fetch one MCP prompt template by name and read its rendered "
                            "messages. The prompt comes back as text for you to read and "
                            "use - it is NOT injected into the conversation automatically, "
                            "so decide whether to follow it."),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string",
                             "description": "Prompt name exactly as list_mcp_prompts reported it."},
                    "server": {"type": "string",
                               "description": ("MCP server name. Optional: required only "
                                               "when several servers expose prompts.")},
                    "arguments": {"type": "object",
                                  "description": ("Values for the prompt's arguments, as "
                                                  "name/value pairs. Omit if it takes none.")}
                },
                "required": ["name"]
            }
        }
    },
    "search_chats": {
        "type": "function",
        "function": {
            "name": "search_chats",
            "description": ("Full-text search across ALL of the user's saved chat history "
                            "(every message in every conversation, not just this chat). "
                            "Returns the best-matching chats with a snippet around each hit "
                            "and the chat_id + message index. Use it for 'what did we decide "
                            "about X' / 'where did we fix Y' questions about PAST work."),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "Words to search for; all must appear (prefix matching)."},
                    "limit": {"type": "integer",
                              "description": f"Max hits to return (1-50). Default {SEARCH_MAX_SNIPPETS}."},
                    "full": {"type": "boolean",
                             "description": "Also return the complete text of each matched message. Default false."}
                },
                "required": ["query"]
            }
        }
    },
    "read_archive": {
        "type": "function",
        "function": {
            "name": "read_archive",
            "description": ("Read messages that an earlier CONTEXT COMPACTION folded out "
                            "of this chat. Compaction replaces old messages with a summary, "
                            "but the originals are kept in the chat's archive and are NOT "
                            "part of the live conversation - so anything they contain is "
                            "invisible to you until you read it here. Use it when the chat "
                            "shows a '[COMPACTED - summary of N earlier messages]' marker "
                            "and the summary is not specific enough: exact code, error text, "
                            "file contents, tool output or a decision made long ago. "
                            "search_chats also covers the archive across all chats and "
                            "reports which compaction event a hit came from."),
            "parameters": {
                "type": "object",
                "properties": {
                    "chat_id": {"type": "string",
                                "description": ("Which chat. Omit for the CURRENT chat "
                                                "(the usual case). Another chat's id comes "
                                                "from search_chats.")},
                    "compaction": {"type": "integer",
                                   "description": ("Which compaction event to read (1-based, "
                                                   "oldest). Omit to list every event with "
                                                   "its message count and date.")},
                    "offset": {"type": "integer",
                               "description": ("Skip this many messages in that event "
                                               "(0-based). Default 0.")},
                    "limit": {"type": "integer",
                              "description": f"Max messages to return (1-{ARCHIVE_PAGE_MSGS_MAX}). Default {ARCHIVE_PAGE_MSGS}."}
                },
                "required": []
            }
        }
    },
    "remember": {
        "type": "function",
        "function": {
            "name": "remember",
            "description": ("Append one durable note to this chat's LEDGER: an append-only "
                            "record that survives context compaction, handoff, and even "
                            "deletion of the chat. Use it AT THE MOMENT something matters, "
                            "not later - a compaction summary is a lossy guess made after "
                            "the fact, so anything you judge important should be written "
                            "down while you still have the full context. Good entries: a "
                            "decision and why, a constraint that must not be re-litigated, "
                            "an exact value/path/command that worked, a failure mode worth "
                            "not repeating, an open question. One entry per fact; keep it "
                            "self-contained (a future session reads it with no surrounding "
                            "context). This never overwrites an earlier entry."),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string",
                             "description": (f"The note. Self-contained and specific. "
                                             f"Max {LEDGER_ENTRY_MAX_CHARS} chars "
                                             f"(longer is refused, not silently cut).")},
                    "tags": {"type": "array", "items": {"type": "string"},
                             "description": (f"Optional labels for later filtering "
                                             f"(max {LEDGER_TAGS_MAX}).")},
                    "global": {"type": "boolean",
                               "description": ("True to write to the standing ledger shared "
                                               "by EVERY chat (user preferences, durable "
                                               "environment facts). Default false = this "
                                               "chat's ledger.")}
                },
                "required": ["text"]
            }
        }
    },
    "read_ledger": {
        "type": "function",
        "function": {
            "name": "read_ledger",
            "description": ("Read a chat's append-only ledger (and the shared global one). "
                            "Only the most recent entries are injected into the system "
                            "prompt automatically, so this is how to reach the older ones - "
                            "the full record is never pruned. Use it when the injected "
                            "LEDGER block mentions entries it omitted for length, or when "
                            "filtering by tag."),
            "parameters": {
                "type": "object",
                "properties": {
                    "chat_id": {"type": "string",
                                "description": ("Which chat's ledger. Omit for the CURRENT "
                                                "chat. Pass \"*\" for the global ledger "
                                                "alone.")},
                    "tag": {"type": "string",
                            "description": "Only entries carrying this tag."},
                    "search": {"type": "string",
                               "description": "Only entries whose text contains this."},
                    "limit": {"type": "integer",
                              "description": f"Max entries (1-{LEDGER_PAGE_MAX}), newest first. Default {LEDGER_PAGE_DEFAULT}."},
                    "offset": {"type": "integer",
                               "description": "Skip this many (newest-first). Default 0."}
                },
                "required": []
            }
        }
    },
    "list_skills": {
        "type": "function",
        "function": {
            "name": "list_skills",
            "description": ("List the Agent Skills DeskPilot can load (folders containing a "
                            "SKILL.md) plus the folders scanned for them. A skill is a "
                            "user-written, reusable procedure; only its name and description "
                            "are in context, so call this to see where they live or "
                            "load_skill to read one in full."),
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    "load_skill": {
        "type": "function",
        "function": {
            "name": "load_skill",
            "description": ("Read one Agent Skill in full: its SKILL.md instructions and the "
                            "files bundled in its folder. Call it BEFORE starting a task that a "
                            "listed skill covers, then follow the instructions. Relative paths in "
                            "the body resolve against the skill folder that is reported."),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string",
                             "description": "The skill name, exactly as listed in the system prompt or by list_skills."}
                },
                "required": ["name"]
            }
        }
    },
    "ask_expert": {
        "type": "function",
        "function": {
            "name": "ask_expert",
            "description": (
                "Ask a second, stronger AI model for a narrow second opinion. Use it where a "
                "fresh pair of eyes genuinely helps: reviewing a diff before you ship it, an API "
                "you are unsure exists, a design trade-off, a bug you cannot explain. "
                "The expert CANNOT see this conversation, your files, your settings or your "
                "history - it receives ONLY the question, the files you name, and any context "
                "text you supply. So the question must be fully self-contained: state the "
                "language, the framework, what you expect, and what specifically worries you. "
                "Files must live inside the workspace. Max 5 calls per turn. "
                "The answer is ADVISORY, not authoritative: an expert can invent APIs too, so "
                "verify any method or signature it names before you use it."),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string",
                                 "description": "The specific question. Self-contained - the expert sees nothing else."},
                    "files": {
                        "type": "array",
                        "description": ("Files the expert should read. Paths MUST be inside the workspace. "
                                        "Prefer this over pasting code: the expert then sees the real "
                                        "current bytes instead of whatever you retyped."),
                        "items": {
                            "type": "object",
                            "properties": {
                                "path":   {"type": "string",  "description": "Path inside the workspace."},
                                "offset": {"type": "integer", "description": "0-based char index to start at. Default 0."},
                                "limit":  {"type": "integer", "description": "Max chars to send. Default 60000."}
                            },
                            "required": ["path"]
                        }
                    },
                    "context": {"type": "string",
                                "description": ("Short excerpt that exists only in your head - a proposed "
                                                "change not yet written to disk, or an error message. "
                                                "Keep it to what the question is actually about.")},
                    "focus": {"type": "string",
                              "enum": ["review", "debug", "design", "second_opinion"],
                              "description": "What kind of answer you want. Default review."}
                },
                "required": ["question"]
            }
        }
    },
}

# ── Permission options ───────────────────────────────────────────────────────
PERM_LABELS = {"always": "Always Allow", "ask": "Ask Permission", "off": "Off"}
PERM_VALUES = {v: k for k, v in PERM_LABELS.items()}
# Click-to-cycle order for the permission chips (v1.1.41): one click per step,
# wrapping, so the bar needs no dropdown per tool - the label's COLOUR is the state.
# PERM_LABELS/PERM_VALUES stay (the long names now surface as a hover tooltip).
PERM_CYCLE = ("always", "ask", "off")
PERM_HINT = {"always": "always allow", "ask": "ask first", "off": "off"}
# Compact spellings for the bar's one-off colour key. The long PERM_LABELS words
# ("Always Allow" / "Ask Permission") cost 215 px of bar width - enough to push the
# last chip onto a second row at a normal window size - so the key uses these.
PERM_KEY = {"always": "● Allow", "ask": "● Ask", "off": "● Off"}
DEFAULT_PERMS = {
    "searxng_search":        "always",
    "exa_search":            "always",
    "firecrawl_scrape":      "always",
    "fetch_url":             "always",
    "list_directory":        "always",
    "search_files":          "always",
    "read_local_file":       "always",
    "get_clipboard_text":    "always",
    "run_javascript":        "ask",
    # Unsandboxed command execution - same posture as run_javascript and the two
    # file-mutating tools. Do not default this to "always".
    "run_shell":             "ask",
    "write_local_file":      "ask",
    "edit_local_file":       "ask",
    "generate_local_image": "ask",
    "capture_screen":        "ask",
    "list_skills":           "always",
    "load_skill":            "always",
    "search_chats":          "always",
    # Read-only view of the app's own archived history (compacted-away messages).
    # Same trust level as search_chats: it exposes nothing the user cannot already
    # read in-app, and it never writes.
    "read_archive":          "always",
    # remember() writes to the app's own ledger file (append-only, never overwrites
    # anything the user made) and read_ledger() only reads it. Neither touches the
    # user's documents, so both sit at the same trust level as search_chats.
    "remember":              "always",
    "read_ledger":           "always",
    "set_clipboard":         "ask",
    "transcribe_audio":      "ask",
    # ask_expert sends the material you name to a THIRD PARTY over the network.
    # Never default this to "always": the permission modal is also the only
    # place the user can see the exact payload before it leaves the machine.
    "ask_expert":            "ask",
    # MCP resources/prompts are read-only views of a server's own data, but the
    # server is an arbitrary local program, so they stay "ask" like MCP tools.
    "list_mcp_resources":    "ask",
    "read_mcp_resource":     "ask",
    "list_mcp_prompts":      "ask",
    "get_mcp_prompt":        "ask",
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
    "firecrawl_api_key": "",    # Firecrawl API key for firecrawl_scrape; blank = tool disabled
    "ui_font_size":    12,    # base UI font size (Settings -> UI Font Size)
    "max_tokens":      0,    # per-reply token cap; 0 = server default (Settings -> Max Tokens)
    "turn_time_limit": TURN_TIME_LIMIT_DEFAULT,  # seconds per user prompt; 0 = no limit
    "file_workspace":    "",    # optional folder confining read/write_local_file; blank = unrestricted
    "handoff_threshold_pct": 75,  # auto-handoff when context usage hits this % (0 = off)
    "handoff_rearm_pct": 10,      # how much further the gauge must grow before a chat may summarize again
    "handoff_notes_dir": "",      # folder for handoff notes; blank = <this folder>/deskpilot_data/handoff_notes
    "compaction_threshold": COMPACTION_THRESHOLD_DEFAULT,   # auto-compact context at this % usage (0 = off); C1
    "compaction_keep_recent": COMPACTION_KEEP_RECENT_DEFAULT,  # most-recent message entries kept verbatim; C1
    "chat_render_window": CHAT_RENDER_WINDOW_DEFAULT,  # message entries rendered per chat view (0 = all); v1.1.36
    "project_memory_chars": PROJECT_MEMORY_DEFAULT,  # chars of the workspace MEMORY.md index injected each request (0 = off); v1.1.66
    "project_memory_file": "",    # explicit index file; blank = <workspace>/MEMORY.md
    "ledger_inject_chars": LEDGER_INJECT_DEFAULT,  # chars of ledger tail in the system prompt (0 = off); Tier 3
    "custom_system_prompt": "",   # user instructions appended to the system prompt each request; blank = built-in only
    "custom_system_prompt_enabled": True,  # gate the prompt above WITHOUT discarding it; False = kept but not sent
    "sampling_defaults": dict(SAMPLING_DEFAULTS),  # sampler params applied to EVERY chat (Settings -> Sampling)
    "show_sampling_note": True,   # render one "what was actually sent" note per user prompt
    "allow_local_network": False,  # fetch_url may reach LAN/localhost/cloud-metadata addresses
    "mcp_servers":       [],   # MCP stdio servers (Settings -> MCP Servers)
    # Servers whose per-tool permission chips are shown in the bar. Absent = collapsed
    # to a single "server (n)" header, so a 25-tool server cannot flood the bar.
    # View state only - permissions are stored and enforced either way.
    "mcp_bar_expanded":  [],
    "skills_dir":        "",   # extra folder scanned for Agent Skills (SKILL.md); blank = defaults only
    # Consultant model (ask_expert). Any OpenAI-compatible endpoint works; these
    # are INDEPENDENT of server_url/api_key/model_name so the local model stays
    # primary. Blank expert_server_url or expert_api_key disables the tool.
    "expert_server_url": "",   # e.g. https://generativelanguage.googleapis.com/v1beta/openai/
    "expert_api_key":    "",   # the consultant's own key (never reuses api_key)
    "expert_model":      "",   # e.g. gemini-3.8-flash - use "List models", do not guess
    "expert_model_history": [],  # previously used expert names, newest first (dropdown)
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
# ── Reddit: read threads through their public Atom (.rss) endpoints ──────────
# Reddit's HTML pages are a JavaScript app, so fetch_url (no JS engine) gets an
# almost empty document once tags are stripped; the .json endpoints 403 plain
# HTTP clients; old.reddit.com bounces to a login wall; and Firecrawl refuses
# reddit.com outright ("we do not support this site", measured 2026-10-04). The
# Atom feeds at <path>.rss are still served publicly and carry the post plus its
# comments, so Reddit URLs are rewritten onto the feed and parsed here.
REDDIT_FEED_HOSTS = ("reddit.com", "redditmedia.com")
REDDIT_ATOM_ACCEPT = "application/atom+xml,application/xml;q=0.9,*/*;q=0.5"
REDDIT_MIN_INTERVAL = 2.5     # seconds between Reddit hits (they 429 aggressively)
REDDIT_MAX_ENTRIES = 60       # feed entries rendered per fetch
REDDIT_ENTRY_CHARS = 1200     # per-entry text cap
REDDIT_OUTPUT_CHARS = 14000   # total cap for a rendered thread (vs 6000 for pages)
_reddit_last_hit = 0.0
_reddit_lock = threading.Lock()
_ATOM = "{http://www.w3.org/2005/Atom}"


def _is_reddit_url(url: str) -> bool:
    """True for reddit.com / redditmedia.com under any subdomain."""
    try:
        host = (urllib.parse.urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    return any(host == h or host.endswith("." + h) for h in REDDIT_FEED_HOSTS)


def _reddit_atom_url(url: str) -> Optional[str]:
    """Rewrite a Reddit page URL onto its public Atom feed; None if unsupported.

    Subdomains are normalised to www: old./np./api. either 403 the request or
    bounce it to a login wall. Query strings are dropped (a ?context=N fragment
    link makes Reddit serve a partial page) except on search feeds, where the
    query IS the search.
    """
    try:
        parts = urllib.parse.urlparse(url)
    except ValueError:
        return None
    path = (parts.path or "").rstrip("/")
    if path in ("", "/"):
        return None                        # reddit.com front page: no Atom feed
    if path.endswith(".rss"):
        feed_path, keep_query = path, True
    elif re.search(r"\.(json|xml|html?|txt|compact|mobile)$", path, re.I):
        # Reddit's format suffix attaches to the last segment (/…/slug.json).
        # Strip it and re-attach the canonical "/.rss" form, which is what was
        # verified to return a full thread (…/slug.rss is not equivalent).
        path = re.sub(r"\.(json|xml|html?|txt|compact|mobile)$", "", path, flags=re.I)
        feed_path, keep_query = path + "/.rss", False
    elif path.endswith("/search"):
        feed_path, keep_query = path + ".rss", True
    else:
        feed_path, keep_query = path + "/.rss", False
    out = "https://www.reddit.com" + feed_path
    if keep_query and parts.query:
        out += "?" + parts.query
    return out


def _reddit_throttle(abort=None) -> bool:
    """Space Reddit requests (anonymous traffic is throttled hard). Returns
    False if the wait was interrupted by Stop, so the caller can bail out."""
    global _reddit_last_hit
    with _reddit_lock:
        now = time.time()
        wait = REDDIT_MIN_INTERVAL - (now - _reddit_last_hit)
        if wait < 0:
            wait = 0.0
        _reddit_last_hit = now + wait      # reserve the slot for this caller
    deadline = time.time() + wait
    while time.time() < deadline:
        if abort is not None and abort():
            return False
        time.sleep(min(0.25, max(0.0, deadline - time.time())))
    return True


def _reddit_strip_html(fragment: str) -> str:
    """Feed content fragment (already XML-unescaped by ElementTree, still HTML)
    into plain text. Tags are stripped BEFORE unescaping so that escaped markup
    in the original post text is never mistaken for feed markup."""
    txt = re.sub(r"(?is)<!--.*?-->", " ", fragment or "")
    txt = re.sub(r"(?is)<br\s*/?>", "\n", txt)
    txt = re.sub(r"(?is)</(p|div|li|tr|blockquote|table)>", "\n\n", txt)
    txt = re.sub(r"(?is)<li[^>]*>", "\n- ", txt)
    txt = re.sub(r"(?s)<[^>]+>", " ", txt)
    txt = html_mod.unescape(txt)
    txt = txt.replace("\u00a0", " ")      # &nbsp; is common in Reddit text
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r" *\n *", "\n", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


def _reddit_external_link(fragment: str) -> str:
    """The submitted URL of a link post: the '[link]' anchor that is NOT reddit.com."""
    for href in re.findall(r'<a href="([^"]+)"\s*>\s*\[link\]\s*</a>', fragment or ""):
        h = html_mod.unescape(href)
        if not _is_reddit_url(h):
            return h
    return ""


# The 'submitted by /u/x [link] [comments]' boilerplate Reddit appends to every
# post entry. Anchored on the reddit.com/user/ anchor so a post whose own text
# happens to contain the words 'submitted by' is not truncated.
_REDDIT_BOILERPLATE = re.compile(
    r'(?is)\s*(?:&#\d+;\s*)?submitted by\s*(?:&#\d+;\s*)?<a href="https?://[^"]*reddit\.com/user/.*$')


def _parse_reddit_atom(raw: str) -> str:
    """Render a Reddit Atom feed as readable text. '' if this is not one."""
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return ""
    entries = root.findall(_ATOM + "entry")
    if not entries:
        return ""
    lines: List[str] = []
    feed_title = (root.findtext(_ATOM + "title") or "").strip()
    if feed_title:
        lines.append(feed_title)
    shown, skipped = 0, 0
    for e in entries:
        if shown >= REDDIT_MAX_ENTRIES:
            skipped += 1
            continue
        eid = e.findtext(_ATOM + "id") or ""
        is_post = eid.startswith("t3_")
        au = e.find(_ATOM + "author")
        author = ((au.findtext(_ATOM + "name") or "") if au is not None else "").strip() or "(unknown)"
        when = (e.findtext(_ATOM + "published") or e.findtext(_ATOM + "updated") or "")[:10]
        c = e.find(_ATOM + "content")
        frag = (c.text or "") if c is not None else ""
        body = _reddit_strip_html(_REDDIT_BOILERPLATE.sub("", frag) if is_post else frag)
        head = ("POST" if is_post else "COMMENT") + f" · {author}" + (f" · {when}" if when else "")
        if is_post:
            title = (e.findtext(_ATOM + "title") or "").strip()
            if title:
                head = f"POST: {title}\n{head}"
            link = _reddit_external_link(frag)
            if link:
                body = (body + "\n" if body else "") + f"link: {link}"
        if len(body) > REDDIT_ENTRY_CHARS:
            body = body[:REDDIT_ENTRY_CHARS] + " [...]"
        lines.append("")
        lines.append(head)
        lines.append(body or "(no text)")
        shown += 1
    if skipped:
        lines.append("")
        lines.append(f"[... {skipped} more entries not shown: Reddit caps the public feed at "
                     f"roughly 100 comments and Deskpilot renders {REDDIT_MAX_ENTRIES}]")
    return "\n".join(lines).strip()


FETCH_BODY_LIMIT = 2 * 1024 * 1024   # max bytes downloaded from a single page (read in chunks)


# Redirect hardening for fetch_url: every 3xx hop is re-validated against the SSRF
# guard and the total hop count is capped (urllib's default opener follows redirects
# blindly, which lets a public URL bounce a fetch into the LAN / cloud metadata).
FETCH_MAX_REDIRECTS = 5       # total 3xx hops allowed per fetch_url request


def _is_model_lookup_error(err) -> bool:
    """True when a failure is about the MODEL NAME, not quota/auth/network.

    Deliberately excludes 429/401/403: a rate limit is not a wrong name, and
    retrying with a different spelling just burns another request against an
    exhausted quota. Verified live: a blank name returns 400 "model is not
    specified", an unknown one returns 404 model_not_found."""
    low = str(err or "").lower()
    if "429" in low or "rate limit" in low or "quota" in low:
        return False
    if "401" in low or "403" in low or "api key not valid" in low:
        return False
    return ("model_not_found" in low or "model is not specified" in low
            or "not a valid model" in low or "invalid model" in low
            or "unknown model" in low or "model id" in low)


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


def _is_mcp_image_msg(m: Any) -> bool:
    """True for the synthetic user message that carries an MCP tool's image output.

    Kept separate from _is_screenshot_msg(): a screenshot is a capture_screen result
    the model asked for, an MCP image is something a server returned unprompted. They
    share the 'only the most recent one is re-sent' policy in
    _prepare_request_messages(), but the migration naming at line ~2677 must keep
    classifying screenshots alone."""
    c = m.get("content") if isinstance(m, dict) else None
    if not isinstance(c, list):
        return False
    for p in c:
        if isinstance(p, dict) and p.get("type") == "text" \
                and str(p.get("text", "")).startswith(MCP_IMAGE_MARKER):
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


# Keys the app stores on a chat message for its OWN use. They must be stripped from
# every outgoing request: strict OpenAI-compatible servers (vLLM, llama.cpp, Unsloth
# Desktop) reject a message carrying an unrecognised field.
#   ts      - render timestamp, shown in the transcript header
#   steered - #10 mid-run steering: marks a message injected DURING a turn rather
#             than one that started it (used for rendering/replay only)
APP_LOCAL_MSG_KEYS = ("ts", "steered")


def _prepare_request_messages(messages: List[dict]) -> List[dict]:
    """Outgoing copy of a message list with screenshot payloads resolved.

    Only the most recent screenshot (the last one whose image is still readable)
    is expanded into a real base64 image_url payload; older ones become text
    placeholders. User-attachment images (image_ref parts in other messages, or
    legacy inline data-URLs) always expand - they are part of the message.

    MCP tool images (_is_mcp_image_msg) share the SAME budget as screenshots: both
    are "the live visual state at that moment", and re-sending every past one would
    bloat the context on every turn. One shared index rather than two, so a request
    never carries more than one live visual of this class no matter which tool made it."""
    def _live_visual(m: Any) -> bool:
        return _is_screenshot_msg(m) or _is_mcp_image_msg(m)

    last_shot = -1
    for i, m in enumerate(messages):
        if not _live_visual(m):
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
        is_shot = _live_visual(m)
        is_mcp = _is_mcp_image_msg(m)
        has_ref = isinstance(c, list) and any(
            isinstance(p, dict) and p.get("type") == "image_ref" for p in (c or []))
        if not is_shot and not has_ref:
            # Strip app-local keys before sending. Strict OpenAI-compatible servers
            # (vLLM, llama.cpp, Unsloth Desktop) reject a message carrying an
            # unrecognised field, so anything the app stores for its own use must not
            # reach the wire. `ts` has always been stripped here; `steered` (#10
            # mid-run steering) is new and must follow the same rule. tool_calls and
            # every other OpenAI field are deliberately NOT touched.
            if any(k in m for k in APP_LOCAL_MSG_KEYS):
                m = {k: v for k, v in m.items() if k not in APP_LOCAL_MSG_KEYS}
            out.append(m)
            continue
        # Placeholder wording must match what the message actually was.
        stale_text = ("[image returned by an MCP tool earlier]" if is_mcp
                      else "[screenshot analyzed earlier]")
        parts: List[dict] = []
        for p in (c or []):
            if not isinstance(p, dict):
                continue
            ptype = p.get("type")
            if ptype == "image_ref":
                if is_shot and i != last_shot:
                    name = Path(str(p.get("path", ""))).name or "image"
                    parts.append({"type": "text", "text": f"{stale_text}: {name}"})
                else:
                    parts.append(_expand_image_ref(p))   # latest screenshot / user attachment
            elif ptype == "image_url":
                url = str((p.get("image_url") or {}).get("url", ""))
                if is_shot and i != last_shot and url.startswith("data:image/"):
                    parts.append({"type": "text", "text": stale_text})
                else:
                    parts.append(p)          # legacy inline payload / user attachment - keep as-is
            else:
                parts.append(p)
        out.append({"role": m.get("role", "user"),
                    "content": parts or [{"type": "text", "text": stale_text}]})
    return out


# ════════════════════════════════════════════════════════════════════════════
#  PERSISTENCE HELPERS
# ════════════════════════════════════════════════════════════════════════════

#  ── Per-turn checkpoints (v1.1.45, SOW #4b) ───────────────────────────────
# The app edits REAL code and _atomic_write keeps only ONE rolling .bak, so
# "undo everything this turn did" was impossible. Before the FIRST write/edit
# that a turn makes to a given file, the file's ORIGINAL bytes are snapshotted
# under <data dir>/checkpoints/<chat_id>/<turn_id>/, and a small manifest entry
# is attached to the chat record. The manifest stores paths + hashes, NEVER the
# content: chats.json is already ~21 MB and save_chats() runs mid-turn, so
# inlining file bodies would grow the history file every single turn.

CHECKPOINT_DIRNAME = "checkpoints"
CHECKPOINT_MAX_FILE_BYTES = 25_000_000   # skip absurd targets rather than duplicate them
CHECKPOINT_MAX_FILES_PER_TURN = 200      # a turn touching this many files is already a red flag
CHECKPOINT_MAX_TURNS = 40                # manifest entries kept PER CHAT (files on disk are
                                         # pruned to match, so this also bounds disk use)


def _checkpoint_root() -> Path:
    r"""Checkpoint store, resolved AT CALL TIME (never cached at import).

    Follows GEN_DIR, which main() relocates to the live data dir (%LOCALAPPDATA%
    \Deskpilot) - the same place images already go. Module-level so a test can
    point it at a temp tree."""
    return GEN_DIR.parent / CHECKPOINT_DIRNAME


def _checkpoint_component(text) -> str:
    """Filesystem-safe component for a chat id / turn id / path segment.

    Same flattening rule as _safe_note_component (chats.json is user-editable, so ids
    and titles are untrusted input and must never become parent-directory hops: every
    character that could build a path is flattened, and a component made only of dots
    collapses to empty and is replaced). It deliberately does NOT call that helper:
    its 40-char cap is tuned for chat-title slugs, and path segments are routinely
    longer - truncating them at 40 would make two deep paths collide onto the same
    snapshot slot. Here the cap is 120 chars, and when truncation happens the suffix
    is a hash of the FULL component, so distinct long paths stay distinct."""
    s = re.sub(r"[^A-Za-z0-9_.-]", "_", str(text or ""))
    s = s.strip("._-")
    if not s:
        return "x"
    if len(s) <= 120:
        return s
    return s[:80] + "_" + hashlib.sha1(str(text).encode("utf-8", "replace")).hexdigest()[:12]


def _checkpoint_rel_parts(p: Path) -> List[str]:
    """Mirror of the target path inside the checkpoint tree, drive-safe.

    Every segment goes through _checkpoint_component (separators, drive colons and
    dots flattened), so a snapshot can never escape the checkpoint root. The drive
    anchor survives as a normal component ('C:\\' -> 'C__'), which keeps two files
    with the same tail on different drives apart. The ORIGINAL absolute path is kept
    in the manifest, so restore never needs to reverse this mapping."""
    try:
        parts = list(p.resolve().parts)
    except Exception:
        parts = list(Path(str(p)).parts)
    out = [_checkpoint_component(x) for x in parts]
    return [x for x in out if x and x != "x"] or ["file"]


def _checkpoint_slot(chat_id: str, turn_id: str, target: Path) -> Path:
    """Snapshot path for one target file within one turn."""
    return (_checkpoint_root()
            / _checkpoint_component(chat_id)
            / _checkpoint_component(turn_id)
            / Path(*_checkpoint_rel_parts(target)))


# Per-path serialisation for _atomic_write: two threads saving the SAME file would
# otherwise race on the shared rolling .bak and on os.replace() (which on Windows
# fails with a sharing violation if the destination is open). Keyed by lowercased
# path; the registry is small (a handful of data files) and never pruned.
_ATOMIC_WRITE_LOCKS: Dict[str, threading.Lock] = {}
_ATOMIC_WRITE_LOCKS_GUARD = threading.Lock()


def _atomic_write(path: Path, text: str) -> None:
    """Write a file atomically (temp file + rename) to avoid corruption.

    If the target exists, the immediately-previous version is kept as
    <name>.bak first - exactly one rolling backup, overwritten on every save.

    Two separate hazards, both handled:
      * a FIXED temp name is shared by every writer of the same file, so two
        concurrent saves write into ONE temp file and can each raise a sharing
        violation on the swap. Hence a unique name (pid + random suffix).
      * even with unique temps, the rolling .bak and the final os.replace() still
        target the SAME paths, and on Windows os.replace fails if the destination
        is open by another thread. So writes to one path are serialised in-process
        by a per-path lock. (Cross-process is out of scope: the single-instance lock
        already prevents two normal instances sharing a data dir, and --multi gives
        each instance its own.)
    """
    key = str(path).lower()
    with _ATOMIC_WRITE_LOCKS_GUARD:
        lock = _ATOMIC_WRITE_LOCKS.get(key)
        if lock is None:
            lock = _ATOMIC_WRITE_LOCKS[key] = threading.Lock()
    with lock:
        tmp = path.with_name(f"{path.name}.{os.getpid()}-{uuid.uuid4().hex[:8]}.tmp")
        try:
            tmp.write_bytes(text.encode("utf-8"))   # binary: no newline translation; keeps CRLF exact
            if path.exists():
                try:
                    shutil.copy2(path, path.with_name(path.name + ".bak"))
                except OSError:
                    pass   # backup is best-effort; the save itself must not fail because of it
            os.replace(tmp, path)
        except Exception:
            # A unique name is never reused, so an abandoned temp would linger
            # forever - remove it. The original file is untouched either way, because
            # the swap is atomic and had not happened yet.
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            raise


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


def _entry_context_length(entry) -> int:
    """Context-window size from a /models entry, across server dialects.

    Unsloth Desktop sends a top-level context_length; Strata nests it as
    meta.n_ctx. Returns 0 when the server reports no window (LM Studio, Ollama)."""
    if not isinstance(entry, dict):
        return 0
    for key in ("context_length", "max_context_length", "native_context_length"):
        v = entry.get(key)
        if v:
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
    meta = entry.get("meta")
    if isinstance(meta, dict):
        v = meta.get("n_ctx") or meta.get("context_length")
        if v:
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
    return 0


def _entry_is_loaded(entry) -> bool:
    """Whether a /models entry is a model currently resident on the server.

    Unsloth Desktop sends a top-level loaded=True boolean; Strata instead nests
    the state as status.value == "loaded". Servers that report neither return
    False, so callers treat the loaded model as unknown rather than guessing."""
    if not isinstance(entry, dict):
        return False
    if entry.get("loaded"):
        return True
    status = entry.get("status")
    if isinstance(status, dict):
        return str(status.get("value", "")).strip().lower() == "loaded"
    if isinstance(status, str):
        return status.strip().lower() == "loaded"
    return False


def loaded_model_ids(entries: List[dict]) -> List[str]:
    """IDs of the entries a server reports as currently loaded (loaded=True).

    Servers that do not send a "loaded" field return [] - callers must then treat
    the loaded model as unknown rather than guessing."""
    out: List[str] = []
    for m in entries or []:
        if isinstance(m, dict) and _entry_is_loaded(m) and m.get("id"):
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


def _search_db_path() -> Path:
    """Where the FTS5 index lives: next to CHATS_FILE (so a --multi scratch profile
    gets its own index). The DB is a CACHE - deleting it is always safe."""
    return Path(CHATS_FILE).with_name(SEARCH_DB_NAME)


def _search_indexable(chats: Dict[str, Any]) -> List[tuple]:
    """Flatten a chats dict into indexable rows:
    (chat_id, title, msg_index, role, text, kind).

    Text comes from the SAME flattening the handoff path uses, so a multimodal
    message indexes its text parts and never its base64 image payload. Assistant
    tool_calls contribute their tool NAME (searching 'firecrawl' should find the
    turns that used it); tool results contribute their output. Corrupt entries are
    skipped rather than raising - chats.json is user-editable.

    v1.1.54: `kind` is "live" for chat["messages"] and "archive:<n>" for the messages
    inside chat["compaction_archive"][<n>]. `idx` is the index WITHIN that list, so an
    archive row is addressed by (chat_id, kind, idx) - never by a live message index,
    which would jump the sidebar to an unrelated message.
    Archived messages are indexed because they are the only record of everything a
    compaction folded away; without them the agent loses its own history permanently."""
    rows: List[tuple] = []
    # v1.1.59: snapshot the outer mapping, exactly as _search_chats_scan does.
    # build_search_index runs on the chat WORKER thread (search_chats tool) as well
    # as the main thread (sidebar), while delete_chat()/new-chat mutate the dict on
    # the main thread -> 'dictionary changed size during iteration' kills the worker.
    # Shallow copy of the pairs only: the store is ~24 MB and the inner message
    # lists are never resized by this pass.
    for cid, chat in list(((chats or {}).get("chats") or {}).items()):
        if not isinstance(chat, dict):
            continue
        title = str(chat.get("title") or "Untitled")
        for i, m in enumerate(chat.get("messages") or []):
            if not isinstance(m, dict):
                continue
            parts: List[str] = []
            body = _flatten_handoff_content(m)
            if body:
                parts.append(body)
            tcs = m.get("tool_calls")
            if isinstance(tcs, list):
                for tc in tcs:
                    if isinstance(tc, dict) and isinstance(tc.get("function"), dict):
                        nm = str(tc["function"].get("name") or "").strip()
                        if nm:
                            parts.append(nm)
            text = "\n".join(p for p in parts if p).strip()
            if not text:
                continue
            rows.append((str(cid), title, int(i), str(m.get("role") or ""),
                         text[:SEARCH_BODY_CAP], "live"))
        # ── archived (compacted-away) messages, v1.1.54 ──
        arch = chat.get("compaction_archive")
        if not isinstance(arch, list):
            continue
        for ev in arch:
            if not isinstance(ev, dict):
                continue
            try:
                num = int(ev.get("compaction_number") or 0)
            except (TypeError, ValueError):
                num = 0
            kind = f"archive:{num}"
            for i, m in enumerate(ev.get("messages") or []):
                if not isinstance(m, dict):
                    continue
                # Same treatment as live rows: a tool call's NAME is indexed alongside
                # its output, so searching 'firecrawl' finds archived turns that used it.
                parts = []
                body = _flatten_handoff_content(m)
                if body:
                    parts.append(body)
                tcs = m.get("tool_calls")
                if isinstance(tcs, list):
                    for tc in tcs:
                        if isinstance(tc, dict) and isinstance(tc.get("function"), dict):
                            nm = str(tc["function"].get("name") or "").strip()
                            if nm:
                                parts.append(nm)
                text = "\n".join(p for p in parts if p).strip()
                if not text:
                    continue
                rows.append((str(cid), title, int(i), str(m.get("role") or ""),
                             text[:SEARCH_BODY_CAP], kind))
    return rows


def build_search_index(chats: Dict[str, Any]) -> tuple:
    """(ok, rows) - rebuild the FTS5 index from scratch into a temp table, then swap.

    Build-then-swap inside one transaction means a reader never sees a half-written
    index and a crash mid-build leaves the previous one intact. Rebuilding wholesale
    is deliberate: it is ~0.4 s on the real 24 MB history (12,808 rows), which is far
    simpler and more obviously correct than incremental invalidation. Returns (False, 0)
    on any failure - search is an enhancement and must never break saving or chatting.

    v1.1.54: a `kind` column distinguishes live messages from archived ones. It is
    appended AFTER body so snippet(msgs, 4, …) still addresses the body column.
    The schema version is recorded in meta so an index written by an older build is
    treated as stale rather than read with the wrong column count.
    Cost note: indexing the archive roughly doubles the row count (6,616 live + 6,192
    archived) and the build went ~0.13 s -> ~0.4 s. Still well inside the 3 s rebuild
    rate limit, and it runs debounced on the main thread / on the worker for the tool."""
    rows = _search_indexable(chats)
    con = None
    try:
        p = _search_db_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(p), timeout=5.0)
        con.execute("PRAGMA journal_mode=WAL")     # readers never block on a rebuild
        con.execute("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT)")
        con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS msgs USING fts5("
                    "cid UNINDEXED, title, idx UNINDEXED, role UNINDEXED, body, "
                    "kind UNINDEXED, tokenize='unicode61')")
        with con:                                  # one transaction
            con.execute("DROP TABLE IF EXISTS msgs_new")
            con.execute("CREATE VIRTUAL TABLE msgs_new USING fts5("
                        "cid UNINDEXED, title, idx UNINDEXED, role UNINDEXED, body, "
                        "kind UNINDEXED, tokenize='unicode61')")
            con.executemany("INSERT INTO msgs_new VALUES(?,?,?,?,?,?)", rows)
            con.execute("DROP TABLE IF EXISTS msgs")
            con.execute("ALTER TABLE msgs_new RENAME TO msgs")
            stamp = str(int((Path(CHATS_FILE).stat().st_mtime if Path(CHATS_FILE).exists()
                             else time.time())))
            con.execute("INSERT OR REPLACE INTO meta VALUES('built_from_mtime',?)", (stamp,))
            con.execute("INSERT OR REPLACE INTO meta VALUES('rows',?)", (str(len(rows)),))
            con.execute("INSERT OR REPLACE INTO meta VALUES('chats',?)",
                        (str(len(chats.get("chats") or {})),))
            con.execute("INSERT OR REPLACE INTO meta VALUES('schema',?)",
                        (SEARCH_SCHEMA_VERSION,))
        return True, len(rows)
    except Exception:
        return False, 0
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:
                pass


def search_index_ready() -> bool:
    """True when the index exists and looks usable."""
    try:
        if not _search_db_path().exists():
            return False
        con = sqlite3.connect(str(_search_db_path()), timeout=3.0)
        try:
            n = con.execute("SELECT count(*) FROM msgs").fetchone()[0]
            return bool(n)
        finally:
            con.close()
    except Exception:
        return False


def search_index_is_stale(chats: Dict[str, Any]) -> bool:
    """True when the index no longer matches the live chats.

    Three signals, because one is not enough: the chats.json mtime catches edits that
    were saved to disk, while the CHAT COUNT catches a chat created (or deleted) in
    memory that has not been written yet - its mtime is still the old one. The SCHEMA
    signal (v1.1.54) catches an index written by an older build of the app: the row
    shape changed when archived messages became indexable, and reading a 5-column
    table with 6-column code must force a rebuild, not degrade silently."""
    try:
        want = str(len((chats or {}).get("chats") or {}))
        con = sqlite3.connect(str(_search_db_path()), timeout=3.0)
        try:
            got = dict(con.execute("SELECT k, v FROM meta").fetchall())
        finally:
            con.close()
        if got.get("schema") != SEARCH_SCHEMA_VERSION:
            return True
        if got.get("chats") != want:
            return True
        stamp = str(int(Path(CHATS_FILE).stat().st_mtime)) if Path(CHATS_FILE).exists() else None
        if stamp is None:
            return True
        return got.get("built_from_mtime") != stamp
    except Exception:
        return True


def _fts5_query(raw: str) -> str:
    """Turn free text into a SAFE FTS5 MATCH expression.

    FTS5's own query syntax (quotes, *, NEAR, AND/OR/NOT, colons, hyphens as
    column filters) would otherwise let a stray character raise a syntax error - or
    let a query mean something the user did not intend. Every word is therefore
    quoted as a phrase and prefix-matched; words are ANDed. Returns '' when nothing
    is searchable."""
    words = re.findall(r"[\w\u00c0-\uffff]+", str(raw or ""), re.UNICODE)
    if not words:
        return ""
    return " ".join('"' + w.replace('"', '""') + '"*' for w in words[:12])


def _search_limit(value) -> int:
    """Clamp a caller/LLM-supplied hit limit. Junk ("", None, "ten", 0, 9999) never
    raises - LLMs routinely send numbers as strings or omit them entirely."""
    try:
        n = int(float(value))
    except (TypeError, ValueError):
        n = SEARCH_MAX_SNIPPETS
    return max(1, min(n, 50))


def search_chats(chats: Dict[str, Any], query: str,
                 limit: int = SEARCH_MAX_SNIPPETS) -> List[dict]:
    """Full-text search over every indexed message. [{chat_id,title,idx,role,kind,snippet}]

    Ranked by FTS5 relevance; snippet() marks the hit. Falls back to a LIKE scan of
    the in-memory chats when the index is missing/unusable, so search still works
    (slower, no ranking) instead of returning nothing.

    `kind` is "live" or "archive:<n>" (v1.1.54). An archive hit is NOT addressable by
    a live message index, so anything that jumps to one must route it to read_archive
    instead of load_chat(jump_to=idx)."""
    q = _fts5_query(query)
    if not q:
        return []
    limit = _search_limit(limit)
    try:
        con = sqlite3.connect(str(_search_db_path()), timeout=3.0)
        try:
            cur = con.execute(
                "SELECT cid, title, idx, role, kind, snippet(msgs, 4, '[', ']', '…', ?) "
                "FROM msgs WHERE msgs MATCH ? ORDER BY rank LIMIT ?",
                (SEARCH_SNIPPET_TOKENS, q, limit))
            out = []
            for cid, title, idx, role, kind, snip in cur.fetchall():
                out.append({"chat_id": str(cid), "title": str(title or "Untitled"),
                            "idx": int(idx), "role": str(role or ""),
                            "kind": str(kind or "live"),
                            "snippet": re.sub(r"\s+", " ", str(snip or "")).strip()})
            return out
        finally:
            con.close()
    except Exception:
        return _search_chats_scan(chats, query, limit)


def _message_body(chats: Dict[str, Any], chat_id: str, idx: int,
                  kind: str = "live") -> str:
    """The full plain text of one message, for search_chats(full=true).

    Read from the in-memory chats dict, so it works even when the chats file is
    outside the File Workspace jail and the model cannot open it directly.

    v1.1.54: `kind` selects the list - "live" is chat["messages"], "archive:<n>" is
    chat["compaction_archive"][<n>]["messages"]. The index is the position WITHIN that
    list; a live index and an archive index are different address spaces and must never
    be confused (jumping to a live message with an archive index would land somewhere
    unrelated and look like a correct answer)."""
    try:
        chat = (chats or {}).get("chats", {}).get(str(chat_id))
        if not isinstance(chat, dict):
            return ""
        if str(kind or "live").startswith("archive:"):
            try:
                num = int(str(kind).split(":", 1)[1])
            except (TypeError, ValueError):
                return ""
            ev = next((a for a in (chat.get("compaction_archive") or [])
                       if isinstance(a, dict) and int(a.get("compaction_number") or 0) == num),
                      None)
            if not isinstance(ev, dict):
                return ""
            msgs = ev.get("messages") or []
        else:
            msgs = chat.get("messages") or []
        if not (0 <= int(idx) < len(msgs)):
            return ""
        return str(_flatten_handoff_content(msgs[int(idx)]) or "")
    except Exception:
        return ""


def _search_group_by_chat(hits: List[dict]) -> List[dict]:
    """Collapse message-level hits into one row per chat: best snippet + hit count.

    A raw message list is a poor shape for "which chats mention X" - one chatty
    session can fill the whole limit with its own tool output. Hits arrive ranked, so
    the FIRST one per chat is its best; later ones only bump the count."""
    order: List[str] = []
    best: Dict[str, dict] = {}
    for h in hits:
        cid = h["chat_id"]
        if cid not in best:
            best[cid] = dict(h)
            best[cid]["hits"] = 1
            order.append(cid)
        else:
            best[cid]["hits"] += 1
    return [best[c] for c in order]


def _search_chats_scan(chats: Dict[str, Any], query: str, limit: int) -> List[dict]:
    """No-index fallback: plain case-insensitive substring scan of the live chats.

    v1.1.54 also scans compaction_archive, so the fallback reaches the same history
    the FTS index does. It cannot rank, and it is slower, but returning only live
    messages here would mean the archive is searchable in one code path and not the
    other - a difference the model has no way to know about."""
    needle = str(query or "").strip().lower()
    if not needle:
        return []
    out: List[dict] = []
    # v1.1.56: snapshot the outer mapping. This runs on the AGENT worker thread
    # while the user may delete or rename a chat on the MAIN thread; iterating the
    # live dict then raises "dictionary changed size during iteration" and kills
    # the turn mid-tool. A shallow copy of the pairs is sufficient: the inner
    # message lists are never resized by this scan, and a CPython list does not
    # raise on a concurrent append (same tolerance as _deterministic_handoff,
    # which snapshots for the identical reason). Deep-copying the whole ~24 MB
    # store per query would be the wrong cure.
    for cid, chat in list((chats.get("chats") or {}).items()):
        if not isinstance(chat, dict):
            continue
        title = str(chat.get("title") or "Untitled")
        groups = [("live", chat.get("messages") or [])]
        arch = chat.get("compaction_archive")
        if isinstance(arch, list):
            for ev in arch:
                if isinstance(ev, dict):
                    try:
                        num = int(ev.get("compaction_number") or 0)
                    except (TypeError, ValueError):
                        num = 0
                    groups.append((f"archive:{num}", ev.get("messages") or []))
        for kind, msgs in groups:
            for i, m in enumerate(msgs or []):
                if not isinstance(m, dict):
                    continue
                body = _flatten_handoff_content(m) or ""
                low = body.lower()
                if needle in low or needle in title.lower():
                    at = low.find(needle)
                    at = at if at >= 0 else 0
                    lo = max(0, at - 60)
                    snip = (("…" if lo else "")
                            + re.sub(r"\s+", " ", body[lo:at + len(needle) + 60]).strip())
                    out.append({"chat_id": str(cid), "title": title, "idx": int(i),
                                "role": str(m.get("role") or ""), "kind": kind,
                                "snippet": snip})
                    if len(out) >= limit:
                        return out
    return out


def _ledger_dir() -> Path:
    """Folder holding the ledgers: <data dir>/ledger. Module-level so a test can
    point it at a temp tree (same convention as _checkpoint_dir / GEN_DIR)."""
    return GEN_DIR.parent / LEDGER_DIR_NAME


def _ledger_path(chat_id: str) -> Path:
    """One append-only file per chat. '' -> the shared global ledger.

    Deliberately OUTSIDE chats.json: deleting a conversation must not take its memory
    with it, and the live record is rewritten by compaction while this is never touched.
    The component is flattened by _checkpoint_component, so a hand-edited chat id in
    chats.json can never become a parent-directory hop."""
    d = _ledger_dir()
    cid = str(chat_id or "").strip()
    return d / (LEDGER_GLOBAL_NAME if not cid else _checkpoint_component(cid) + ".jsonl")


def _ledger_read_raw(chat_id: str) -> List[dict]:
    """Every entry in a ledger, oldest first. Never raises.

    Reads the TAIL of the file when it exceeds LEDGER_SCAN_MAX_BYTES (an append-only log
    grows forever by design), dropping the first line in that case because it may be
    partial. A malformed line is skipped, not fatal - the file is plain text a user can
    open and edit."""
    p = _ledger_path(chat_id)
    try:
        size = p.stat().st_size
        if size <= 0:
            return []
        with open(p, "rb") as fh:
            if size > LEDGER_SCAN_MAX_BYTES:
                fh.seek(size - LEDGER_SCAN_MAX_BYTES)
                fh.readline()          # discard the possibly-partial first line
            data = fh.read()
    except Exception:
        return []
    out: List[dict] = []
    for line in data.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        if isinstance(e, dict) and str(e.get("text") or "").strip():
            out.append(e)
    return out


def _ledger_append(chat_id: str, text: str, tags: Optional[List[str]] = None,
                   source: str = "model") -> Optional[dict]:
    """Append one entry. Returns the stored entry, or None when nothing was written.

    Append-only: this function NEVER rewrites or truncates the file, so an entry cannot
    be silently lost the way a compaction summary loses the text it replaced. Entries
    are written as single-line JSON, which keeps the file greppable and hand-editable.
    Runs on the worker thread (no Tk); the lock serialises concurrent appends from the
    worker and the main thread so lines cannot interleave."""
    body = re.sub(r"\s+", " ", str(text or "")).strip()
    if not body:
        return None
    if len(body) > LEDGER_ENTRY_HARD_MAX:
        return None                      # caller reports this; never silently truncate here
    if len(body) > LEDGER_ENTRY_MAX_CHARS:
        body = body[:LEDGER_ENTRY_MAX_CHARS] + "…"
    clean: List[str] = []
    for t in (tags or []):
        t = re.sub(r"\s+", "-", str(t or "")).strip("-").lower()[:40]
        if t and t not in clean:
            clean.append(t)
        if len(clean) >= LEDGER_TAGS_MAX:
            break
    entry = {"ts": datetime.now().isoformat(timespec="seconds"),
             "chat_id": str(chat_id or ""),
             "source": str(source or "model")[:40],
             "tags": clean,
             "text": body}
    line = json.dumps(entry, ensure_ascii=False)
    if "\n" in line:                     # json.dumps cannot emit one, but a line-based
        return None                      # format is only safe if that stays true
    try:
        p = _ledger_path(chat_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        with _LEDGER_LOCK:
            with open(p, "a", encoding="utf-8", newline="\n") as fh:
                fh.write(line + "\n")
    except Exception:
        return None
    return entry


def _ledger_inject_text(chat_id: str, budget_chars: int) -> str:
    """The ledger block for the system prompt: newest entries, hard-bounded.

    The STORE is unbounded; only this VIEW is capped - an uncapped "remember everything"
    block would just hit the same context wall compaction exists to dodge. Newest first
    in the file order, because the most recent notes are the ones a continuation needs.
    Returns '' when the feature is off or there is nothing to show."""
    budget = max(0, min(int(budget_chars or 0), LEDGER_INJECT_MAX))
    if budget <= 0:
        return ""
    merged = _ledger_read_raw("") + _ledger_read_raw(chat_id)
    if not merged:
        return ""
    merged = merged[-LEDGER_INJECT_ENTRIES:]
    lines: List[str] = []
    used = 0
    dropped = 0
    for e in reversed(merged):
        ts = str(e.get("ts") or "")[:16].replace("T", " ")
        tags = (" #" + " #".join(e.get("tags") or [])) if e.get("tags") else ""
        src = "" if e.get("source") in (None, "model") else f" [{e.get('source')}]"
        one = f"- {ts}{src}{tags}: {e.get('text')}"
        if lines and used + len(one) > budget:
            dropped += 1
            continue
        lines.append(one)
        used += len(one)
    lines.reverse()
    if dropped:
        lines.insert(0, f"[{dropped} older ledger entries omitted for length - "
                        f"read_ledger() returns all of them]")
    return ("LEDGER (append-only notes that survive compaction and chat deletion; "
            "add to it with remember(), page through it with read_ledger()):\n"
            + "\n".join(lines))


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
    Offsets then advance by exactly the characters inserted; newlines inside a
    segment bump ln0 and reset off0, so URLs on later lines are keyed to the
    right registry entry (the old flat offset made every URL after the first
    line of a multi-line message unclickable)."""
    try:
        ln0, off0 = (int(x) for x in widget.index("end-1c").split(".")[:2])
    except (tk.TclError, ValueError):
        ln0, off0 = 1, 0

    def _advance(chunk: str) -> None:
        nonlocal ln0, off0
        nl = chunk.count("\n")
        if nl:
            ln0 += nl
            off0 = len(chunk) - chunk.rfind("\n") - 1
        else:
            off0 += len(chunk)

    pos = 0
    for m in URL_RE.finditer(seg):
        if m.start() > pos:
            widget.insert("end", seg[pos:m.start()], tags)
            _advance(seg[pos:m.start()])
        raw = m.group(0)
        url = raw.rstrip(r".,;:!?)\]]")
        if "://" in url and len(url.split("://", 1)[1]) > 0:
            if app is not None:
                try:
                    app._links[(ln0, off0)] = url
                except Exception:
                    pass
            widget.insert("end", url, list(tags) + ["link"])
            _advance(url)
            # Punctuation stripped from the link target is still part of the text.
            if len(raw) > len(url):
                widget.insert("end", raw[len(url):], tags)
                _advance(raw[len(url):])
        else:
            widget.insert("end", raw, tags)            # degenerate (scheme only) - plain text
            _advance(raw)
        pos = m.end()
    if pos < len(seg):
        widget.insert("end", seg[pos:], tags)
        _advance(seg[pos:])


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


def parse_tool_arguments(args_raw: str) -> tuple:
    """(args_dict, error_kind) for one tool call's raw JSON arguments.

    error_kind is None on success, else 'malformed' (unparseable) or the type
    name of a value that parsed but is not an object.

    json.loads happily returns a list, string, number or null for arguments the
    model wrote as '["What is this code?"]'. Everything downstream assumes a
    mapping - args_preview's .items(), _permission_preview's .get(),
    handler(**args) - so a non-dict raises AttributeError on the worker thread
    and kills the turn. Reject it at the single point where the shape is known
    instead of patching each consumer.

    The '_raw_arguments' unmasking lives here too: the stream finalizer masks
    malformed model JSON as {"_raw_arguments": ...} so history stays sendable to
    strict servers, and leaving that in place would reach the handler and die
    with a confusing TypeError rather than the clear error below.
    """
    try:
        args = json.loads(args_raw)
    except Exception:
        return {}, "malformed"
    if isinstance(args, dict) and "_raw_arguments" in args:
        orig_raw = str(args["_raw_arguments"])
        if orig_raw.strip() not in ("", "{}"):
            return {}, "malformed"
        args = {}
    if not isinstance(args, dict):
        return {}, type(args).__name__
    return args, None


def args_preview(args: Dict[str, Any]) -> str:
    """Compact one-line preview of tool arguments for accordion headers.

    Tolerates a non-dict on purpose. A tool call whose arguments parsed to a
    list/string/number is stored in chat history exactly as the model sent it,
    so load_chat replays it through here long after the dispatch guard was
    added - and .items() on a list raises AttributeError inside a root.after
    callback, where nothing catches it. Display code must not be able to kill
    a render."""
    if not isinstance(args, dict):
        s = str(args).replace("\n", " ")
        return (s[:60] + "…") if len(s) > 60 else (s or "no args")
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
        loaded = [m for m in info if _entry_is_loaded(m)]
        if len(loaded) == 1:
            entry = loaded[0]
    ctx = _entry_context_length(entry)
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



def project_memory_file(settings: dict) -> Path:
    """Which file is the workspace index. Explicit setting wins; else <workspace>/MEMORY.md."""
    raw = str((settings or {}).get("project_memory_file") or "").strip()
    if raw:
        try:
            return Path(raw).expanduser()
        except Exception:
            pass
    return default_workspace_root() / PROJECT_MEMORY_NAME


def _project_memory_text(settings: dict) -> str:
    """The workspace memory INDEX, for the system prompt. '' when off or absent.

    Bounded on purpose: this rides on EVERY request, so an unbounded index is an
    unbounded per-turn cost. Unlike the ledger there is no store to page through -
    the file is the store - so a truncated read says so explicitly and names the
    tool that can get the whole thing, rather than letting a half-index pass as
    the full project list.
    """
    try:
        budget = int(float((settings or {}).get("project_memory_chars",
                                               PROJECT_MEMORY_DEFAULT)))
    except (TypeError, ValueError):
        budget = PROJECT_MEMORY_DEFAULT
    budget = max(0, min(budget, PROJECT_MEMORY_MAX))
    if budget <= 0:
        return ""
    p = project_memory_file(settings)
    try:
        if not p.is_file():
            return ""
        raw = p.read_bytes()[:PROJECT_MEMORY_READ_MAX].decode("utf-8", "replace")
    except Exception:
        return ""
    raw = raw.strip()
    if not raw:
        return ""
    note = ""
    total = len(raw)
    if total > budget:
        raw = raw[:budget].rstrip()
        note = (f"\n\n[... index truncated at {budget} of {total} chars - "
                f"read_local_file the whole file at {p} ...]")
    return (f"PROJECT MEMORY INDEX (workspace convention: every project has its own "
            f"folder under {default_workspace_root() / PROJECT_MEMORY_SUBDIR} with its "
            f"own MEMORY.md; read that file before working in a project, and update it "
            f"when anything durable changes):\n{raw}{note}")


def effective_custom_prompt(settings: dict) -> str:
    """The custom system prompt as it should reach the model THIS request.

    The Settings checkbox gates INJECTION only - the text itself is never cleared,
    so switching a persona off for one session and back on later costs no retyping.
    Gating here rather than inside build_system_message() keeps that function pure
    (the suites call it directly with a literal string) and gives both call sites
    ONE place to read the flag, so they cannot drift apart.

    Defaults to enabled when the key is absent, so a settings file written before
    this feature behaves exactly as it did before.
    """
    s = settings or {}
    if not s.get("custom_system_prompt_enabled", True):
        return ""
    return str(s.get("custom_system_prompt") or "")


def build_system_message(file_workspace: str = "", exa_key: str = "", firecrawl_key: str = "",
                         custom_prompt: str = "", skills_index: str = "",
                         ledger: str = "", project_memory: str = "") -> dict:
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
    if off is not None:
        total_min = int(off.total_seconds()) // 60   # whole minutes (fractional seconds irrelevant)
        sign = "+" if total_min >= 0 else "-"
        h, m = divmod(abs(total_min), 60)
        off_s = f"UTC{sign}{h}" + (f":{m:02d}" if m else "")   # e.g. UTC+5:30, UTC-4
    else:
        off_s = "unknown offset"
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
                    "walls) - use it whenever a page needs a browser engine or resists plain fetching. "
                    "fetch_url remains the faster first choice for simple static pages.")
    si = (skills_index or "").strip()
    if si:
        # Placed BEFORE the user's standing instructions so skills read as available
        # capability, not as rules competing with the persona.
        content += "\n\n" + si
    pm = (project_memory or "").strip()
    if pm:
        # The index is a standing instruction about where to look, so it sits with
        # the skills block and ahead of the ledger (facts) and the user's own text.
        content += "\n\n" + pm
    led = (ledger or "").strip()
    if led:
        # The ledger is FACTS, not instructions, and it sits after the skills block but
        # before the user's standing prompt: it must not be able to displace the persona
        # or the workspace jail, and it must stay above anything the user wrote.
        content += "\n\n" + led
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


def default_workspace_root() -> Path:
    """The workspace folder itself (NOT the notes subfolder, NOT the data store).

    Same resolution order the handoff-notes path uses: USER_WORKSPACE_ROOT when it
    exists on this machine, else the script's own folder - so a copy of this
    script elsewhere still finds a sensible root instead of a hardcoded path that
    does not exist."""
    try:
        if USER_WORKSPACE_ROOT.is_dir():
            return USER_WORKSPACE_ROOT
    except Exception:
        pass
    return BASE_DIR


def default_handoff_notes_dir() -> Path:
    """The notes folder used when Settings -> Handoff notes folder is blank (never AppData)."""
    return default_workspace_root() / DEFAULT_HANDOFF_NOTES_SUBDIR


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
        "## Project & memory files\nWhich project folder this work belongs to (e.g. "
        "projects/<name>/), and the exact paths of any project MEMORY.md or history/ file "
        "that was READ or UPDATED during the session. Those files live on disk, not in this "
        "conversation, so naming them is what lets the next session pick up the thread - "
        "say \"none yet\" if no memory file exists for this work yet.\n\n"
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


# ── Agent Skills: the open SKILL.md standard, progressive disclosure ─────────
# A skill is a FOLDER holding a SKILL.md file: YAML frontmatter with at least
# `name` and `description`, then a markdown body of instructions, optionally with
# scripts/ and references/ beside it.  Only name + description go into the system
# prompt (a few dozen tokens each); the body is fetched on demand by the
# load_skill tool.  That split is the entire point of the format - capabilities
# without paying context for the ones not in use.  Standard: agentskills.io
# (Anthropic's Agent Skills, published cross-platform in Dec 2025).
SKILLS_SUBDIR = "skills"                 # default folder name under the app workspace
SKILL_FILE_NAME = "SKILL.md"             # the only file that makes a folder a skill
SKILL_NAME_MAX = 64                      # frontmatter name length cap
SKILL_DESC_MAX = 1024                    # frontmatter description length cap
SKILL_INDEX_MAX = 60                     # skills named in the system prompt
SKILL_INDEX_CHARS = 8000                 # hard cap on the whole injected index
SKILL_BODY_MAX = 40000                   # chars of SKILL.md body returned by load_skill
SKILL_SCAN_DEPTH = 3                     # how deep below a root to look for SKILL.md
SKILL_SCAN_MAX = 400                     # directories visited per scan (bounded walk)
SKILL_SKIP_DIRS = frozenset({            # never descended into while scanning
    "node_modules", ".git", "__pycache__", ".venv", "venv", "site-packages"})


def _skill_frontmatter(text: str) -> tuple:
    """Split a SKILL.md into (metadata_dict, body_str).

    The standard requires the file to open with a '---'-delimited YAML frontmatter
    block carrying at least name and description.  PyYAML is not a dependency this
    app should gain, so only the flat 'key: value' subset the standard actually
    needs is parsed, with the conveniences real skills rely on:
      * a value may continue on following INDENTED lines (folded to one line) -
        descriptions are routinely wrapped in the wild;
      * '#' starts a comment only at the start of a line, so 'name: a#b' survives;
      * quoted values keep their content, and '|' / '>' block scalars are accepted.
    Nested blocks (metadata:, compatibility:) fold into their parent key, which is
    harmless because only name/description are read.  A file with no frontmatter
    returns ({}, whole_text) so a plain markdown note is still usable."""
    lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if not lines or lines[0].strip() != "---":
        return {}, text or ""
    end = -1
    for i in range(1, len(lines)):
        if lines[i].strip() in ("---", "..."):
            end = i
            break
    if end < 0:
        return {}, text or ""          # unterminated frontmatter: treat as plain body
    meta: Dict[str, str] = {}
    key: Optional[str] = None
    for raw in lines[1:end]:
        if not raw.strip():
            key = None
            continue
        if raw[0] in " \t":
            if key:                    # folded continuation of the previous key
                meta[key] = (meta[key] + " " + raw.strip()).strip()
            continue
        if raw.lstrip().startswith("#") or ":" not in raw:
            key = None
            continue
        k, v = raw.split(":", 1)
        k, v = k.strip(), v.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", k):
            key = None
            continue
        if v in ("|", ">", "|-", ">-", "|+", ">+"):
            v = ""                     # block scalar: content arrives as indented lines
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        meta[k] = v
        key = k
    return meta, "\n".join(lines[end + 1:]).strip()


def _skill_name_from_dir(folder: Path) -> str:
    """Fallback skill name from its folder name (a skill with no frontmatter name)."""
    s = re.sub(r"[^A-Za-z0-9_-]+", "-", str(folder.name).strip()).strip("-").lower()
    return s[:SKILL_NAME_MAX] or "skill"


def _skill_dirs(settings) -> List[Path]:
    r"""Skill folders to scan, in priority order, read from Settings AT CALL TIME.

    Settings -> 'Skills folder' (one absolute path) wins, then the defaults:
    <app workspace>/skills and <script folder>/skills.  Non-existent folders and
    duplicates are dropped, so an unset option costs nothing.  Scanning is
    read-only: this never creates a directory (a typo should not silently make one)."""
    out: List[Path] = []
    seen = set()

    def add(cand) -> None:
        try:
            p = Path(str(cand)).expanduser().resolve()
            ok = p.is_dir()
        except Exception:
            return
        if not ok:
            return
        k = str(p).lower()
        if k not in seen:
            seen.add(k)
            out.append(p)

    raw = (settings or {}).get("skills_dir", "")
    if isinstance(raw, str) and raw.strip():
        add(raw.strip())
    for base in (USER_WORKSPACE_ROOT, BASE_DIR):
        try:
            if base.is_dir():
                add(base / SKILLS_SUBDIR)
        except Exception:
            pass
    return out


def _discover_skills(settings) -> List[dict]:
    """Scan the skill folders for */SKILL.md -> [{name, description, path, dir}].

    Bounded on purpose (SKILL_SCAN_DEPTH / SKILL_SCAN_MAX) so a skills folder
    pointed at a huge tree cannot stall every request.  The FIRST skill found for a
    name wins, which is what makes the Settings folder override the defaults.  Never
    raises - an unreadable folder or file is skipped."""
    found: List[dict] = []
    by_name = set()
    visited = 0
    for root_dir in _skill_dirs(settings):
        stack: List[tuple] = [(root_dir, 0)]
        while stack and len(found) < SKILL_INDEX_MAX * 4:
            d, depth = stack.pop()
            if visited >= SKILL_SCAN_MAX or depth > SKILL_SCAN_DEPTH:
                continue
            visited += 1
            try:
                entries = sorted(d.iterdir(), key=lambda e: e.name.lower())
            except Exception:
                continue
            for e in entries:
                try:
                    if e.is_dir():
                        if e.name.lower() not in SKILL_SKIP_DIRS:
                            stack.append((e, depth + 1))
                        continue
                    if e.name.lower() != SKILL_FILE_NAME.lower():
                        continue
                    raw = e.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                meta, _body = _skill_frontmatter(raw)
                name = str(meta.get("name") or "").strip()[:SKILL_NAME_MAX] or _skill_name_from_dir(d)
                low = name.lower()
                if low in by_name:
                    continue
                by_name.add(low)
                found.append({"name": name,
                              "description": re.sub(r"\s+", " ", str(meta.get("description") or "")).strip()[:SKILL_DESC_MAX],
                              "path": str(e), "dir": str(d)})
                if len(found) >= SKILL_INDEX_MAX * 4:
                    break
    return found


def _skills_index_text(settings) -> str:
    """The 'available skills' block for the system prompt ('' when there are none).

    Deliberately tiny: one line per skill.  This is progressive disclosure - a
    skill's body costs context only once the model actually asks for it."""
    skills = _discover_skills(settings)
    if not skills:
        return ""
    lines = ["", "",
             f"Available skills (folders holding a {SKILL_FILE_NAME}): a skill is reusable "
             "step-by-step procedure the user wrote down. If one fits the task, call "
             "load_skill(name) to read it BEFORE acting, then follow it. list_skills() "
             "reports where they live:"]
    used = n = 0
    for s in skills:
        if n >= SKILL_INDEX_MAX:
            break
        line = f"- {s['name']}: {s['description']}" if s["description"] else f"- {s['name']}"
        if used + len(line) > SKILL_INDEX_CHARS:
            break
        lines.append(line)
        used += len(line)
        n += 1
    if len(skills) > n:
        lines.append(f"- [... {len(skills) - n} more - call list_skills() to see them all]")
    return "\n".join(lines)


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
HANDOFF_SUMMARY_MAX_TOKENS = 4096   # bound on the summary itself (some servers reject it -> retried without)
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

HANDOFF_RECENT_PROMPTS   = 10       # last N REAL user prompts copied VERBATIM into every note
HANDOFF_PROMPT_MAX_CHARS = 700      # per-prompt cap there (a pasted file must not flood the note)

# ── Context compaction (C2/C3): in-place summarization so a turn keeps going ─────
# v1.1.33: the old COMPACTION_MAX_PER_CHAT cap is GONE. Compaction re-arms as many times as
# the gauge demands; chat["compaction_archive"] stays the authoritative record of every event.
COMPACTION_MIN_OLD      = 4         # fewer old messages than this -> not worth compacting (split too small)
COMPACTION_SUMMARY_MAX_TOKENS = 4096   # bound on the compaction summary itself (some servers reject it -> retried without)
# v1.1.58: thinking models spend max_tokens on REASONING, not on the answer.
# Measured on the user's qwen3.8-flash-next-iq3_s (llama.cpp): at max_tokens=1024 a
# real archived transcript returned finish_reason='length' with content EMPTY and
# reasoning_content 3,751 chars - the whole budget went to thinking. Removing the cap
# produced a proper 4,845-char summary (reasoning 11,349 chars). So a fixed cap cannot
# work for a thinking model; the request escalates instead. See empty_attempts in
# _generate_compaction_summary and _handoff_worker, and SUMMARY_EMPTY_RETRIES below.
# v1.1.68: the FIRST rung used to be 1024, which that same measurement already proved
# is too small for this model - so every compaction burned a guaranteed-failing call
# before escalating. Rungs raised to 4096 -> 16384 -> uncapped. Uncapped is still the
# final answer for a thinking model, because the reasoning budget needed scales with
# the transcript (11,349 reasoning chars for a 648-message one) and no fixed number
# covers it; the ladder exists to give the server a chance to reject a cap cleanly.
SUMMARY_RETRY_TOKENS      = 16384   # first escalation step when a summary comes back empty
SUMMARY_EMPTY_RETRIES     = 2       # 4096 -> 16384 -> no cap, then give up on the model
# v1.1.53: how much of the PREVIOUS compaction summaries may be carried forward verbatim
# into the live summary message. Carrying everything stops the 96%-per-generation loss, but
# an uncapped chain eventually eats the window it was meant to free (measured on a real
# 6-event chat: ~90 KB, i.e. ~22K tokens - the entire window of a small model). So the view
# is bounded while the STORE (chat["compaction_archive"]) stays complete: the newest
# summaries are kept, older ones remain readable in-app and are named in the marker.
CARRY_FRACTION_OF_WINDOW = 0.25   # of the context window, in chars (window*4*0.25)
CARRY_MIN_CHARS          = 6000   # floor: even a tiny window keeps some history
CARRY_MAX_CHARS          = 60000  # ceiling: never let the summary alone crowd a big window


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


def _is_synthetic_prompt(text: str) -> bool:
    """A "user" message the user never typed: a compaction marker/summary, or the intro a
    handoff seeds into a continued session. Neither is a real prompt, so neither belongs in
    the note's prompt log."""
    t = (text or "").lstrip()
    return t.startswith(COMPACTION_MARKER) or t.startswith(SESSION_HANDOFF_MARKER)



def _recent_prompts_section(messages: List[dict],
                            limit: int = HANDOFF_RECENT_PROMPTS) -> str:
    """Markdown section with the chat's most recent REAL user prompts, VERBATIM.

    A summary is what the model chose to remember; the prompts are what the user actually
    asked, in their own words - the most reliable record of intent for the session that
    continues this chat. Chronological (oldest first), newest prompts always kept, each one
    length-capped so a pasted file cannot flood the note. "" when there is nothing to say."""
    prompts: List[str] = []
    for m in messages or []:
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        c = (_flatten_handoff_content(m) or "").strip()
        if not c or _is_synthetic_prompt(c):
            continue
        if len(c) > HANDOFF_PROMPT_MAX_CHARS:
            c = c[:HANDOFF_PROMPT_MAX_CHARS].rstrip() + " [\u2026truncated]"
        prompts.append(c)
    if not prompts:
        return ""
    picked = prompts[-max(1, int(limit)):] if limit else prompts
    out = ("\n\n## Recent prompts (verbatim, oldest first - last "
           f"{len(picked)} of {len(prompts)})\n")
    for i, p in enumerate(picked, 1):
        out += f"{i}. {p}\n"
    return out


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


# ── Context compaction helpers (C2/C3) ─────────────────────────────────────────
# A compacted message's content starts with this marker. It is what the UI finds to draw
# the "📦 Context compacted" accordion, what _is_synthetic_prompt skips in handoff prompt
# logs, and what the re-compaction detector in _generate_compaction_summary looks for.
# (The EVENT count lives in chat["compaction_archive"] - never in marker counts: a
# re-compaction folds the previous summary, and its marker, into the new one.)
COMPACTION_MARKER = "[COMPACTED"
SESSION_HANDOFF_MARKER = "[Session Handoff]"   # intro a handoff seeds into a continued session


def _compaction_split_index(messages: List[dict], keep_recent: int) -> int:
    """Index at which to split messages into [old | recent] for compaction (C2).

    We want to keep the last `keep_recent` entries verbatim, but the split must NOT
    land in the middle of a tool phase: an assistant message carrying tool_calls has
    to stay together with the tool-result messages that follow it, or the next request
    would present orphaned tool calls and the server would reject it. So we start at
    len - keep_recent and walk BACKWARD past any assistant-with-tool_calls (and the
    results that belong to it) until the message just before the split is clean.

    A second, subtler case: an assistant message may fire SEVERAL tool calls in one
    parallel phase, producing a run of consecutive `tool` results. If the boundary
    lands in the middle of that run, the first recent message is a tool result whose
    owning assistant sits in the OLD half - the next request then presents an orphaned
    tool_call_id and strict servers reject it. So we also step back over any leading
    `tool` result (its owner always precedes it, index < idx, so this never drops a
    still-recent message). The recent tail may therefore keep slightly more than
    `keep_recent` entries when the trailing tool phase is large - that is deliberate:
    a valid request beats an exact count.
    Returns 0 when no safe boundary exists (caller then declines to compact)."""
    n = len(messages or [])
    idx = max(0, n - int(keep_recent or 0))
    while idx > 0:
        cur = messages[idx]
        prev = messages[idx - 1]
        if (isinstance(cur, dict) and cur.get("role") == "tool") or \
                (isinstance(prev, dict) and prev.get("role") == "assistant"
                 and prev.get("tool_calls")):
            idx -= 1
        else:
            break
    return idx


def _clip_transcript_to_budget(transcript: str, budget_chars: int) -> tuple:
    """Trim a formatted compaction transcript to `budget_chars`, keeping the NEWEST
    lines. Returns (kept_text, dropped_line_count).

    v1.1.53: the compaction summary request had NO budget at all, unlike the handoff
    path (`_handoff_budget_chars` + `_trim_for_handoff` + the halving ladder). It fires
    precisely BECAUSE the window is already ~70% full, then sends the whole folded
    history - measured on real chats, 3.4K to 60K tokens - so the summary request
    overflowed too and 93% of compactions silently fell back to the digest.

    Keeps the newest lines because that is what a continuation needs most (the same
    choice `_deterministic_handoff` makes); trims on line boundaries so a `[tool:…]`
    block is never cut mid-sentence. `dropped_line_count` lets the caller say so in
    the prompt instead of letting the model invent an overview of text it never saw."""
    budget = max(1000, int(budget_chars or 0))
    if len(transcript) <= budget:
        return transcript, 0
    lines = transcript.split("\n")
    kept: List[str] = []
    used = 0
    for ln in reversed(lines):
        # +1 for the newline joining lines back together.
        if used + len(ln) + 1 > budget:
            break
        kept.append(ln)
        used += len(ln) + 1
    kept.reverse()
    return "\n".join(kept), len(lines) - len(kept)


def _reply_reasoning(message: Any) -> str:
    """The reasoning/thinking text of a NON-streaming reply, if the server sent any.

    The streaming loop already reads reasoning_content (DeepSeek-R1 / Qwen style);
    the summary paths did not, which is how a thinking model's output was thrown
    away entirely. openai's ChatMessage keeps it in model_extra.
    """
    extra = getattr(message, "model_extra", None) or {}
    return str(getattr(message, "reasoning_content", None)
               or extra.get("reasoning_content") or "")


def _reply_finish_reason(resp: Any) -> str:
    try:
        return str(resp.choices[0].finish_reason or "")
    except Exception:
        return ""


def _create_summary_reply(client: Any, kwargs: Dict[str, Any]) -> Any:
    """Non-streaming create, retrying once WITHOUT max_tokens if the server rejects it.

    Some OpenAI-compatible servers (certain Ollama/LM Studio builds) error on the
    parameter itself rather than on the value.
    """
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as e:
        emsg = str(e).lower()
        if "max_tokens" in emsg or "max_completion_tokens" in emsg:
            kwargs.pop("max_tokens", None)
            return client.chat.completions.create(**kwargs)
        raise


def _format_messages_for_compaction(messages: List[dict]) -> str:
    """Render a slice of chat history as a compact transcript for the summary prompt (C3).

    Tool calls are folded into their assistant line and tool results capped, so a long
    agentic phase does not blow the summary request's own context. Images become a
    placeholder (the summary model may be non-vision)."""
    def _clip(s: str, n: int) -> str:
        s = str(s or "")
        return s if len(s) <= n else s[:n] + "…"

    out: List[str] = []
    for m in (messages or []):
        if not isinstance(m, dict):
            continue
        role = m.get("role", "?")
        content = str(m.get("content") or "")
        if role == "user" and content.lstrip().startswith(COMPACTION_MARKER):
            # v1.1.53: a PREVIOUS compaction summary must not be clipped to 600 chars.
            # It is the only surviving record of everything folded up earlier, and the
            # 600-char user cap destroyed ~96% of it on every re-compaction - including
            # the tail, which is where the newest material sits. These are carried
            # forward VERBATIM by _maybe_compact_context instead, so they are skipped
            # here rather than summarized (re-summarizing a summary compounds loss).
            continue
        if role == "assistant":
            tcs = m.get("tool_calls") or []
            calls = "; ".join(
                f"{((tc.get('function') or {}) if isinstance(tc, dict) else {}).get('name', '?')}"
                f"({_clip(str(((tc.get('function') or {}) if isinstance(tc, dict) else {}).get('arguments', '')), 120)})"
                for tc in tcs)
            line = _clip(content.strip(), 400)
            if calls:
                line = (line + " ") if line else ""
                out.append(f"[assistant] {line}\u2192 ran: {calls}")
            elif line:
                out.append(f"[assistant] {line}")
        elif role == "tool":
            name = str(m.get("name") or "")
            out.append(f"[tool:{name}] {_clip(content.strip(), 400)}")
        else:  # user (and anything unexpected)
            line = _clip(content.strip(), 600)
            if line:
                out.append(f"[user] {line}")
    return "\n".join(out)


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


def fmt_msg_time(ts: Optional[str]) -> str:
    """Format a message timestamp for display in the chat pane.

    Always shows 'Mon DD, YYYY - HH:MM:SS' (e.g. 'Oct 03, 2026 - 00:19:42').
    Returns '' when ts is missing or unparseable (legacy chats)."""
    if not ts:
        return ""
    try:
        dt = datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return ""
    return dt.strftime("%b %d, %Y - %H:%M:%S")


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
        # v1.1.52 (#10 mid-run steering): the MAIN thread enqueues while a turn runs and
        # the WORKER drains it at a step boundary. Dedicated lock - this is the
        # MAIN->WORKER direction, the mirror image of _post() (worker->main), which must
        # NOT be used for it. Entries are the same content shapes a normal user message
        # uses (str, or a multimodal parts list).
        self._steer_queue: List[Any] = []
        self._steer_lock = threading.Lock()
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
        self._perm_combos: Dict[str, tuple] = {}   # tool -> (chip label, state) - name kept for the UI tests
        self._mic_active = False
        self._mic_phrases = 0          # phrases transcribed this session (shown in the status line)
        self._mic_stop = threading.Event()   # set => stop continuous dictation
        self._stop_event = threading.Event()  # set => user pressed Stop; abort the in-flight turn
        # v1.1.45 (#4b): chat_id -> id of the turn currently running, so all tool
        # calls of one user prompt share one checkpoint namespace. In-memory only:
        # a restart starts a fresh turn, and the manifest of the previous run keeps
        # its own id (restore works across restarts; the id is just not reused).
        # WORKER THREAD ONLY - see the thread-ownership note above the checkpoint
        # methods; the main thread must never read or write these two.
        self._turn_ids: Dict[str, str] = {}
        self._cp_done: set = set()          # (chat, turn, path) already snapshotted this turn
        # v1.1.48: vision messages built by an MCP tool handler (e.g. read_mcp_resource
        # returning an image). WORKER THREAD ONLY - the tool loop drains this after each
        # tool call, so it never needs a lock and the main thread never touches it.
        self._pending_mcp_images: List[dict] = []
        self._tts_stop = threading.Event()   # set => halt TTS playback (send/close/toggle off)
        self._stt_thread: Optional[threading.Thread] = None
        self._ui_queue = queue.Queue()   # thread-safe handoff: worker threads -> UI
        self._sidebar_visible = True   # Python-side source of truth (pane.slaves() unreliable)
        self._drag_state: Optional[dict] = None   # sidebar drag-and-drop reordering state
        self._chat_visible: List[str] = []         # chat ids currently shown in the listbox (search-filtered)
        self._links: Dict[tuple, str] = {}         # (line, char offset) -> URL of clickable links in chat_text
        self._render_window_override: Dict[str, int] = {}   # cid -> widened view after "Load earlier messages"
        self._msg_marks: Dict[str, Dict[int, str]] = {}     # cid -> {msg index: Text index} (search jump)
        self._search_result_widgets: List[tk.Widget] = []  # full-text result rows in the sidebar
        self._search_last_build: float = 0.0               # rate limit on index rebuilds
        self._search_debounce = None                       # after() id for the body search
        self._archive_expanded: Dict[str, set] = {}         # cid -> {compaction_number} drawers opened this session
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
        # v1.1.59: the spawn gate. _mcp_connect_pending + the worker's while-loop
        # already prevented duplicate SUBPROCESSES (the pass itself is serialized by
        # _mcp_connect_lock), but they never prevented duplicate THREADS: N rapid
        # Saves spawned N threads, N-1 of which mostly queued on the lock and then
        # each still ran one extra pass. Four clicks could mean five reconnect passes.
        # A plain bool is not enough on its own - the check-and-set must be atomic,
        # and the worker must clear it under the SAME lock or a save landing between
        # the worker's final pending-check and its exit is lost forever (no thread
        # left to consume it). Hence _mcp_connect_spawn_lock around both decisions.
        self._mcp_connect_spawn_lock = threading.Lock()
        self._mcp_connect_running = False
        self._mcp_tool_map: Dict[str, tuple] = {}      # mcp_<server>_<tool> -> (server, raw tool name)
        self._mcp_tool_desc: Dict[str, str] = {}       # mcp_<server>_<tool> -> tooltip-sized description
        self._mcp_perm_widgets: List[tk.Widget] = []   # MCP section widgets in the permission bar

        # ── root window ───────────────────────────────────────────────────
        self.root = tk.Tk()
        self.root.title(f"{APP_NAME} — AI Desktop Assistant")
        self._set_window_icon()
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

    def _set_window_icon(self) -> None:
        """Give the window/taskbar the app icon.

        Without this a Tk app shows the interpreter's default feather, which is the
        detail that makes an installed program look unofficial next to everything
        else on the taskbar. Searched in the same places _image_cli_dir looks,
        because in a onefile build BASE_DIR is the temp extraction folder and the
        .ico may also simply sit next to the exe. Every failure is silent and
        harmless: a missing icon must never stop the app from starting.
        """
        bases = [getattr(sys, "_MEIPASS", None), str(BASE_DIR)]
        if getattr(sys, "frozen", False):
            bases.append(str(Path(sys.executable).resolve().parent))
        else:
            bases.append(str(Path(__file__).resolve().parent))
        for b in bases:
            if not b:
                continue
            p = Path(b) / "app_icon.ico"
            if not p.is_file():
                continue
            try:
                # default= makes the icon apply to every window, including the
                # permission/elicitation dialogs, not just the root one.
                self.root.iconbitmap(default=str(p))
                return
            except Exception:
                continue

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
        # v1.1.42: full-text hits render here, as clickable rows. A Frame, NOT the
        # Label above - a Label draws its own text UNDER packed children, so packing
        # rows into it overlaps them (verified). Packed above the chat list, which is
        # the expand=True widget and therefore absorbs the extra height.
        self.chat_search_hits = tk.Frame(side, bg=COL["bg_deep"])
        self.chat_search_hits.pack(fill="x", padx=12)

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

        # v1.1.46 (#4c): per-turn checkpoint revert. Deliberately a TOP-BAR control and
        # not a button on each tool-output accordion:
        #  * revert is a WHOLE-TURN operation. A button per tool call would imply "undo
        #    just this edit", which is not well defined (later edits in the same turn
        #    were applied on top of it) and is not what _checkpoint_restore does.
        #  * accordions are rebuilt on every render and store absolute Tk indices;
        #    embedding a live widget in an elided body shifts them (see the note in
        #    _render_compaction_accordion). A top-bar widget has no index at stake.
        # Hidden unless the chat actually has checkpointed turns (_refresh_revert_button).
        self.revert_btn = tk.Button(top, text="↺ Revert turn", command=self._on_revert_clicked,
                                    bg=COL["bg_raised"], fg=COL["text_dim"],
                                    activebackground=COL["danger"], activeforeground="#FFFFFF",
                                    relief="flat", bd=0, padx=8, cursor="hand2", font=F(11))
        self._revert_packed = False          # track pack state so refreshes are idempotent

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

    # ── Tool permissions bar (v1.1.41: click-to-cycle chips, flow-wrapped) ────
    # One dropdown per tool made this bar ~125 px wide PER TOOL (the Combobox, not
    # the label, set the cell width), so 15 tools needed ~2085 px and everything
    # past ~8 tools hid behind scroll buttons. The chip IS the control now: click a
    # name to cycle always -> ask -> off, and the name's COLOUR is the state. That
    # drops a cell to ~60-90 px, and the chips re-flow into as many rows as the
    # window allows instead of scrolling sideways.
    PERM_BAR_MIN_H = 40      # canvas height with a single row of chips
    PERM_BAR_MAX_ROWS = 3    # rows before the bar stops growing and the wheel scrolls
    PERM_CHIP_PADX = 4       # horizontal padding per chip (also the gap between chips)

    @staticmethod
    def _perm_row_break(x_used: int, avail: int, cell_w: int, col: int) -> bool:
        """Pure wrap decision: does a cell of width cell_w have to start a new row?

        Extracted (and kept free of any Tk call) so the wrap arithmetic is testable
        without a mapped window - the bar's own width is floored by the pane minsize,
        which makes narrow-window behaviour otherwise impossible to exercise."""
        return col > 0 and (x_used + cell_w) > avail

    def _perm_color(self, perm: str) -> str:
        """The one place a permission state's colour is defined."""
        return {"always": COL["success"], "ask": COL["warning"], "off": COL["danger"]}.get(
            perm, COL["text_dim"])

    def _perm_get(self, name: str) -> str:
        return (self.settings.get("tool_permissions") or {}).get(name, "ask")

    def _perm_tooltip(self, widget, text_fn) -> None:
        """Attach a hover tooltip to a permission chip.

        The words 'always'/'ask'/'off' no longer sit beside every tool, so the long
        PERM_LABELS name survives here. text_fn is called when the tip appears, never
        at attach time, so it can never show a stale state after a click.

        v1.1.61: the tip also carries a sentence describing the tool, which makes it
        several lines long. Two consequences handled here: the Label must WRAP
        (wraplength) or a 300-char description renders as one enormous off-screen
        line, and the window must be clamped to the screen or a chip near the right
        edge puts the tip beyond the desktop boundary where it cannot be read.
        """
        tip: Dict[str, Any] = {"win": None, "after": None}

        def hide(_e=None) -> None:
            if tip["after"] is not None:
                try:
                    widget.after_cancel(tip["after"])
                except Exception:
                    pass
                tip["after"] = None
            if tip["win"] is not None:
                try:
                    tip["win"].destroy()
                except Exception:
                    pass
                tip["win"] = None

        def show() -> None:
            tip["after"] = None
            if not widget.winfo_exists():
                return
            try:
                x = widget.winfo_rootx()
                y = widget.winfo_rooty() + widget.winfo_height() + 2
            except tk.TclError:
                return
            body = text_fn()
            # Wrap to a readable column, then keep the whole tip on screen. Sized
            # from the label's OWN effective point size (F() already folds in the
            # user's UI font scale) so a larger font widens the tip instead of
            # just adding lines. Note: there is no per-instance font-size accessor
            # on the app - the scale lives in the module-level FONT_DELTA - so do
            # not reach for one here; a bad call would raise inside the hover
            # handler, where nothing catches it and the tip simply never appears.
            fsize = F(9)[1]
            wrap = max(300, min(560, fsize * 44))
            win = tk.Toplevel(widget)
            win.wm_overrideredirect(True)          # no title bar / no focus steal
            try:
                scr = win.winfo_screenwidth()
                if x + wrap + 8 > scr:
                    x = max(0, scr - wrap - 12)
            except Exception:
                pass
            win.wm_geometry(f"+{x}+{y}")
            tk.Label(win, text=body, bg=COL["bg_raised"], fg=COL["text"],
                     font=F(9), padx=7, pady=4, bd=0, justify="left",
                     anchor="w", wraplength=wrap,
                     highlightthickness=1, highlightbackground=COL["border"]).pack()
            tip["win"] = win

        def show_later(_e=None) -> None:
            hide()                                  # one pending timer per chip
            try:
                tip["after"] = widget.after(450, show)
            except tk.TclError:
                pass

        widget.bind("<Enter>", show_later)
        widget.bind("<Leave>", hide)
        # No <Destroy> cleanup needed, contrary to how this first read: the tip is
        # created as tk.Toplevel(widget), which DOES make it a child of the chip in
        # Tk's hierarchy, so destroying the chip destroys the tip too (verified by
        # probe: tip.winfo_exists() is 0 after the chip is destroyed). A tip still
        # PENDING when the chip dies is covered by show()'s winfo_exists() guard.
        return hide

    def _perm_bind_wheel(self, w: tk.Widget) -> None:
        """Give one bar widget the wheel, so it is not a scroll dead-zone.

        ⚠️ Tk does NOT propagate <MouseWheel> to ancestors, so binding the canvas
        alone leaves every chip (a Frame + Label under `inner`) unhandled - the
        pointer is over a chip for most of the bar's area, which is why the wheel
        appeared to do nothing. Same trap the Settings dialog already works around
        with _bind_body_wheel.
        """
        try:
            w.bind("<MouseWheel>", self._perm_wheel, add="+")
            w.bind("<Button-4>", lambda _e: self._perm_scroll(-1), add="+")
            w.bind("<Button-5>", lambda _e: self._perm_scroll(1), add="+")
        except tk.TclError:
            pass

    def _tool_tip_text(self, name: str) -> str:
        """What this tool does, for the permission-chip tooltip ('' if unknown).

        MCP tools read the server's own description (captured at connect time in
        _mcp_schemas); built-ins read TOOL_SCHEMAS. Both are the exact strings the
        model is given, so the tip cannot drift from the tool's real behaviour."""
        desc = self._mcp_tool_desc.get(name)
        if not desc:
            desc = tool_description(name)
        return desc or ""

    def _add_perm_chip(self, name: str, text: str, in_flow: bool = True) -> tk.Frame:
        """One clickable permission chip: a label whose colour shows the state.

        Returns the cell frame (positioned later by _perm_reflow). The click handler
        cycles the permission, recolours the chip, saves settings, and reports the
        change in the status line - the status line is what makes a colour-only bar
        readable, since the words 'always'/'ask'/'off' no longer sit next to each tool.

        in_flow=False builds the chip but keeps it OUT of _perm_cells, so _perm_reflow
        never grid-manages it and it takes no space (a collapsed MCP tool). It is still
        registered in _perm_combos, so its permission is stored, displayed on expand and
        enforced at execution time exactly as before - collapsing is a view state, never
        a grant.
        """
        cell = tk.Frame(self._perm_inner, bg=COL["bg_deep"])
        lab = tk.Label(cell, text=text, bg=COL["bg_deep"], font=F(10),
                       fg=self._perm_color(self._perm_get(name)), cursor="hand2")
        lab.pack(anchor="w")

        def tip_text() -> str:
            # Built at hover time, so the description and the permission state are
            # both current (a click must never leave a stale state on screen).
            desc = self._tool_tip_text(name)
            return ((desc + "\n\n") if desc else "") + (
                f"{text} · {PERM_LABELS.get(self._perm_get(name), '?')}\n"
                "click to cycle Allow → Ask → Off")

        hide_tip = self._perm_tooltip(cell, tip_text)

        def cycle(_e=None, name=name, lab=lab) -> None:
            hide_tip()          # a stale tooltip must never outlive the click
            cur = self._perm_get(name)
            try:
                nxt = PERM_CYCLE[(PERM_CYCLE.index(cur) + 1) % len(PERM_CYCLE)]
            except ValueError:
                nxt = PERM_CYCLE[0]
            self.settings.setdefault("tool_permissions", {})[name] = nxt
            lab.configure(fg=self._perm_color(nxt))
            save_settings(self.settings)
            self._set_status(f"🔐 {text}: {PERM_HINT.get(nxt, nxt)}")

        for w in (cell, lab):
            w.bind("<Button-1>", cycle)
            w.configure(cursor="hand2")
            self._perm_bind_wheel(w)      # chips must not be wheel dead-zones
        # _perm_combos keeps its (frame, label) shape: the UI smoke suite reads
        # index [1] as the label (friendly-name check + font re-scale check).
        self._perm_combos[name] = (cell, lab)
        if in_flow:
            self._perm_cells.append(cell)
        return cell

    def _perm_reflow(self, _event=None) -> None:
        """Lay the chips out in as many rows as fit, then size the canvas to them.

        Replaces the sideways scrollbar: chips wrap, so nothing is hidden unless the
        bar would exceed PERM_BAR_MAX_ROWS - and then the wheel scrolls the rest."""
        canvas, inner = getattr(self, "_perm_canvas", None), getattr(self, "_perm_inner", None)
        if canvas is None or inner is None:
            return
        try:
            avail = int(canvas.winfo_width())
            if avail <= 1:
                return                      # not mapped yet; the <Configure> will reflow
            # Realize pending geometry FIRST: winfo_reqwidth() on a chip that has
            # never been mapped returns a bogus ~1px, which would pile a whole row's
            # worth of freshly-added chips (an MCP reconnect) onto row 0.
            inner.update_idletasks()
            # Skip only when BOTH the available width and the chips' own widths are
            # unchanged. A font change leaves the canvas width identical while every
            # chip resizes, so a width-only guard would keep a stale (too-wide) wrap.
            widths = []
            for cell in getattr(self, "_perm_cells", []):
                if not cell.winfo_exists():
                    continue
                cell.update_idletasks()          # measure before ungridding
                widths.append((cell, int(cell.winfo_reqwidth())))
            sig = (avail, tuple(w for _, w in widths))
            if getattr(self, "_perm_last_sig", None) == sig:
                # Also what stops the recursion: resizing the canvas below re-fires
                # <Configure> on it, and that pass finds nothing changed and returns.
                return
            self._perm_last_sig = sig
            pad2 = 2 * self.PERM_CHIP_PADX

            # Row height first: place() needs y before the loop can position chips.
            max_h = 0
            for _cell, _reqw in widths:
                max_h = max(max_h, int(_cell.winfo_reqheight()))
            row_h = (max_h + 4) if max_h else 24   # + both pady sides

            # ── Flow layout with place(), NOT grid ───────────────────────────────
            # ⚠ The bug (user screenshot): chips that wrapped to a second row spread
            # the FIRST row apart with large uneven gaps. Cause: Tk sizes a grid
            # COLUMN to the widest cell in it, and every row shared one column set -
            # so a wide chip on row 1 (an MCP header like "Photocraft (20) -") widened
            # that column for row 0 too, pushing the built-in chips apart.
            #
            # Per-row column OFFSETS do not fix this: the columns are still shared by
            # the whole frame, so offsetting row 1 past row 0's columns simply shifts
            # row 1 to the right by row 0's entire width (measured: x=1405 in a 735px
            # canvas). place() removes the coupling entirely - each chip goes exactly
            # where the wrap arithmetic says, with a uniform gap.
            x = col = row = 0
            row_w = 0
            max_row_w = 0
            rows_used = 1
            for cell, reqw in widths:
                w = reqw + pad2
                if self._perm_row_break(x, avail, w, col):
                    max_row_w = max(max_row_w, row_w)
                    row += 1
                    col = 0
                    x = 0
                    row_w = 0
                cell.grid_forget()          # clear any legacy grid management
                cell.place(x=x + self.PERM_CHIP_PADX, y=row * row_h + 2)
                x += w
                row_w = x
                col += 1
                rows_used = max(rows_used, row + 1)
            max_row_w = max(max_row_w, row_w)

            inner.update_idletasks()
            rows = min(rows_used, self.PERM_BAR_MAX_ROWS)
            # place() children do NOT propagate to the parent's requested size, so the
            # inner frame must be sized explicitly - otherwise winfo_reqheight() and
            # bbox("all") collapse to 0, the bar can never grow for wrapped rows, and
            # the wheel-scroll fallback stops engaging.
            content_h = rows_used * row_h + 4
            inner.configure(width=max_row_w, height=content_h)
            want = content_h + 8
            new_h = max(self.PERM_BAR_MIN_H, min(want, rows * row_h + 10))
            # Only resize on an actual change: configuring the height re-fires
            # <Configure> on the canvas, which would otherwise re-enter this method.
            if int(canvas.cget("height")) != new_h:
                canvas.configure(height=new_h)
            canvas.configure(scrollregion=(0, 0, max_row_w, content_h))
        except tk.TclError:
            pass

    def _perm_scroll(self, direction: int) -> None:
        """Scroll the bar by one notch. Vertical first (the bar is capped at
        PERM_BAR_MAX_ROWS); horizontal only if a single row still overflows."""
        canvas = getattr(self, "_perm_canvas", None)
        if canvas is None:
            return
        try:
            bb = canvas.bbox("all")
            if not bb:
                return
            if bb[3] > int(canvas.cget("height")):
                canvas.yview_moveto(max(0.0, min(1.0,
                    canvas.yview()[0] + direction * 0.15)))
            elif bb[2] > int(canvas.winfo_width()):
                canvas.xview_moveto(max(0.0, min(1.0,
                    canvas.xview()[0] + direction * 0.08)))
        except tk.TclError:
            pass

    def _perm_wheel(self, event) -> None:
        """Wheel over the bar scrolls it - only reachable once the row cap is hit."""
        # Windows convention (verified on this Tk build): delta +120 scrolls toward
        # the TOP, -120 toward the bottom.
        self._perm_scroll(-1 if getattr(event, "delta", 120) > 0 else 1)

    def _build_permissions_bar(self) -> None:
        """Dynamic per-tool permission bar at the bottom of the window."""
        bar = tk.Frame(self.root, bg=COL["bg_deep"])
        bar.grid(row=1, column=0, sticky="ew")
        tk.Label(bar, text="🔐 Tools", bg=COL["bg_deep"], fg=COL["text_dim"],
                 font=F(11, "bold")).pack(side="left", padx=(12, 6), pady=7)
        # The colour key lives here once, instead of a word on every tool.
        key = tk.Frame(bar, bg=COL["bg_deep"])
        key.pack(side="left", padx=(0, 6), pady=7)
        for perm in PERM_CYCLE:
            tk.Label(key, text=PERM_KEY[perm], bg=COL["bg_deep"],
                     fg=self._perm_color(perm), font=F(9)).pack(side="left", padx=(0, 6))

        canvas = tk.Canvas(bar, bg=COL["bg_deep"], height=self.PERM_BAR_MIN_H,
                           highlightthickness=0)
        canvas.pack(side="left", fill="both", expand=True)

        self._perm_inner = tk.Frame(canvas, bg=COL["bg_deep"])
        inner = self._perm_inner
        canvas.create_window((0, 0), window=inner, anchor="nw")
        self._perm_canvas = canvas
        self._perm_cells: List[tk.Widget] = []
        self._perm_last_sig = None
        canvas.bind("<MouseWheel>", self._perm_wheel)
        # Tk 9 on Windows delivers <MouseWheel>; Linux sends Button-4/5.
        canvas.bind("<Button-4>", lambda _e: self._perm_scroll(-1))
        canvas.bind("<Button-5>", lambda _e: self._perm_scroll(1))
        # Re-flow whenever the bar changes width (window resize, sidebar drag, font
        # change) - that is what keeps the chips visible without sideways scrolling.
        canvas.bind("<Configure>", self._perm_reflow)

        for name, pretty in TOOLS:
            self._add_perm_chip(name, pretty)

        # MCP tools (discovered from connected servers) are appended after the
        # built-in tools; _mcp_rebuild_permissions_bar() owns that section.
        self._mcp_rebuild_permissions_bar()
        self._perm_reflow()

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
                loaded = [m for m in entries if _entry_is_loaded(m)]
                if len(loaded) == 1:
                    entry = loaded[0]
            self._ctx_total = _entry_context_length(entry)
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

    def _ledger_inject(self, chat_id: str) -> str:
        """The ledger block for one request. Safe on any thread.

        Reads the setting at call time (so turning it off takes effect immediately) and
        swallows every failure: a ledger is an enhancement and must never stop a request
        the way a broken index must never stop a save."""
        try:
            budget = int(float(self.settings.get("ledger_inject_chars",
                                                 LEDGER_INJECT_DEFAULT)))
        except (TypeError, ValueError):
            budget = LEDGER_INJECT_DEFAULT
        if budget <= 0:
            return ""
        try:
            return _ledger_inject_text(str(chat_id or ""), budget)
        except Exception:
            return ""

    def _ctx_window(self) -> int:
        """Context window size as seen from a worker thread.

        _ctx_total is main-thread state; read through getattr so half-built instances
        (tests that skip __init__) never raise AttributeError inside the agentic loop."""
        try:
            return int(getattr(self, "_ctx_total", 0) or 0)
        except (TypeError, ValueError):
            return 0

    def _ctx_used_last(self) -> int:
        """Last server-reported session tokens as seen from a worker thread.

        _ctx_used is main-thread state (set per completed step via _set_ctx_used and reset to
        0 on a chat switch); read through getattr so half-built instances (tests that skip
        __init__) never raise AttributeError inside the agentic loop. At turn start this is the
        PREVIOUS turn's real server figure - the accurate one, unlike estimate_prompt_tokens()
        which omits the system prompt and tool schemas."""
        try:
            return int(getattr(self, "_ctx_used", 0) or 0)
        except (TypeError, ValueError):
            return 0

    # ── Context compaction (C2/C3): in-place summarization ───────────────────────
    def _maybe_compact_context(self, chat_id: str, used: int, total: int,
                               messages: Optional[List[dict]] = None) -> bool:
        """Summarize the older part of this chat IN PLACE so the turn can keep going.

        Called from the worker at a step boundary, BEFORE the mid-turn handoff gate. When
        the context gauge crosses `compaction_threshold` and there is enough old history to
        fold up, the oldest messages are replaced by a single summary message (the recent
        tail stays verbatim) and the turn continues - no fork, no button, no user action.
        The original messages are preserved in chat["compaction_archive"] so the user can
        still read every word in-app (C4).

        `messages` is the worker's LOCAL message list - the very list the agentic loop builds
        its next request from. Compacting it IN PLACE (`messages[:] = ...`) is what actually
        frees context for the NEXT request: the persisted chat["messages"] record alone is a
        copy that the running loop never re-reads, so updating only that (the pre-v1.1.25
        bug) left the worker sending all the old tokens again and firing a 2nd compaction.

        Returns True when compaction was performed (the caller should continue the turn);
        False when it declined and the existing handoff gate should run instead. Deliberately
        inert - no mutation, no summary call - when the feature is off, the gauge is below
        threshold, the window is unknown, too few messages exist to split, or the per-chat
        there is no per-chat compaction cap any more (removed in v1.1.33).

        Runs on the worker thread. It rebinds the caller's LOCAL list in place (a plain local,
        safe here) and posts ONE callback that mutates chat["compaction_archive"] +
        chat["messages"] and calls save_chats() on the main thread - the worker never touches
        self.chats directly (invariant #4). The persist callback reads chat["messages"] LIVE at
        drain time, so it saves the full local list including any tool results appended after
        compaction; the queued note is posted AFTER it and renders once the record is current."""
        if self._closed or self._stop_event.is_set():
            return False
        try:
            threshold = int(float(self.settings.get("compaction_threshold", COMPACTION_THRESHOLD_DEFAULT)))
        except (TypeError, ValueError):
            threshold = COMPACTION_THRESHOLD_DEFAULT
        if threshold <= 0:
            return False                       # feature explicitly off (Settings -> Compaction threshold % = 0)
        if total <= 0 or used <= 0:
            return False                       # no measurable window to gate on (budget ladder owns that path)
        if (used / total) * 100.0 < float(threshold):
            return False                       # not yet at the compaction threshold
        chat = self.chats["chats"].get(chat_id)
        if not chat:
            return False
        if messages is None:                     # defensive default; the agentic loop always passes its local list
            messages = chat.get("messages") or []
        # The archive list is the authoritative record of how many times this chat has been
        # compacted. Counting [COMPACTED markers in live messages would NOT work: each
        # re-compaction folds the previous summary (and its marker) into the new one, so at
        # most ONE marker survives no matter how many compactions have already happened.
        # Read the CURRENT length on the worker only for the numbering; the list itself is
        # never mutated here - the entry is appended inside _persist_compaction on the MAIN
        # thread (invariant #4: the worker never mutates self.chats).
        archive_now = chat.get("compaction_archive")
        archive_len = len(archive_now) if isinstance(archive_now, list) else 0
        # v1.1.33: no cap. A chat may be compacted as many times as its gauge fills; the
        # handoff gate (fork) still runs downstream, but only when compaction declines.
        try:
            keep_recent = int(float(self.settings.get("compaction_keep_recent", COMPACTION_KEEP_RECENT_DEFAULT)))
        except (TypeError, ValueError):
            keep_recent = COMPACTION_KEEP_RECENT_DEFAULT
        # Clamp here too, not just on Settings-save (line ~8070): a hand-edited or
        # corrupted settings file with keep_recent <= 0 would make _compaction_split_index
        # index messages[n] (out of range -> IndexError). Same 2..50 bounds as the save path.
        keep_recent = max(2, min(50, keep_recent))
        split = _compaction_split_index(messages, keep_recent)
        if split < COMPACTION_MIN_OLD:
            return False                       # not enough old messages to be worth compacting
        old_msgs = list(messages[:split])
        recent_msgs = list(messages[split:])
        # v1.1.53: carry any PREVIOUS compaction summary forward VERBATIM instead of
        # re-summarizing it. It is the only surviving record of everything folded up
        # earlier; running it through the model again (and through the 600-char user cap
        # in _format_messages_for_compaction) destroyed ~96% of it per generation, keeping
        # the head while the newest material sat in the tail. Splitting it out here means
        # the model only ever summarizes genuinely new text.
        # Partition by IDENTITY, not equality: two compaction summaries can be
        # byte-identical, and `m not in list` would drop both on value comparison.
        prior_summaries = [m for m in old_msgs
                           if isinstance(m, dict)
                           and str(m.get("content") or "").lstrip().startswith(COMPACTION_MARKER)]
        _prior_ids = {id(m) for m in prior_summaries}
        new_msgs = [m for m in old_msgs if id(m) not in _prior_ids]
        summary_text = self._generate_compaction_summary(new_msgs, chat)
        if not summary_text and not prior_summaries:
            return False                       # nothing usable produced -> let the handoff gate handle it
        if prior_summaries:
            # Newest prior summary first inside the block, so the most recent material
            # sits closest to the live tail. Bounded: keep newest-first up to the carry
            # budget, and name what was left out so the text does not silently pretend
            # to be the whole record (the archive still holds all of it).
            carry_budget = min(
                CARRY_MAX_CHARS,
                max(CARRY_MIN_CHARS,
                    int(self._ctx_window() * 4 * CARRY_FRACTION_OF_WINDOW)))
            kept: List[str] = []
            used = 0
            dropped = 0
            for m in reversed(prior_summaries):
                txt = str(m.get("content") or "")
                if kept and used + len(txt) > carry_budget:
                    dropped += 1
                    continue
                if not txt.strip():
                    continue
                kept.append(txt)
                used += len(txt)
            if dropped:
                kept.append(f"[{dropped} earlier compaction summary/suppressed from this "
                            "view for length - the full original messages remain readable "
                            "under the \U0001F4E6 'Context compacted' drawers in this chat]")
            summary_text = ((summary_text + "\n\n" + "\n\n".join(reversed(kept)))
                            if summary_text else "\n\n".join(reversed(kept)))
        compaction_number = archive_len + 1
        marker = f"{COMPACTION_MARKER} \u2014 summary of {len(old_msgs)} earlier messages]"
        # v1.1.54: point at the archive. Without this the model has no signal that the
        # originals still exist and are readable, so it treats the summary as the whole
        # record and cannot recover detail the summary dropped.
        marker += (f" (the {len(old_msgs)} original messages are kept in this chat's "
                   f"archive - retrieve them with read_archive(compaction="
                   f"{compaction_number}))")
        summary_msg = {"role": "user", "content": marker + "\n" + summary_text}
        # Archive is a LIST (one entry per compaction event - unlimited since v1.1.33):
        # the user can re-open every original message from in-app (C4), so an earlier
        # compaction's originals must survive a later one - a single dict would be clobbered.
        # Build the entry on the worker; the APPEND happens in _persist_compaction below.
        archive_entry = {
            "messages": old_msgs,
            "timestamp": datetime.now().isoformat(),
            "summary_text": summary_text,
            "compaction_number": compaction_number,
        }
        # Rebind the worker's LOCAL list IN PLACE so the NEXT request in this same loop
        # carries [summary] + recent instead of the full old history. This is the fix for
        # the stale-local-list bug: without it the loop kept sending every old token and a
        # 2nd compaction fired immediately (the gauge never dropped).
        compacted = [summary_msg] + recent_msgs
        messages[:] = compacted
        # Persist on the MAIN thread only (invariant #4: the worker never mutates self.chats,
        # and the main thread must not read the worker's live list either). Capture a snapshot
        # of the compacted list HERE on the worker thread; the callback assigns it. The archive
        # entry is APPENDED inside the callback (at drain time, on the main thread) so the
        # worker touches no shared chat state at all. The next _ui_sync_messages() (posted
        # after this) overwrites chat["messages"] with the fuller post-compaction list, so this
        # save only needs to persist the archive + a consistent messages snapshot atomically.
        # The note is posted AFTER the persist callback (FIFO queue) so it renders once the
        # record is current.
        def _persist_compaction(chat=chat, entry=archive_entry, snap=list(compacted)):
            archive = chat.get("compaction_archive")
            if not isinstance(archive, list):
                archive = []
            archive.append(entry)
            chat["compaction_archive"] = archive
            chat["messages"] = list(snap)
            save_chats(self.chats)
        self._post(_persist_compaction)
        # Tier 3: mirror the summary into the append-only ledger. The live record can be
        # re-compacted and the chat can be deleted; the ledger survives both, so the
        # automatic record is no longer the only one and is no longer the fragile one.
        # Marked source="compaction" so it reads as a lower-trust, model-written note
        # next to the ones the model chose to write itself.
        # A long summary is truncated AT A SENTENCE-ish BOUNDARY and says so, pointing at
        # read_archive for the rest - silently cutting the tail is the exact Tier 1 defect.
        # The disclosure MUST be budgeted for: _ledger_append truncates defensively at
        # LEDGER_ENTRY_MAX_CHARS, so a 4000-char body plus a ~150-char marker got the marker
        # itself chopped off in testing. Reserve room for it instead of appending after.
        _mirror = summary_text
        if len(_mirror) > LEDGER_ENTRY_MAX_CHARS:
            _note = (f" […truncated; the full {len(summary_text)}-char summary and all "
                     f"{len(old_msgs)} original messages are recoverable with "
                     f"read_archive(compaction={compaction_number})]")
            _room = max(200, LEDGER_ENTRY_MAX_CHARS - len(_note))
            _cut = _mirror.rfind(". ", 0, _room)
            _cut = _cut if _cut > _room // 2 else _room
            # +1: rfind returns the index OF the period, so [:_cut] would drop it and the
            # kept text would end mid-sentence ("...the retry ladder") - which reads like
            # an arbitrary cut rather than a clean one.
            _mirror = _mirror[:_cut + 1].rstrip() + _note
        _ledger_append(chat_id, _mirror,
                       tags=[f"compaction{compaction_number}", "auto"], source="compaction")
        self._post(lambda n=len(old_msgs), k=compaction_number:
                   self.render_note(f"\U0001F4E6 Context compacted - {n} earlier messages summarized "
                                    f"(compaction {k}); continuing…"))
        return True

    def _generate_compaction_summary(self, old_messages: List[dict], chat: dict) -> str:
        """Produce the summary text for a compaction (C3).

        One non-streaming model call over the transcript of the old messages. On any failure
        (auth, connection, overflow, empty reply) it falls back to _deterministic_handoff -
        the existing model-free digest - so a compaction always yields something usable and
        never strands the turn.

        v1.1.53 BUDGET (this is what was broken). The request is now sized to the window
        with `_handoff_budget_chars` + `_clip_transcript_to_budget`, and halved-and-retried
        if the server still refuses it - the same ladder the handoff path has had since
        v1.1.14. Before this, compaction had NO budget: it fired precisely because the
        window was already ~70% full, then sent the entire folded history (measured on the
        user's real chats: 3.4K-60K tokens) into that same window. The summary request
        overflowed too, the except caught it, and 93% of compactions (25 of 27 events in
        the live chats.json) silently degraded to the crude digest. The only user-visible
        sign was a note that scrolled past.

        v1.1.53 NO RE-SUMMARIZING A SUMMARY. A previous compaction summary is carried
        forward VERBATIM by _maybe_compact_context, not fed back through the model. It used
        to go through _format_messages_for_compaction, whose 600-char user cap kept ~3.7%
        of it - and specifically the HEAD, while _deterministic_handoff deliberately puts
        the newest, most-needed material at the TAIL. So each re-compaction quietly threw
        away almost all of its own memory while the digest grew and nested itself."""
        client = self._get_client()
        transcript = _format_messages_for_compaction(old_messages)
        if not transcript.strip():
            return ""
        model = (self.settings.get("model_name") or "gpt-4o-mini").strip()
        # Budget ladder, mirroring _handoff_worker. The window is known here: the caller
        # already returned False when total <= 0.
        budget = _handoff_budget_chars(self._ctx_window())
        attempts = 0
        empty_attempts = 0
        salvage_reason = ""      # best reasoning text seen, used only if the ladder fails
        last_err = ""
        while client is not None:
            body, dropped = _clip_transcript_to_budget(transcript, budget)
            system = ("You are summarizing a conversation for continuity. Produce a concise summary "
                      "preserving: key decisions, facts established, file paths mentioned, code changes "
                      "made, and any open questions. Also state which project folder the work belongs to "
                      "and which project MEMORY.md or history/ files were read or updated - those files "
                      "stay on disk after this summary, so naming them keeps the thread findable. "
                      "Be specific. No preamble. If tool calls were made, "
                      "note what was fetched or changed and the key findings - do not include raw output.")
            if dropped:
                # Say so, or the model will invent an overview of text it was never shown.
                system += (f" Note: the OLDEST {dropped} lines of this transcript were omitted for "
                           "length; summarize only what is shown and do not guess at the rest.")
            # The cap must be derived from empty_attempts, NOT mutated into kwargs:
            # this loop rebuilds kwargs on every pass, so escalating the dict and
            # then `continue`-ing would silently discard the escalation. (A first
            # draft of this fix did exactly that and would have retried at the same
            # size twice before giving up.)
            cap = (COMPACTION_SUMMARY_MAX_TOKENS if empty_attempts == 0
                   else (SUMMARY_RETRY_TOKENS if empty_attempts == 1 else None))
            kwargs: Dict[str, Any] = dict(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": body},
                ],
                stream=False,
                temperature=0.3,      # factual summary - low sampling noise
            )
            if cap:
                kwargs["max_tokens"] = cap
            try:
                resp = _create_summary_reply(client, kwargs)
                summary = (resp.choices[0].message.content or "").strip()
                if summary:
                    return summary
                # v1.1.58: an empty answer is NOT automatically a dead server. A
                # thinking model can spend the entire completion cap on
                # reasoning_content and still finish with finish_reason='length'
                # and content='' - reproduced on the user's own model. Two things
                # are wrong with the old behaviour: it read only .content (so real
                # reasoning text was discarded), and it gave up after ONE empty
                # reply while an overflow gets a whole retry ladder.
                # v1.1.68 ORDERING FIX. v1.1.58 returned salvaged reasoning
                # IMMEDIATELY, which short-circuited the escalation ladder below:
                # a thinking model that spent COMPACTION_SUMMARY_MAX_TOKENS (1024)
                # on reasoning and answered content='' was never retried at 4096 or
                # uncapped, so the raw thinking trace became the permanent summary.
                # Observed directly: compaction 7 of this very chat begins "We need
                # answer user's request: summarize conversation for continuity."
                # Reasoning is now remembered and used only as the LAST resort,
                # preferred over the model-free digest but never over a real summary.
                reason = _reply_reasoning(resp.choices[0].message).strip()
                if reason and not salvage_reason:
                    salvage_reason = reason
                last_err = ("the model returned an empty summary"
                            + (f" (finish_reason={_reply_finish_reason(resp)})" if _reply_finish_reason(resp) else ""))
                if empty_attempts < SUMMARY_EMPTY_RETRIES:
                    empty_attempts += 1
                    nxt = (SUMMARY_RETRY_TOKENS if empty_attempts == 1 else None)
                    self._post(lambda c=nxt: self.render_note(
                        f"\U0001F4E6 Compaction summary was empty - retrying with a "
                        f"larger completion budget ({c if c else 'no cap'})\u2026"))
                    continue
                if salvage_reason:
                    # Ladder exhausted. A thinking trace still beats a digest.
                    self._post(lambda n=len(salvage_reason): self.render_note(
                        f"\U0001F4E6 Compaction summary stayed empty after "
                        f"{SUMMARY_EMPTY_RETRIES} retries; carrying over the model's "
                        f"{n} chars of reasoning instead of a digest."))
                    return salvage_reason
                break
            except Exception as e:
                last_err = str(e)
                # Overflow AND room left to shrink -> retry smaller. Auth/connection errors
                # are not fixed by a shorter transcript: stop trying.
                if (is_context_overflow_error(last_err)
                        and attempts < HANDOFF_RETRY_OVERFLOW
                        and budget // 2 >= HANDOFF_FALLBACK_MIN_CHARS):
                    attempts += 1
                    budget //= 2
                    self._post(lambda b=budget: self.render_note(
                        f"\U0001F4E6 Compaction summary too large - retrying with a smaller "
                        f"transcript budget ({b} chars)…"))
                    continue
                break
        # Loud, not a note that scrolls past: a digest is materially worse than a model
        # summary, and silently accepting that was how this went unnoticed for 50 releases.
        # An overflow is EXPECTED here (that is why the ladder exists), so it is a warning;
        # anything else (auth, connection, empty reply) means something is genuinely wrong.
        if client is None:
            pass          # no client at all: the caller's own path reports this
        elif not is_context_overflow_error(last_err):
            self._post(lambda m=last_err: self.render_error(
                f"\U0001F4E6 Compaction summary FAILED ({m}) - falling back to a "
                f"deterministic digest, which preserves far less. Check the model/server."))
        else:
            self._post(lambda m=last_err: self.render_note(
                f"\U0001F4E6 Compaction summary still too large after {attempts} shrink(s) "
                f"- using a deterministic digest instead (it keeps much less detail)."))
        # Fallback: model-free digest of the old slice (never blocks, never needs the server).
        return _deterministic_handoff({"messages": old_messages},
                                      max(HANDOFF_FALLBACK_MIN_CHARS,
                                          min(8000, HANDOFF_FALLBACK_MAX_CHARS)))

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
        empty_attempts = 0
        summary = ""
        salvage_reason = ""      # best reasoning text seen; used only if the ladder fails
        last_err = ""
        while True:
            history = _trim_for_handoff(sanitized, budget)
            # Cap derived from empty_attempts, not mutated into kwargs: this loop
            # rebuilds kwargs every pass (v1.1.58 - a thinking model can spend the
            # whole cap on reasoning and answer with an empty content).
            hcap = (HANDOFF_SUMMARY_MAX_TOKENS if empty_attempts == 0
                    else (SUMMARY_RETRY_TOKENS if empty_attempts == 1 else None))
            kwargs: Dict[str, Any] = dict(
                model=model,
                messages=[
                    build_system_message(str(self.settings.get("file_workspace") or ""),
                                         str(self.settings.get("exa_api_key") or ""),
                                         str(self.settings.get("firecrawl_api_key") or ""),
                                         effective_custom_prompt(self.settings),
                                         _skills_index_text(self.settings),
                                         self._ledger_inject(chat_id),
                                         _project_memory_text(self.settings)),
                    *history,
                    {"role": "user", "content": _handoff_prompt(title)},
                ],
                stream=False,
                temperature=0,      # factual summary - no sampling noise
            )
            if hcap:
                kwargs["max_tokens"] = hcap
            try:
                resp = _create_summary_reply(client, kwargs)
                summary = (resp.choices[0].message.content or "").strip()
                if summary:
                    break
                # v1.1.58: same salvage + escalation as _generate_compaction_summary.
                # v1.1.68: reasoning is remembered, not returned - returning it here
                # skipped the escalation ladder and baked a raw thinking trace into
                # the handoff note that every later session reads as its first message.
                reason = _reply_reasoning(resp.choices[0].message).strip()
                if reason and not salvage_reason:
                    salvage_reason = reason
                last_err = ("the model returned an empty summary"
                            + (f" (finish_reason={_reply_finish_reason(resp)})"
                               if _reply_finish_reason(resp) else ""))
                if empty_attempts < SUMMARY_EMPTY_RETRIES:
                    empty_attempts += 1
                    continue
                if salvage_reason:
                    summary = salvage_reason
                    self._post(lambda n=len(salvage_reason): self.render_note(
                        f"\U0001F4E6 Handoff summary stayed empty after "
                        f"{SUMMARY_EMPTY_RETRIES} retries; carrying over the model's "
                        f"{n} chars of reasoning instead of a digest."))
                    break
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
                _atomic_write(fp, header + summary
                                + _recent_prompts_section(sanitized) + "\n")
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

        # v1.1.33: sampling carries over with the handoff. A continued session is the SAME
        # work in a new chat, so it must not silently drop back to TEMP_DEFAULT and the
        # default thinking level mid-task. new_chat() already consumed any welcome-state
        # _pending_* values and synced the combos from the blank chat, so re-sync after
        # the copy (the half-built test app has no combos - guard on the widget).
        for _k in ("temperature", "thinking"):
            if src.get(_k) not in (None, ""):
                fresh[_k] = src[_k]
        if getattr(self, "temp_combo", None) is not None:
            self._sync_temp_combo(fresh)
            self._sync_thinking_combo(fresh)
            self._update_temp_topbar()
        if summary:
            intro = (SESSION_HANDOFF_MARKER + " The previous session reached its context limit. "
                     "Here is a summary of the work so far - continue from here:\n\n" + summary)
        else:
            # Notes-only handoff (the condensed summary is gone, e.g. a chat record written before
            # summaries were persisted): announce that plainly instead of promising a summary that is
            # not there, and point at the notes as the only record.
            intro = (SESSION_HANDOFF_MARKER + " The previous session reached its context limit. Its condensed"
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
        _intro_ts = datetime.now().isoformat(timespec="seconds")
        fresh["messages"].append({"role": "user", "content": intro, "ts": _intro_ts})
        base = (src.get("title") or "Session").strip()[:24]
        fresh["title"] = f"Continued: {base}"
        save_chats(self.chats)
        self._refresh_chat_list()
        self.render_user_message(intro, _intro_ts)
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
            self._refresh_revert_button()   # v1.1.46: disable revert while the turn runs
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
        # v1.1.45 (#4b): this chat's checkpoint tree belongs to it alone - no other
        # record names those turns, so it goes with it. Same ordering rule as the
        # images above: after save_chats(), so a crash mid-deletion never leaves
        # persisted history pointing at snapshots that are already gone. Retention is
        # deliberately ONLY here and in _checkpoint_prune - no filesystem sweeps
        # (standing policy 2026-10-06: the user cleans up manually).
        try:
            shutil.rmtree(_checkpoint_root() / _checkpoint_component(cid), ignore_errors=True)
        except Exception:
            pass
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
            if was_current:
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

    def _chat_list_drag(self, event) -> Optional[str]:
        """While dragging, move the grabbed chat to the row under the pointer.

        Once a reorder drag is active this returns "break" so the Listbox CLASS
        <B1-Motion> binding does not also run. That built-in handler does
        drag-select and, when the pointer nears the top/bottom edge, auto-scrolls
        the listbox rapidly while keeping a stale selection anchor. Left running
        alongside our reorder it fights the drag: on a long list (one with a
        scrollbar) the list jumps and scrolls really fast and the item can't be
        placed. Suppressing it here is the fix. A plain click (not yet a drag)
        still returns None so the class binding can do normal click-selection.
        """
        st = self._drag_state
        if not st:
            return
        if not st["active"]:
            if abs(event.y - st["y0"]) < 6:
                return                      # still a plain click (let the class binding select)
            st["active"] = True
            try:
                self.chat_list.configure(cursor="fleur")
            except tk.TclError:
                pass
        # Drag is active: suppress the built-in drag-select / edge auto-scroll for
        # the remainder of this motion event by returning "break" on every path.
        try:
            idx = self.chat_list.nearest(event.y)
        except tk.TclError:
            return "break"
        if not (0 <= idx < len(self._chat_visible)):
            return "break"
        order = self.chats["order"]
        cur = order.index(st["cid"])
        if idx != cur:
            order.insert(idx, order.pop(cur))
            self._refresh_chat_list(select_cid=st["cid"])   # keep the dragged row selected
        return "break"

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

    def load_chat(self, cid: str, force: bool = False, jump_to: Optional[int] = None) -> None:
        """Render one chat. jump_to = a message index to scroll to and select once the
        view is built (used by full-text search). The render window is widened first if
        that message would otherwise fall outside it."""
        chat = self.chats["chats"].get(cid)
        if not chat or (not force and cid == self.current_chat_id):
            return
        self.current_chat_id = cid
        # clear display + embedded widgets
        try:
            self.chat_text.delete("1.0", "end")
        except tk.TclError:
            pass
        # v1.1.35 PERFORMANCE FIX - leaked Tk tags.
        # Every accordion creates two tags named acc_<id>_hdr / acc_<id>_body
        # (_add_accordion). delete("1.0","end") removes the TEXT but NOT the tags,
        # so they accumulated for the lifetime of the process. Tk's Text renderer
        # walks the tag list per character, so cost grew with every chat switch:
        # measured on the real 13.6MB data file, loading the same 1MB chat went
        # 5.4s -> 20.5s -> 36.7s -> 47.2s after 1/2/3/4 switches, and the tag
        # count climbed 590 -> 4378. That is the 10-20s freeze on chat clicks.
        # tag_delete() also drops the tag's <Button-1> binding, so the accordion
        # handler lambdas (which close over the Text widget) are released too.
        # Only the acc_* tags are dynamic; the style tags configured in
        # _configure_text_tags() are static and must survive.
        try:
            leaked = [t for t in self.chat_text.tag_names() if t.startswith("acc_")]
            if leaked:
                self.chat_text.tag_delete(*leaked)
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
            if jump_to is not None:
                self._widen_view_for(cid, len(msgs), int(jump_to))
            self._render_chat_messages(cid, chat, msgs)
        self._stick_bottom = True
        self.chat_text.see("end")
        if jump_to is not None:
            self._jump_to_message(cid, int(jump_to))
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
        # v1.1.46 (#4c): the revert control is per-chat - a chat with checkpointed
        # turns shows it, one without hides it. Last so it reflects the record as
        # just loaded (and stays correct when switching to a chat mid-turn).
        self._refresh_revert_button()

    # ── windowed rendering (v1.1.36) ──────────────────────────────────
    # Rendering a whole thread is O(history): the fattest chats put 8,000+ Text
    # lines and 280+ accordions into the widget on every click (4.9-6.9s each).
    # Only the trailing window is rendered, so switching stays roughly constant
    # no matter how much history accumulates (measured: 4.90s -> 0.30s at keep=60).
    # Older entries are never dropped from the data - the "Load earlier messages"
    # button widens the view.

    def _view_is_live(self) -> bool:
        """True while a turn is streaming into the chat on screen. A full
        re-render would clobber the partial reply, so view-widening actions are
        refused until the turn ends."""
        return bool(self._busy) and self._run_chat_id == self.current_chat_id

    def _chat_render_window(self, cid: str, total: int) -> int:
        """Trailing message entries to render for this view (0 = render all).
        A per-chat override (set by 'Load earlier messages') only ever WIDENS the
        view - it never shrinks below the configured window."""
        try:
            w = int(float(self.settings.get("chat_render_window", CHAT_RENDER_WINDOW_DEFAULT)))
        except (TypeError, ValueError):
            w = CHAT_RENDER_WINDOW_DEFAULT
        w = max(0, w)
        ov = self._render_window_override.get(cid) or 0
        if ov > w:
            w = ov
        return total if (w <= 0 or w >= total) else w

    def _render_chat_messages(self, cid: str, chat: dict, msgs: List[dict]) -> None:
        """Render a chat's message entries into chat_text (was inline in load_chat).

        The compaction marker sequence is counted over the FULL list, not just the
        rendered slice, so a windowed view still pairs each marker with the right
        compaction_archive entry."""
        window = self._chat_render_window(cid, len(msgs))
        start = max(0, len(msgs) - window) if window else 0
        if start > 0:
            self._render_earlier_button(start)
        # v1.1.42: remember where each message begins in the Text widget so a search
        # hit can be scrolled to and selected. Keys are message INDICES, values are
        # Text index strings - rebuilt on every render, never reused across chats.
        marks: Dict[int, str] = {}
        _compaction_seq = 0   # 1-based index of the [COMPACTED marker seen so far (C4)
        for i, m in enumerate(msgs):
            if not isinstance(m, dict):
                continue      # corrupt entry - skip (rendering never raises)
            role = m.get("role")
            is_marker = role == "user" and str(m.get("content") or "").startswith(COMPACTION_MARKER)
            if is_marker:
                _compaction_seq += 1
            if i < start:
                # Outside the render window; the marker count still advances. The
                # drawer is rendered anyway (it is lazy, so this costs one header
                # line) - otherwise compaction would make the archived messages
                # unreachable once the marker scrolled out of the window.
                if is_marker:
                    self._render_compaction_accordion(chat, _compaction_seq, cid)
                continue
            marks[i] = self.chat_text.index("end-1c")
            if is_marker:
                self._render_compaction_accordion(chat, _compaction_seq, cid)
                continue
            if role == "user":
                self.render_user_message(m.get("content"), m.get("ts"))
            elif role == "assistant":
                content = m.get("content")
                tcs = m.get("tool_calls") or []
                if content:
                    self.render_markdown_full(content, m.get("ts"))
                for tc in tcs:
                    try:
                        a = json.loads(tc["function"].get("arguments", "{}") or "{}")
                    except Exception:
                        a = {}
                    self._ui_tool_call_accordion(tc["function"]["name"], a)
            elif role == "tool":
                self._ui_tool_output_accordion(m.get("name", "tool"), m.get("content", ""))
        self._msg_marks[cid] = marks

    def _widen_view_for(self, cid: str, total: int, idx: int) -> None:
        """Make sure message `idx` will be inside the render window (never shrinks it).

        A search hit can sit far back in a long chat; without this the jump would land
        on a message that was never rendered."""
        need = max(1, total - max(0, idx))
        cur = self._render_window_override.get(cid) or 0
        if need > cur:
            self._render_window_override[cid] = need

    def _jump_to_message(self, cid: str, idx: int) -> None:
        """Scroll to a rendered message and select it using Tk's built-in `sel` tag.

        `sel` is used deliberately: a custom tag with background= would override the
        widget's selectbackground (invariant #8) and this needs no new tag at all."""
        marks = self._msg_marks.get(cid) or {}
        at = marks.get(int(idx))
        if not at:
            return
        try:
            self.chat_text.see(at)
            # Text.search() returns a bare index string here (tkinter returns a
            # (index, count) tuple only for some argument shapes) - take the value
            # directly and never index into it, or "9.25"[0] becomes the invalid
            # index "9".
            end = self.chat_text.search("\n\n", at, "end", nocase=True)
            if isinstance(end, tuple):
                stop = end[0] if end else "end"
            else:
                stop = end or "end"
            self.chat_text.tag_remove("sel", "1.0", "end")
            self.chat_text.tag_add("sel", at, stop)
        except tk.TclError:
            pass

    def _render_earlier_button(self, hidden: int) -> None:
        """'Load earlier messages' button, inserted at the TOP of the render.
        Placed before any indexed content so the append-only index invariant that
        _links / _image_thumbs / accordion a0-a1 depend on still holds."""
        try:
            btn = tk.Button(
                self.chat_text,
                text=f"\u2191 Load earlier messages ({hidden} earlier not shown)",
                command=self._load_earlier_messages,
                bg=COL["bg_raised"], fg=COL["text_dim"],
                activebackground=COL["accent"], activeforeground="#FFFFFF",
                relief="flat", bd=0, padx=10, pady=3, cursor="hand2", font=F(11))
            self._code_buttons.append(btn)          # prevent GC (same as code buttons)
            self.chat_text.window_create("end-1c", window=btn)
            self.chat_text.insert("end", "\n")
        except Exception:
            pass

    def _load_earlier_messages(self) -> None:
        """Widen the current chat's render window and re-render.

        A full re-render (NOT an insertion at the top) is required: _links,
        _image_thumbs and every accordion store ABSOLUTE Tk indices, so inserting
        text ahead of them would invalidate all of them."""
        cid = self.current_chat_id
        if not cid or self._view_is_live():
            return
        chat = self.chats["chats"].get(cid)
        if not chat:
            return
        total = len(chat.get("messages") or [])
        cur = self._chat_render_window(cid, total)
        if cur >= total:
            return
        self._render_window_override[cid] = min(total, cur + max(cur, 60))
        self.load_chat(cid, force=True)

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

    def _ensure_search_index(self, force: bool = False) -> bool:
        """Rebuild the FTS5 index when it is missing or older than chats.json.

        Cheap enough to run inline (0.13 s on the real 20 MB history) but rate-limited
        so a burst of saves while streaming cannot rebuild it repeatedly. Returns
        False when the index is unavailable - callers then fall back to a LIKE scan."""
        now = time.time()
        if not force and not self._search_index_stale():
            return True
        # Serialised: this runs on the main thread (sidebar search) AND the chat
        # worker (search_chats tool). The rate limiter alone does not make the check
        # and the build atomic, so two threads can both pass it and race DROP/CREATE
        # on the same scratch table - one build then fails and returns False, which
        # silently degrades that caller to a LIKE scan. Holding the lock across the
        # whole check+build also means the loser reuses the winner's fresh index
        # instead of rebuilding.
        with _SEARCH_BUILD_LOCK:
            now = time.time()
            if not force and not self._search_index_stale():
                return True
            if (now - getattr(self, "_search_last_build", 0.0)) < SEARCH_INDEX_MIN_INTERVAL:
                return search_index_ready()
            self._search_last_build = now
            ok, _n = build_search_index(self.chats)
            return ok

    def _search_index_stale(self) -> bool:
        """Missing index, or built from a different chats.json mtime than the live file."""
        if not search_index_ready():
            return True
        return search_index_is_stale(self.chats)

    def _show_search_results(self, query: str, hits: List[dict]) -> None:
        """Render full-text hits as clickable rows under the search box.

        One row per chat (best snippet + hit count). Clicking opens that chat and
        scrolls to the matching message."""
        for w in getattr(self, "_search_result_widgets", []):
            try:
                if w.winfo_exists():
                    w.destroy()
            except tk.TclError:
                pass
        self._search_result_widgets = []
        if not query:
            return
        parent = self.chat_search_hits
        if not hits:
            lab = tk.Label(parent, text="No message matches found", bg=COL["bg_deep"],
                           fg=COL["text_dim"], font=F(9), justify="left", wraplength=230)
            lab.pack(anchor="w", pady=(2, 0))
            self._search_result_widgets.append(lab)
            return
        head = tk.Label(parent, text=f"{len(hits)} chat(s) mention “{query[:24]}”",
                        bg=COL["bg_deep"], fg=COL["text_dim"], font=F(9, "bold"),
                        justify="left", wraplength=230)
        head.pack(anchor="w", pady=(3, 0))
        self._search_result_widgets.append(head)
        # Capped hard: the chat Listbox is the expand=True widget in this sidebar, so
        # an unbounded result list would squeeze it out of the window entirely.
        shown = hits[:SEARCH_SIDEBAR_HITS]
        for h in shown:
            cid, idx = h["chat_id"], h["idx"]
            kind = str(h.get("kind") or "live")
            archived = kind.startswith("archive:")
            title = h["title"][:34]
            extra = f" · {h['hits']} hits" if h.get("hits", 1) > 1 else ""
            # An archived hit's idx addresses the compaction event, NOT chat["messages"].
            # Jumping to it would scroll to an unrelated live message and look like a
            # correct answer, so the row is labelled and the jump is suppressed.
            src = (f"archived (compaction {kind.split(':', 1)[1]})" if archived else "live")
            row = tk.Frame(parent, bg=COL["bg_main"], cursor="hand2")
            tk.Label(row, text=f"{title}{extra}", bg=COL["bg_main"], fg=COL["accent"],
                     font=F(10, "bold"), anchor="w", justify="left",
                     wraplength=230).pack(fill="x", padx=4, pady=(2, 0))
            tk.Label(row, text=(h["snippet"][:SEARCH_SIDEBAR_SNIPPET] or "(no text)"),
                     bg=COL["bg_main"], fg=COL["text_dim"], font=F(9), anchor="w",
                     justify="left", wraplength=230).pack(fill="x", padx=4, pady=(0, 2))
            if archived:
                tk.Label(row, text=f"↳ {src} - open the 📦 drawer in the chat to read it",
                         bg=COL["bg_main"], fg=COL["text_dim"], font=F(8, "italic"),
                         anchor="w", justify="left",
                         wraplength=230).pack(fill="x", padx=4, pady=(0, 2))

            def open_hit(_e=None, cid=cid, idx=idx, archived=archived) -> None:
                self._clear_chat_search()
                # jump_to only for live hits (see above).
                self.load_chat(cid, force=True,
                               jump_to=(None if archived else idx))

            for w in (row,) + tuple(row.winfo_children()):
                w.configure(cursor="hand2")
                w.bind("<Button-1>", open_hit)
            row.pack(fill="x", pady=(0, 3))
            self._search_result_widgets.append(row)
        if len(hits) > len(shown):
            more = tk.Label(parent, text=f"… {len(hits) - len(shown)} more (keep typing to narrow)",
                            bg=COL["bg_deep"], fg=COL["text_dim"], font=F(9),
                            justify="left", wraplength=230)
            more.pack(anchor="w", pady=(0, 2))
            self._search_result_widgets.append(more)

    def _clear_search_results(self) -> None:
        self._show_search_results("", [])

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
        """Filter the sidebar by title AND search message bodies (v1.1.42).

        The title filter is instant and always applied. The BODY search is debounced:
        the first lookup after chats.json changes rebuilds the index (~0.2 s on the
        real 20 MB history), and firing that on every keystroke would stutter while
        typing. It runs once the user pauses, and only past a 2-character query."""
        try:
            q = self._chat_search_query().lower()
            self._refresh_chat_list()
            if self._search_debounce is not None:
                try:
                    self.root.after_cancel(self._search_debounce)
                except Exception:
                    pass
                self._search_debounce = None
            if not q:
                self.chat_search_results.configure(text="")
                self._clear_search_results()
                return
            self.chat_search_results.configure(
                text=f"{len(self._chat_visible)} of {len(self.chats['order'])} chats by title")
            if len(q) < 2:
                self._clear_search_results()
                return
            self._search_debounce = self.root.after(SEARCH_BODY_DEBOUNCE_MS,
                                                   lambda qq=q: self._run_body_search(qq))
        except tk.TclError:
            pass
        except Exception as e:
            try:
                self.chat_search_results.configure(text=f"Search error: {str(e)[:80]}")
            except tk.TclError:
                pass

    def _run_body_search(self, q: str) -> None:
        """The debounced half of _on_chat_search: index lookup + result rows."""
        self._search_debounce = None
        try:
            # The query may have changed (or been cleared) during the debounce wait.
            if q != self._chat_search_query().lower():
                return
            if not self._ensure_search_index():
                self._show_search_results(q, _search_group_by_chat(
                    _search_chats_scan(self.chats, q, SEARCH_MAX_SNIPPETS)))
                return
            self._show_search_results(q, _search_group_by_chat(
                search_chats(self.chats, q, SEARCH_MAX_SNIPPETS)))
        except tk.TclError:
            pass
        except Exception as e:
            try:
                self.chat_search_results.configure(text=f"Search error: {str(e)[:80]}")
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

    def render_user_message(self, content: Any, ts: Optional[str] = None) -> None:
        self.chat_text.insert("end", "\n")
        self.chat_text.insert("end", "You", ["user_hdr"])
        t = fmt_msg_time(ts)
        if t:
            self.chat_text.insert("end", f"  {t}", ["dim"])
        self.chat_text.insert("end", "\n")
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

    def render_markdown_full(self, text: str, ts: Optional[str] = None) -> None:
        """Render a complete markdown document (used when replaying history)."""
        self.chat_text.insert("end", "\n")
        self.chat_text.insert("end", APP_NAME, ["asst_hdr"])
        t = fmt_msg_time(ts)
        if t:
            self.chat_text.insert("end", f"  {t}", ["dim"])
        self.chat_text.insert("end", "\n")
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
        self.chat_text.insert("end", APP_NAME, ["asst_hdr"])
        _now_s = datetime.now().strftime("%b %d, %Y - %H:%M:%S")
        self.chat_text.insert("end", f"  {_now_s}", ["dim"])
        self.chat_text.insert("end", "\n")
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
        # An archive drawer's open/closed state also lives in _archive_expanded
        # (which decides whether the body is RENDERED on the next load). Collapsing
        # here is elide-only, so the tracker must be updated too - otherwise the
        # drawer would "forget" it was collapsed and re-render expanded on return.
        arch = info.get("archive")
        if arch:
            _ac, _an = arch
            _opened = self._archive_expanded.setdefault(_ac, set())
            if expanded:
                _opened.add(_an)
            else:
                _opened.discard(_an)
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

    def _render_compaction_accordion(self, chat: dict, compaction_number: int,
                                     cid: Optional[str] = None) -> None:
        """Render a '📦 Context compacted' accordion for one in-place compaction (C4).

        The archived original messages are shown verbatim (role-labelled plain text) inside an
        elidable body so the user can read every word of what was summarized - the same access
        they have to any other chat. Collapsed by default. No tag here sets a background (that
        would hide text selection - see invariants). Plain text rather than nested accordions:
        nesting live code-copy buttons / tool accordions inside an elided region would shift the
        stored Tk indices those widgets rely on.

        v1.1.36 LAZY BODY: the body is NOT inserted unless this drawer has been opened.
        These archives hold up to ~10,000 hidden Text lines (4.7MB of stored archive;
        one chat rendered 6,447 lines nobody ever saw), which dominated load time
        (compaction deskpilot-3: 1.55s -> 0.12s once the body became lazy). Because the
        accordion sits mid-buffer, its body cannot be inserted on click without shifting
        the absolute indices in _links / _image_thumbs / accordion a0-a1, so expanding
        re-renders the whole view instead (see _toggle_archive_expansion)."""
        archive = chat.get("compaction_archive") or []
        entry = next((a for a in archive if isinstance(a, dict)
                      and int(a.get("compaction_number") or 0) == int(compaction_number)), None)
        if not entry:
            return
        old_msgs = entry.get("messages") or []
        ts = str(entry.get("timestamp") or "")[:16].replace("T", " ")
        title = f"\U0001F4E6 Context compacted \u00b7 {len(old_msgs)} messages summarized" + (f" ({ts})" if ts else "")
        expanded = compaction_number in self._archive_expanded.get(cid or "", set())
        sid = self._add_accordion(title, COL["text_dim"], expanded=expanded, font=F(11))
        if cid:
            # Marks this drawer as archive-backed so _toggle_accordion keeps
            # _archive_expanded in sync when the body is collapsed by elide.
            self._accordions[sid]["archive"] = (cid, compaction_number)
        if not expanded:
            # Body not rendered: clicking re-renders the view WITH the body.
            try:
                self.chat_text.tag_bind(
                    sid + "_hdr", "<Button-1>",
                    lambda e, c=cid, n=compaction_number: self._toggle_archive_expansion(c, n))
            except tk.TclError:
                pass
            return
        body_tag = sid + "_body"
        for m in old_msgs:
            if not isinstance(m, dict):
                continue
            role = m.get("role", "?")
            content = str(m.get("content") or "")
            if role == "assistant":
                tcs = m.get("tool_calls") or []
                calls = "; ".join(
                    f"{((tc.get('function') or {}) if isinstance(tc, dict) else {}).get('name', '?')}(...)"
                    for tc in tcs)
                label = "[assistant]" + (f" \u2192 ran: {calls}" if calls else "")
            elif role == "tool":
                label = f"[tool:{m.get('name') or 'tool'}]"
            else:
                label = "[user]"
            self.chat_text.insert("end", label + "\n", [body_tag, "dim"])
            if content.strip():
                self.chat_text.insert("end", content.rstrip() + "\n", [body_tag])
        self._autoscroll()

    def _toggle_archive_expansion(self, cid: Optional[str], compaction_number: int) -> None:
        """Open/close a compaction drawer by re-rendering the view (never by
        inserting mid-buffer, which would invalidate stored Tk indices)."""
        if not cid or self._view_is_live():
            return
        opened = self._archive_expanded.setdefault(cid, set())
        if compaction_number in opened:
            opened.discard(compaction_number)
        else:
            opened.add(compaction_number)
        self.load_chat(cid, force=True)

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
        # v1.1.60: flush rather than drop. This used to assign None outright, which
        # silently discarded MarkdownStream.pending - the last line of a step that
        # had no trailing newline. A model that says "I'll search for that now"
        # and immediately calls a tool had that sentence vanish from the transcript
        # (it was still in history, and reappeared on re-render, but the live view
        # lied about what the model said). The primary fix is the explicit flush at
        # the end of the stream; this is the safety net for any path that reaches a
        # new step with a stream still open. Late is better than lost.
        self._ui_finish_active_md()
        self._reasoning_sid = None
        self._pending_tool_output = None

    def _ui_finish_active_md(self) -> None:
        """MAIN THREAD. Flush the active markdown stream's pending partial line and
        close any open code fence, then release it.

        MarkdownStream only renders COMPLETE lines (so partial tokens never produce
        broken formatting), which means the final line of a step sits in .pending
        until finish() is called. Every path that ends a step must call this, or
        that line is never shown.
        """
        if self._active_md is None:
            return
        try:
            self._active_md.finish()
        except tk.TclError:
            pass      # widget gone (chat reloaded / app closing): nothing to flush into
        self._active_md = None

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

    def _render_steered_message(self, content: Any) -> None:
        """MAIN THREAD. Show a mid-run steering message in the transcript.

        ⚠️ Guarded by _viewing_run_chat(): the user may have switched to another chat
        while the turn runs, and render_note()/render_user_message() insert into
        self.chat_text unconditionally - unguarded, a steered message would be painted
        into an unrelated conversation. Same guard _ui_stream_chunk() uses.

        Any open assistant block is finished first so a steered message cannot land
        inside an unterminated code fence. Safe for stored indices: everything here
        inserts at "end", so nothing in _links / _image_thumbs / accordion a0-a1 shifts
        (the same reason render_note() is safe)."""
        if self._closed or not self._viewing_run_chat():
            return
        self._ui_finish_active_md()
        self.render_note("\u2699 Steered mid-run")
        self.render_user_message(content)

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
        self._ui_finish_active_md()
        self._finish_turn_ui()
        self._maybe_trigger_handoff()   # auto-summary once context usage crosses the threshold
        if self.settings.get("tts_enabled"):
            self.speak(full_content)

    def _ui_turn_stopped(self) -> None:
        """Finish the UI after a user-initiated stop.

        Like _ui_turn_complete() but without TTS (no point reading an aborted
        reply aloud) and without the handoff check (the context did not grow)."""
        self._ui_finish_active_md()
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
        # v1.1.46 (#4c): a turn that just wrote files makes this chat revertable, and
        # _busy just went False so the button must leave the disabled state.
        self._refresh_revert_button()

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

    def _steer_enqueue(self, content: Any) -> int:
        """MAIN THREAD (#10 mid-run steering): hold a message for the next step
        boundary. Returns the number now queued, or -1 if the queue is full.

        Deliberately does NOT touch the chat record: the worker's per-step
        `_ui_sync_messages()` REPLACES chat["messages"] with its own local list, so a
        main-thread append here would be silently overwritten. The worker is the only
        writer of the running turn's message list - same discipline as compaction."""
        with self._steer_lock:
            if len(self._steer_queue) >= MAX_STEER_QUEUE:
                return -1
            self._steer_queue.append(content)
            return len(self._steer_queue)

    def _steer_drain(self) -> List[Any]:
        """WORKER THREAD: take everything queued, atomically. Called only at a step
        boundary, never mid-tool-batch (inserting a user message between an assistant
        tool_calls message and its tool results breaks tool_call_id pairing)."""
        with self._steer_lock:
            out, self._steer_queue = list(self._steer_queue), []
        return out

    def _steer_clear(self) -> None:
        """Drop any queued steering. Called at turn START so a message left over from
        an aborted turn can never leak into the next one."""
        with self._steer_lock:
            self._steer_queue = []

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
        if OpenAI is None:
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

        # ── #10 mid-run steering: while a turn is running, queue instead of refuse ──
        # The message is injected into the conversation at the NEXT step boundary by
        # the worker (never mid-tool-batch). It does not restart the turn and does not
        # reset the step budget or the wall-clock deadline.
        # ⚠️ Only the chat that OWNS the running turn can be steered. If the user is
        # looking at a different chat, queueing there would inject into a conversation
        # they are not watching (and the worker only ever drains for its own chat), so
        # the old refuse-to-send behaviour is kept for that case.
        if self._busy and self._viewing_run_chat():
            n = self._steer_enqueue(content)
            if n < 0:
                self._set_status(f"⚙ Steering queue is full ({MAX_STEER_QUEUE}) - "
                                 "wait for the current step or press Stop")
                return
            self.input_text.delete("1.0", "end")
            self._attachments.clear()
            self._render_chips()
            self._set_status(f"⚙ Queued for the next step ({n} waiting)")
            return
        if self._busy:
            self._set_status("⚙ A turn is already running in another chat - "
                             "open that chat to steer it, or press Stop")
            return

        chat = self.current_chat()
        if chat is None:                        # fresh launch state: no thread selected yet
            self.new_chat()                    # start a brand-new conversation for this message
            chat = self.current_chat()
        _ts = datetime.now().isoformat(timespec="seconds")
        chat["messages"].append({"role": "user", "content": content, "ts": _ts})

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
        self.render_user_message(content, _ts)

        self._busy = True
        self._stop_event.clear()
        self._steer_clear()      # never carry a stale steering message into a new turn
        # The Send button becomes the Stop button for the duration of the turn:
        # it must stay ENABLED so the user can click it to abort.
        self.send_btn.configure(text="⏹ Stop", bg=COL["danger"], activebackground="#B91C1C",
                                command=self.stop_generation)
        self.speed_label.configure(text="⚡ …")
        self._refresh_revert_button()   # v1.1.46: disable revert while the turn runs
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
        # Queued steering is moot once the user has asked to stop: drop it now so it
        # cannot be injected on the way out and cannot leak into the next turn.
        self._steer_clear()
        # Tools now poll this event, so the turn unwinds within STOP_POLL_S; say what is
        # happening while they do (a killed node/CLI subprocess takes a moment to die).
        self._set_status("⏹ Stopping… (aborting any running tool)")

    def _client_for(self, url: str, key: str) -> Optional[Any]:
        """Cached OpenAI client for an explicit (url, key) pair.

        Split out of _get_client so a SECOND endpoint (the ask_expert consultant)
        can have its own client without touching the primary one. The cache is
        keyed on the pair, so the two never collide and each keeps its own warm
        connection pool."""
        if OpenAI is None:
            return None
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

    def _get_client(self) -> Optional[Any]:
        """Return a cached OpenAI client for the current server/key.

        Reusing one client across turns keeps its HTTP connection pool warm
        (keep-alive), so each request skips the TCP+TLS handshake."""
        if OpenAI is None:
            return None
        url = (self.settings.get("server_url") or "").strip()
        key = (self.settings.get("api_key") or "").strip() or "sk-local"
        return self._client_for(url, key)

    def _invalidate_clients(self) -> None:
        """Drop cached OpenAI clients (called when server URL / API key change)."""
        for c in self._clients.values():
            try:
                c.close()
            except Exception:
                pass
        self._clients.clear()

    # ── ask_expert: the consultant ────────────────────────────────────
    def _expert_root(self) -> Path:
        """The one directory the consultant may read from.

        file_workspace when it is set (that IS the user's working directory);
        otherwise USER_WORKSPACE_ROOT if it exists on this machine, else BASE_DIR
        - the same resolution order the handoff-notes path uses, so a copy of
        this script on any other machine gets a real, sensible root instead of a
        hardcoded path that does not exist (which would refuse every file and
        print someone else's username in the error).

        NOT an intersection: requiring both would refuse every file whenever
        file_workspace points somewhere other than under USER_WORKSPACE_ROOT.
        The point is that there is ALWAYS a confinement root, even when the
        optional setting is blank - unlike _safe_path, which goes unrestricted
        in that case."""
        ws_raw = str((self.settings or {}).get("file_workspace") or "").strip()
        if ws_raw:
            base = Path(os.path.expanduser(ws_raw))
        elif USER_WORKSPACE_ROOT.is_dir():
            base = USER_WORKSPACE_ROOT
        else:
            base = BASE_DIR
        return Path(base).resolve()

    def _expert_safe_path(self, raw_path: str) -> Path:
        """Resolve a path for the consultant, ALWAYS confined to the workspace.

        Deliberately does NOT rely on the optional File Workspace setting being
        set: _safe_path confines only when file_workspace is set, and blank
        means unrestricted for the other tools. A tool that ships local file
        contents to a third party must not inherit that optionality, so the
        floor here is _expert_root() and it is never absent.

        _safe_path already resolves symlinks, so a link inside the workspace
        pointing outside resolves outside and is then refused by the prefix
        check - that case is covered, not assumed."""
        p = self._safe_path(raw_path)          # traversal + system-dir + optional jail
        try:
            root = self._expert_root()
        except Exception as e:
            raise PermissionError(f"cannot resolve the workspace root: {e}")
        p_str, r_str = str(p).lower(), str(root).lower()
        if p_str != r_str and not p_str.startswith(_dir_prefix(r_str)):
            raise PermissionError(
                f"ask_expert may only read inside the workspace ({root}): {p}")
        return p

    def _expert_read_file(self, spec: dict) -> tuple:
        """(label, text) for one requested file; text starts with DENIED: if refused.

        Char offsets mirror read_local_file so a region can be targeted instead
        of shipping a whole 700 KB file to an outside model. Reading from disk
        rather than pasting code into the prompt is deliberate: a small model
        retyping code misquotes it (this session produced a citation to a test
        file that never existed), and the expert would then review something
        that is not real."""
        raw = str((spec or {}).get("path") or "")
        if not raw.strip():
            return "(empty path)", "DENIED: no path given"
        try:
            p = self._expert_safe_path(raw)
        except PermissionError as e:
            return Path(raw).name or raw, f"DENIED: {e}"
        if not p.is_file():
            return p.name, "DENIED: not a file"
        try:
            body = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return p.name, f"DENIED: could not read ({e})"
        try:
            off = max(0, int(spec.get("offset") or 0))
        except (TypeError, ValueError):
            off = 0
        try:
            lim = int(spec.get("limit") or 0)
        except (TypeError, ValueError):
            lim = 0
        if lim <= 0 or lim > EXPERT_FILE_CHARS:
            lim = EXPERT_FILE_CHARS
        chunk = body[off:off + lim]
        # Label relative to the workspace root: the expert needs to know WHICH
        # file it is looking at, but an absolute path also hands it the OS
        # username and the whole directory layout, which is not part of the
        # question. Fall back to the name if the path is not under the root.
        try:
            rel = p.relative_to(self._expert_root())
        except Exception:
            rel = Path(p.name)
        label = f"{rel} (chars {off}-{off + len(chunk)} of {len(body)})"
        if off + len(chunk) < len(body):
            chunk += f"\n\n[... truncated: {len(body) - off - len(chunk)} more chars not sent]"
        return label, chunk

    def _expert_payload(self, question: str, files, context: str, focus: str) -> list:
        """Build the consultant's messages FROM SCRATCH.

        This is the isolation guarantee, so it is worth being blunt about why it
        does not reuse build_system_message(): that function injects the ledger,
        the skills index, the custom system prompt and the workspace path, and
        the main loop passes them in. Reusing it would ship the user's durable
        notes about their machine to a third party. The consultant gets one
        hardcoded line and nothing else - no history, no persona, no tools.
        """
        _FOCUS = {
            "review":          "Review the code for bugs, unsafe assumptions and API calls that may not exist.",
            "debug":           "Diagnose the failure and say what to check next.",
            "design":          "Compare the approaches and recommend one with reasons.",
            "second_opinion":  "Give an independent view; disagree if the current approach is wrong.",
        }
        sys_txt = (
            "You are a consulting engineer answering ONE narrow question. You have no "
            "tools, no filesystem access and no access to any conversation - everything "
            "you know is in this message. Be concrete and specific. If you name a "
            "function, method or signature, state the library and version you believe "
            "provides it, and say so plainly if you are unsure it exists. "
            "Answer directly; do not ask for more context.\n\n"
            + _FOCUS.get(str(focus or "review"), _FOCUS["review"])
        )
        parts = ["QUESTION:\n" + str(question or "").strip()]
        ctx = str(context or "").strip()
        if ctx:
            parts.append("CONTEXT (supplied verbatim by the asking model):\n" + ctx)
        denials, sent = [], 0
        for spec in (files or []):
            if isinstance(spec, str):
                spec = {"path": spec}
            label, text = self._expert_read_file(spec)
            if text.startswith("DENIED:"):
                denials.append(f"{label}: {text}")
            else:
                sent += 1
                parts.append(f"FILE {label}:\n```\n{text}\n```")
        return [{"role": "system", "content": sys_txt},
                {"role": "user", "content": "\n\n".join(parts)}], denials, sent

    def _tool_ask_expert(self, question: str = "", files=None, context: str = "",
                         focus: str = "review") -> str:
        """Ask the consultant. Single-shot, no tools, sees only what is named.

        Guards, each load-bearing:
          * per-turn call limit (a small model over-delegates; five 60-second
            thinking calls in one turn is a very long wait)
          * no `tools` kwarg, and any tool_calls in the reply are DISCARDED -
            giving the expert the tool schemas would start a second agent loop
            with only one of them under the permission system
          * empty-content retry: a thinking model can spend the entire cap on
            reasoning and return content='' (v1.1.58). Returning that blank is
            the worst possible failure, because it happens on hard questions.
        """
        url = str(self.settings.get("expert_server_url") or "").strip()
        key = str(self.settings.get("expert_api_key") or "").strip()
        model = str(self.settings.get("expert_model") or "").strip()
        # All three are required. Checking only url+key let a blank model through
        # and sent model="" - verified against the live endpoint: it answers
        # 400 "model is not specified", which reads to the model as a broken
        # expert rather than an unconfigured one.
        if not url or not key or not model:
            return ("ERROR: ask_expert is not configured. Set Expert Server URL, "
                    "Expert API Key and Expert Model in Settings "
                    + ("(Expert Model is blank)" if (url and key and not model) else "") + ".")
        if not str(question or "").strip():
            return "ERROR: ask_expert needs a question."

        used = int(getattr(self, "_expert_calls_this_turn", 0) or 0)
        if used >= EXPERT_MAX_CALLS_PER_TURN:
            return (f"ERROR: ask_expert limit reached ({EXPERT_MAX_CALLS_PER_TURN} "
                    f"per turn). Work from what you already have.")

        msgs, denials, sent_files = self._expert_payload(question, files, context, focus)
        # Files were named but every one was refused, and there is no context to
        # fall back on: do not spend a consult shipping a prompt whose only
        # content is a list of denials. Say so and return WITHOUT charging the
        # budget - nothing left the machine, so it is not one of the five.
        if denials and not sent_files and not str(context or "").strip():
            return ("ERROR: no files could be sent to the expert:\n  "
                    + "\n  ".join(denials))
        # Total-size clamp: the payload is built from user-named files, and one
        # huge file would otherwise ride out unbounded.
        total = sum(len(str(m.get("content") or "")) for m in msgs)
        if total > EXPERT_TOTAL_CHARS:
            return (f"ERROR: payload too large for one consult ({total} chars; "
                    f"limit {EXPERT_TOTAL_CHARS}). Narrow the file ranges or ask "
                    f"about fewer files at once.")

        # Charged only once a request is actually going out.
        self._expert_calls_this_turn = used + 1

        client = self._client_for(url, key)
        if client is None:
            return "ERROR: could not create the expert client (is 'openai' installed?)"

        label = f"{model} @ {url.split('//')[-1].rstrip('/')}"
        kw = dict(model=model, messages=msgs, stream=False,
                  max_tokens=EXPERT_MAX_TOKENS)
        # NO tools / tool_choice here. That is the whole safety property.
        txt, usage, err = _probe_create(client, kw, EXPERT_TIMEOUT_S)
        # Gemini's /models lists IDs as "models/<name>" but its own chat examples
        # send the bare "<name>". Which one the endpoint accepts was not
        # verifiable here (the account was rate-limited), so try the other form
        # once on a model error rather than making the user guess. A 429 is NOT a
        # model error and must not trigger this.
        if err and _is_model_lookup_error(err):
            alt = model[7:] if model.startswith("models/") else "models/" + model
            if alt and alt != model:
                kw_alt = dict(kw)
                kw_alt["model"] = alt
                t2, u2, e2 = _probe_create(client, kw_alt, EXPERT_TIMEOUT_S)
                if not e2 or not _is_model_lookup_error(e2):
                    txt, usage, err = t2, u2, e2
                    model = alt
                    label = f"{alt} @ {url.split('//')[-1].rstrip('/')}"
        # Thinking-model recovery: empty content but the model clearly worked.
        if not err and not txt.strip():
            kw2 = dict(model=model, messages=msgs, stream=False)
            txt, usage, err = _probe_create(client, kw2, EXPERT_TIMEOUT_S)
            if not err and not txt.strip():
                return (f"ERROR: {label} returned an empty answer twice (it may have "
                        f"spent the whole budget thinking). Narrow the question or "
                        f"send less code.")
        if err:
            return f"ERROR: expert call failed ({label}): {_probe_first_line(err)}"
        if denials:
            # Partial refusal: some files went, others did not. The expert must
            # be told which are missing or it will review an incomplete picture
            # and say nothing about it.
            txt = (txt.rstrip() + "\n\n[Note: these files were requested but NOT "
                   "sent to the expert, so the answer cannot cover them:\n  "
                   + "\n  ".join(denials) + "]")

        self._post(lambda t=txt, l=label, n=used + 1:
                   self._render_expert_note(t, l, n))
        return (f"EXPERT CONSULT #{used + 1}/{EXPERT_MAX_CALLS_PER_TURN} ({label}) - "
                f"advisory only, verify anything it names:\n\n{txt.strip()}")

    def _render_expert_note(self, text: str, label: str, n: int) -> None:
        """MAIN THREAD. Show the consult as its own accordion, so it is always
        visible which parts of the reasoning came from this machine and which
        came from a third party.

        Guarded by _viewing_run_chat(): the user may have switched chats while
        the consult was in flight, and painting into a chat that is no longer
        open corrupts the wrong transcript (same reason _render_steered_message
        is guarded)."""
        if not self._viewing_run_chat():
            return
        body = (text or "").strip() or "(empty answer)"
        title = f"\U0001f9e0 {label} - consulted (#{n})"
        sid = self._add_accordion(title, COL["accent"], expanded=False)
        self.chat_text.insert("end", body + "\n", [sid + "_body", "code_block"])
        self._autoscroll()

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
            # v1.1.52 (#10): the ONE choke point every turn passes through, so the
            # steering queue can never carry a stale message into a new turn - a
            # per-caller clear would be missed by any future turn-start path.
            # v1.1.68: INSIDE the try. It used to sit above it, so an exception
            # here (a missing _steer_lock, a future refactor) escaped the whole
            # function with _busy stuck True - the exact wedge this net exists to
            # prevent, reachable only because the guarded call was outside the guard.
            self._steer_clear()
            self._chat_worker_inner(chat_id, messages)
        except Exception as e:
            traceback.print_exc()
            self._post(lambda m=str(e): self.render_error(
                f"Internal error during generation:\n{m}"))
            self._post(self._finish_turn_ui)
        finally:
            # v1.1.45 (#4b): worker-thread-only state, cleared on EVERY exit path
            # (normal, Stop, and the escaping-exception net above). The manifest of
            # the turn that just ended is already posted/persisted; only the
            # in-memory namespace closes.
            # v1.1.68: guarded. An exception raised HERE replaces the original one
            # and escapes the thread - and because it runs in `finally` it would do
            # so even on the clean path. _busy is already cleared by then, so it
            # cannot wedge the UI, but it would bury the real error. Never let
            # cleanup noise mask the failure that started the turn.
            try:
                self._end_turn_checkpoints(chat_id)
            except Exception:
                traceback.print_exc()

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
        # v1.1.45 (#4b): open this prompt's checkpoint namespace. Allocated here (not in
        # send_message) because the worker is the only thing that always runs for a turn,
        # including the handoff/retry paths. Pruning happens at TURN START, never at turn
        # end, so a turn that fails midway cannot delete anything.
        self._begin_turn_checkpoints(chat_id)
        # ask_expert's per-turn budget resets here, not in send_message: the
        # worker is the only thing that always runs for a turn (the same reason
        # the checkpoint namespace is opened here).
        self._expert_calls_this_turn = 0
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
            # ── #10 mid-run steering: drain the queue at the STEP BOUNDARY ──────
            # This is the only legal injection point. It is BEFORE the request is
            # built (so the model sees it on this step) and AFTER the previous step's
            # assistant+tool messages were appended (so a user message never lands
            # between an assistant tool_calls message and its tool results, which
            # would break tool_call_id pairing and is rejected by strict servers).
            # The worker is the sole writer of this list - _ui_sync_messages() later
            # REPLACES chat["messages"] with it, so the main thread must not append.
            # Injecting does NOT reset `step` or `turn_deadline`: steering is guidance
            # within the same turn, not a new turn.
            if not self._stop_event.is_set():
                _steered = self._steer_drain()
                for _sc in _steered:
                    messages.append({"role": "user", "content": _sc,
                                     "ts": datetime.now().isoformat(timespec="seconds"),
                                     "steered": 1})
                    # Persist + render. Posted, never inline (self.chats is main-thread
                    # only, invariant #4). Snapshot the list so the callback cannot see
                    # later worker mutations.
                    self._post(lambda cid=chat_id, m=list(messages):
                               self._ui_sync_messages(cid, m))
                    self._post(lambda c=_sc: self._render_steered_message(c))
                if _steered:
                    # Otherwise the "N waiting" status lingers for the rest of the turn
                    # even though the message has already been applied.
                    self._post(lambda: self._set_status(
                        "\u2699 Steering applied \u2014 continuing this turn"))
            # ── context compaction (C8 / Option A): pre-request check at turn start ──
            # The end-of-iteration gate below only fires AFTER a tool phase, so a purely
            # conversational turn (no tools) never reaches it - and yet the context still
            # grows across turns. Check here, BEFORE the first request of this prompt: if the
            # outgoing history already crosses the compaction threshold, summarize it in place
            # now so the request goes out smaller. Gate on the LARGER of the previous turn's
            # real server figure (_ctx_used_last) and a char estimate of this outgoing list -
            # the estimate alone omits the system prompt + tool schemas, so it understates a long
            # conversational session and the gate would never trip (C10). _ctx_window() is 0 until
            # /models reports context_length, in which case _maybe_compact_context declines
            # (total <= 0) and we proceed exactly as before. Only at step == 0: mid-turn growth is
            # already handled by the end-of-iteration gate, so this must not double-fire within a turn.
            if step == 0:
                self._maybe_compact_context(
                    chat_id, max(estimate_prompt_tokens(messages), self._ctx_used_last()),
                    self._ctx_window(), messages)
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
                                   effective_custom_prompt(self.settings),
                                   _skills_index_text(self.settings),
                                   self._ledger_inject(chat_id),
                                   _project_memory_text(self.settings))]
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
                    thinking_ok = False           # server rejects the extension -> retry: kwargs is
                    continue                        # rebuilt at the top of every step and gated by
                                                    # this flag (the old in-place pop was inert)
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
                # Always close the SSE stream when this step's loop exits (Stop, app
                # close, normal end, or exception). A bare `break` on Stop left the
                # connection half-consumed in the keep-alive pool: a single-slot server
                # like Strata never learns the client left, keeps generating into an
                # unread socket, and the NEXT request queues behind that orphaned
                # generation. The worker then blocks inside create() before it can
                # honour Stop, so _busy stays True and the app looks dead until restart.
                # Closing the stream aborts the in-flight generation and frees the slot.
                try:
                    stream.close()
                except Exception:
                    pass

            if self._stop_event.is_set():
                # Stopped mid-stream: keep whatever was generated so far (the
                # partial reply is persisted as a normal assistant message, so
                # the next turn can continue from it), skip tool execution.
                if full_content.strip():
                    messages.append({"role": "assistant", "content": full_content,
                                     "ts": datetime.now().isoformat(timespec="seconds")})
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

            # ── FLUSH the streamed text before anything else renders ────────
            # v1.1.60: MarkdownStream holds the final line in .pending until
            # finish() is called. Without this, a step that ends in a tool call
            # never flushes: the model's "I'll search for that now" disappears
            # from the transcript while the tool accordions render in its place.
            # It must happen HERE, not at the next step's reset, so the text lands
            # BEFORE the tool call/output accordions and in the order the model
            # actually produced it. Safe on the final-answer branch too -
            # _ui_turn_complete's own flush then becomes a no-op.
            self._post(self._ui_finish_active_md)

            if not tool_calls:
                # ── final answer → persist + finish ───────────────────────
                messages.append({"role": "assistant", "content": full_content,
                                 "ts": datetime.now().isoformat(timespec="seconds")})
                self._post(lambda cid=chat_id, m=list(messages): self._ui_sync_messages(cid, m))
                if not full_content.strip():
                    self._post(lambda: self.render_note("(model returned an empty response)"))
                self._post(lambda c=full_content: self._ui_turn_complete(c))
                return

            # ── tool phase: record assistant msg, execute each call ───────
            tool_phase_start = len(messages)   # rollback point if the user stops mid-phase
            messages.append({"role": "assistant", "content": full_content or None,
                             "tool_calls": tool_calls,
                             "ts": datetime.now().isoformat(timespec="seconds")})

            for tc in tool_calls:
                if self._closed or self._stop_event.is_set():
                    break
                name = tc["function"]["name"]
                args_raw_dispatch = tc["function"].get("arguments") or "{}"
                args, args_err = parse_tool_arguments(args_raw_dispatch)
                if args_err:
                    # Malformed (or wrongly-shaped) tool-call arguments, common
                    # with local models on large payloads: never silently run the
                    # tool with {} - that executes a no-op and reads as "the tool
                    # produced no output".
                    self._post(lambda n=name: self._ui_tool_call_accordion(n, {"_malformed_arguments": True}))
                    if args_err == "malformed":
                        err = (f"ERROR: The {name} tool call had malformed JSON arguments and "
                               f"could not be parsed - it was NOT executed. Re-issue the call "
                               f"with valid JSON (keep large payloads like code concise).")
                    else:
                        err = (f"ERROR: The {name} tool call's arguments were a "
                               f"{args_err}, not a JSON object - it was NOT executed. "
                               f"Re-issue the call as an object of named parameters, "
                               f"e.g. {{\"question\": \"...\"}}.")
                    messages.append({"role": "tool", "tool_call_id": tc["id"],
                                     "name": name, "content": err})
                    self._post(lambda n=name, r=err: self._ui_tool_output_accordion(n, err))
                    continue
                self._post(lambda n=name, a=args: self._ui_tool_call_accordion(n, a))
                self._post(lambda s=f"🛠 Running {name}…": self._set_status(s))
                self._post(lambda n=name: self._ui_tool_output_start(n))

                _mcp_imgs: List[dict] = []
                self._pending_mcp_images = []      # drain buffer for this call only
                result = self._execute_tool(name, args, images_out=_mcp_imgs)   # may block on permission modal
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "name": name, "content": result})
                self._post(lambda n=name, r=result: self._ui_tool_output_accordion(n, r))
                # A tool handler that produced its own vision message (read_mcp_resource
                # on an image resource) queues it here; the loop appends it AFTER the
                # tool result so the ordering the server expects is preserved.
                if self._pending_mcp_images:
                    messages.extend(self._pending_mcp_images)
                    self._pending_mcp_images = []

                if name == "capture_screen" and result.startswith("OK"):
                    vision = self._build_vision_message(result)
                    if vision is not None:
                        messages.append(vision)
                        self._post(lambda: self.render_note("📸 Screenshot queued for vision analysis"))
                if _mcp_imgs:
                    # MCP servers can return image content (browser / computer-use
                    # servers live by it). The tool message itself must stay a plain
                    # string - a tool result carrying image parts is not portable
                    # across servers - so the pixels ride in a follow-up user message,
                    # the same shape capture_screen already uses.
                    vision = self._build_mcp_vision_message(name, _mcp_imgs)
                    if vision is not None:
                        messages.append(vision)
                        self._post(lambda n=name: self.render_note(
                            f"\U0001F5BC Image from {n} queued for visual analysis"))

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
            # ── context compaction (C5): try to free space IN PLACE before forking ──
            # If the gauge crossed the compaction threshold and there is enough old history,
            # summarize it in place and CONTINUE the turn - no fork, no button, no user
            # action. When _maybe_compact_context() declines (feature off / below threshold /
            # no measurable window / too few messages / per-chat cap reached) it returns False
            # with NO mutation, so the existing handoff gate below runs exactly as before.
            if not self._maybe_compact_context(chat_id, _gate_used, self._ctx_window(), messages):
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

    def _build_mcp_vision_message(self, tool_name: str, images: List[dict]) -> Optional[dict]:
        """Turn an MCP tool's image content into a queued vision message.

        Bytes are decoded, downscaled and saved under GEN_DIR, then referenced as
        lightweight 'image_ref' parts - identical to how capture_screen works. That
        reuse is deliberate: it keeps base64 out of deskpilot_chats.json (which is
        already ~21 MB), and it means delete_chat()'s existing image reclaim already
        covers these files because _chat_image_paths() walks every image_ref part.

        Returns None when nothing could be saved (bad base64, unwritable store) - the
        tool's text result still stands on its own, so the turn must not fail.
        """
        parts: List[dict] = []
        for img in images[:MCP_MAX_IMAGES_PER_CALL]:
            try:
                b64 = str((img or {}).get("data") or "")
                if not b64.strip():
                    continue
                # Cap BEFORE decoding: base64 is ~4/3 of the payload, and a hostile or
                # buggy server could otherwise hand us a huge string to inflate in RAM.
                if len(b64) > MCP_MAX_IMAGE_BYTES * 2:
                    parts.append({"type": "text",
                                  "text": "[image skipped: larger than the "
                                          f"{MCP_MAX_IMAGE_BYTES}-byte cap]"})
                    continue
                raw = base64.b64decode(b64, validate=False)
                if not raw:
                    continue
                data, mime = _downscale_image_bytes(raw)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
                ext = ".jpg" if mime == "image/jpeg" else ".png"
                out_path = GEN_DIR / f"mcp_{ts}{ext}"
                out_path.write_bytes(data)          # BYTES (invariant #11)
                parts.append({"type": "image_ref", "path": str(out_path)})
            except Exception:
                continue          # one bad image must not lose the good ones
        if not parts:
            return None
        return {
            "role": "user",
            "content": [
                {"type": "text",
                 "text": MCP_IMAGE_MARKER + f" ({tool_name}) is attached below for your "
                         "visual analysis."},
            ] + parts,
        }

    # ── permission gate + tool dispatch (worker thread) ─────────────────────

    def _permission_preview(self, name: str, args: dict) -> str:
        """The text shown in the permission modal.

        For ask_expert the arguments ARE the payload that leaves the machine,
        so the stock 600-char JSON dump is not enough to actually review what
        is being sent to a third party. This spells out the destination, the
        whole question, and every file with the range that will be shipped.
        """
        if name != "ask_expert":
            p = json.dumps(args, ensure_ascii=False)
            return p[:600] + "…" if len(p) > 600 else p
        # Belt and braces: the dispatch loop already rejects non-dict arguments,
        # so this is not the load-bearing guard - but this function calls
        # args.get() repeatedly and must not raise wherever it is called from.
        if not isinstance(args, dict):
            args = {}
        url = str(self.settings.get("expert_server_url") or "")
        model = str(self.settings.get("expert_model") or "")
        host = url.split("//")[-1].split("/")[0] if url else "(not configured)"
        out = [f"SEND TO: {model} @ {host}",
               "The expert sees NOTHING else - no chat history, no settings.\n"]
        out.append("QUESTION:\n" + str((args or {}).get("question") or "").strip())
        ctx = str((args or {}).get("context") or "").strip()
        if ctx:
            shown = ctx if len(ctx) <= 1200 else ctx[:1200] + "\n… [truncated in this preview]"
            out.append("\nCONTEXT TEXT (" + str(len(ctx)) + " chars):\n" + shown)
        files = (args or {}).get("files") or []
        if files:
            out.append("\nFILES TO SEND:")
            for spec in files:
                if isinstance(spec, str):
                    spec = {"path": spec}
                off = spec.get("offset") or 0
                lim = spec.get("limit") or EXPERT_FILE_CHARS
                out.append(f"  {spec.get('path')}  chars {off}-{off + lim}")
        else:
            out.append("\nFILES TO SEND: none")
        return "\n".join(out)

    def _ask_permission_modal(self, name: str, args: dict) -> bool:
        """Modal Allow/Deny prompt, marshalled to the main thread; blocks this
        worker until the user answers (or 5 minutes pass).

        v1.1.57 (#5 of the external review): this used messagebox.askyesno. That
        dialog is built by Tk's C tk_dialog and is NOT a Python Toplevel - a probe
        confirmed root.winfo_children() reports ZERO toplevels while it is on
        screen, so nothing in Python can close it. Consequence: when Stop aborted
        the worker's sliced wait, the worker unwound and returned, but the dialog
        stayed modal on screen until the user clicked it. It is now a real Toplevel
        we hold a reference to, and it self-closes (answering No) as soon as Stop
        or app-close is seen - the same shape _elicit_dialog already uses.
        """
        preview = self._permission_preview(name, args)
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
            top = tk.Toplevel(self.root)
            top.title(f"{APP_NAME} — Tool Permission")
            top.configure(bg=COL["bg_main"])
            top.resizable(False, False)
            try:
                top.transient(self.root)
                top.grab_set()
            except tk.TclError:
                pass
            done = {"v": False, "timer": None}

            def answer(val: bool):
                # Idempotent: the Stop poll and a real click can race.
                if done["v"]:
                    return
                done["v"] = True
                ans["ok"] = bool(val)
                # Cancel the pending stop_poll first. Leaving it queued makes Tk
                # run a script whose command was deleted with the window, which
                # prints 'invalid command name "...stop_poll"' to stderr.
                try:
                    if done["timer"] is not None:
                        top.after_cancel(done["timer"])
                except Exception:
                    pass
                try:
                    top.destroy()
                except Exception:
                    pass
                ev.set()

            tk.Label(top, text="Deskpilot wants to run the tool:",
                     bg=COL["bg_main"], fg=COL["text"], font=F(11, "bold")).pack(
                padx=16, pady=(14, 2), anchor="w")
            tk.Label(top, text=name, bg=COL["bg_main"], fg=COL["accent"],
                     font=F(12, "bold")).pack(padx=16, pady=(0, 6), anchor="w")
            tk.Label(top, text="Arguments:", bg=COL["bg_main"], fg=COL["text_dim"],
                     font=F(10)).pack(padx=16, anchor="w")
            box = tk.Text(top, height=(18 if name == "ask_expert" else 9), width=64,
                          bg=COL["bg_raised"], fg=COL["text"],
                          insertbackground=COL["text"], relief="flat", font=F(10),
                          wrap="word", padx=8, pady=6)
            box.insert("1.0", preview)
            box.configure(state="disabled")
            box.pack(padx=16, fill="both")
            row = tk.Frame(top, bg=COL["bg_main"])
            row.pack(pady=14)
            tk.Button(row, text="Allow", command=lambda: answer(True), bg=COL["accent"],
                      fg="#FFFFFF", relief="flat", padx=14, cursor="hand2",
                      font=F(11)).pack(side="left", padx=4)
            tk.Button(row, text="Deny", command=lambda: answer(False), bg=COL["bg_raised"],
                      fg=COL["text_dim"], relief="flat", padx=14, cursor="hand2",
                      font=F(11)).pack(side="left", padx=4)
            top.bind("<Return>", lambda e: answer(True))
            top.bind("<Escape>", lambda e: answer(False))
            # Closing with the title-bar X is a refusal, never a grant.
            top.protocol("WM_DELETE_WINDOW", lambda: answer(False))

            def stop_poll():
                # The worker's wait already unwinds on Stop, so a dialog still on
                # screen would be a modal nobody is waiting for. Polled rather than
                # event-driven: the stop flag is set from another thread and Tk has
                # no hook for it.
                if done["v"]:
                    return
                if self._closed or self._stop_event.is_set():
                    answer(False)
                    return
                try:
                    if top.winfo_exists():
                        done["timer"] = top.after(400, stop_poll)
                except Exception:
                    pass

            try:
                done["timer"] = top.after(400, stop_poll)
            except Exception:
                pass

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

    def _execute_tool(self, name: str, args: dict, images_out: Optional[list] = None) -> str:
        """Run one tool. Returns its text result.

        images_out: optional list that an MCP tool's image content is appended to
        (as {"mime","data"} dicts, straight from the server). Kept an OUT PARAMETER
        rather than changing the return type: every built-in tool and the existing
        permission tests rely on _execute_tool returning a string, and only MCP
        tools can produce images at all."""
        perm = self.settings.get("tool_permissions", {}).get(name, "ask")
        if perm == "off":
            return f"ERROR: Tool '{name}' is disabled by the user (permission set to Off)."
        if perm == "ask" and not self._ask_permission_modal(name, args):
            return f"ERROR: User denied permission to run '{name}'."
        if name.startswith("mcp_") and name not in TOOL_SCHEMAS:
            # Namespaced MCP tools are mcp_<server>_<tool>. The prefix test alone is
            # not enough: a BUILT-IN tool whose name starts with mcp_ would be routed
            # here and never reach its own handler. TOOL_SCHEMAS is the authority on
            # what is built in, so it wins over the prefix.
            result = self._mcp_dispatch(name, args, images_out=images_out)
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

        There is NO counter in this code: the only thing that ever limited this
        tool was the wording below. The user's plan carries a generous monthly
        allowance (1,000 calls), so the credit-hoarding steering was removed on
        request - fetch_url is still tried first for simple static pages because
        it is faster and free, not because a scrape is scarce.
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
        # Reddit: rewrite onto the public Atom feed (see the REDDIT_* helpers).
        # The guard above ran on the ORIGINAL url; the rewritten one stays on the
        # same host, so it needs no separate SSRF check.
        reddit_url = _reddit_atom_url(url) if _is_reddit_url(url) else None
        fetch_target = reddit_url or url
        req = urllib.request.Request(
            fetch_target,
            headers={"User-Agent": "Mozilla/5.0 (Deskpilot)",
                     "Accept": (REDDIT_ATOM_ACCEPT if reddit_url else "text/html,application/xhtml+xml,*/*;q=0.8")})
        capped = False
        _FETCH_REDIRECT_HANDLER._allow_local = bool(self.settings.get("allow_local_network"))
        if reddit_url and not _reddit_throttle(abort=self._abort_requested):
            return (f"ERROR: Fetch was stopped by the user before it finished "
                    f"(Reddit request for {url} was still waiting its turn).")
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
            if e.code == 429 and reddit_url:
                return ("ERROR: Reddit is rate-limiting this IP (HTTP 429). Wait about a "
                        "minute before retrying - repeated requests extend the block. "
                        f"(feed URL tried: {fetch_target})")
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
        if reddit_url:
            text = _parse_reddit_atom(raw)
            if not text:
                return (f"ERROR: Reddit returned a response Deskpilot could not parse as an "
                        f"Atom feed for {url}. It may have changed its feed format, or the "
                        "request was intercepted (a login or bot-check page).")
            out = f"Reddit thread {url} (via its Atom feed; {REDDIT_MAX_ENTRIES}-entry cap, no vote counts):\n\n"
            out += text[:REDDIT_OUTPUT_CHARS]
            if len(text) > REDDIT_OUTPUT_CHARS:
                out += f"\n[... truncated, {len(text)} chars total]"
            if capped:
                out += f"\n[... download stopped at {FETCH_BODY_LIMIT // (1024 * 1024)} MB]"
            return out
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
                        "past it - escalate to it when the content is needed.")
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

    def _subprocess_cwd(self) -> str:
        """Working directory for subprocess tools (run_javascript).

        The File Workspace, when configured — the SAME root _safe_path() jails the
        read/write/edit tools to. Without this the subprocess inherits
        _run_cancellable's default (the system temp dir), so a script told to use
        relative paths looks in Temp and fails with ENOENT on files the model just
        wrote into the workspace. Falls back to the app's own cwd, and to the temp
        dir if that is unusable.
        """
        ws = str((self.settings or {}).get("file_workspace") or "").strip()
        for cand in (ws, os.getcwd()):
            if not cand:
                continue
            try:
                p = Path(os.path.expanduser(cand))
                if p.is_dir():
                    return str(p)
            except Exception:
                continue
        return tempfile.gettempdir()

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
                                      abort=self._abort_requested,
                                      cwd=self._subprocess_cwd())
            if p is None:
                if self._abort_requested():
                    return ("ERROR: JavaScript execution was stopped by the user before it "
                            f"finished (the node process was killed after running under "
                            f"{JS_TIMEOUT} s). Re-issue the call with shorter/cheaper code.")
                return f"ERROR: JavaScript execution timed out after {JS_TIMEOUT} seconds."
            out = (p.stdout or "")
            # Mark the cap so a cut-off result is never mistaken for "no output".
            if len(out) > JS_STDOUT_CAP:
                out = (out[:JS_STDOUT_CAP] + f"\n[... stdout truncated at {JS_STDOUT_CAP} chars - "
                       "print less per call, or write to a file and read it back with "
                       "read_local_file offset/limit]")
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

    def _tool_run_shell(self, command: str = "", timeout: Any = None) -> str:
        """Run ONE shell command line, stateless (SOW #6).

        SECURITY POSTURE - read before changing anything here. This is
        deliberately UNSANDBOXED and has NO path jail: the command runs with the
        user's full privileges and can touch anything the account can, anywhere.
        `_safe_path()` guards only the read/write/edit tools, and node already
        reaches outside the workspace via fs/child_process, so this adds no new
        capability - only less friction. The controls are:
          * DEFAULT_PERMS["run_shell"] = "ask" (never widen that default);
          * the tool description states the exposure so the model can decline;
          * the per-turn checkpoint caveat below.
        A command denylist was considered and REJECTED: it is trivially
        bypassable (`python -c "import os; os.system(...)"`), so it would trade
        real utility for false confidence.

        ⚠️ CHECKPOINT GAP (accepted, SOW #6): `_checkpoint_before_write` snapshots
        only what write_local_file / edit_local_file mutate. A shell command that
        rewrites files bypasses that, so ↺ Revert turn cannot undo shell damage.
        The turn is flagged (`_turn_used_shell`) so the revert dialog can say so.

        Stateless by design: no persistent session, so `cd`/env do not carry over.
        That keeps this entirely on the existing `_run_cancellable` path (Stop
        kills it, timeout applies, pipes are drained so a chatty child cannot
        deadlock the poll loop). A persistent session is a separate decision.
        """
        if not (command or "").strip():
            return ("ERROR: No command was received. Re-issue run_shell with the full "
                    "'command' argument.")
        try:
            limit = float(timeout) if timeout is not None else float(SHELL_TIMEOUT)
        except (TypeError, ValueError):
            limit = float(SHELL_TIMEOUT)
        if limit <= 0:
            limit = float(SHELL_TIMEOUT)
        limit = min(limit, float(SHELL_TIMEOUT_MAX))

        if os.name == "nt":
            # cmd.exe parses the command line itself; passing it as ONE list
            # element avoids Python doing a second, different quoting pass.
            # ⚠️ cmd.exe /c strips a matched pair of OUTER quotes, so a command
            # that begins with " needs the `cmd /c "..."` form handled by the
            # caller; documented in the tool description rather than patched here.
            argv = [os.environ.get("ComSpec") or "cmd.exe", "/c", command]
        else:
            argv = ["/bin/sh", "-c", command]

        try:
            p = self._run_cancellable(argv, limit,
                                      abort=self._abort_requested,
                                      cwd=self._subprocess_cwd())
        except FileNotFoundError as e:
            return f"ERROR: Shell executable not found ({e})."
        except Exception as e:
            return f"ERROR: Failed to run shell command: {e}"

        if p is None:
            if self._abort_requested():
                return ("ERROR: Shell command was stopped by the user before it "
                        "finished (the process tree was killed). Re-issue with a "
                        "cheaper command, or split it into steps.")
            return (f"ERROR: Shell command timed out after {limit:.0f} s and its "
                    f"process tree was killed. Raise 'timeout' (max "
                    f"{SHELL_TIMEOUT_MAX}) or run the long part in the background.")

        # Record that this turn ran a shell command, so the revert dialog can warn
        # that shell-driven file changes are NOT covered by checkpoints. Posted to
        # the main thread because self.chats is main-thread-only (invariant #4).
        self._mark_turn_used_shell()

        out = p.stdout or ""
        if len(out) > SHELL_STDOUT_CAP:
            out = (out[:SHELL_STDOUT_CAP]
                   + f"\n[... stdout truncated at {SHELL_STDOUT_CAP} chars - redirect "
                     "to a file and read it back with read_local_file offset/limit]")
        err = p.stderr or ""
        if len(err) > SHELL_STDOUT_CAP:
            err = (err[:SHELL_STDOUT_CAP]
                   + f"\n[... stderr truncated at {SHELL_STDOUT_CAP} chars]")

        parts = []
        if out.strip():
            parts.append(out)
        if err.strip():
            parts.append("[stderr]\n" + err)
        body = "\n".join(parts) or "(no output)"
        rc = p.returncode
        if rc == 0:
            return body
        # The exit code must be visible: a silent non-zero exit is how a failed
        # build or a missing binary gets read as success.
        return f"{body}\n\n[exit code {rc}]"

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

    def _tool_search_files(self, path: str = "", pattern: str = "", regex: bool = False,
                           ignore_case: bool = False, glob: str = "",
                           context: int = 0, max_matches: int = 0) -> str:
        """Bounded grep over a file / directory tree (SOW 2026-09-30, P3).

        Literal matching by default (regex opt-in); binary files and noise dirs
        are skipped. Output is relpath:lineno: trimmed-line grouped by file,
        with '-'-separated context lines when requested; per-file and global
        match caps plus the shared TOOL_OUTPUT_LIMIT backstop keep it bounded.
        Every candidate file is re-checked against the File Workspace jail so a
        symlink cannot widen scope (mirrors _safe_path / _tool_list_directory).
        """
        if not str(pattern or "").strip():
            return "ERROR: No search pattern provided."
        try:
            rx = re.compile(str(pattern) if _as_bool(regex) else re.escape(str(pattern)),
                            re.IGNORECASE if _as_bool(ignore_case) else 0)
        except re.error as e:
            return f"ERROR: Invalid regex pattern {pattern!r}: {e}"
        try:
            root = self._safe_path(path or ".")
        except PermissionError as e:
            return f"ERROR: Access denied (blocked path): {e}"
        if not root.exists():
            return f"ERROR: Path not found: {root}"
        try:
            ctx = max(0, min(int(context or 0), SEARCH_CONTEXT_MAX))
        except (TypeError, ValueError):
            ctx = 0
        try:
            cap = int(max_matches) if max_matches else SEARCH_DEFAULT_MAX_MATCHES
            cap = max(1, min(cap, SEARCH_MAX_MATCHES_CAP))
        except (TypeError, ValueError):
            cap = SEARCH_DEFAULT_MAX_MATCHES

        # Workspace jail re-check for every CANDIDATE file: a symlink inside an
        # allowed directory must not widen the search scope.
        ws_raw = str((getattr(self, "settings", None) or {})
                     .get("file_workspace") or "").strip()
        ws_str = ""
        if ws_raw:
            try:
                ws_str = str(Path(os.path.expanduser(ws_raw)).resolve()).lower()
            except Exception as e:
                return f"ERROR: Invalid file workspace configured ({ws_raw!r}): {e}"

        def in_scope(f: Path) -> bool:
            if not ws_str:
                return True
            try:
                fs = str(f.resolve()).lower()
            except OSError:
                return False
            return fs == ws_str or fs.startswith(_dir_prefix(ws_str))

        files: List[Path] = []
        if root.is_file():
            files.append(root)
        else:
            for f in root.rglob("*"):
                try:
                    if not f.is_file():
                        continue
                except OSError:
                    continue
                if any(part.lower() in SEARCH_SKIP_DIRS
                       for part in f.relative_to(root).parts[:-1]):
                    continue
                files.append(f)
            files.sort(key=lambda f: str(f).lower())

        groups: List[str] = []
        total = 0
        n_files = scanned = skipped_big = 0
        any_cut = False
        glob_pat = (glob or "").strip()
        for f in files:
            if total >= cap:
                break
            if glob_pat and not fnmatch.fnmatch(f.name, glob_pat):
                continue
            if not in_scope(f):
                continue
            try:
                if f.stat().st_size > EDIT_FILE_MAX:
                    skipped_big += 1
                    continue
                data = f.read_bytes()
            except OSError:
                continue
            scanned += 1
            if b"\x00" in data[:1024]:
                continue                      # binary: not searchable text
            flines = data.decode("utf-8", "replace").splitlines()
            hits = [i for i, ln in enumerate(flines, 1) if rx.search(ln)]
            if not hits:
                continue
            n_files += 1
            take = hits[:min(SEARCH_PER_FILE_CAP, cap - total)]
            if len(hits) > len(take):
                any_cut = True      # more matches existed than were reported
            total += len(take)
            rel = str(f) if root.is_file() else f.relative_to(root).as_posix()
            hitset = set(take)
            emit: Dict[int, bool] = {}
            for ln in take:
                lo, hi = ((ln, ln) if ctx == 0 else
                          (max(1, ln - ctx), min(len(flines), ln + ctx)))
                for j in range(lo, hi + 1):
                    emit[j] = j in hitset
            chunk: List[str] = []
            prev = 0
            for j in sorted(emit):
                if ctx and j != prev + 1 and chunk:
                    chunk.append("--")        # grep-style gap separator
                sep = ":" if emit[j] else "-"
                chunk.append(f"{rel}{sep}{j}{sep} {flines[j - 1].strip()[:200]}")
                prev = j
            groups.append("\n".join(chunk))

        head = (f"{total} matches in {n_files} files "
                f"({scanned} scanned) in {root}:")
        notes: List[str] = []
        if total >= cap:
            notes.append(f"stopped at max_matches={cap}")
        elif any_cut:
            notes.append(f"per-file cap of {SEARCH_PER_FILE_CAP} matches/file reached")
        if skipped_big:
            notes.append(f"{skipped_big} file(s) over the {EDIT_FILE_MAX}-byte limit skipped")
        body = "\n\n".join(groups)
        out = head + ("\n\n" + body if body else "")
        if notes:
            out += "\n[... " + "; ".join(notes) + "]"
        return out[:TOOL_OUTPUT_LIMIT]

    def _tool_read_local_file(self, path: str = "", offset: int = 0,
                              limit: int = 0) -> str:
        """Read a file with optional chunked windowing (SOW 2026-09-30, P2).

        offset/limit slice the decoded text by character index; PDF pages are
        joined then windowed the same way. The header always reports the
        returned range and total length so a follow-up read can resume at
        `end`; a PDF whose pages ran out before the window was filled reports
        its total with a '+' suffix (it is a lower bound)."""
        try:
            p = self._safe_path(path)
            if not p.is_file():
                return f"ERROR: File not found: {p}"
            start = max(0, int(offset))
            n = int(limit) if limit else FILE_READ_LIMIT
            n = min(max(n, 1), FILE_READ_LIMIT)
            plus = ""                       # '+' when a PDF total is a lower bound
            if p.suffix.lower() == ".pdf":
                if PdfReader is None:
                    return "ERROR: 'pypdf' is not installed. Run:  pip install pypdf"
                reader = PdfReader(str(p))
                parts, acc = [], 0
                for page in reader.pages:
                    t = page.extract_text() or ""
                    parts.append(t)
                    acc += len(t) + (1 if len(parts) > 1 else 0)   # '\n' joiners count
                    if acc >= start + n:    # window covered; stop decoding pages
                        plus = "+"          # more pages exist: total is a lower bound
                        break
                full = "\n".join(parts)
                total_n = len(full)
                body = full[start:start + n]
            else:
                data = p.read_bytes()
                if b"\x00" in data[:1024]:
                    return (f"ERROR: {p.name} appears to be a binary file. "
                            f"Only text files and PDFs are supported.")
                full = data.decode("utf-8", "replace")
                total_n = len(full)
                body = full[start:start + n]
        except PermissionError as e:
            return f"ERROR: Access denied (blocked path): {e}"
        except Exception as e:
            return f"ERROR: Failed to read file: {e}"
        if start and start >= total_n:      # explicit, self-correcting past-EOF
            return (f"ERROR: offset {start} past end of file; "
                    f"file has {total_n}{plus} chars")
        end = start + len(body)
        footer = (f"\n[... more content follows at offset {end}]"
                  if end < total_n else "")
        return (f"Contents of {p.name} (chars {start}\u2013{end} of "
                f"{total_n}{plus}):\n\n{body}{footer}")

    #  ── Per-turn checkpoints (v1.1.45, SOW #4b) ────────────────────────
    #
    #  THREAD OWNERSHIP (this is the part that must not be broken later):
    #    * self.chats is mutated on the MAIN THREAD ONLY. That is a pre-existing
    #      invariant of this app - the worker never touches it, it posts
    #      _ui_sync_messages() instead. The manifest therefore goes in via _post().
    #      Mutating it from the worker would race with save_chats() serializing the
    #      same dict on the main thread (dictionary-changed-size-during-iteration,
    #      or a torn manifest on disk).
    #    * _turn_ids / _cp_done are WORKER-THREAD ONLY (never read or written from
    #      the main thread, so they need no lock).
    #    * Snapshot FILES are written by the worker and deleted by the main thread;
    #      only one turn runs at a time (_busy), and prune is posted before any
    #      snapshot of that turn, so the two cannot interleave.

    def _turn_id(self, chat_id: str) -> str:
        """Id for the turn currently running, allocated on first use.

        There was no turn identifier at all: _chat_worker_inner has 'step' (which
        resets per turn and does not survive a restart) and nothing that groups the
        tool calls of one user prompt. Allocated lazily so any caller - the worker,
        a tool, a test - works without the worker having set anything."""
        tid = self._turn_ids.get(chat_id)
        if not tid:
            tid = uuid.uuid4().hex[:12]
            self._turn_ids[chat_id] = tid
        return tid

    def _begin_turn_checkpoints(self, chat_id: str) -> str:
        """Open a turn's checkpoint namespace (worker thread, at turn start)."""
        tid = self._turn_id(chat_id)
        self._cp_done = set()
        # The CURRENT turn id is passed to the prune on purpose. The prune runs on the
        # main thread when the UI queue drains, which can be AFTER this turn already
        # wrote its first snapshot (snapshots are written synchronously on the worker;
        # their manifest entries are queued behind the prune). A prune that only kept
        # ids named by the manifest would therefore delete the live turn's own
        # snapshot and leave the manifest pointing at a missing file.
        self._post(lambda c=chat_id, t=tid: self._ui_checkpoint_prune(c, t))
        return tid

    def _end_turn_checkpoints(self, chat_id: str) -> None:
        """Close the namespace (worker thread) so the next turn starts fresh."""
        self._turn_ids.pop(chat_id, None)
        self._cp_done = set()

    def _ui_checkpoint_prune(self, chat_id: str, current_turn_id: str = "") -> None:
        """MAIN THREAD. Drop checkpoint turns no longer named by the manifest.

        Runs at TURN START, never at turn end, and never walks the filesystem
        looking for orphans. The standing policy (2026-10-06) is that the user
        cleans up manually - an automatic sweep could delete something wanted later.
        A chat's whole checkpoint tree is removed by delete_chat().

        Why here at all: the manifest is capped at CHECKPOINT_MAX_TURNS, and without
        this the oldest entries would fall off the end of the JSON while their
        snapshot files stayed on disk forever, unreferenced and undiscoverable.

        current_turn_id is always kept: this turn's manifest entries may still be
        queued behind this callback (see _begin_turn_checkpoints)."""
        chat = self.chats["chats"].get(chat_id)
        if not chat:
            return
        keep = set()
        for t in (chat.get("checkpoints") or []):
            tid = str((t or {}).get("turn_id") or "")
            if tid:
                keep.add(_checkpoint_component(tid))
        if current_turn_id:
            keep.add(_checkpoint_component(current_turn_id))
        try:
            root = _checkpoint_root() / _checkpoint_component(chat_id)
            if not root.is_dir():
                return
            for sub in root.iterdir():
                if sub.is_dir() and sub.name not in keep:
                    shutil.rmtree(sub, ignore_errors=True)
        except Exception:
            pass      # housekeeping must never break a turn

    def _ui_add_checkpoint(self, chat_id: str, turn_id: str, entry: dict) -> None:
        """MAIN THREAD. Attach one manifest entry to the chat record.

        Posted by _checkpoint_before_write once the snapshot bytes are safely on
        disk, so the manifest never names a snapshot that does not exist (the
        reverse ordering, manifest-then-file, is what would leave a restore
        pointing at nothing). Paths + hashes only - never content."""
        try:
            chat = self.chats["chats"].get(chat_id)
            if chat is None:
                return                       # chat deleted mid-turn: nothing to attach to
            turns = chat.setdefault("checkpoints", [])
            turn = None
            for t in turns:
                if isinstance(t, dict) and t.get("turn_id") == turn_id:
                    turn = t
                    break
            if turn is None:
                if len(turns) >= CHECKPOINT_MAX_TURNS:
                    return                   # manifest cap reached for this chat
                turn = {"turn_id": turn_id,
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "files": []}
                turns.append(turn)
            files = turn.setdefault("files", [])
            if len(files) >= CHECKPOINT_MAX_FILES_PER_TURN:
                return
            files.append(entry)
        except Exception:
            pass

    def _checkpoint_before_write(self, p: Path) -> None:
        """Snapshot p's ORIGINAL bytes once per (chat, turn, file). Never raises.

        Called immediately before the two file-mutating tools write to disk. Every
        failure mode is swallowed: a checkpoint that cannot be taken must never
        block, delay or fail the user's actual write.
        """
        try:
            cid = self._run_chat_id or ""
            if not cid:
                return                       # no chat context (e.g. a direct tool call)
            tid = self._turn_id(cid)
            key = str(p)
            dedup = cid + "\x00" + tid + "\x00" + key
            if dedup in self._cp_done:
                return                       # already snapshotted this turn
            if len(self._cp_done) >= CHECKPOINT_MAX_FILES_PER_TURN:
                return
            self._cp_done.add(dedup)         # set BEFORE the work: a failure must not retry
            if not p.is_file():
                # The file did not exist before this turn: restoring means DELETING it.
                self._post(lambda c=cid, t=tid, e={"path": key, "existed": False}:
                           self._ui_add_checkpoint(c, t, e))
                return
            data = p.read_bytes()
            if len(data) > CHECKPOINT_MAX_FILE_BYTES:
                self._post(lambda c=cid, t=tid,
                           e={"path": key, "existed": True, "skipped": "too large"}:
                           self._ui_add_checkpoint(c, t, e))
                return
            slot = _checkpoint_slot(cid, tid, p)
            slot.parent.mkdir(parents=True, exist_ok=True)
            # BYTES, not write_text: a checkpoint that mangles newlines is worse than none.
            slot.write_bytes(data)
            self._post(lambda c=cid, t=tid, e={"path": key,
                                               "existed": True,
                                               "cp": str(slot),
                                               "bytes": len(data),
                                               "sha256": hashlib.sha256(data).hexdigest()[:16]}:
                       self._ui_add_checkpoint(c, t, e))
        except Exception:
            return

    def _checkpoint_restore(self, chat_id: str, turn_id: str) -> str:
        """MAIN THREAD. Put every file this turn touched back to its pre-turn state.

        Returns a human-readable report (also what the 4c button shows). Partial on
        purpose: one locked file must not abort the rest, and every line says what
        actually happened. Never raises.

        Containment: the manifest lives in chats.json, which is USER-EDITABLE, so
        both 'cp' and 'path' are untrusted here. Two checks, not the file-tool
        workspace rule: (1) the snapshot must resolve to somewhere INSIDE
        _checkpoint_root() - otherwise 'cp' is pointing at an arbitrary file on this
        machine and we would be copying it over something; (2) the snapshot's sha256
        must match the manifest, so a truncated/tampered snapshot is refused rather
        than silently written. Restoring OUTSIDE the current workspace is allowed on
        purpose: the workspace setting may have changed since the turn ran, and this
        is a user-initiated click, not a model tool call.

        Restore is itself destructive, so the bytes it displaces are kept once as
        <cp>.pre-restore - a revert of a revert."""
        chat = self.chats["chats"].get(chat_id)
        if not chat:
            return "ERROR: conversation not found."
        turn = None
        for t in (chat.get("checkpoints") or []):
            if isinstance(t, dict) and t.get("turn_id") == turn_id:
                turn = t
                break
        if turn is None:
            return "ERROR: no checkpoint found for that turn (it may have aged out or been deleted)."
        files = turn.get("files") or []
        if not files:
            return "Nothing to revert: that turn recorded no file changes."

        root_resolved = _checkpoint_root().resolve()
        restored = deleted = skipped = failed = 0
        lines: List[str] = []
        for f in files:
            if not isinstance(f, dict):
                skipped += 1
                continue
            target = str(f.get("path") or "")
            if not target:
                skipped += 1
                continue
            try:
                if not f.get("existed"):
                    # The file was NOT there before the turn: reverting means removing it.
                    tp = Path(target)
                    if tp.is_file():
                        tp.unlink()
                        deleted += 1
                        lines.append(f"removed  {tp.name}  (did not exist before that turn)")
                    else:
                        skipped += 1
                        lines.append(f"already absent: {tp.name}")
                    continue
                cp = f.get("cp")
                if not cp:
                    skipped += 1
                    lines.append(f"no snapshot recorded for {Path(target).name}"
                                 + (f" ({f.get('skipped')})" if f.get("skipped") else ""))
                    continue
                cpp = Path(cp)
                # (1) the snapshot must live inside the checkpoint store
                if root_resolved not in cpp.resolve().parents:
                    failed += 1
                    lines.append(f"REFUSED (snapshot outside the checkpoint store): {Path(target).name}")
                    continue
                if not cpp.is_file():
                    failed += 1
                    lines.append(f"MISSING snapshot: {Path(target).name}")
                    continue
                data = cpp.read_bytes()
                want = str(f.get("sha256") or "")
                got = hashlib.sha256(data).hexdigest()[:16]
                # (2) integrity: never write bytes we cannot verify
                if want and got != want:
                    failed += 1
                    lines.append(f"REFUSED (snapshot hash mismatch, expected {want} got {got}): "
                                 f"{Path(target).name}")
                    continue
                tp = Path(target)
                tp.parent.mkdir(parents=True, exist_ok=True)
                if tp.is_file():
                    try:
                        cpp.with_name(cpp.name + ".pre-restore").write_bytes(tp.read_bytes())
                    except Exception:
                        pass      # the safety copy is best-effort; the restore still proceeds
                tmp = tp.with_name(tp.name + ".deskpilot-restore.tmp")
                try:
                    tmp.write_bytes(data)          # BYTES - never write_text (invariant #11)
                    os.replace(tmp, tp)
                except Exception:
                    # v1.1.56: a failed write used to leave the half-written temp
                    # sitting next to the target forever. Unlike _atomic_write the
                    # name is FIXED (not pid+uuid), so it is not littered once per
                    # attempt - but it still lingers indefinitely and can be
                    # mistaken for real content. The target is untouched either way,
                    # because the swap is atomic and had not happened yet. Re-raise
                    # so the outer handler still reports FAILED.
                    try:
                        tmp.unlink(missing_ok=True)
                    except Exception:
                        pass
                    raise
                restored += 1
                lines.append(f"restored {tp.name}  ({len(data)} bytes)")
            except Exception as e:
                failed += 1
                lines.append(f"FAILED {Path(target).name}: {e}")
        head = (f"Reverted turn: {restored} restored, {deleted} removed, "
                f"{skipped} skipped, {failed} failed")
        return head + ("\n" + "\n".join(lines) if lines else "")

    #  ── #4c: the revert control (v1.1.46) ──────────────────────────────

    def _mark_turn_used_shell(self) -> None:
        """WORKER THREAD. Flag the current turn as having run a shell command.

        Only meaningful when the turn ALSO has checkpointed files: a turn that
        used shell alone has nothing to revert and no button. The flag therefore
        exists to downgrade the revert promise from "everything goes back" to
        "the tool-file writes go back; shell changes do not".

        Posted, never applied inline - self.chats is main-thread-only. Best-effort:
        a failure here must not change the command's result."""
        try:
            cid = self._run_chat_id or ""
            if not cid:
                return                       # no chat context (e.g. a direct tool call)
            tid = self._turn_id(cid)
            self._post(lambda c=cid, t=tid: self._ui_mark_turn_shell(c, t))
        except Exception:
            pass

    def _ui_mark_turn_shell(self, chat_id: str, turn_id: str) -> None:
        """MAIN THREAD. Set shell=1 on this turn's checkpoint record.

        Creates the turn record if needed (same shape _ui_add_checkpoint builds) so
        a shell-only turn is still recognisable - though _revertable_turns will not
        offer it, since it has no files."""
        try:
            chat = self.chats["chats"].get(chat_id)
            if chat is None:
                return                       # chat deleted mid-turn
            turns = chat.setdefault("checkpoints", [])
            for t in turns:
                if isinstance(t, dict) and t.get("turn_id") == turn_id:
                    t["shell"] = 1
                    return
            if len(turns) >= CHECKPOINT_MAX_TURNS:
                return
            turns.append({"turn_id": turn_id,
                          "ts": datetime.now().isoformat(timespec="seconds"),
                          "files": [], "shell": 1})
        except Exception:
            pass

    def _revertable_turns(self, chat: Optional[dict]) -> List[dict]:
        """Checkpointed turns of this chat that actually have something to undo.

        A turn entry with no files (or only entries that recorded a skip) is not
        offered - showing a live button that would do nothing is worse than none."""
        out: List[dict] = []
        for t in ((chat or {}).get("checkpoints") or []):
            if not isinstance(t, dict) or not t.get("turn_id"):
                continue
            files = [f for f in (t.get("files") or []) if isinstance(f, dict)]
            if files:
                out.append({"turn_id": str(t["turn_id"]),
                            "ts": str(t.get("ts") or ""),
                            "count": len(files),
                            "shell": bool(t.get("shell"))})
        return out

    def _refresh_revert_button(self) -> None:
        """Show the top-bar revert button only when the open chat can be reverted.

        Idempotent: tracks its own pack state so repeated calls (chat switch, turn
        end, after a revert) do not stack widgets. Never raises - a broken top bar
        must not take the app down."""
        try:
            turns = self._revertable_turns(self.current_chat())
            n = len(turns)
            if n == 0:
                if self._revert_packed:
                    self.revert_btn.pack_forget()
                    self._revert_packed = False
                return
            if not self._revert_packed:
                # Packed AFTER the labels with side="right", so it lands to their left
                # (status | speed | context | temp | model are rightmost; revert sits
                # just left of status). No `before=` needed - pack order does it.
                self.revert_btn.pack(side="right", padx=(0, 12), pady=10)
                self._revert_packed = True
            self.revert_btn.configure(
                text=("\u21ba Revert turn" if n == 1 else f"\u21ba Revert turn ({n})"),
                state=("disabled" if self._busy else "normal"))
        except tk.TclError:
            pass
        except Exception:
            pass

    def _on_revert_clicked(self) -> None:
        """Confirm, then undo the most recent checkpointed turn of this chat.

        Deliberately scoped to the LATEST turn only, with no turn picker:
        restoring an OLDER turn would also clobber every edit made by the turns
        after it, which is not what a user asking to 'undo that one change' expects,
        and _checkpoint_restore has no way to merge the two. The latest turn is the
        only state that is unambiguously 'back to how things were'.

        Blocked while a turn is running: the worker may be mid-write, and restoring
        underneath it would race the agent's own edits."""
        if self._closed or self._busy:
            self._set_status("\u21ba Revert is disabled while a turn is running")
            return
        chat = self.current_chat()
        turns = self._revertable_turns(chat)
        if not turns:
            self._set_status("No checkpointed turns to revert in this conversation")
            return
        t = turns[-1]
        when = t["ts"].replace("T", " ")[:19] or "unknown time"
        shell_note = (
            "\n\u26a0\ufe0f This turn ALSO ran shell commands. Files changed by the "
            "shell are NOT checkpointed and will NOT be undone - only the files "
            "written/edited through the file tools go back.\n"
            if t.get("shell") else "")
        if not messagebox.askyesno(
                APP_NAME,
                f"Undo the file changes from the most recent tool turn?\n\n"
                f"  when:   {when}\n"
                f"  files:  {t['count']}\n\n"
                f"Every file that turn wrote or edited goes back to how it was BEFORE "
                f"the turn, and files it created are deleted. This does not touch the "
                f"conversation text.{shell_note}\n\nProceed?", parent=self.root):
            self._set_status("Revert cancelled - nothing changed")
            return
        report = self._checkpoint_restore(chat.get("id") or self.current_chat_id or "", t["turn_id"])
        # The report goes into the transcript (not just the status bar): it lists per
        # file what was restored/removed/refused, and the status bar is one line.
        self.render_note("\u21ba " + report)
        self._set_status("\u21ba " + report.split("\n")[0])
        self._refresh_revert_button()

    def _tool_edit_local_file(self, path: str = "", edits=None,
                               dry_run: bool = False) -> str:
        """All-or-nothing in-place text replacements (see SOW 2026-09-30, P1).

        Validates EVERY edit against the current text before mutating anything;
        on any mismatch writes nothing and reports all failures together.
        Dominant line ending (CRLF) is preserved; non-UTF-8/binary files are
        refused outright - never lossily round-tripped."""
        if not (path or "").strip():
            return "ERROR: No target path provided."
        if not isinstance(edits, list) or not edits:
            return "ERROR: 'edits' must be a non-empty list of {old_text, new_text} objects."
        try:
            p = self._safe_path(path)
            if not p.is_file():
                return (f"ERROR: File not found: {p} - edit_local_file never creates "
                        f"files; use write_local_file for that.")
            data = p.read_bytes()
            if len(data) > EDIT_FILE_MAX:
                return (f"ERROR: {p.name} is {len(data)} bytes, over the "
                        f"{EDIT_FILE_MAX}-byte edit limit; split the work into smaller edits "
                        f"via run_javascript.")
            if b"\x00" in data[:1024]:
                return (f"ERROR: {p.name} appears to be a binary file. "
                        f"Only text files are supported.")
            try:
                raw_text = data.decode("utf-8")   # strict: no silent replacement chars
            except UnicodeDecodeError:
                return (f"ERROR: {p.name} is not valid UTF-8 text; refusing to edit it "
                        f"(use write_local_file to rewrite the whole file).")
        except PermissionError as e:
            return f"ERROR: Access denied (blocked path): {e}"
        except Exception as e:
            return f"ERROR: Failed to read file for editing: {e}"

        crlf = "\r\n" in raw_text
        text = raw_text.replace("\r\n", "\n") if crlf else raw_text

        # ── validate ALL edits first, sequentially against a simulation
        # copy so dependent edits (later edits matching earlier edits' output)
        # validate correctly; collect every failure, write nothing ──
        failures: List[str] = []
        plan: List[tuple] = []          # (index, old, new, count)
        work = text                     # simulation copy: validation == application
        for i, ed in enumerate(edits):
            if not isinstance(ed, dict):
                failures.append(f"edit #{i}: not an object")
                continue
            old = ed.get("old_text")
            new = ed.get("new_text")
            if not isinstance(old, str) or not isinstance(new, str):
                failures.append(f"edit #{i}: old_text/new_text must be strings")
                continue
            if old == "":
                failures.append(f"edit #{i}: old_text is empty (would match everywhere)")
                continue
            allow = _as_bool(ed.get("allow_multiple", False))
            count = work.count(old)
            if not ((count >= 1) if allow else (count == 1)):
                failures.append(f"edit #{i}: old_text found {count} time(s), needs "
                                f"{'>=1' if allow else 'exactly 1'}: {old[:80]!r}")
                continue
            plan.append((i, old, new, count))
            work = work.replace(old, new)
        if failures:
            return (f"ERROR: no changes written ({len(failures)} problem(s), all-or-nothing):\n"
                    + "\n".join(failures))

        # ── dry run: report validation counts without touching disk ──
        if _as_bool(dry_run):
            report = "; ".join(f"edit #{i}: {c} match(es)" for (i, _o, _n, c) in plan)
            return (f"DRY RUN OK: {p.name} - all {len(plan)} edit(s) validate; "
                    f"nothing written. {report}")

        # ── apply sequentially (later edits see earlier edits' output) ──
        counts: List[str] = []
        for (i, old, new, n) in plan:
            text = text.replace(old, new)
            counts.append(f"edit #{i}: {n}")
        out = text.replace("\n", "\r\n") if crlf else text

        self._checkpoint_before_write(p)      # v1.1.45: snapshot BEFORE mutating
        try:
            _atomic_write(p, out)
        except PermissionError as e:
            return f"ERROR: Access denied (blocked path): {e}"
        except Exception as e:
            return f"ERROR: Failed to write file: {e}"

        # v1.1.44: a unified diff so the model (and the user, in the tool-output
        # drawer) can see exactly what changed without re-reading the file.
        diff = _unified_diff_text(raw_text, out)
        # compact receipt: first changed line lets the model self-verify cheaply
        old_lines, new_lines = raw_text.splitlines(), out.splitlines()
        first_change = ""
        for a, b in zip(old_lines, new_lines):
            if a != b:
                first_change = b.strip()[:100]
                break
        if not first_change and len(new_lines) > len(old_lines):
            first_change = new_lines[len(old_lines)].strip()[:100]
        delta = len(out.encode("utf-8")) - len(raw_text.encode("utf-8"))
        eol = ", CRLF preserved" if crlf else ""
        ctx = f"; first change: {first_change!r}" if first_change else ""
        return (f"OK: edited {p.name} - {len(plan)} edit(s) applied{eol}; "
                + "; ".join(counts) + f"; net {delta:+d} bytes{ctx}\n" + _diff_block(diff))

    def _tool_write_local_file(self, path: str = "", content: str = "") -> str:
        if not (path or "").strip():
            return "ERROR: No target path provided."
        body = content if isinstance(content, str) else ("" if content is None else str(content))
        try:
            p = self._safe_path(path)
            existed = p.is_file()
            before = p.read_bytes() if existed else b""
            p.parent.mkdir(parents=True, exist_ok=True)
            # INVARIANT #11 (FIXED v1.1.44): this used p.write_text(), which on
            # Windows translates every '\n' to os.linesep. Content that ALREADY had
            # CRLF - the normal result of read_local_file (read_bytes + decode, no
            # translation) followed by an unchanged write, or any run_javascript
            # fs.writeFileSync of CRLF text - came back out as '\r\r\n', silently
            # corrupting the file. _atomic_write was fixed for exactly this in
            # 2026-09-30; this path was missed. Write BYTES.
            self._checkpoint_before_write(p)  # v1.1.45: snapshot BEFORE mutating
            # v1.1.56 (#1 of the external review): route through _atomic_write,
            # exactly as the sibling edit_local_file already does. Two fixes in
            # one: (a) ATOMICITY - a crash or disk-full mid-write used to leave a
            # truncated file where the old content was; now the temp is completed
            # first and swapped in one step, so the target is either the old bytes
            # or the new ones. (b) the per-path lock, which stops two concurrent
            # writers of the same path racing on the swap. _atomic_write encodes
            # to BYTES itself, so invariant #11 (no newline translation) holds.
            # The .bak it also writes is redundant with the checkpoint snapshot,
            # but harmless and consistent with edit_local_file.
            _atomic_write(p, body)
        except PermissionError as e:
            return f"ERROR: Access denied (blocked path): {e}"
        except Exception as e:
            return f"ERROR: Failed to write file: {e}"
        verb = "overwrote" if existed else "created"
        size = len(body.encode("utf-8"))
        head = f"OK: {verb} {p} - {len(body)} characters"
        if not existed:
            # No diff for a brand-new file: it would just echo back content the model
            # already emitted. Overwrites are the case where the prior state matters.
            return head + f" (new file, {size} bytes)"
        diff = _unified_diff_text(before.decode("utf-8", "replace"), body)
        delta = f"; net {size - len(before):+d} bytes"
        if not diff:
            return head + delta + "; content unchanged"
        return head + delta + "\n" + _diff_block(diff)

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
            # uuid suffix: two captures in the same second must not overwrite each other
            # (an earlier chat message's image_ref would then point at the newer shot).
            ts = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
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

    def _tool_list_skills(self) -> str:
        """Report the folders scanned for skills and every skill found in them."""
        dirs = _skill_dirs(self.settings)
        if not dirs:
            return ("No skills folder exists yet. Create one at "
                    f"{USER_WORKSPACE_ROOT / SKILLS_SUBDIR} with a <skill name>/{SKILL_FILE_NAME} "
                    "inside it, or point Settings -> Skills folder at an existing one.")
        skills = _discover_skills(self.settings)
        lines = ["Skill folders scanned (highest priority first):"]
        lines += [f"  {d}" for d in dirs]
        if not skills:
            return "\n".join(lines) + (
                f"\n\nNo skills found. A skill is <folder>/{SKILL_FILE_NAME} opening with YAML "
                "frontmatter that has at least 'name:' and 'description:'.")
        lines.append(f"\n{len(skills)} skill(s):")
        for s in skills:
            lines.append(f"  {s['name']}" + (f" - {s['description']}" if s["description"] else ""))
            lines.append(f"      {s['path']}")
        return "\n".join(lines)

    def _tool_load_skill(self, name: str = "") -> str:
        """Return one skill's SKILL.md body plus the files bundled with it."""
        want = str(name or "").strip().lower()
        if not want:
            return "ERROR: load_skill needs the skill's name (see list_skills)."
        skills = _discover_skills(self.settings)
        hit = next((s for s in skills if s["name"].lower() == want), None)
        if hit is None:
            avail = ", ".join(sorted(s["name"] for s in skills)) or "(none installed)"
            near = [s["name"] for s in skills if want in s["name"].lower()]
            hint = f" Partial matches: {', '.join(near)}." if near else ""
            return f"ERROR: No skill named {name!r}.{hint} Available: {avail}"
        try:
            raw = Path(hit["path"]).read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"ERROR: could not read {hit['path']}: {e}"
        _meta, body = _skill_frontmatter(raw)
        if len(body) > SKILL_BODY_MAX:
            body = body[:SKILL_BODY_MAX] + f"\n[... skill body truncated at {SKILL_BODY_MAX} chars]"
        # Bundled resources are why skills beat prompts: the body refers to scripts/
        # and references/ by RELATIVE path. Listing them saves a list_directory hop.
        extras: List[str] = []
        try:
            for p in sorted(Path(hit["dir"]).iterdir(), key=lambda x: x.name.lower()):
                if p.name.lower() != SKILL_FILE_NAME.lower():
                    extras.append(p.name + ("/" if p.is_dir() else ""))
        except Exception:
            pass
        head = (f"# Skill: {hit['name']}\nSource: {hit['path']}\n"
                f"Skill folder (resolve any relative path in the body against it): {hit['dir']}")
        if extras:
            head += "\nAlso in the skill folder: " + ", ".join(extras[:60])
        return head + "\n\n" + (body or "(this skill has no instructions in its body)")

    def _tool_search_chats(self, query: str = "", limit: int = SEARCH_MAX_SNIPPETS,
                           full: bool = False) -> str:
        """Full-text search across ALL of the user's chat history."""
        q = str(query or "").strip()
        if not q:
            return "ERROR: search_chats needs a query."
        if not self._ensure_search_index():
            hits = _search_group_by_chat(_search_chats_scan(self.chats, q, SEARCH_MAX_SNIPPETS))
            note = " (no FTS index available - plain substring scan)"
        else:
            hits = _search_group_by_chat(search_chats(self.chats, q, SEARCH_MAX_SNIPPETS))
            note = ""
        if not hits:
            return f"No chat history matches {q!r}."
        lines = [f"{len(hits)} chat(s) match {q!r}{note} (best hit each, most relevant first):"]
        for h in hits:
            archived = str(h.get("kind") or "live").startswith("archive:")
            where = (f"archived message #{h['idx']} of compaction "
                     f"{str(h['kind']).split(':', 1)[1]}" if archived
                     else f"message #{h['idx']}")
            lines.append(f"\n• {h['title']}  [{h['hits']} hit(s)]  chat_id={h['chat_id']}")
            lines.append(f"    {where} ({h['role']}): {h['snippet'][:240]}")
            if _as_bool(full):
                # The chats file usually sits OUTSIDE the File Workspace jail, so
                # telling the model to go read it would just earn a denial. Hand it
                # the matched message instead - from the right address space.
                body = _message_body(self.chats, h["chat_id"], h["idx"],
                                     h.get("kind") or "live")
                if body:
                    lines.append(f"    full message: {body[:SEARCH_BODY_CAP]}")
            if archived:
                lines.append("    (this message was folded out of the conversation by a "
                             "compaction - read_archive(chat_id=…, compaction=…) pages "
                             "through the rest of that event)")
        lines.append("\nUse full=true to get the whole matched message, refine the query "
                     "for a different part of the history, or read_archive to page through "
                     "a compaction's archived messages.")
        return "\n".join(lines)

    def _tool_read_archive(self, chat_id: str = "", compaction: int = 0,
                           offset: int = 0, limit: int = ARCHIVE_PAGE_MSGS) -> str:
        """Page through the messages a compaction folded out of a chat.

        The archive is the only complete record of a long chat: compaction replaces old
        messages with a summary, and a summary is a lossy view of what actually happened.
        This makes the originals reachable by the agent, not just by the user reading the
        📦 drawer in the UI."""
        cid = str(chat_id or "").strip() or str(self.current_chat_id or "")
        chat = (self.chats.get("chats") or {}).get(cid)
        if not isinstance(chat, dict):
            return (f"ERROR: no chat with id {cid!r}. Omit chat_id to read the current "
                    f"chat, or get an id from search_chats.")
        arch = [a for a in (chat.get("compaction_archive") or []) if isinstance(a, dict)]
        if not arch:
            return (f"Chat {chat.get('title') or cid!r} has no compaction archive - "
                    f"nothing has been folded out of it yet.")
        arch.sort(key=lambda a: int(a.get("compaction_number") or 0))

        # No event selected -> describe what is available. Listing first is what makes
        # the tool usable without guessing numbers, and it is one cheap call.
        try:
            want = int(compaction) if compaction else 0
        except (TypeError, ValueError):
            want = 0
        if want == 0:
            lines = [f"{len(arch)} compaction event(s) archived in "
                     f"{chat.get('title') or cid!r}:"]
            for a in arch:
                n = int(a.get("compaction_number") or 0)
                ms = a.get("messages") or []
                ts = str(a.get("timestamp") or "")[:19].replace("T", " ")
                lines.append(f"  compaction {n}: {len(ms)} messages archived {ts}")
            lines.append("\nRead one with read_archive(compaction=N, offset=…, limit=…). "
                         "search_chats reports which event a hit came from, so start there "
                         "if you are looking for something specific.")
            return "\n".join(lines)

        ev = next((a for a in arch
                   if int(a.get("compaction_number") or 0) == want), None)
        if ev is None:
            nums = ", ".join(str(int(a.get("compaction_number") or 0)) for a in arch)
            return f"ERROR: chat has no compaction event {want}. Available: {nums}."
        msgs = [m for m in (ev.get("messages") or []) if isinstance(m, dict)]
        try:
            off = max(0, int(offset or 0))
        except (TypeError, ValueError):
            off = 0
        try:
            lim = int(limit) if limit else ARCHIVE_PAGE_MSGS
        except (TypeError, ValueError):
            lim = ARCHIVE_PAGE_MSGS
        lim = max(1, min(lim, ARCHIVE_PAGE_MSGS_MAX))
        if off >= len(msgs):
            return (f"Compaction {want} holds {len(msgs)} messages; offset {off} is past "
                    f"the end. Valid offsets are 0..{max(0, len(msgs) - 1)}.")
        page = msgs[off:off + lim]
        lines = [f"Compaction {want} of {chat.get('title') or cid!r} - "
                 f"{len(msgs)} archived messages, showing {off}..{off + len(page) - 1} "
                 f"(archived {str(ev.get('timestamp') or '')[:19].replace('T', ' ')}):"]
        for i, m in enumerate(page):
            role = str(m.get("role") or "?")
            body = (_flatten_handoff_content(m) or "").strip()
            tag = ""
            if role == "assistant" and isinstance(m.get("tool_calls"), list):
                names = [str(((tc.get("function") or {}) if isinstance(tc, dict) else {})
                             .get("name") or "?") for tc in m["tool_calls"]]
                if names:
                    tag = f" [ran: {', '.join(names[:8])}]"
            elif role == "tool" and m.get("name"):
                tag = f" [output of {m['name']}]"
            cut = ""
            if len(body) > ARCHIVE_MSG_CAP:
                body = body[:ARCHIVE_MSG_CAP]
                cut = " […truncated, read the rest with a smaller limit]"
            lines.append(f"\n#{off + i} {role}{tag}:\n{body}{cut}")
        remaining = len(msgs) - (off + len(page))
        if remaining > 0:
            lines.append(f"\n{remaining} more message(s) in this event - "
                         f"continue with offset={off + len(page)}.")
        else:
            lines.append("\nEnd of this compaction event.")
        return "\n".join(lines)

    def _tool_remember(self, text: str = "", tags: Optional[List[str]] = None,
                       global_: bool = False, **kwargs) -> str:
        """Append one durable note to the ledger.

        `global_` because `global` is a Python keyword; the schema still exposes
        "global", so the dispatcher's **kwargs absorbs it. Accepting the unknown key
        rather than raising matters: a model that writes global=true must not get a
        TypeError for following the documented interface."""
        if "global" in kwargs:
            global_ = _as_bool(kwargs["global"])
        body = str(text or "").strip()
        if not body:
            return "ERROR: remember() needs text."
        # The schema promises "longer is refused, not silently cut", so honour it: refuse
        # anything over the per-entry cap. _ledger_append still truncates as a defensive
        # net for internal callers, but a model calling this tool must never lose the tail
        # of a note without being told. HARD_MAX only changes the wording, flagging what
        # looks like a whole-file paste rather than an over-long note.
        if len(body) > LEDGER_ENTRY_MAX_CHARS:
            if len(body) > LEDGER_ENTRY_HARD_MAX:
                return (f"ERROR: that is {len(body)} chars - far over the "
                        f"{LEDGER_ENTRY_MAX_CHARS}-char entry limit, and too large for a "
                        f"ledger note. Do not paste a file: record the PATH plus what you "
                        f"learned from it, in one or two sentences.")
            return (f"ERROR: that is {len(body)} chars, over the {LEDGER_ENTRY_MAX_CHARS}-char "
                    f"limit. Split it into separate remember() calls, one fact each - nothing "
                    f"was written.")
        cid = "" if _as_bool(global_) else str(self.current_chat_id or "")
        if not cid and not _as_bool(global_):
            return "ERROR: no current chat to attach this note to."
        entry = _ledger_append(cid, body, tags, source="model")
        if entry is None:
            return "ERROR: the ledger could not be written (check the data folder is writable)."
        scope = "global" if _as_bool(global_) else f"chat {cid}"
        where = _ledger_path(cid)
        total = len(_ledger_read_raw(cid))
        return (f"Remembered ({scope}, entry {total}): {entry['text'][:200]}\n"
                f"Appended to {where} - this file is never rewritten or pruned, and it "
                f"survives compaction, handoff and deletion of this chat.\n"
                f"Recent entries are injected into the system prompt automatically; "
                f"read_ledger() reaches the older ones.")

    def _tool_read_ledger(self, chat_id: str = "", tag: str = "", search: str = "",
                          limit: int = LEDGER_PAGE_DEFAULT, offset: int = 0) -> str:
        """Page through a ledger, newest first, optionally filtered."""
        want_star = str(chat_id or "").strip() == "*"
        cid = "" if want_star else (str(chat_id or "").strip() or str(self.current_chat_id or ""))
        if not want_star and not cid:
            return "ERROR: no current chat. Pass chat_id, or \"*\" for the global ledger."
        entries = _ledger_read_raw(cid)
        scope = "global" if want_star else f"chat {cid}"
        if not entries:
            return (f"The {scope} ledger is empty. Add to it with remember(text=…); "
                    f"write global=true for a fact that should apply to every chat.")
        tg = re.sub(r"\s+", "-", str(tag or "")).strip("-").lower()
        if tg:
            entries = [e for e in entries if tg in [str(x).lower() for x in (e.get("tags") or [])]]
        needle = str(search or "").strip().lower()
        if needle:
            entries = [e for e in entries if needle in str(e.get("text") or "").lower()]
        try:
            lim = int(limit) if limit else LEDGER_PAGE_DEFAULT
        except (TypeError, ValueError):
            lim = LEDGER_PAGE_DEFAULT
        lim = max(1, min(lim, LEDGER_PAGE_MAX))
        try:
            off = max(0, int(offset or 0))
        except (TypeError, ValueError):
            off = 0
        entries.reverse()                      # newest first, matching the injected view
        if off >= len(entries):
            return f"{len(entries)} entry(ies) match; offset {off} is past the end."
        page = entries[off:off + lim]
        head = f"{scope} ledger: {len(entries)} matching entr{'y' if len(entries)==1 else 'ies'} "
        head += f"(showing {off}..{off + len(page) - 1} newest-first)\n"
        lines = [head]
        for e in page:
            ts = str(e.get("ts") or "")[:16].replace("T", " ")
            tags = (" #" + " #".join(e.get("tags") or [])) if e.get("tags") else ""
            src = f" [{e.get('source')}]" if e.get("source") not in (None, "model") else ""
            lines.append(f"\n{ts}{src}{tags}:\n{e.get('text')}")
        remaining = len(entries) - (off + len(page))
        if remaining > 0:
            lines.append(f"\n{remaining} more entr{'y' if remaining==1 else 'ies'} - "
                         f"continue with offset={off + len(page)}.")
        else:
            lines.append("\nEnd of ledger.")
        return "\n".join(lines)

    def _tool_set_clipboard(self, text: str = "") -> str:
        """Put text on the system clipboard. Tk access is main-thread only, so this
        marshals a callback and waits, exactly like the clipboard read."""
        if text is None or str(text) == "":
            return "ERROR: set_clipboard needs text."
        payload = str(text)
        if len(payload) > CLIPBOARD_LIMIT:
            payload = payload[:CLIPBOARD_LIMIT]
            truncated = True
        else:
            truncated = False
        result: Dict[str, Any] = {}
        ev = threading.Event()
        # Set by the worker the moment its wait is cut short. The callback checks it
        # when it finally runs: _stop_event alone is not enough, because a NEW turn
        # may have cleared it before this queued callback is drained, letting a
        # cancelled write clobber the clipboard anyway.
        cancelled = {"v": False}

        def do():
            # A cancelled turn must not still mutate a shared system resource: the
            # callback can sit in the UI queue after Stop already released the worker
            # (same reasoning as _ask_permission_modal's guard).
            if cancelled["v"] or self._closed or self._stop_event.is_set():
                result["error"] = "cancelled"
                ev.set()
                return
            try:
                self.root.clipboard_clear()
                self.root.clipboard_append(payload)
                result["ok"] = True
            except Exception as e:
                result["error"] = str(e)
            finally:
                ev.set()

        # Tools normally run on the agentic worker thread, where _post + wait is the
        # only safe way to touch Tk. Called from the MAIN thread that pattern would
        # deadlock (the queue is drained by this very thread), so run inline there.
        if threading.current_thread() is threading.main_thread():
            do()
        else:
            self._post(do)
            # Bounded wait: the main thread runs the UI queue, so this returns fast;
            # a Stop press must not leave the worker parked behind it.
            if not _interruptible_wait(ev, 10,
                                       lambda: self._closed or self._stop_event.is_set()):
                cancelled["v"] = True
                return "ERROR: clipboard write did not complete (cancelled or UI busy)."
        if not result.get("ok"):
            return f"ERROR: could not write the clipboard ({result.get('error', 'unknown')})"
        note = ""
        if truncated:
            note = (f" [truncated to {CLIPBOARD_LIMIT} of {len(str(text))} chars - the input "
                    "was too large for the clipboard]")
        return (f"Clipboard set to {len(payload)} chars.{note} "
                "Paste it into the target application with Ctrl+V.")

    def _tool_transcribe_audio(self, path: str = "", language: str = "") -> str:
        """Transcribe an audio file on this machine with the same local Whisper model
        the microphone uses. Audio never leaves the computer."""
        if WhisperModel is None:
            return ("ERROR: local transcription is unavailable - pip install "
                    "faster-whisper (the mic feature uses it too).")
        try:
            p = self._safe_path(path)
        except PermissionError as e:
            return f"ERROR: Access denied (blocked path): {e}"
        if not p.exists():
            return f"ERROR: File not found: {p}"
        if not p.is_file():
            return f"ERROR: Not a file: {p}"
        try:
            size = p.stat().st_size
            if size > AUDIO_FILE_MAX_BYTES:
                return (f"ERROR: file is {size/1048576:.1f} MB, over the "
                        f"{AUDIO_FILE_MAX_BYTES//1048576} MB limit - split it first.")
            data = p.read_bytes()
        except Exception as e:
            return f"ERROR: could not read {p}: {e}"
        try:
            pcm, seconds = decode_audio_bytes(data)
        except ValueError as e:
            return f"ERROR: {e} ({p.name})"
        except Exception as e:
            return f"ERROR: could not decode {p.name}: {e}"
        if seconds and seconds > AUDIO_MAX_SECONDS:
            keep = int(AUDIO_MAX_SECONDS * AUDIO_SAMPLE_RATE)
            pcm = pcm[:keep]
            cut = f" (cut to the first {AUDIO_MAX_SECONDS}s of {seconds/60:.1f} min)"
        else:
            cut = ""
        if self._abort_requested():
            return "ERROR: cancelled before transcribing."
        try:
            lang = (str(language or "").strip() or None)
            if lang and lang.lower() in ("auto", "detect"):
                lang = None
            t0 = time.time()
            text = _transcribe_pcm(pcm, language=lang or "")
        except Exception as e:
            return f"ERROR: transcription failed: {e}"
        dur = time.time() - t0
        if not text:
            return (f"Transcribed {p.name} ({seconds:.1f}s of audio in {dur:.1f}s): "
                    "no speech detected (silence, music, or unintelligible).")
        head = (f"Transcription of {p.name} "
                f"({seconds:.1f}s audio, {dur:.1f}s to process{cut}, "
                f"{_WHISPER_DEVICE_LABEL}):")
        if len(text) > TOOL_OUTPUT_LIMIT - len(head) - 200:
            text = text[:TOOL_OUTPUT_LIMIT - len(head) - 200] + "\n[... truncated]"
        return head + "\n" + text

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

        # v1.1.43: the _post + blocking-wait pattern DEADLOCKS when called from the
        # main thread, because the queue is drained by that same thread - the wait
        # simply runs out its 10 s and reports "no clipboard access". Run inline there.
        if threading.current_thread() is threading.main_thread():
            do()
        else:
            self._post(do)
            # Backlog #7: this wait used to be a plain ev.wait(10) with no abort
            # predicate, so Stop could not cut it short.
            _interruptible_wait(ev, 10, lambda: self._closed or self._stop_event.is_set())
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

    def _mic_status(self) -> str:
        """Status-line text while dictation is active.

        Always names the engine AND the device (GPU/CPU) so the device is
        verifiable at a glance. Dictated text is deliberately NOT echoed here -
        it already lands in the input box, and echoing it hid the device label.
        A phrase count gives per-phrase feedback without stealing the label.
        """
        if WhisperModel is not None:
            # The device label is only known once the model has actually loaded;
            # before that it is the placeholder "local", which would read as the
            # redundant "local Whisper (local)" - omit it until it is GPU/CPU.
            dev = _WHISPER_DEVICE_LABEL if _WHISPER_DEVICE_LABEL in ("GPU", "CPU") else None
            eng = "local Whisper" + (f" ({dev})" if dev else "")
        else:
            eng = "Google (cloud)"
        n = ""
        if self._mic_phrases:
            n = f" \u00b7 {self._mic_phrases} phrase{'' if self._mic_phrases == 1 else 's'}"
        return (f"\U0001F3A4 Listening via {eng}{n}\u2026 speak freely; "
                f"hit Send (or click the mic) to stop")

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
        self._mic_phrases = 0
        try:
            self.mic_btn.configure(bg=COL["accent"], fg="#FFFFFF")
        except tk.TclError:
            pass
        self._set_status(self._mic_status())
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
            self._mic_phrases += 1
            self._set_status(self._mic_status())
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
                    # "Loading local Whisper model…" while we waited). This is the
                    # first point at which the GPU/CPU label is known.
                    self._post(lambda: self._set_status(self._mic_status()))
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
                _sm = str(e)
                if "pyaudio" in _sm.lower():
                    _sm = ("Dictation needs PyAudio, which is not available in this build "
                           "(pip install pyaudio; compiled exe: rebuild with --hidden-import pyaudio)")
                self._post(lambda m=_sm: self._set_status("\U0001F3A4 " + m))
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
        # Chips just changed size, so the row count may have too. The signature guard
        # in _perm_reflow compares chip widths too, so this reflows even though the
        # canvas width is unchanged.
        self._perm_reflow()

    def open_settings(self) -> None:
        # Hardening (v1.1.12): the modal grab is taken at map time, BEFORE the ~600 lines
        # of dialog build. If that build throws, the half-built dialog has no Cancel button
        # and the grab outlives it; destroying the Toplevel releases the grab, so report
        # and clean up here instead of leaving a frozen shell on screen.
        try:
            self._open_settings_inner()
        except Exception as e:
            traceback.print_exc()
            title = f"{APP_NAME} \u2014 Settings"
            try:
                for w in self.root.winfo_children():
                    if isinstance(w, tk.Toplevel) and w.winfo_exists() and w.title() == title:
                        w.destroy()
            except tk.TclError:
                pass
            try:
                self._set_status("Settings dialog failed to build: " + str(e)[:120])
            except Exception:
                pass

    def _open_settings_inner(self) -> None:
        top = tk.Toplevel(self.root)
        top.title(f"{APP_NAME} — Settings")
        top.configure(bg=COL["bg_main"])
        top.transient(self.root)

        # v1.1.11 Option A: every settings row lives inside a scrollable Canvas body;
        # the Save/Test/Cancel footer is pinned OUTSIDE it, so shrinking the window
        # can never clip rows off the bottom - they scroll into view instead.
        # (Layout binds + geometry land after the footer exists - end of method.)
        body = tk.Frame(top, bg=COL["bg_main"])
        canvas = tk.Canvas(body, bg=COL["bg_main"], highlightthickness=0, bd=0)
        vbar = tk.Scrollbar(body, orient="vertical", command=canvas.yview,
                            bg=COL["bg_raised"], troughcolor=COL["bg_deep"],
                            highlightthickness=0, borderwidth=0)
        canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(canvas, bg=COL["bg_main"])
        _inner_win = canvas.create_window((0, 0), window=inner, anchor="nw")
        # P7b: col-0 row labels collected so the resize handler can re-wrap them
        # (labels built later in this method are appended to this same list).
        row_labels: List[tk.Label] = []

        def _sync_scrollregion(*_a) -> None:
            # The scroll region must be the inner frame's NATURAL size: left free,
            # Tk stretches a "nsew" canvas window to the viewport height, which pins
            # the scrollbar thumb at 100% and scrolling stops working.
            try:
                canvas.configure(scrollregion=canvas.bbox("all") or (0, 0, 1, 1))
            except tk.TclError:
                pass

        def _on_canvas_resize(e) -> None:
            # Rows follow the window width (labels + fields re-flow, never clipped).
            try:
                canvas.itemconfigure(_inner_win, width=max(e.width, 1))
            except tk.TclError:
                pass
            # P7b: the label column must shrink like the field column does.
            # Unwrapped col-0 labels pin the grid's minimum width (the longest is
            # ~90 chars), so the window can never get narrower than their natural
            # width. Re-wrap them against the current viewport (~42% share, floor
            # 180px): wide window -> one line each (unchanged look); narrow window
            # -> labels reflow to 2-3 lines and the label column collapses.
            try:
                wl = max(180, int(e.width * 0.42))
            except (AttributeError, TypeError, ValueError):
                wl = 180
            for _lbl in row_labels:
                try:
                    _lbl.configure(wraplength=wl)
                except tk.TclError:
                    pass

        def _wheel(e):
            # Widget-local (add="+") binds on the body widgets only - no root-level
            # handler exists, so nothing can leak onto other windows after destroy.
            # Windows: <MouseWheel> delta=120/notch (~3 lines ~ one settings row);
            # X11 wheels: Button-4/5 (~1 line); touchpads send many small deltas.
            try:
                if getattr(e, "num", None) == 4:
                    d = 1        # X11 wheel up
                elif getattr(e, "num", None) == 5:
                    d = -1       # X11 wheel down
                else:
                    d = (e.delta > 0) - (e.delta < 0)
            except (AttributeError, ValueError, TypeError):
                d = 0
            if d:
                try:
                    # Fraction-based stepping: yview_scroll("units") means pixels on
                    # some Tk builds and lines on others - fractions are exact. ~60px
                    # per Windows notch (~4 rows), ~18px for small trackpad deltas.
                    bb = canvas.bbox("all")
                    total = (bb[3] - bb[1]) if bb else 0
                    px = 60 if abs(int(getattr(e, "delta", 0) or 0)) >= 120 else 18
                    if total > 0:
                        top_ = canvas.yview()[0] - d * (px / total)
                        canvas.yview_moveto(max(0.0, min(top_, 1.0)))
                except tk.TclError:
                    pass
            return "break"   # own the wheel; no propagation to siblings / root

        def _bind_body_wheel(w) -> None:
            for c in w.winfo_children():
                if isinstance(c, (tk.Text, tk.Listbox)):
                    continue        # multi-line boxes keep their native scrolling
                _bind_body_wheel(c)
            try:
                w.bind("<MouseWheel>", _wheel, add="+")
                w.bind("<Button-4>", _wheel, add="+")
                w.bind("<Button-5>", _wheel, add="+")
            except tk.TclError:
                pass

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
            ("Compaction threshold % (summarize old messages in-place at this usage; keep <= Handoff threshold so it fires first; 0 = off)", "compaction_threshold"),
            ("Compaction keep-recent (most recent messages kept verbatim when compacting)", "compaction_keep_recent"),
            ("Chat render window (messages shown per chat view; older ones behind 'Load earlier messages'; 0 = show all)", "chat_render_window"),
            ("Ledger in system prompt (chars of append-only notes injected each request; 0 = off, read_ledger() still reaches them)", "ledger_inject_chars"),
            ("Project memory index (chars of the workspace MEMORY.md injected each request; 0 = off)", "project_memory_chars"),
            ("Project memory file (blank = <workspace>/MEMORY.md)", "project_memory_file"),
            ("Custom System Prompt (appended to the built-in system prompt on every request; blank = none)", "custom_system_prompt"),
            ("Exa API Key (blank = Exa Search disabled)", "exa_api_key"),
            ("Firecrawl API Key (blank = Firecrawl Scrape disabled)", "firecrawl_api_key"),
            ("Allow Local Network (fetch_url)", "allow_local_network"),
            ("MCP Servers (stdio subprocesses; tools appear in the permission bar)", "mcp_servers"),
            ("Skills folder (extra folder scanned for SKILL.md skills; blank = <workspace>/skills only)", "skills_dir"),
            # Consultant (ask_expert). Independent of the three rows above: the
            # local model stays primary and these only ever get consulted.
            ("Expert Server URL (ask_expert; blank = tool off)", "expert_server_url"),
            ("Expert API Key (ask_expert; blank = tool off)", "expert_api_key"),
            ("Expert Model (exact ID, e.g. gemini-3.8-flash)", "expert_model"),
        ]
        entries: Dict[str, tk.Entry] = {}
        model_combo: Optional[ttk.Combobox] = None
        expert_combo: Optional[ttk.Combobox] = None
        local_net_var: Optional[tk.BooleanVar] = None
        _note_var: Optional[tk.BooleanVar] = None
        custom_prompt_text: Optional[tk.Text] = None
        cp_enabled_var: Optional[tk.BooleanVar] = None
        for i, (label, key) in enumerate(rows):
            if key == "mcp_servers":
                # Managed list of stdio MCP servers (Add/Remove); connecting is
                # live - discovered tools appear in the permission bar.
                _lbl0 = tk.Label(inner, text=label, bg=COL["bg_main"], fg=COL["text_dim"],
                                 font=F(11))
                _lbl0.grid(row=i, column=0, sticky="w", padx=(16, 8), pady=6)
                row_labels.append(_lbl0)   # P7b: shrinkable label column
                mcp_frame = tk.Frame(inner, bg=COL["bg_main"])
                mcp_frame.grid(row=i, column=1, columnspan=2, sticky="ew", pady=6, padx=(0, 8))
                self._draw_mcp_server_rows(mcp_frame, top)
                continue
            _lbl0 = tk.Label(inner, text=label, bg=COL["bg_main"], fg=COL["text_dim"],
                             font=F(11))
            _lbl0.grid(row=i, column=0, sticky="w", padx=(16, 8), pady=6)
            row_labels.append(_lbl0)   # P7b: shrinkable label column
            if key == "sampling_defaults":
                # Sampler parameters (Top P / Top K / Min P / Repetition / Presence / Frequency),
                # GLOBAL for every chat session by design - the chat UI stays uncluttered.
                # Built in a sub-frame so these six rows never appear among the dialog's own
                # children (the smoke suite counts plain Entries there positionally).
                # Label row: the generic tk.Label(inner, ...) above this branch already placed it.
                samp_frame = tk.Frame(inner, bg=COL["bg_main"])
                samp_frame.grid(row=i, column=1, columnspan=2, sticky="ew", pady=6, padx=(0, 8))
                # P7: weight the entry column so the verify-results box (sticky ew)
                # stretches to the SAME right edge as every other field (its fixed
                # width=58 chars used to leave a black void beside it).
                samp_frame.columnconfigure(1, weight=1)
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
                # P7: the combo shares its cell (col 1, spanning to the old col 2
                # edge) with the Refresh / Use-loaded buttons built below, so the
                # buttons sit adjacent to the field and follow it on resize.
                model_row = tk.Frame(inner, bg=COL["bg_main"])
                model_row.grid(row=i, column=1, columnspan=2, sticky="ew",
                               pady=6, padx=(0, 8))
                model_combo = ttk.Combobox(model_row, values=[], width=44,
                                           style="Tool.TCombobox", font=F(11))
                model_combo.set(self.settings.get("model_name", ""))
                model_combo.pack(side="left", fill="x", expand=True)
                continue
            if key == "expert_model":
                # Editable combobox, same shape as Model Name: the exact ID
                # matters (a wrong one is a 400/404), and "List models" fills it
                # from the expert endpoint's own /models so the name is never
                # guessed. Free typing still works for providers that do not
                # list models.
                exp_row = tk.Frame(inner, bg=COL["bg_main"])
                exp_row.grid(row=i, column=1, columnspan=2, sticky="ew",
                             pady=6, padx=(0, 8))
                expert_combo = ttk.Combobox(exp_row, values=[], width=44,
                                            style="Tool.TCombobox", font=F(11))
                expert_combo.set(self.settings.get("expert_model", ""))
                expert_combo.pack(side="left", fill="x", expand=True)
                continue
            if key == "custom_system_prompt":
                # ONE row, two controls: the enable checkbox sits directly above the
                # text box in the same col-1 cell, because the toggle and the text are
                # one logical control. A separate row made them read as two unrelated
                # settings.
                #
                # Why the checkbox is NOT in col 0 (where the label usually goes):
                #  - the col-0 label is what p7b's re-wrap machinery sizes and shrinks
                #    (row_labels); a Checkbutton there is not in that list, so its long
                #    text would pin the label column's minimum width - the exact bug
                #    P7b fixed.
                #  - test_p7_settings_layout asserts one col-0 label PER ROW, so a row
                #    without one fails the census.
                # Keeping the label and stacking the controls in col 1 satisfies both.
                #
                # Gates INJECTION only: unticking keeps the text, so a persona can be
                # switched off for one session and back on later with no retyping - the
                # whole reason the control exists.
                cp_frame = tk.Frame(inner, bg=COL["bg_main"])
                cp_frame.grid(row=i, column=1, sticky="ew", pady=6, padx=(0, 8))
                cp_frame.columnconfigure(0, weight=1)
                cp_enabled_var = tk.BooleanVar(
                    value=bool(self.settings.get("custom_system_prompt_enabled", True)))
                tk.Checkbutton(cp_frame, text="Enable (unticked = the text below is kept but NOT sent)",
                               variable=cp_enabled_var, bg=COL["bg_main"], fg=COL["text"],
                               activebackground=COL["bg_main"], activeforeground=COL["text"],
                               selectcolor=COL["bg_deep"], font=F(10)) \
                    .grid(row=0, column=0, sticky="w", pady=(0, 4))
                # Multi-line: persona/style instructions can run to several lines.
                custom_prompt_text = tk.Text(cp_frame, width=44, height=5, wrap="word",
                                             bg=COL["bg_deep"], fg=COL["text"],
                                             insertbackground=COL["text"], relief="flat",
                                             font=F(10))
                custom_prompt_text.insert("1.0", self.settings.get("custom_system_prompt", ""))
                custom_prompt_text.grid(row=1, column=0, sticky="ew")
                continue
            if key == "allow_local_network":
                local_net_var = tk.BooleanVar(value=bool(self.settings.get("allow_local_network", False)))
                tk.Checkbutton(inner, text="Allow fetch_url to reach local network addresses "
                               "(localhost, LAN, cloud metadata) - off by default",
                               variable=local_net_var, bg=COL["bg_main"], fg=COL["text"],
                               activebackground=COL["bg_main"], activeforeground=COL["text"],
                               selectcolor=COL["bg_deep"], font=F(11)) \
                    .grid(row=i, column=1, sticky="w", pady=6, padx=(0, 8))
                continue
            # P7: the two folder-picker rows get their sub-frame HERE (at their row
            # position in the loop) so the dialog's child order stays exactly the row
            # order - the smoke suite walks winfo_children() and counts plain Entries
            # positionally; frames appended after the loop would reorder the census.
            _rowf: Optional[tk.Frame] = None
            if key == "file_workspace":
                ws_rowf = tk.Frame(inner, bg=COL["bg_main"])
                ws_rowf.grid(row=i, column=1, columnspan=2, sticky="ew", pady=6, padx=(0, 8))
                _rowf = ws_rowf
            elif key == "handoff_notes_dir":
                nd_rowf = tk.Frame(inner, bg=COL["bg_main"])
                nd_rowf.grid(row=i, column=1, columnspan=2, sticky="ew", pady=6, padx=(0, 8))
                _rowf = nd_rowf
            elif key == "skills_dir":
                sk_rowf = tk.Frame(inner, bg=COL["bg_main"])
                sk_rowf.grid(row=i, column=1, columnspan=2, sticky="ew", pady=6, padx=(0, 8))
                _rowf = sk_rowf
            e = tk.Entry(_rowf or inner, width=44, bg=COL["bg_deep"], fg=COL["text"],
                         insertbackground=COL["text"], relief="flat",
                         show="*" if key in ("api_key", "exa_api_key", "firecrawl_api_key", "expert_api_key") else "")
            e.insert(0, self.settings.get(key, ""))
            if _rowf is None:
                e.grid(row=i, column=1, sticky="ew", pady=6, padx=(0, 8))
            entries[key] = e

        # File Workspace row: folder picker ADJACENT to its entry (P7). Entry and
        # button share the ws_rowf sub-frame (built at this row's position in the
        # loop above), so the button sits right of the field and moves with it on
        # resize (was a col-2 grid child pinned to the far edge - it drifted away
        # from the entry when widened).
        def _browse_ws() -> None:
            cur = entries["file_workspace"].get().strip()
            d = filedialog.askdirectory(parent=top, title="Choose the File Workspace folder",
                                        initialdir=(cur or str(Path.home())))
            if d:
                entries["file_workspace"].delete(0, "end")
                entries["file_workspace"].insert(0, d)

        entries["file_workspace"].pack(in_=ws_rowf, side="left", fill="x", expand=True)
        tk.Button(ws_rowf, text="Browse...", command=_browse_ws, bg=COL["bg_raised"], fg=COL["text"],
                  activebackground="#4B5563", relief="flat", bd=0, padx=10, pady=2,
                  cursor="hand2", font=F(10)).pack(side="left", padx=(8, 0))

        # Handoff notes folder row: picker + the EFFECTIVE folder shown live, so it is always
        # obvious where notes are being written (blank field = the app default working folder).
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

        # P7: same treatment as File Workspace - entry, picker and the live
        # "Notes will be written to" note share one sub-frame in col 1; the note
        # gets its own line below the field (was packed BESIDE the Browse button,
        # whose 260px wrap warped the row width on every resize).
        entries["handoff_notes_dir"].pack(in_=nd_rowf, side="top", fill="x")
        nd_btns = tk.Frame(nd_rowf, bg=COL["bg_main"])
        nd_btns.pack(side="top", anchor="w", pady=(2, 0))
        tk.Button(nd_btns, text="Browse...", command=_browse_notes, bg=COL["bg_raised"], fg=COL["text"],
                  activebackground="#4B5563", relief="flat", bd=0, padx=10, pady=2,
                  cursor="hand2", font=F(10)).pack(side="left")
        notes_dir_lbl = tk.Label(nd_rowf, text="", bg=COL["bg_main"], fg=COL["text_dim"],
                                 font=F(9), justify="left", wraplength=260)
        notes_dir_lbl.pack(side="top", anchor="w", padx=(8, 0))
        _update_notes_note()
        entries["handoff_notes_dir"].bind("<KeyRelease>", lambda _e: _update_notes_note())

        # Skills folder row: picker + the folders that will ACTUALLY be scanned, so the
        # defaults (<workspace>/skills, <script dir>/skills) are visible without guessing.
        def _browse_skills() -> None:
            cur = entries["skills_dir"].get().strip()
            d = filedialog.askdirectory(parent=top, title="Choose the extra skills folder",
                                        initialdir=(cur or str(USER_WORKSPACE_ROOT / SKILLS_SUBDIR)))
            if d:
                entries["skills_dir"].delete(0, "end")
                entries["skills_dir"].insert(0, d)
                _update_skills_note()

        def _update_skills_note() -> None:
            try:
                found = _discover_skills(self.settings)
                scanned = _skill_dirs({"skills_dir": entries["skills_dir"].get().strip()})
                skills_dir_lbl.configure(
                    text="Scanning:\n" + "\n".join(str(p) for p in scanned)
                         + f"\n{len(found)} skill(s) currently installed")
            except tk.TclError:
                pass
            except Exception:
                try:
                    skills_dir_lbl.configure(text="(skills scan failed - check the path)")
                except tk.TclError:
                    pass

        entries["skills_dir"].pack(in_=sk_rowf, side="top", fill="x")
        sk_btns = tk.Frame(sk_rowf, bg=COL["bg_main"])
        sk_btns.pack(side="top", anchor="w", pady=(2, 0))
        tk.Button(sk_btns, text="Browse...", command=_browse_skills, bg=COL["bg_raised"], fg=COL["text"],
                  activebackground="#4B5563", relief="flat", bd=0, padx=10, pady=2,
                  cursor="hand2", font=F(10)).pack(side="left")
        skills_dir_lbl = tk.Label(sk_rowf, text="", bg=COL["bg_main"], fg=COL["text_dim"],
                                  font=F(9), justify="left", wraplength=260)
        skills_dir_lbl.pack(side="top", anchor="w", padx=(8, 0))
        _update_skills_note()
        entries["skills_dir"].bind("<KeyRelease>", lambda _e: _update_skills_note())

        # Model Name row: Refresh button re-fetches {server_url}/models in a
        # background thread (same pattern as Test Connection). The request carries
        # no model name, so it works even with a stale/wrong saved model_name.
        # (P7: the buttons live in model_row next to the combobox - no grid anchor needed.)
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

        # P7: both buttons pack into model_row (built with the combobox above),
        # adjacent to the field - no longer col-2/col-3 grid children adrift in
        # the far-right dead space of the stretched row.
        refresh_btn = tk.Button(model_row, text="Refresh models",
                                command=lambda: _refresh_models(False, auto_fix=False),
                                bg=COL["bg_raised"], fg=COL["text"], activebackground="#4B5563",
                                relief="flat", bd=0, padx=10, pady=2, cursor="hand2", font=F(10))
        refresh_btn.pack(side="left", padx=(8, 0))

        # "Use loaded model": copies the single loaded model's exact server ID into
        # Model Name - the reliable fix when you do not know the name yourself.
        use_btn = tk.Button(model_row, text="Use loaded model", state="disabled",
                            bg=COL["bg_raised"], fg=COL["text"], activebackground="#4B5563",
                            relief="flat", bd=0, padx=10, pady=2, cursor="hand2", font=F(10))
        use_btn.pack(side="left", padx=(8, 0))

        def _use_loaded() -> None:
            try:
                if len(loaded_ids) == 1 and top.winfo_exists():
                    model_combo.set(loaded_ids[0])
                    self._set_status(f"Model Name set to loaded model: {loaded_ids[0]}")
            except tk.TclError:
                pass

        use_btn.configure(command=_use_loaded)

        # ── Expert model list (ask_expert) ───────────────────────────
        # The expert endpoint's own /models is the only reliable source for the
        # exact ID; guessing it produced a name that is not in Google's list at
        # all. Fetched on the expert URL+key, NOT the primary ones.
        def _refresh_expert_models() -> None:
            eurl = entries["expert_server_url"].get().strip()
            ekey = entries["expert_api_key"].get().strip()
            if not eurl or not ekey:
                exp_lbl.configure(text="Set Expert Server URL and API Key first.")
                return
            try:
                exp_btn.configure(text="Loading...")
                expert_combo.configure(state="disabled")
            except tk.TclError:
                return

            def _eworker():
                info = fetch_model_info(eurl, ekey, timeout=20)
                # Gemini's /models returns "models/<name>" while its own chat
                # examples use the bare "<name>". Offer the bare form (what the
                # docs show); _tool_ask_expert retries the other form if the
                # endpoint disagrees, so either one works.
                ids = []
                for m in info:
                    mid = str(m.get("id") or "").strip()
                    if not mid:
                        continue
                    bare = mid[7:] if mid.startswith("models/") else mid
                    if bare not in ids:
                        ids.append(bare)

                def _edone():
                    try:
                        if not top.winfo_exists():
                            return
                        hist = [h for h in (self.settings.get("expert_model_history") or [])
                                if isinstance(h, str)]
                        merged = list(ids) + [h for h in hist if h not in ids]
                        expert_combo.configure(values=merged, state="normal")
                        exp_btn.configure(text="List models")
                        if ids:
                            pro = [i for i in ids if "pro" in i.lower()]
                            exp_lbl.configure(
                                text=f"{len(ids)} models from the expert endpoint"
                                     + (f" - Pro: {', '.join(pro[:4])}" if pro else ""))
                        else:
                            exp_lbl.configure(text="No model list from that endpoint "
                                                  "(check the URL/key) - type the ID.")
                    except tk.TclError:
                        pass
                self._post(_edone)

            threading.Thread(target=_eworker, daemon=True).start()

        exp_btn = tk.Button(exp_row, text="List models",
                            command=_refresh_expert_models,
                            bg=COL["bg_raised"], fg=COL["text"], activebackground="#4B5563",
                            relief="flat", bd=0, padx=10, pady=2, cursor="hand2", font=F(10))
        exp_btn.pack(side="left", padx=(8, 0))
        exp_lbl = tk.Label(exp_row, text="", bg=COL["bg_main"], fg=COL["text_dim"],
                           font=F(9), justify="left")
        exp_lbl.pack(side="left", padx=(8, 0))
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
                # v1.1.12: leave other applications alone. The poller exists to undo OUR
                # OWN main window stealing keyboard focus back after a Toplevel opens (the
                # documented Windows quirk), which always happens with our PID in the
                # foreground. If the Windows foreground window belongs to another process
                # (Alt-Tab away), stay out of it. Any failure -> False = original behaviour.
                foreign = False
                if os.name == "nt":
                    try:
                        # Imported here, not at module level: on non-Windows builds
                        # where the 'v' (VARIANT_BOOL) typecode is unregistered,
                        # ctypes.wintypes raises ValueError at IMPORT time. CPython
                        # registered 'v' everywhere in Oct 2020 (3.10 cycle), so this
                        # only bites on older interpreters - but a top-level import
                        # would then kill startup on Linux/macOS. Inside this nt-only
                        # branch it can never be reached on those platforms.
                        import ctypes.wintypes
                        fg_hwnd = int(ctypes.windll.user32.GetForegroundWindow())
                        fg_pid = ctypes.wintypes.DWORD()
                        ctypes.windll.user32.GetWindowThreadProcessId(
                            fg_hwnd, ctypes.byref(fg_pid))
                        foreign = bool(fg_hwnd) and int(fg_pid.value) != os.getpid()
                    except Exception:
                        foreign = False
                if not inside and not child_open and not foreign:
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
            # Skills folder: like the notes folder, blank = the app defaults. Anything
            # else must be an ABSOLUTE folder that already EXISTS - skills are read-only
            # input, so creating a missing one would only hide a typo in the path.
            sk_raw = entries["skills_dir"].get().strip() if "skills_dir" in entries else ""
            sk_p: Optional[Path] = None
            if sk_raw:
                try:
                    sk_p = Path(os.path.expanduser(sk_raw)).resolve()
                    _skhit = _forbidden_dir_hit(str(sk_p).lower())
                    if _skhit is not None:
                        raise PermissionError(f"that folder is inside a protected system directory ({_skhit})")
                    if not sk_p.is_dir():
                        raise FileNotFoundError("that folder does not exist")
                except Exception as e:
                    messagebox.showwarning(
                        f"{APP_NAME} - Skills folder",
                        "That folder cannot be used for skills:\n\n"
                        f"{e}\n\nLeave the field blank to scan only the app defaults: "
                        f"{USER_WORKSPACE_ROOT / SKILLS_SUBDIR}")
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
            if cp_enabled_var is not None:
                self.settings["custom_system_prompt_enabled"] = bool(cp_enabled_var.get())
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
            # Expert Model comes from its own combobox (not in `entries`, so the
            # loop above cannot see it) - without this the field would never save.
            exp_val = str(expert_combo.get()).strip() if expert_combo is not None else ""
            self.settings["expert_model"] = exp_val
            if exp_val:
                eh = [h for h in (self.settings.get("expert_model_history") or [])
                      if isinstance(h, str) and h != exp_val]
                eh.insert(0, exp_val)
                self.settings["expert_model_history"] = eh[:MODEL_HISTORY_MAX]
            if model_val:
                hist = [h for h in (self.settings.get("model_history") or [])
                        if isinstance(h, str) and h != model_val]
                hist.insert(0, model_val)               # newest first
                self.settings["model_history"] = hist[:MODEL_HISTORY_MAX]
            if ws_raw:
                self.settings["file_workspace"] = str(ws_p)   # normalized absolute path
            if nd_p is not None:
                self.settings["handoff_notes_dir"] = str(nd_p)   # normalized absolute path
            if sk_p is not None:
                self.settings["skills_dir"] = str(sk_p)          # normalized absolute path
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
            # Compaction threshold: numeric percent, clamped to 0..95 (invalid -> default).
            # Capped at 95 so the summary call itself always fits in the remaining window.
            try:
                ct = int(float(self.settings.get("compaction_threshold", COMPACTION_THRESHOLD_DEFAULT)))
            except (TypeError, ValueError):
                ct = COMPACTION_THRESHOLD_DEFAULT
            self.settings["compaction_threshold"] = max(0, min(95, ct))
            # Compaction keep-recent: message count, clamped to 2..50 (invalid -> default).
            try:
                ck = int(float(self.settings.get("compaction_keep_recent", COMPACTION_KEEP_RECENT_DEFAULT)))
            except (TypeError, ValueError):
                ck = COMPACTION_KEEP_RECENT_DEFAULT
            self.settings["compaction_keep_recent"] = max(2, min(50, ck))
            # Chat render window: message count, clamped to 0..2000 (invalid -> default).
            try:
                _rw = int(float(self.settings.get("chat_render_window", CHAT_RENDER_WINDOW_DEFAULT)))
            except (TypeError, ValueError):
                _rw = CHAT_RENDER_WINDOW_DEFAULT
            self.settings["chat_render_window"] = max(0, min(2000, _rw))
            # Ledger injection budget: chars, clamped to 0..LEDGER_INJECT_MAX (invalid ->
            # default). 0 means OFF (nothing injected; read_ledger() still reaches every
            # entry). _ledger_inject clamps again at read time, so a hand-edited settings
            # file cannot make a request carry an unbounded block.
            try:
                _lg = int(float(self.settings.get("ledger_inject_chars", LEDGER_INJECT_DEFAULT)))
            except (TypeError, ValueError):
                _lg = LEDGER_INJECT_DEFAULT
            self.settings["ledger_inject_chars"] = max(0, min(LEDGER_INJECT_MAX, _lg))
            # Project memory index budget: same clamp shape as the ledger, and 0 = OFF.
            # _project_memory_text clamps again at read time, so a hand-edited settings
            # file cannot make every request carry an unbounded index.
            try:
                _pm = int(float(self.settings.get("project_memory_chars",
                                                  PROJECT_MEMORY_DEFAULT)))
            except (TypeError, ValueError):
                _pm = PROJECT_MEMORY_DEFAULT
            self.settings["project_memory_chars"] = max(0, min(PROJECT_MEMORY_MAX, _pm))
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
            _save_settings_geometry()
            top.destroy()
            self._apply_ui_font()   # live re-scale of all fonts (no restart needed)
            self._set_status("⚙ Settings saved")

        btns = tk.Frame(top, bg=COL["bg_main"])
        body.grid(row=0, column=0, sticky="nsew")      # scrollable rows above the footer
        btns.grid(row=1, column=0, sticky="ew", pady=(14, 10))   # pinned at the bottom
        top.columnconfigure(0, weight=1)
        top.rowconfigure(0, weight=1)     # body absorbs resize; footer keeps its height
        inner.columnconfigure(1, weight=1)    # entry column stretches with the window width
        canvas.bind("<Configure>", _on_canvas_resize)
        inner.bind("<Configure>", _sync_scrollregion)
        _bind_body_wheel(canvas)          # AFTER every body widget exists (recursive)
        # Expose the binder so widgets built LATER (MCP Add/Remove row redraws) can join
        # the set: Tk does NOT propagate <MouseWheel> to ancestors, so an unbound new
        # child is a scroll dead-zone (probe _probe_wheel_propagation.py).
        self._settings_bind_wheel = _bind_body_wheel

        # -- Persistent geometry (v1.1.11 Option A, second half) ------
        top.update_idletasks()            # inner/btns natural request sizes valid now
        reqw = (inner.winfo_reqwidth() + vbar.winfo_reqwidth() + 24)   # rows + bar + pad
        reqh = inner.winfo_reqheight() + btns.winfo_reqheight() + 28   # + footer padding
        cap_w, cap_h = top.winfo_screenwidth(), top.winfo_screenheight()
        saved_geo = str(self.settings.get("settings_geometry") or "").strip()
        m_ = re.fullmatch(r"(-?\d+)x(-?\d+)([+-]\d+)([+-]\d+)", saved_geo)
        if m_:
            # Restore the exact position + size of the previous session.
            gw, gh, gx, gy = int(m_[1]), int(m_[2]), int(m_[3]), int(m_[4])
            sw, sh = cap_w, cap_h
            # Unplugged-monitor / off-screen geometry: recentre on the main screen.
            if not (gx > -gw + 60 and gx < sw - 60 and gy > -gh + 60 and gy < sh - 60):
                gx, gy = max(0, (sw - gw) // 2), max(0, (sh - gh) // 2)
            top.geometry(f"{gw}x{gh}{gx:+d}{gy:+d}")
        else:
            # FIRST open ever: the dialog's natural FULL size - exactly what it
            # rendered at before scrollbars existed - capped to (nearly) the full
            # screen height, so a tall dialog on a short display scrolls with the
            # footer visible instead of pushing rows under the bottom edge.
            # No offset part in the string -> the WM centers the window.
            top.geometry(f"{min(reqw, cap_w - 40)}x{min(reqh, cap_h - 40)}")

        def _save_settings_geometry() -> None:
            # Routes BOTH Close buttons (Save/Cancel) and WM_DELETE_WINDOW: remembers
            # where the dialog was; settings CONTENTS are untouched by this path.
            try:
                if top.winfo_exists():
                    self.settings["settings_geometry"] = top.geometry()
            except tk.TclError:
                pass

        def _close_settings() -> None:
            _save_settings_geometry()
            top.destroy()
            # Flush to disk NOW. Contents at this point are exactly the last-saved
            # settings plus the new geometry, so Cancel semantics are untouched; before
            # v1.1.12 an unclean app kill lost the geometry with no second chance.
            try:
                save_settings(self.settings)
            except Exception:
                pass

        top.protocol("WM_DELETE_WINDOW", _close_settings)
        loaded_lbl.pack(in_=btns, side="top", pady=(0, 4))   # "Loaded on server: <exact id>"

        tk.Label(btns, text=f"Data files saved in: {SETTINGS_FILE.parent}", bg=COL["bg_main"],
                 fg=COL["text_dim"], font=F(9)).pack(side="top")

        def test():
            # Read every field on the MAIN thread: Test must try the key as TYPED in the
            # dialog, not the previously saved settings (a freshly pasted key was tested
            # against the stale saved one -> bogus 401 before v1.1.12).
            url = (entries["server_url"].get().strip() or "http://localhost:11434/v1")
            key = entries["api_key"].get().strip()
            threading.Thread(target=self._test_connection, args=(url, key, top), daemon=True).start()

        tk.Button(btns, text="Save", command=save, bg=COL["accent"], fg="#FFFFFF",
                  activebackground="#2563EB", relief="flat", bd=0, padx=18, pady=6,
                  cursor="hand2", font=F(11, "bold")).pack(side="left", padx=6)
        tk.Button(btns, text="Test Connection", command=test, bg=COL["bg_raised"], fg=COL["text"],
                  activebackground="#4B5563", relief="flat", bd=0, padx=12, pady=6,
                  cursor="hand2", font=F(11)).pack(side="left", padx=6)
        tk.Button(btns, text="Cancel", command=_close_settings, bg=COL["bg_raised"], fg=COL["text"],
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
        new_desc: Dict[str, str] = {}
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
                # Same fallback mcp_tool_schema uses, so the tip matches what the
                # model was actually told about this tool.
                new_desc[ns] = _one_line_desc(
                    str(t.get("description") or "").strip()
                    or f"MCP tool {raw} from server {client.name}")
                out.append(mcp_tool_schema(client.name, raw, t))
        self._mcp_tool_map = new_map   # full rebuild: stale entries drop out
        self._mcp_tool_desc = new_desc
        return out

    def _mcp_clients_snapshot(self) -> List["MCPClient"]:
        """Connected clients, copied under the lock (the Settings worker may mutate
        the dict on another thread; iterating it live can raise RuntimeError)."""
        with self._mcp_lock:
            return [c for c in self._mcp_clients.values() if c.ready()]

    def _mcp_pick_server(self, want: str, have_attr: str) -> tuple:
        """(client, error_text) for a resources/prompts tool call.

        Blank `want` is honoured only when EXACTLY one connected server offers the
        feature - guessing between two would silently read the wrong server's data.
        """
        clients = self._mcp_clients_snapshot()
        if not clients:
            return (None, "ERROR: no MCP server is connected. Configure one in "
                          "Settings -> MCP Servers.")
        capable = [c for c in clients if getattr(c, have_attr, None)]
        if not capable:
            return (None, "ERROR: none of the connected MCP servers exposes this "
                          f"({have_attr}). They may not implement that part of MCP.")
        want = (want or "").strip()
        if not want:
            if len(capable) == 1:
                return (capable[0], None)
            return (None, "ERROR: several MCP servers expose this - name one: "
                          + ", ".join(c.name for c in capable))
        for c in capable:
            if c.name == want:
                return (c, None)
        return (None, f"ERROR: no connected MCP server named {want!r}. "
                      f"Available: {', '.join(c.name for c in capable)}")

    def _tool_list_mcp_resources(self, server: str = "") -> str:
        client, err = self._mcp_pick_server(server, "resources")
        if err:
            return err
        try:
            res = client.list_resources()
        except Exception as e:
            return f"ERROR: resources/list on '{client.name}' failed: {e}"
        items = [r for r in (res.get("resources") or []) if isinstance(r, dict)]
        if not items:
            return f"MCP server '{client.name}' reports no resources."
        lines = [f"Resources from MCP server '{client.name}' ({len(items)}):"]
        for r in items[:MCP_LIST_CAP]:
            uri = str(r.get("uri") or "")
            name = str(r.get("name") or "")
            mime = str(r.get("mimeType") or "")
            desc = str(r.get("description") or "")[:160]
            lines.append(f"- {uri}" + (f"  [{name}]" if name else "")
                         + (f"  ({mime})" if mime else "")
                         + (f"  {desc}" if desc else ""))
        if len(items) > MCP_LIST_CAP:
            lines.append(f"[... {len(items) - MCP_LIST_CAP} more; pass a cursor to "
                         "the server for the rest]")
        if res.get("nextCursor"):
            lines.append(f"(more available; nextCursor={str(res['nextCursor'])[:60]})")
        return "\n".join(lines)

    def _tool_read_mcp_resource(self, uri: str = "", server: str = "") -> str:
        if not (uri or "").strip():
            return "ERROR: No resource URI provided. Call list_mcp_resources first."
        client, err = self._mcp_pick_server(server, "resources")
        if err:
            return err
        try:
            text, images = client.read_resource(uri.strip())
        except Exception as e:
            return f"ERROR: resources/read on '{client.name}' failed: {e}"
        if images:
            # Same path as MCP tool images: save and queue for visual analysis.
            vision = self._build_mcp_vision_message(f"read_mcp_resource {uri}", images)
            if vision is not None:
                self._pending_mcp_images.append(vision)
                self._post(lambda u=uri: self.render_note(
                    f"\U0001F5BC Resource {u} attached for visual analysis"))
        body = text or "(resource returned no text content)"
        if len(body) > TOOL_OUTPUT_LIMIT // 2:
            body = body[:TOOL_OUTPUT_LIMIT // 2] + f"\n[... truncated, {len(text)} chars total]"
        return f"Resource {uri} (from MCP server '{client.name}'):\n\n{body}"

    def _tool_list_mcp_prompts(self, server: str = "") -> str:
        client, err = self._mcp_pick_server(server, "prompts")
        if err:
            return err
        try:
            res = client.list_prompts()
        except Exception as e:
            return f"ERROR: prompts/list on '{client.name}' failed: {e}"
        items = [p for p in (res.get("prompts") or []) if isinstance(p, dict)]
        if not items:
            return f"MCP server '{client.name}' reports no prompts."
        lines = [f"Prompts from MCP server '{client.name}' ({len(items)}) - "
                 "fetch one with get_mcp_prompt:"]
        for p in items[:MCP_LIST_CAP]:
            name = str(p.get("name") or "")
            desc = str(p.get("description") or "")[:200]
            args = [str(a.get("name")) + ("" if a.get("required") else "?")
                    for a in (p.get("arguments") or []) if isinstance(a, dict)]
            lines.append(f"- {name}" + (f"({', '.join(args)})" if args else "()")
                         + (f"  {desc}" if desc else ""))
            lines.append("    (? = optional argument)")
        if len(items) > MCP_LIST_CAP:
            lines.append(f"[... {len(items) - MCP_LIST_CAP} more]")
        return "\n".join(lines)

    def _tool_get_mcp_prompt(self, name: str = "", server: str = "",
                             arguments: Optional[dict] = None) -> str:
        if not (name or "").strip():
            return "ERROR: No prompt name provided. Call list_mcp_prompts first."
        client, err = self._mcp_pick_server(server, "prompts")
        if err:
            return err
        args = arguments if isinstance(arguments, dict) else {}
        try:
            desc, msgs = client.get_prompt(name.strip(), args)
        except Exception as e:
            return f"ERROR: prompts/get {name!r} on '{client.name}' failed: {e}"
        if not msgs:
            return f"Prompt {name!r} returned no messages."
        lines = [f"Prompt {name!r} from MCP server '{client.name}'"
                 + (f": {desc}" if desc else "") + " (for you to read and use; "
                 "not injected into the conversation)"]
        for m in msgs[:20]:
            lines.append(f"\n--- {m.get('role','user')} ---\n{m.get('text','')}")
        if len(msgs) > 20:
            lines.append(f"[... {len(msgs) - 20} more messages]")
        return "\n".join(lines)

    def _mcp_dispatch(self, name: str, args: dict, images_out: Optional[list] = None) -> str:
        """Execute a namespaced MCP tool call; returns its text output.

        Any image content the server returned is appended to images_out (as
        {"mime","data"} base64 dicts) for the caller to save and hand to the vision
        path. The tool's TEXT result is what goes back into the conversation as the
        tool message - images ride in a separate follow-up user message, exactly like
        capture_screen does."""
        entry = self._mcp_tool_map.get(name)
        if entry is None:
            return f"ERROR: MCP tool '{name}' is not available (server not connected)."
        server_name, raw = entry
        client = self._mcp_clients.get(server_name)
        if client is None or not client.ready():
            return (f"ERROR: MCP server '{server_name}' is not running. "
                    f"Reconnect it in Settings -> MCP Servers.")
        try:
            text, images = client.call_tool_parts(raw, args, abort=self._abort_requested)
        except Exception as e:
            return f"ERROR: {name} failed: {e}"
        if images and images_out is not None:
            if len(images) > MCP_MAX_IMAGES_PER_CALL:
                images_out.extend(images[:MCP_MAX_IMAGES_PER_CALL])
                return (text + f"\n[{len(images) - MCP_MAX_IMAGES_PER_CALL} further image(s) "
                               f"from this call were dropped - cap is {MCP_MAX_IMAGES_PER_CALL}]")
            images_out.extend(images)
        elif images:
            # Caller cannot accept images: say so instead of silently discarding them.
            return text + f"\n[{len(images)} image(s) returned but not shown to the model]"
        return text

    def _mcp_connect_all(self) -> None:
        """(Re)connect every configured MCP server on a background thread.

        A second call while a connect is already running just marks the work
        pending; the in-flight worker re-runs once when it finishes, so rapid
        double-Saves cannot spawn duplicate (leaked) server subprocesses.

        v1.1.59: the thread itself is now also gated. The pending flag + worker
        loop serialized the PASSES but still started a new Thread on every call,
        so four rapid Saves meant four threads and up to five sequential reconnect
        passes (the in-flight one loops once to consume the flag, and each of the
        other three runs one pass of its own after queueing on the lock).
        """
        self._mcp_connect_pending.set()
        with self._mcp_connect_spawn_lock:
            if self._mcp_connect_running:
                return          # the live worker will see the pending flag
            self._mcp_connect_running = True
        threading.Thread(target=self._mcp_connect_worker_loop, daemon=True).start()

    def _mcp_connect_worker_loop(self) -> None:
        """Run connect passes until no newer save is pending (serialized)."""
        cleared_here = False
        try:
            while True:
                with self._mcp_connect_lock:
                    try:
                        self._mcp_connect_worker()
                    except Exception as e:
                        print(f"[{APP_NAME}] MCP connect pass failed: {e}")
                # 'Am I done?' and 'release the gate' MUST be one critical section.
                # If they were separate, a save landing between the pending check
                # and the flag clear would set pending with no worker left to
                # consume it - the servers would never reconnect (lost wakeup).
                # The finally below deliberately does NOT re-acquire the lock on
                # this path: the normal exit already released it, and threading.Lock
                # is not reentrant.
                with self._mcp_connect_spawn_lock:
                    if not self._mcp_connect_pending.is_set():
                        self._mcp_connect_running = False
                        cleared_here = True
                        return
                    self._mcp_connect_pending.clear()
        finally:
            if not cleared_here:
                # Any unexpected exit must still release the gate, or MCP
                # reconnects would be dead for the rest of the session.
                with self._mcp_connect_spawn_lock:
                    self._mcp_connect_running = False

    def _mcp_connect_worker(self) -> None:
        # A bundled Node (installed next to the exe by the Windows installer) has
        # to be on PATH before any server is spawned, or `npx -y ...` fails with
        # "not found on PATH". Cheap and idempotent; a system Node wins if present.
        _ensure_bundled_node_on_path()
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
            # Server-initiated traffic needs somewhere to go. on_elicitation runs ON
            # THE READER THREAD, so it must marshal to the main thread itself (see
            # _mcp_elicit); on_notification lets a list_changed refresh the bar.
            client.on_elicitation = self._mcp_elicit
            client.on_notification = self._mcp_notification
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

    def _mcp_notification(self, method: str, params: dict) -> None:
        """A server notification (worker/reader thread). Only list-changed matters:
        re-run discovery so the new tools/resources/prompts appear without a restart.
        Called on the reader thread - keep it cheap and never block."""
        if method.endswith("list_changed"):
            self._post(lambda: self._mcp_refresh_after_change())

    def _mcp_refresh_after_change(self) -> None:
        """MAIN THREAD. Re-pull tools/resources/prompts from live servers."""
        if self._closed:
            return
        for client in self._mcp_clients_snapshot():
            threading.Thread(target=self._mcp_repoll, args=(client,), daemon=True).start()

    def _mcp_repoll(self, client: "MCPClient") -> None:
        """Worker thread: re-run the list calls, then refresh the UI once."""
        try:
            if "resources" in client.server_caps:
                rl = client.list_resources()
                client.resources = [r for r in (rl.get("resources") or []) if isinstance(r, dict)]
            if "prompts" in client.server_caps:
                pl = client.list_prompts()
                client.prompts = [p for p in (pl.get("prompts") or []) if isinstance(p, dict)]
        except Exception:
            pass
        self._post(lambda: self._mcp_connect_done([]))

    #  ── Elicitation: the server asks the USER a structured question ───────

    def _mcp_elicit(self, server_name: str, params: dict) -> dict:
        """Answer an elicitation/create request. CALLED ON THE READER THREAD.

        Returns an ElicitResult: {"action":"accept","content":{...}} | {"action":
        "decline"} | {"action":"cancel"}.

        The dialog must be built on the main thread (Tk), so this posts it and waits.
        The wait is sliced and abort-aware for exactly the reason _ask_permission_modal
        documents: Stop or app-close while the request sits in the queue must not leave
        the reader parked behind a dialog nobody will answer. A cancelled wait DECLINES
        - the safe default is to refuse to invent an answer for the server.

        ⚠️ While this blocks, the reader thread is not parsing further lines. That is
        acceptable: the server is itself blocked waiting for this response, so there is
        nothing else in flight to service."""
        res: Dict[str, Any] = {"action": "decline"}
        ev = threading.Event()

        def show():
            if self._closed or self._abort_requested():
                res["action"] = "decline"
                ev.set()
                return
            try:
                res.update(self._elicit_dialog(server_name, params))
            except Exception:
                res["action"] = "decline"
            finally:
                ev.set()

        self._post(show)
        if not _interruptible_wait(ev, 300, lambda: self._closed or self._abort_requested()):
            return {"action": "decline"}
        return res

    def _elicit_dialog(self, server_name: str, params: dict) -> dict:
        """MAIN THREAD. Build a small form from requestedSchema and collect answers.

        Supported: flat object schemas whose properties are string / number /
        integer / boolean, with enum rendered as a dropdown. Anything deeper is
        declined rather than guessed at - sending a half-understood answer to a
        server is worse than refusing.
        """
        message = str(params.get("message") or "The MCP server needs more information.")
        schema = params.get("requestedSchema")
        if not isinstance(schema, dict) or schema.get("type") not in (None, "object"):
            return {"action": "decline"}
        props = schema.get("properties")
        if not isinstance(props, dict) or not props:
            # No fields: the server is only asking for confirmation.
            ok = messagebox.askyesno(f"{APP_NAME} — MCP question", f"{server_name}:\n\n{message}",
                                     parent=self.root)
            return {"action": "accept", "content": {}} if ok else {"action": "decline"}
        required = set(schema.get("required") or [])
        props_all = props
        # A required field the form cannot show is a form that can never validate, so
        # decline rather than trap the user: the property cap would otherwise drop it.
        if len(props_all) > 12 and any(k in required for k in list(props_all)[12:]):
            return {"action": "decline"}
        props = dict(list(props_all.items())[:12])

        top = tk.Toplevel(self.root)
        top.title(f"{APP_NAME} — {server_name} needs input")
        top.configure(bg=COL["bg_main"])
        top.resizable(False, False)
        try:
            top.transient(self.root)
            top.grab_set()
        except tk.TclError:
            pass
        tk.Label(top, text=message, bg=COL["bg_main"], fg=COL["text"],
                 font=F(11), wraplength=460, justify="left").pack(padx=14, pady=(12, 8),
                                                                  anchor="w")
        fields: List[tuple] = []          # (key, kind, widget)
        for key, spec in props.items():
            spec = spec if isinstance(spec, dict) else {}
            label = str(spec.get("title") or key)
            desc = str(spec.get("description") or "")
            row = tk.Frame(top, bg=COL["bg_main"])
            row.pack(fill="x", padx=14, pady=2)
            tk.Label(row, text=label + ("" if key in required else "  (optional)"),
                     bg=COL["bg_main"], fg=COL["text_dim"], font=F(10), width=22,
                     anchor="w").pack(side="left")
            enum = spec.get("enum")
            if isinstance(enum, list) and enum:
                vals = [str(v) for v in enum]
                combo = ttk.Combobox(row, values=vals, state="readonly", width=28)
                combo.set(vals[0])
                combo.pack(side="left", fill="x", expand=True)
                fields.append((key, "enum", combo))
            else:
                ent = tk.Entry(row, bg=COL["bg_raised"], fg=COL["text"],
                               insertbackground=COL["text"], relief="flat", width=30,
                               font=F(11))
                ent.pack(side="left", fill="x", expand=True)
                kind = str(spec.get("type") or "string")
                fields.append((key, "bool" if kind == "boolean" else kind, ent))
            if desc:
                tk.Label(top, text=desc, bg=COL["bg_main"], fg=COL["text_dim"],
                         font=F(9), wraplength=460, justify="left").pack(padx=18, anchor="w")

        out: Dict[str, Any] = {"action": "decline"}
        # v1.1.59 (#2 of the third review): same flaw _ask_permission_modal had. This
        # dialog is a real Toplevel, so it CAN be closed - but nothing closed it when
        # the user pressed Stop. _mcp_elicit's sliced wait unwinds the READER thread
        # and returns "decline", while this window stayed modal on the main thread
        # until clicked. NB: the guard is set inside _close(), NOT at the top of
        # submit() - submit() legitimately returns early on a validation error with
        # the dialog still open, and a guard at entry would make the second attempt
        # a no-op that never closes anything.
        done = {"v": False, "timer": None}

        def _close(action: str, content: Optional[dict] = None) -> None:
            if done["v"]:
                return
            done["v"] = True
            out["action"] = action
            if content is not None:
                out["content"] = content
            try:
                if done["timer"] is not None:
                    top.after_cancel(done["timer"])
            except Exception:
                pass
            try:
                top.destroy()
            except Exception:
                pass

        def submit():
            content: Dict[str, Any] = {}
            blanks: List[str] = []
            for key, kind, w in fields:
                raw = w.get() if hasattr(w, "get") else ""
                raw = "" if raw is None else str(raw).strip()
                if not raw:
                    # A blank REQUIRED field is not an answer. Accepting it as "" would
                    # hand the server an empty string for a value it demanded, which is
                    # worse than refusing and letting the user fill it in.
                    if key in required:
                        blanks.append(key)
                    continue              # optional and left blank: omit it
                if kind == "boolean":
                    content[key] = raw.lower() in ("true", "1", "yes", "y", "on")
                elif kind in ("number", "integer"):
                    try:
                        content[key] = float(raw) if kind == "number" else int(float(raw))
                    except ValueError:
                        messagebox.showerror(
                            APP_NAME, f"{key} must be a number.", parent=top)
                        return
                else:
                    content[key] = raw
            missing = blanks + [k for k in required if k not in content and k not in blanks]
            if missing:
                messagebox.showerror(
                    APP_NAME, "Please fill in: " + ", ".join(missing), parent=top)
                return
            _close("accept", content)

        def cancel():
            _close("cancel")

        def stop_poll():
            # v1.1.59: Stop (or app close) must tear this down. _mcp_elicit's sliced
            # wait already unwinds the reader thread on Stop, so from that moment on
            # this window is a modal nobody is waiting for - it would sit on screen
            # blocking the main UI until the user clicked it. Polled because the stop
            # flag is set from another thread and Tk has no hook for it. Cancelling is
            # the safe answer: the server gets "cancel" rather than an invented reply.
            if done["v"]:
                return
            if self._closed or self._abort_requested():
                _close("cancel")
                return
            try:
                if top.winfo_exists():
                    done["timer"] = top.after(400, stop_poll)
            except Exception:
                pass

        btns = tk.Frame(top, bg=COL["bg_main"])
        btns.pack(pady=12)
        tk.Button(btns, text="Submit", command=submit, bg=COL["accent"], fg="#FFFFFF",
                  relief="flat", padx=12, cursor="hand2", font=F(11)).pack(side="left", padx=4)
        tk.Button(btns, text="Cancel", command=cancel, bg=COL["bg_raised"], fg=COL["text_dim"],
                  relief="flat", padx=12, cursor="hand2", font=F(11)).pack(side="left", padx=4)
        top.bind("<Return>", lambda e: submit())
        top.bind("<Escape>", lambda e: cancel())
        # Closing the window with the title-bar X must count as cancel, not accept.
        top.protocol("WM_DELETE_WINDOW", cancel)
        try:
            done["timer"] = top.after(400, stop_poll)
        except Exception:
            pass
        top.wait_window()
        return out

    #  ── MCP section of the permission bar: one COLLAPSIBLE header per server ──
    #
    #  Why grouped: a single server can advertise a lot of tools (Playwright MCP
    #  ships 25). One chip each pushed the bar from 23 to 48 cells - 4+ rows, and
    #  unusable. A server now contributes ONE header chip showing its tool count;
    #  clicking it expands that server's per-tool chips inline. Permissions are
    #  unaffected either way: every chip is still registered in _perm_combos and
    #  every tool still defaults to Ask - collapsing is a VIEW state, not a grant.
    #  Collapsed by default (the common case is "leave them all at Ask").

    def _mcp_expanded_cfg(self) -> list:
        """Servers the user has explicitly expanded (persisted).

        The list holds EXPANDED servers, not collapsed ones, so the default for a
        never-seen server is COLLAPSED - a freshly connected 25-tool server must not
        blow up the bar before the user has chosen to look at it."""
        raw = self.settings.get("mcp_bar_expanded")
        return [str(x) for x in raw] if isinstance(raw, list) else []

    def _mcp_is_collapsed(self, server: str) -> bool:
        return server not in self._mcp_expanded_cfg()

    def _mcp_toggle_collapsed(self, server: str) -> None:
        cur = self._mcp_expanded_cfg()
        if server in cur:
            cur.remove(server)          # -> collapse
        else:
            cur.append(server)          # -> expand
        self.settings["mcp_bar_expanded"] = cur
        save_settings(self.settings)
        self._mcp_rebuild_permissions_bar()

    def _mcp_rebuild_permissions_bar(self) -> None:
        """(Re)draw the MCP section: one collapsible header per server, each
        folding that server's per-tool chips. Tools default to Ask Permission
        (MCP servers are arbitrary local programs - never silently auto-approved)."""
        for w in getattr(self, "_mcp_perm_widgets", []):
            try:
                if w.winfo_exists():
                    w.destroy()
            except tk.TclError:
                pass
        self._mcp_perm_widgets = []
        for n in [k for k in list(self._perm_combos) if k.startswith("mcp_")]:
            del self._perm_combos[n]
        # _perm_cells holds the same widgets (chips AND the MCP section headers).
        # Drop every destroyed one so _perm_reflow never lays out dead widgets.
        self._perm_cells = [c for c in getattr(self, "_perm_cells", [])
                            if c.winfo_exists()]
        self._perm_last_sig = None
        inner = getattr(self, "_perm_inner", None)
        if inner is None:
            return
        names = self._mcp_tool_names()
        if not names:
            return

        # Group in discovery order, one bucket per server.
        by_server: Dict[str, List[str]] = {}
        for name in names:
            server_name = self._mcp_tool_map[name][0]
            by_server.setdefault(server_name, []).append(name)

        for server_name, group in by_server.items():
            collapsed = self._mcp_is_collapsed(server_name)
            # Headers are part of the flow, not separately placed: _perm_reflow owns
            # every grid position in `inner`, so anything grid'ed by hand here would
            # collide with a chip.
            hdr = tk.Label(inner,
                           text=f"🔌 {server_name} ({len(group)}) "
                                + ("\u25b8" if collapsed else "\u25be"),
                           bg=COL["bg_deep"], fg=COL["accent"],
                           font=F(10, "bold"), cursor="hand2")
            hdr.bind("<Button-1>", lambda _e, s=server_name: self._mcp_toggle_collapsed(s))
            self._perm_bind_wheel(hdr)
            # v1.1.61: the header had no tooltip at all, so a collapsed server was
            # opaque - you could not see which tools it offered without expanding it.
            # Names only here: 25 descriptions would be unreadable in a hover tip, and
            # each tool chip already carries its own description.
            _tools = [self._mcp_tool_map[n][1] for n in group]
            _shown = ", ".join(_tools[:12]) + (f" \u2026 (+{len(_tools) - 12} more)"
                                               if len(_tools) > 12 else "")
            self._perm_tooltip(
                hdr, lambda s=server_name, k=len(group), sh=_shown, c=collapsed:
                f"MCP server: {s}\n{k} tool(s): {sh}\n\n"
                f"click to {'expand' if c else 'collapse'} this server's tools\n"
                "each tool keeps its own permission (default: Ask)")
            self._mcp_perm_widgets.append(hdr)
            self._perm_cells.append(hdr)
            for name in group:
                _srv, raw = self._mcp_tool_map[name]
                # MCP servers are arbitrary local programs - never silently
                # auto-approved: the default stays Ask Permission (DEFAULT_PERMS
                # has no entry for them).
                cell = self._add_perm_chip(name, f"{raw} ({server_name})",
                                           in_flow=not collapsed)
                self._mcp_perm_widgets.append(cell)
        # re-flow: the chip list changed, so the row count (and bar height) may too
        self._perm_reflow()

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
                  command=lambda: self._mcp_add_server_dialog(parent_top, frame),
                  bg=COL["bg_raised"], fg=COL["text"], activebackground="#4B5563",
                  relief="flat", bd=0, padx=8, pady=2, cursor="hand2", font=F(10)) \
            .pack(anchor="w", pady=(4, 0))
        # Rows were destroyed + rebuilt above: re-apply the dialog wheel binding. The very
        # first build lands here BEFORE the binder exists (getattr -> None); at the end of
        # open_settings the canvas-wide bind covers everything, so nothing is missed.
        _b = getattr(self, "_settings_bind_wheel", None)
        if _b is not None:
            _b(frame)

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

    def _mcp_add_server_dialog(self, parent: tk.Toplevel,
                               redraw_frame: Optional[tk.Frame] = None) -> None:
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
                  ("Env vars (space-separated KEY=VALUE; quote a value with spaces, "
                   "e.g. PATH=\"C:\\Program Files\\bin\")", "env")]
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
            # Show the new row IMMEDIATELY in the open dialog (pre-v1.1.12 the list only
            # refreshed after closing and reopening Settings).
            if redraw_frame is not None:
                try:
                    if redraw_frame.winfo_exists():
                        self._draw_mcp_server_rows(redraw_frame, parent)
                except tk.TclError:
                    pass

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
                               "Sampling verification in progress")
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
            self._verify_report(out_box, [], "Verifying sampling parameters", clear=True)
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
            self._verify_report(out_box, ["(evidence only: a server can accept a key and ignore it)"])
            self._verify_report(out_box, [], "Sampling verification finished")
        except Exception as e:
            traceback.print_exc()
            # str(e) is often EMPTY (TypeError, KeyError, AttributeError with no args),
            # which rendered as "Verification aborted: " and told the user nothing.
            # The type name is the informative part when the message is not.
            detail = _probe_first_line(f"{type(e).__name__}: {e}") or type(e).__name__
            self._verify_report(out_box, ["Verification aborted: " + detail],
                                "Sampling verification aborted")
        finally:
            self._verify_running = False
            self._verify_set_button(btn, "normal")

    def _test_connection(self, url: str, key: str, top: tk.Toplevel) -> None:
        # `key` comes from the LIVE dialog entry, captured on the main thread by the
        # caller; this method runs on a worker thread and must not touch widgets.
        ok = False
        detail = ""
        try:
            key = (key or "").strip()
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