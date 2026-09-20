# Deskpilot — Linux Install (copy & paste)

_Clean Arch / CachyOS machine → working app. Do the steps **in order**.
Each step is one block to copy-paste into a terminal._

**Also needed (separately):** an OpenAI-compatible LLM server (e.g. LM Studio + a model).
Deskpilot is the client — it needs a server to talk to.

---

## Step 1 — System packages (terminal, one line)

Open a **terminal** and paste:

```bash
sudo pacman -S --needed python python-tk portaudio nodejs curl
```

(Python, tkinter, the mic library PyAudio builds against, Node.js for the `run_javascript` tool, and curl for Step 4.)

## Step 2 — Put the app file in its folder (terminal)

Copy `deskpilot.py` to this machine (USB / scp / whatever), then paste — **replace the first path** with where you actually put it:

```bash
mkdir -p ~/Deskpilot && cp /tmp/deskpilot.py ~/Deskpilot/
```

## Step 3 — Python packages (terminal, one line)

```bash
cd ~/Deskpilot && python -m venv .venv && source .venv/bin/activate && pip install --upgrade pip && pip install openai pypdf Pillow kokoro-onnx sounddevice SpeechRecognition PyAudio faster-whisper numpy
```

(Creates an isolated environment and installs everything the app can use. Takes a few minutes.)

> Every **new** terminal window needs: `source ~/Deskpilot/.venv/bin/activate` first.

## Step 4 — Add the TTS model files (terminal, one line)

```bash
mkdir -p ~/Deskpilot/kokoro_models && cd ~/Deskpilot/kokoro_models && curl -L -O https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx && curl -L -O https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

(Downloads the two TTS files, ~116 MB total, from the official kokoro-onnx GitHub release. If you'd rather use a browser: open <https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0>, download `kokoro-v1.0.int8.onnx` and `voices-v1.0.bin`, and put them in `~/Deskpilot/kokoro_models/`.)

(The dictation Whisper model downloads itself automatically on first use — nothing to do.)

## Step 5 — Test it runs (terminal)

```bash
cd ~/Deskpilot && source .venv/bin/activate && python deskpilot.py
```

The Deskpilot window should open. Close it when you see it.

## Step 6 — Run it from now on

```bash
source ~/Deskpilot/.venv/bin/activate && python ~/Deskpilot/deskpilot.py
```

(Or make a launcher: right-click your desktop → *Create Launcher* → Command = the line above.)

First run: Settings → Server URL → **Test Connection** → Model name → Save. Done.

---

## Notes & if something goes wrong

| Problem | Fix |
|-|-|
| `source .venv/bin/activate` says "no such file" | You're in the wrong folder — run `cd ~/Deskpilot` first, or use the full path from Step 6. |
| PyAudio build fails in Step 3 | `sudo pacman -S --needed portaudio`, then re-run Step 3's pip line. |
| TTS button says "model files not found" | Step 4 — both files must be in `~/Deskpilot/kokoro_models/`. |
| Dictation stuck on "Loading local Whisper model…" | First use needs internet (one-time ~150 MB download). |
| Screen-capture tool errors | It needs an X11 display; on a pure Wayland session it can't grab the screen (use an X11 session, or take screenshots manually with `grim`). Everything else works fine. |
| Settings/chats not persisting | Data lives next to the script (`~/Deskpilot/`) — the startup banner prints the exact paths; check that folder is writable. |

**Linux vs Windows differences:** no exe compilation needed (run via Python directly); the "already running" guard doesn't refuse on Linux (its liveness check is Windows-only) — `--multi` and `.bak` backups work the same.
