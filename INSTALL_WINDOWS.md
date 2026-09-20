# Deskpilot — Windows Install (copy & paste)

_Clean Windows 10/11 machine → working `DeskPilot.exe`. Do the steps **in order**.
Each step is one thing to click or one block to copy-paste._

**Also needed (separately):** an OpenAI-compatible LLM server (e.g. LM Studio + a model).
Deskpilot is the client — it needs a server to talk to.

---

## Step 1 — Install Python (web)

1. Go to <https://www.python.org/downloads/windows/> → download **"Windows installer (64-bit)"** for **3.12.x**.
2. Run it. On the first screen **tick "Add python.exe to PATH"** ← important, then click *Install Now*.

## Step 2 — Install Node.js (web)

Go to <https://nodejs.org/> → download the green **LTS** button → run the installer with all defaults.

## Step 3 — Install VC++ Redistributable (web)

Download and run: <https://aka.ms/vs/17/release/vc_redist.x64.exe> (click through, done).

## Step 4 — Install all Python packages (PowerShell)

Open **PowerShell** (Start menu → type "powershell" → Enter). Paste this one line and press Enter:

```powershell
pip install openai pypdf Pillow kokoro-onnx sounddevice SpeechRecognition PyAudio faster-whisper numpy pyinstaller
```

(Installs everything the app can use, plus the compiler. Takes a few minutes.)

## Step 5 — Put the app file in its folder (PowerShell)

Copy `deskpilot.py` to this machine (USB / OneDrive / whatever), then paste — **replace the first path** with where you actually put it:

```powershell
mkdir C:\Deskpilot; copy "C:\Users\YOU\Downloads\deskpilot.py" C:\Deskpilot\
```

## Step 6 — Add the TTS model files (PowerShell, one line)

```powershell
New-Item -ItemType Directory -Force "$env:LOCALAPPDATA\Deskpilot\kokoro_models" | Out-Null; Invoke-WebRequest "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx" -OutFile "$env:LOCALAPPDATA\Deskpilot\kokoro_models\kokoro-v1.0.int8.onnx"; Invoke-WebRequest "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin" -OutFile "$env:LOCALAPPDATA\Deskpilot\kokoro_models\voices-v1.0.bin"
```

(Downloads the two TTS files, ~116 MB total, from the official kokoro-onnx GitHub release. Browser alternative: open <https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0>, download `kokoro-v1.0.int8.onnx` + `voices-v1.0.bin`, and put them in `%LOCALAPPDATA%\Deskpilot\kokoro_models\`.)

(The dictation Whisper model downloads itself automatically on first use — nothing to do.)

## Step 7 — Test it runs (PowerShell)

```powershell
python C:\Deskpilot\deskpilot.py
```

The Deskpilot window should open. Close it when you see it.

## Step 8 — Compile the exe (PowerShell, two lines)

```powershell
cd C:\Deskpilot
pyinstaller --noconfirm --clean --onefile --windowed --name DeskPilot --collect-all kokoro_onnx --collect-all faster_whisper --hidden-import pypdf --hidden-import PIL.ImageGrab --hidden-import sounddevice --hidden-import speech_recognition --hidden-import pyaudio --hidden-import numpy deskpilot.py
```

Takes ~1 minute. When it finishes, your app is at: **`C:\Deskpilot\dist\DeskPilot.exe`**
(make a Desktop shortcut to it if you like).

## Step 9 — Run it

- **Normal:** `DeskPilot.exe` (a second normal launch shows "already running" and exits)
- **Second isolated window:** `DeskPilot.exe --multi`

First run: Settings → Server URL → **Test Connection** → Model name → Save. Done.

---

## If something goes wrong

| Problem | Fix |
|---|---|
| `pip` not recognized in a new window | Re-run the Python installer → *Modify* → tick "Add python.exe to PATH" → open a **new** PowerShell. |
| Step 7 shows a red error about a missing module | Paste the exact module name here and I'll give you the one-line fix (usually `pip install <name>`). |
| Exe is slow on first launch / Windows warns "unknown publisher" | Normal for unsigned single-file exes — *More info → Run*. First start extracts to temp (~a few seconds). |
| TTS button says "model files not found" | Step 6 — both files must be in `%LOCALAPPDATA%\Deskpilot\kokoro_models\`. |
| Dictation stuck on "Loading local Whisper model…" | First use needs internet (one-time ~150 MB download). |
