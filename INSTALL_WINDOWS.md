# Deskpilot — Windows Install (copy & paste)

_Clean Windows 10/11 machine → working `DeskPilot.exe`. Do the steps **in order**.
Each step is one thing to click or one block to copy-paste._

**Also needed (separately):** an OpenAI-compatible LLM server (e.g. LM Studio + a model).
Deskpilot is the client — it needs a server to talk to.

---

## Step 1 — Install Python (web)

1. Go to <https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe> → download **"Windows installer (64-bit)"** for **3.12.x or 3.13.x** (both fully supported - avoid 3.14: PyAudio has no Windows wheels for it yet). ⚠️ The big green button on that page is the newest version (currently 3.14) - if you want 3.13 instead, click **"All other download options"** and pick the latest **3.13.x** installer (and use `py -3.13` in every step below).
2. Run it. On the first screen **tick "Add python.exe to PATH"** ← important, then click *Install Now*.
3. Verify in PowerShell: `python --version` must print 3.12 or 3.13 - if it prints 3.14 (or an older Python is on PATH), use the `py` launcher instead for every step below: `py -3.13 -m pip ...`, `py -3.13 deskpilot.py`, etc.

## Step 2 — Install Node.js (web)

Go to <https://nodejs.org/> → download the green **LTS** button → run the installer. On the "Tools for Native Modules" screen leave **"Automatically install the necessary tools" UNCHECKED** (that only pulls in Chocolatey + Visual Studio Build Tools for compiling npm native modules - Deskpilot just runs plain JS, so the base Node runtime is enough) → *Next* with all other defaults.

## Step 3 — Install VC++ Redistributable (web)

Download and run: <https://aka.ms/vs/17/release/vc_redist.x64.exe> (click through, done).

## Step 4 — Install all Python packages (PowerShell)

Open **PowerShell** (Start menu → type "powershell" → Enter). Paste this one line and press Enter:

```powershell
py -0p          # first: lists every installed Python and its path (sanity check)
py -3.12 -m pip install --upgrade pip
py -3.12 -m pip install openai pypdf Pillow kokoro-onnx sounddevice SpeechRecognition PyAudio faster-whisper numpy pyinstaller
```

(Installs everything the app can use, plus the compiler. Takes a few minutes. `py -3.12` pins the exact Python so this works even with several Pythons installed - if you used 3.13 instead, replace `-3.12` with `-3.13` in **every** step below.)

## Step 5 — Put the app file in its folder

Copy `deskpilot.py` to c:\Deskpilot


## Step 6 — Add the TTS model files (PowerShell, one line)

```powershell
New-Item -ItemType Directory -Force "$env:LOCALAPPDATA\Deskpilot\kokoro_models" | Out-Null; Invoke-WebRequest "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx" -OutFile "$env:LOCALAPPDATA\Deskpilot\kokoro_models\kokoro-v1.0.int8.onnx"; Invoke-WebRequest "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin" -OutFile "$env:LOCALAPPDATA\Deskpilot\kokoro_models\voices-v1.0.bin"
```

(Downloads the two TTS files, ~116 MB total, from the official kokoro-onnx GitHub release. Browser alternative: open <https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0>, download `kokoro-v1.0.int8.onnx` + `voices-v1.0.bin`, and put them in `%LOCALAPPDATA%\Deskpilot\kokoro_models\`.)

(The dictation Whisper model downloads itself automatically on first use — nothing to do.)

## Step 7 — Test it runs (PowerShell)

```powershell
py -3.12 C:\Deskpilot\deskpilot.py
```

The Deskpilot window should open. Close it when you see it.

## Step 8 — Optional: Compile an .exe file to launch from anywhere on your PC (PowerShell, two lines)

```powershell
cd C:\Deskpilot
py -3.12 -m PyInstaller --noconfirm --clean --onefile --windowed --name DeskPilot --collect-all kokoro_onnx --collect-all faster_whisper --hidden-import pypdf --hidden-import PIL.ImageGrab --hidden-import sounddevice --hidden-import speech_recognition --hidden-import pyaudio --hidden-import numpy deskpilot.py
```
Create and exe to leave in C:\Deskpilot\dist\Deskpilot folder for faster launch times (will Deskpilot.exe plus subfolder and files instead of just the one single .exe file)

```powershell
cd C:\Deskpilot
py -3.12 -m PyInstaller --noconfirm --clean --windowed --name DeskPilot --collect-all kokoro_onnx --collect-all faster_whisper --hidden-import pypdf --hidden-import PIL.ImageGrab --hidden-import sounddevice --hidden-import speech_recognition --hidden-import pyaudio --hidden-import numpy deskpilot.py
```

Takes ~1 minute. When it finishes, your app is at: **`C:\Deskpilot\dist\DeskPilot.exe`**
(make a Desktop shortcut to it if you like).

## Step 9 — Run it

- **Normal:** `DeskPilot.exe` (a second normal launch shows "already running" and exits)
- **To Run a Second Instance:** `DeskPilot.exe --multi`

First run: Settings → Server URL → **Test Connection** → Model name → Save. Done.

---

## If something goes wrong

| Problem | Fix |
|---|---|
| `pip` not recognized in a new window | Re-run the Python installer → *Modify* → tick "Add python.exe to PATH" → open a **new** PowerShell. |
| Step 4 fails building PyAudio: "Cannot open include file: 'portaudio.h'" | You are on Python 3.14 - it has no prebuilt PyAudio wheel, so pip tries to compile from source. Install Python 3.13.x (see Step 1) and re-run Step 4 with `py -3.13 -m pip install ...`. |
| Wrong Python version is being used (log shows `C:\PythonXXX` paths or `cp3XX` wheel names that don't match what you installed) | Multiple Pythons on PATH and the old one wins. Check with `py -0p` (lists all installed versions), then pin explicitly in every step: `py -3.12 -m pip ...`, `py -3.12 deskpilot.py`, `py -3.12 -m PyInstaller ...`. |
| `pip` / `pyinstaller` "not recognized as the name of a cmdlet, function, script file..." | That command's Scripts folder isn't on PATH (common with user installs + multiple Pythons). Use the module form instead: `py -3.12 -m pip ...`, `py -3.12 -m PyInstaller ...`. |
| Step 7 shows a red error about a missing module | Paste the exact module name here and I'll give you the one-line fix (usually `pip install <name>`). |
| Exe is slow on first launch / Windows warns "unknown publisher" | Normal for unsigned single-file exes — *More info → Run*. First start extracts to temp (~a few seconds). |
| TTS button says "model files not found" | Step 6 — both files must be in `%LOCALAPPDATA%\Deskpilot\kokoro_models\`. |
| Dictation stuck on "Loading local Whisper model…" | First use needs internet (one-time ~150 MB download). |
