# Deskpilot — Windows Install

_Get Windows 10/11 → a working Deskpilot. There are **two ways** to install; pick one._

**Needed either way (not bundled):** an OpenAI-compatible LLM server — llama.cpp, LM
Studio, Ollama, or a hosted API key. Deskpilot is the client; it does not ship a model.

---

## Which route?

**A. Installer** (recommended) 

**B. From source** |

**Both routes need the same optional extras for the full feature set** — see
"Optional extras" below. Neither route can include them: together they are ~45 GB.

---

# Route A — Install with the installer file

## A1 — Download the installer

Grab the newest **`DeskPilot-Setup-<version>.exe`** from the project's GitHub
**Releases** page. It is a normal Windows installer

## A2 — Run it

Double-click it. Choose **"Install for me"** when asked — no admin rights or
password are needed, because Deskpilot keeps its settings and chat history in
`%LOCALAPPDATA%\Deskpilot`.

It installs to:

```
%LOCALAPPDATA%\Programs\Deskpilot\        (normally C:\Users\<you>\AppData\Local\Programs\Deskpilot)
```

If you instead choose **"Install for all users"**, it goes to
`C:\Program Files\Deskpilot\` 

⚠️ **First run — SmartScreen.** Windows will say "_Windows protected your PC_"
because the installer is not code-signed. That is expected: click
**More info → Run anyway**.

## A3 — Point it at a model

Start Deskpilot from the Start Menu. Open **Settings**, set
**Server URL (OpenAI-compatible base)** — e.g. `http://localhost:8080/v1` —
click **Test Connection**, then set **Model Name** and **Save**.

That is enough for chat, web search, file editing, MCP servers and history.
Everything below is optional.

## A4 — Optional extras

Each one unlocks exactly one feature. The app tells you what is missing when
you try to use it. Two folders are used below:

```
DATA = %LOCALAPPDATA%\Deskpilot        settings, chat history, downloaded models
APP  = %LOCALAPPDATA%\Programs\Deskpilot    where the installer put the exe
```

| Feature | What to add | Size |
|---|---|---|
| **Voice output** (🔊) | two Kokoro files in `DATA\kokoro_models` | ~115 MB |
| **GPU dictation** (🎤) | NVIDIA cuBLAS in `APP\nvidia\cublas\bin` | ~736 MB |
| **Dictation model** (🎤) | **nothing** — downloads itself on first use | ~1.5 GB GPU / ~140 MB CPU |
| **Image generation** | your own `ai-imagegen` folder in `APP` | ~43 GB + big GPU |
| **MCP servers using `npx`** | **nothing** — Node.js is already bundled | — |

### A4a — Voice output (text-to-speech)

One line in PowerShell — downloads both files into the right folder:

```powershell
New-Item -ItemType Directory -Force "$env:LOCALAPPDATA\Deskpilot\kokoro_models" | Out-Null; Invoke-WebRequest "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx" -OutFile "$env:LOCALAPPDATA\Deskpilot\kokoro_models\kokoro-v1.0.int8.onnx"; Invoke-WebRequest "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin" -OutFile "$env:LOCALAPPDATA\Deskpilot\kokoro_models\voices-v1.0.bin"
```

Browser alternative: from <https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0>
download `kokoro-v1.0.int8.onnx` (88 MB) + `voices-v1.0.bin` (27 MB) and put
them in `%LOCALAPPDATA%\Deskpilot\kokoro_models\`. (That release also carries
larger `fp16` / full-precision variants — you want the **int8** one.)

### A4b — GPU dictation (only if you have an NVIDIA GPU)

Without this, dictation still works — it just runs on the CPU, slower and less
accurate. You need three DLLs placed **next to the exe**. No Python required:
download the NVIDIA wheel (it is just a zip) and unpack it.

1. Download `nvidia_cublas_cu12-12.9.2.10-py3-none-win_amd64.whl` (528 MB):
   <https://pypi.org/project/nvidia-cublas-cu12/#files>

2. Then, in PowerShell (one line — it unpacks the download into the app folder):

```powershell
$w=(Get-ChildItem "$env:USERPROFILE\Downloads\nvidia_cublas_cu12*win_amd64.whl")[0]; if(-not $w){Write-Host "no wheel found - re-download it"; break}; Copy-Item $w.FullName "$env:TEMP\deskpilot_cublas.zip" -Force; Expand-Archive "$env:TEMP\deskpilot_cublas.zip" "$env:TEMP\deskpilot_cublas_x" -Force; New-Item -ItemType Directory -Force "$env:LOCALAPPDATA\Programs\Deskpilot\nvidia\cublas\bin" | Out-Null; Copy-Item "$env:TEMP\deskpilot_cublas_x\nvidia\cublas\bin\*.dll" "$env:LOCALAPPDATA\Programs\Deskpilot\nvidia\cublas\bin"; Get-ChildItem "$env:LOCALAPPDATA\Programs\Deskpilot\nvidia\cublas\bin" | Select-Object Name,Length
```

It should list `cublas64_12.dll`, `cublasLt64_12.dll`, `nvblas64_12.dll`.
Needs a current NVIDIA driver (CUDA 12-capable) — you do **not** need to
install the CUDA Toolkit itself.

### A4c — Dictation model: nothing to do

The first time you press 🎤 it downloads automatically — **~1.5 GB**
(`large-v3-turbo`) on an NVIDIA GPU, **~140 MB** (`base.en`) otherwise. Cached
in `DATA\whisper_models`. Offline machine? Copy that folder in from another PC.

### A4d — Local image generation (FLUX / SDXL)

Deliberately not in the installer: ~43 GB of model weights and it wants a
large GPU (~16 GB VRAM). If you have that set up, place the folder next to the
exe so Deskpilot finds it:

```
APP\ai-imagegen\generate.py
APP\ai-imagegen\.venv\Scripts\python.exe
```

Without it, the image tool returns a clear "not found" message and nothing
else is affected.

## A5 — Check dictation is using the GPU (30 seconds)

Click **🎤** and read the **amber status text in the top bar**:

- `🎤 Listening via local Whisper (GPU)…` ← working as well as it gets
- `🎤 Listening via local Whisper (CPU)…` ← cuBLAS not found; A4b was skipped or the folder is not next to the exe

## A6 — Uninstall

Normal Windows uninstall (Settings → Apps). **Your settings, chats and
downloaded models in `%LOCALAPPDATA%\Deskpilot` are never deleted**, so
reinstalling picks up exactly where you left off.

---

# Route B — Install from source

Use this if you want to edit `deskpilot.py` or build your own exe. Do the steps
**in order**; each is one thing to click or one block to paste.

## Step 1 — Install Python

Install **Python 3.12.x or 3.13.x** — both fully supported. Direct 64-bit
installers (verified links):

- **3.12.10:** <https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe>
- **3.13.16:** <https://www.python.org/ftp/python/3.13.16/python-3.13.16-amd64.exe>

⚠️ **Avoid 3.14** — PyAudio has no Windows wheels for it yet, so the mic install
in Step 4 fails.

Run the installer. On the first screen **tick "Add python.exe to PATH"** ←
important, then click *Install Now*.

Verify in PowerShell:

```powershell
py -0p
```

That lists every Python installed. **Pick one and use it in every step below**
(`py -3.12` throughout this guide; substitute `py -3.13` if that is what you
installed). Pinning the version is what keeps this working when several Pythons
are present.

## Step 2 — Install Node.js

Go to <https://nodejs.org/> → the green **LTS** button → run the installer. On the
"Tools for Native Modules" screen leave **"Automatically install the necessary
tools" UNCHECKED** (that only pulls in Chocolatey + Visual Studio Build Tools for
compiling native npm modules — Deskpilot runs plain JS, so the base runtime is
enough). *Next* through the rest with defaults.

Node is needed only for MCP servers launched with `npx`. Skip it if you will not
use those.

## Step 3 — Install the VC++ Redistributable

Download and run: <https://aka.ms/vs/17/release/vc_redist.x64.exe> (click through,
done). Needed by the ONNX/GPU libraries.

## Step 4 — Install the Python packages (PowerShell, one line)

```powershell
py -3.12 -m pip install --upgrade pip; py -3.12 -m pip install openai pypdf Pillow "kokoro-onnx>=0.6.1" sounddevice SpeechRecognition PyAudio "faster-whisper>=1.0" "av<19" numpy nvidia-cublas-cu12 pyinstaller
```

Takes a few minutes. Notes:

- **`av<19` is not optional.** faster-whisper needs `av.open(metadata_errors=...)`,
  which av 19 removed. Without the pin, pip installs av 19 and **dictation breaks**.
- **`PyAudio` is what opens the microphone.** It is not a declared dependency of
  SpeechRecognition, so it must be installed by name.
- **`nvidia-cublas-cu12` (528 MB download)** is the GPU math library for fast
  dictation. Skip it only if you have no NVIDIA GPU — dictation then runs on the CPU.
- **`kokoro-onnx` has no 1.x release** (latest is 0.6.1). If you use
  `pip install -r requirements.txt`, note that it must ask for `>=0.6.1`, not `>=1.0`.
- You do **not** need to install pip separately; it ships with Python. The `-m pip`
  form is used throughout because it works even when bare `pip` is not on PATH. If
  you ever get "No module named pip", run `py -3.12 -m ensurepip --upgrade` and
  repeat this step.

## Step 5 — Put the app file in its folder

Create `C:\Deskpilot` and copy `deskpilot.py` into it.

## Step 6 — Test that it runs (PowerShell)

```powershell
py -3.12 C:\Deskpilot\deskpilot.py
```

The Deskpilot window should open. Close it. Then add the optional extras from
**A4** above as wanted — when running from the script, cuBLAS is found
automatically in Step 4's packages, so **A4b is not needed for Route B**.

## Step 7 — Optional: compile your own exe

Recommended build (a folder — starts faster, and everything stays together):

```powershell
cd C:\Deskpilot
py -3.12 -m PyInstaller --noconfirm --clean --windowed --name DeskPilot --collect-all kokoro_onnx --collect-all faster_whisper --collect-all phonemizer --collect-all espeakng_loader --hidden-import pypdf --hidden-import PIL.ImageGrab --hidden-import sounddevice --hidden-import speech_recognition --hidden-import pyaudio --hidden-import numpy deskpilot.py
```

Your app is then at **`C:\Deskpilot\dist\DeskPilot\DeskPilot.exe`**. Takes a few
minutes.

⚠️ **`--collect-all phonemizer --collect-all espeakng_loader` are required.** A
build without them produces an exe whose text-to-speech silently does not work.

If you prefer a single file, add `--onefile`; it re-extracts to a temp folder on
every launch, so it starts more slowly.

**Either way, a compiled exe needs cuBLAS beside it** (it has no site-packages of
its own). One line, which finds every `DeskPilot.exe` under `dist\` and drops the
DLLs next to it:

```powershell
py -3.12 -c "import site,glob,shutil,os; exes=glob.glob(r'C:\Deskpilot\dist'+os.sep+'**'+os.sep+'DeskPilot.exe',recursive=True); src=[f for sp in site.getsitepackages() for f in glob.glob(sp+os.sep+'nvidia'+os.sep+'cublas'+os.sep+'bin'+os.sep+'*.dll')]; [([os.makedirs(os.path.join(os.path.dirname(e),'nvidia','cublas','bin'),exist_ok=True), [shutil.copy(f,os.path.join(os.path.dirname(e),'nvidia','cublas','bin')) for f in src]] if src else None) for e in exes]; print('exes:',exes); print('dlls found:',len(src))"
```

Should print `dlls found: 3`. If it prints `0`, Step 4's `nvidia-cublas-cu12` did
not install.

## Step 8 — Run it

- **Normal launch:** `DeskPilot.exe` — a second launch shows "Already running" and
  exits, because two instances would share one data file.
- **Deliberately run a second, independent instance:** `DeskPilot.exe --multi`
  (gets its own scratch profile; nothing it does is saved to your real history).

First run: **Settings → Server URL → Test Connection → Model Name → Save**.

Then check dictation with **A5** above.

---

## Optional-extras sizes, for reference

| Extra | Downloaded | Where it goes |
|---|---|---|
| Kokoro voice (int8) | 88 MB + 27 MB = **~115 MB** | `DATA\kokoro_models` |
| cuBLAS (GPU dictation) | **528 MB** wheel, ~736 MB unpacked | `APP\nvidia\cublas\bin` |
| Whisper `large-v3-turbo` (GPU) | **~1.5 GB** | `DATA\whisper_models` |
| Whisper `base.en` (CPU) | **~140 MB** | `DATA\whisper_models` |
| Image generation (FLUX + SDXL) | **~43 GB** | `APP\ai-imagegen` |

---

## If something goes wrong

| Problem | Fix |
|---|---|
| Windows says "Windows protected your PC" | Expected — the build is not code-signed. **More info → Run anyway**. |
| `pip` not recognized in a new window | Re-run the Python installer → *Modify* → tick "Add python.exe to PATH" → open a **new** PowerShell. Or skip PATH entirely: `py -3.12 -m pip ...`. |
| `py -3.12 -m pip ...` says **"No module named pip"** | pip was unticked, or this is a minimal/embedded Python. Bootstrap it, then repeat Step 4: `py -3.12 -m ensurepip --upgrade` |
| Step 4 fails building PyAudio: "Cannot open include file: 'portaudio.h'" | You are on Python 3.14 — no prebuilt wheel, so pip compiles from source. Use 3.12 or 3.13 and re-run with `py -3.13 -m pip install ...`. |
| Wrong Python used (log shows `C:\PythonXXX` paths or `cp3XX` wheel names you did not install) | Several Pythons on PATH and an old one wins. Check `py -0p`, then pin explicitly in **every** step: `py -3.12 -m pip ...`, `py -3.12 ...\deskpilot.py`, `py -3.12 -m PyInstaller ...`. |
| `pip` / `pyinstaller` "not recognized as the name of a cmdlet" | That Python's Scripts folder is not on PATH. Use the module form: `py -3.12 -m pip ...`, `py -3.12 -m PyInstaller ...`. |
| Step 6 shows a red error about a missing module | `py -3.12 -m pip install <that-name>` then re-run Step 6. |
| Mic does nothing / error when pressing 🎤 | PyAudio is missing — it is **not** pulled in automatically. `py -3.12 -m pip install PyAudio` (Route B). |
| Dictation is slow / status shows **(CPU)** on an NVIDIA machine | cuBLAS is not reachable. Route A: A4b. Route B: re-run Step 4 (needs `nvidia-cublas-cu12`), and for a compiled exe the Step 7 copy step so `nvidia\cublas\bin\` is **in the same folder as the exe**. |
| Status shows **(GPU)** but dictation errors on the first phrase | Usually an old GPU driver — ctranslate2 needs **CUDA 12** support. Update the NVIDIA driver; you do *not* need the CUDA Toolkit itself. |
| Dictation stuck on "Loading local Whisper model…" | First use downloads the model — ~1.5 GB (GPU) or ~140 MB (CPU). Needs internet; the first click can take a minute. |
| 🔊 speaks nothing / "model files not found" | A4a — both files must be in `%LOCALAPPDATA%\Deskpilot\kokoro_models\`. |
| 🔊 works from the script but not in a compiled exe | You built without `--collect-all phonemizer --collect-all espeakng_loader` (Step 7). Rebuild with them. |
| **Dictation broke after a `pip install --upgrade`** | `av` 19 removed an API faster-whisper needs. Pin it: `py -3.12 -m pip install "av<19"` |
| Exe is slow to start / "unknown publisher" | Normal for an unsigned build. A `--onefile` exe also re-extracts to temp on every launch — the folder build in Step 7 starts faster. |
| No NVIDIA GPU at all | Skip A4b / `nvidia-cublas-cu12`. Dictation runs on the CPU automatically and the status shows **(CPU)**. |
| "Deskpilot - Already running" appears on launch | Another instance holds the data folder. Close it, or use `DeskPilot.exe --multi` for a separate throwaway session. |

---

## Where your data lives

Everything Deskpilot writes — settings, chat history, downloaded models,
generated images, checkpoints — is in **`%LOCALAPPDATA%\Deskpilot`**. It survives
uninstalling, and it is the folder to copy if you move machines.
