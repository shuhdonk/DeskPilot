# Deskpilot — Linux Install (copy & paste)

_Clean machine → working app. Do the steps **in order** for your distro._

**Pick your section:** [Arch / CachyOS](#arch--cachyos) · [Fedora (KDE Plasma)](#fedora-kde-plasma)

**Also needed (separately):** an OpenAI-compatible LLM server (e.g. LM Studio + a model).
Deskpilot is the client — it needs a server to talk to.

**GPU dictation (optional):** if your machine has an NVIDIA GPU, the
pip line in Step 3 also installs the GPU math libraries, which makes microphone
dictation fast and much more accurate. No NVIDIA GPU? Delete `nvidia-cublas-cu12
nvidia-cudnn-cu12` from that line — everything still works, dictation just runs on the CPU.
**Step 7** shows how to confirm which one you got.

**Pasting tip (fish / Konsole):** select only the code line(s) themselves. fish runs every pasted line immediately, so a paste that carries extra blank lines makes it execute a flood of empty commands - you will see endless `❯` prompts. If that happens: Ctrl+C, then re-paste just the single command (or type it). Also note: fish does not support heredocs (`<<`) - every command in this doc is a single line for that reason.

---

## Arch / CachyOS

> ⚠️ **Read this first:** Arch's official `python` package is now **Python 3.14** (CachyOS mirrors it at 3.14.7), which Deskpilot does NOT support — `kokoro-onnx` requires `<3.14` and PyAudio has no 3.14 wheels, so Step 3 fails outright with "Could not find a version that satisfies the requirement kokoro-onnx". This section therefore installs **`python313`** and uses the **`python3.13`** binary explicitly — never bare `python`. Also, tkinter comes from the `tk` package (Arch has no separate `python-tk`), and PyAudio compiles from source, so build packages are needed up front.

### Step 1 — System packages (terminal)

Open a **terminal** and paste:

```bash
sudo pacman -S --needed tk portaudio nodejs curl git base-devel python-pip && sudo pacman -S --needed python313 && python3.13 -c "import tkinter; print('tkinter OK:', tkinter.TkVersion)"
```

(`tk` = the Tk library Python's tkinter links against — Arch has no separate `python-tk` package, and `tk` is literally the optional dependency that enables tkinter; `portaudio` + `base-devel` = what PyAudio compiles its mic bindings against; Node.js for the `run_javascript` tool; curl for Step 4; git for Step 2. The second command installs the **3.13 interpreter**: **CachyOS ships `python313` in its own repos**, so it just works there. The last part verifies the interpreter and tkinter — it should print "tkinter OK: …".)

**On plain Arch** (not CachyOS), `python313` is not in the official repos — pacman will say `repository does not contain 'python313'`. Build it from the AUR instead (single line, this one needs `git`, which the command above already installed), then re-run the verify part:

```bash
git clone https://aur.archlinux.org/python313.git /tmp/py313 && cd /tmp/py313 && makepkg -si
```

(An AUR helper works too — `paru -S python313`. `paru` is still installable from the CachyOS repos, but note current CachyOS ISOs ship **Shelly** as the default helper instead, so the steps above deliberately do not assume `paru` exists. `pyenv` or `uv` are alternatives if you would rather not touch the AUR.)

(You do **not** need to install the CUDA Toolkit or any NVIDIA system packages — Step 3's pip line brings its own GPU libraries. All you need is a working NVIDIA **driver**, which Arch already has if `nvidia-smi` prints a table.)

### Step 2 — Get the app onto this machine (terminal)

Easiest: clone it from GitHub:

```bash
git clone https://github.com/shuhdonk/DeskPilot.git ~/Deskpilot
```

(No internet? Copy `deskpilot.py` over via USB / scp, then run `mkdir -p ~/Deskpilot && cp /tmp/deskpilot.py ~/Deskpilot/` instead.)

### Step 3 — Python packages (terminal, one line)

```bash
cd ~/Deskpilot && python3.13 -m venv .venv && .venv/bin/pip install --upgrade pip && .venv/bin/pip install openai pypdf Pillow kokoro-onnx sounddevice SpeechRecognition PyAudio faster-whisper "av<19" numpy nvidia-cublas-cu12 nvidia-cudnn-cu12
```

(Creates an isolated **3.13** environment and installs everything the app can use. Takes a few minutes — PyAudio builds from source here, so expect a minute of compiler output; that's normal. No `source .../activate` needed - the commands call the venv's own pip directly, so this works in bash, zsh **and fish** (CachyOS's default shell). If you'd rather activate the env for your prompt: `source .venv/bin/activate.fish` in fish, or `source .venv/bin/activate` in bash/zsh.)

(⚠️ **`"av<19"` is not optional.** faster-whisper calls `av.open(metadata_errors=...)`, which PyAV **19 removed**, and faster-whisper itself puts no upper bound on `av`. Without the pin you get av 19 and dictation dies with `TypeError: open() got an unexpected keyword argument 'metadata_errors'`. The pinned 18.x ships `cp311-abi3` manylinux wheels, so it installs cleanly on 3.12/3.13 with no compiling.)

(`nvidia-cublas-cu12` + `nvidia-cudnn-cu12` = the GPU libraries for dictation, **~1.3 GB of downloads** (554 MB + 731 MB for x86_64) — Step 3 will look stalled for a while, that is normal. Deskpilot finds and loads them by itself from the venv; you do not have to set `LD_LIBRARY_PATH`. No NVIDIA GPU? Drop those two names from the line.)

### Step 4 — Add the TTS model files (terminal, one line)

```bash
mkdir -p ~/Deskpilot/kokoro_models && cd ~/Deskpilot/kokoro_models && curl -L -O https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx && curl -L -O https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

(Downloads the two TTS files, ~116 MB total, from the official kokoro-onnx GitHub release. If you'd rather use a browser: open <https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0>, download `kokoro-v1.0.int8.onnx` and `voices-v1.0.bin`, and put them in `~/Deskpilot/kokoro_models/`.)

(The dictation Whisper model downloads itself automatically on first use — nothing to do. Size depends on your GPU: **~1.5 GB** on a GPU machine, **~140 MB** on CPU. See Step 7.)

### Step 5 — Test it runs (terminal)

```bash
cd ~/Deskpilot && .venv/bin/python deskpilot.py
```

The Deskpilot window should open. Close it when you see it.

### Step 6 — Run it from now on

```bash
~/Deskpilot/.venv/bin/python ~/Deskpilot/deskpilot.py
```

**Desktop shortcut (KDE Plasma):** Plasma 6 has no "Create Launcher" in the desktop right-click menu, so create a launcher file instead - paste this single line into your terminal (replace `username` with your user name):

```bash
printf '[Desktop Entry]\nType=Application\nName=Deskpilot\nExec=/home/username/Deskpilot/.venv/bin/python /home/username/Deskpilot/deskpilot.py\nIcon=utilities-terminal\nTerminal=false\n' > ~/Desktop/Deskpilot.desktop && chmod +x ~/Desktop/Deskpilot.desktop
```

The icon appears on the desktop. KDE marks new launcher files untrusted: right-click it once and choose **Allow Executing** (or *Trust File*), then double-click launches Deskpilot. (Optional: `cp ~/Desktop/Deskpilot.desktop ~/.local/share/applications/` also puts it in the application menu - Super key, type "Deskpilot" - where you can right-click → *Pin to Task Manager* for a permanent panel icon. On older Plasma 5 systems you could instead right-click the desktop → *Create Launcher*.)

First run: Settings → Server URL → **Test Connection** → Model name → Save. Done.

### Step 7 — Check that GPU dictation is active (30 seconds)

Click the **🎤** button and look at the **amber status text in the top bar** (top-right, next to the app title). It reads either:

- `🎤 Listening via local Whisper (GPU)…` ← **GPU is working.** Best accuracy and speed.
- `🎤 Listening via local Whisper (CPU)…` ← the GPU libraries were not found; dictation still works, just slower and less accurate.

If it says **(CPU)** in upper right on an NVIDIA machine, run this to see what the app sees:

```bash
cd ~/Deskpilot && .venv/bin/python -c "import ctranslate2 as c; print('CUDA GPUs visible:', c.get_cuda_device_count())"
```

`0` means the driver/CUDA is the problem (not Deskpilot) — see the table at the bottom.

---

## Fedora (KDE Plasma)

> ⚠️ **Read this first:** on current Fedora (43 and 44) the default `python3` is **Python 3.14**, which Deskpilot does NOT support (PyAudio has no 3.14 wheels, kokoro-onnx caps at <3.14). This section therefore uses the **`python3.13`** binary explicitly in every step - never bare `python3`. Also, tkinter is a *separate* package on Fedora (`python3.13-tkinter`), and PyAudio compiles from source, so a few build packages are needed up front.

### Step 1 — System packages (terminal)

Open a **terminal** (Konsole) and paste:

```bash
sudo dnf install -y python3.13 python3.13-tkinter python3.13-devel portaudio-devel gcc nodejs curl git && python3.13 -c "import tkinter; print('tkinter OK:', tkinter.TkVersion)"
```

(`python3.13` = the supported interpreter, `python3.13-tkinter` = the Tk GUI library it links against, `python3.13-devel` + `portaudio-devel` + `gcc` = what PyAudio needs to compile its mic bindings, plus Node.js for the `run_javascript` tool, curl for Step 4 and git for Step 2. The last part verifies tkinter — it should print "tkinter OK: …".)

(You do **not** need to install the CUDA Toolkit or any NVIDIA system packages — Step 3's pip line brings its own GPU libraries. All you need is a working NVIDIA **driver**: `rpm -qa '*nvidia*'` shows one, or `nvidia-smi` prints a table.)

### Step 2 — Get the app onto this machine (terminal)

Easiest: clone it from GitHub:

```bash
git clone https://github.com/shuhdonk/DeskPilot.git ~/Deskpilot
```

(No internet? Copy `deskpilot.py` over via USB / scp, then run `mkdir -p ~/Deskpilot && cp /tmp/deskpilot.py ~/Deskpilot/` instead.)

### Step 3 — Python packages (terminal, one line)

```bash
cd ~/Deskpilot && python3.13 -m venv .venv && .venv/bin/pip install --upgrade pip && .venv/bin/pip install openai pypdf Pillow kokoro-onnx sounddevice SpeechRecognition PyAudio faster-whisper "av<19" numpy nvidia-cublas-cu12 nvidia-cudnn-cu12
```

(Creates an isolated 3.13 environment and installs everything the app can use. Takes a few minutes — PyAudio builds from source here, so expect a minute of compiler output; that's normal. No `source .../activate` needed - the commands call the venv's own pip directly, which works in bash (Fedora's default) and fish alike.)

(⚠️ **`"av<19"` is not optional.** faster-whisper calls `av.open(metadata_errors=...)`, which PyAV **19 removed**, and faster-whisper itself puts no upper bound on `av`. Without the pin you get av 19 and dictation dies with `TypeError: open() got an unexpected keyword argument 'metadata_errors'`. The pinned 18.x ships `cp311-abi3` manylinux wheels, so it installs cleanly on 3.13 with no compiling.)

(`nvidia-cublas-cu12` + `nvidia-cudnn-cu12` = the GPU libraries for dictation, **~1.3 GB of downloads** (554 MB + 731 MB for x86_64) — Step 3 will look stalled for a while, that is normal. Deskpilot finds and loads them by itself from the venv; you do not have to set `LD_LIBRARY_PATH`. No NVIDIA GPU? Drop those two names from the line.)

### Step 4 — Add the TTS model files (terminal, one line)

```bash
mkdir -p ~/Deskpilot/kokoro_models && cd ~/Deskpilot/kokoro_models && curl -L -O https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx && curl -L -O https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

(Downloads the two TTS files, ~116 MB total, from the official kokoro-onnx GitHub release. If you'd rather use a browser: open <https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0>, download `kokoro-v1.0.int8.onnx` and `voices-v1.0.bin`, and put them in `~/Deskpilot/kokoro_models/`.)

(The dictation Whisper model downloads itself automatically on first use — nothing to do. Size depends on your GPU: **~1.5 GB** on a GPU machine, **~140 MB** on CPU. See Step 7.)

### Step 5 — Test it runs (terminal)

```bash
cd ~/Deskpilot && .venv/bin/python deskpilot.py
```

The Deskpilot window should open. Close it when you see it.

### Step 6 — Run it from now on

```bash
~/Deskpilot/.venv/bin/python ~/Deskpilot/deskpilot.py
```

**Desktop shortcut (KDE Plasma):** same as the Arch section — paste this single line (replace `username` with your user name):

```bash
printf '[Desktop Entry]\nType=Application\nName=Deskpilot\nExec=/home/username/Deskpilot/.venv/bin/python /home/username/Deskpilot/deskpilot.py\nIcon=utilities-terminal\nTerminal=false\n' > ~/Desktop/Deskpilot.desktop && chmod +x ~/Desktop/Deskpilot.desktop
```

Right-click the new icon once → **Allow Executing**, then double-click launches Deskpilot. (Optional: `cp ~/Desktop/Deskpilot.desktop ~/.local/share/applications/` puts it in the application menu too.)

First run: Settings → Server URL → **Test Connection** → Model name → Save. Done.

### Step 7 — Check that GPU dictation is active (30 seconds)

Same as the Arch section: click **🎤** and read the amber top-bar status — `local Whisper (GPU)` vs `local Whisper (CPU)`. Diagnostic one-liner if it says CPU:

```bash
cd ~/Deskpilot && .venv/bin/python -c "import ctranslate2 as c; print('CUDA GPUs visible:', c.get_cuda_device_count())"
```

---

## Notes & if something goes wrong (both distros)

| Problem | Fix |
|-|-|
| `.venv/bin/pip` / `.venv/bin/python` says "no such file" | You're in the wrong folder — run `cd ~/Deskpilot` first, or use the full paths from Step 6. |
| Sourcing `.venv/bin/activate` errors with '"case" builtin not inside of switch block' | You're in fish - it can't read the sh-style activate script. Use `source .venv/bin/activate.fish` instead (or skip activation; the steps above don't need it). |
| Endless empty `❯` prompts after pasting | The paste carried extra newlines and fish ran each one as an empty command. Ctrl+C, check for a partial clone (`ls ~/Deskpilot`, `rm -rf` it if incomplete), then re-paste just the single command line. |
| PyAudio build fails in Step 3 (portaudio.h not found) | Arch: `sudo pacman -S --needed portaudio base-devel`. Fedora: `sudo dnf install -y portaudio-devel gcc python3.13-devel`. Then re-run Step 3's pip line. (PyAudio publishes **no Linux wheels at all** - only Windows ones - so it always compiles from source on Linux. That is expected, not a broken install.) |
| Step 3 fails with "Could not find a version that satisfies the requirement kokoro-onnx" | You are on Python 3.14, which kokoro-onnx excludes (`requires_python: <3.14`). Use the pinned interpreter: Arch/CachyOS `sudo pacman -S --needed python313` (or build it from the AUR, see Step 1) then `python3.13 -m venv .venv`; Fedora `sudo dnf install -y python3.13` then `python3.13 -m venv .venv`. Do not use bare `python` / `python3`. |
| Dictation broke after a `pip install --upgrade` (TypeError: open() got an unexpected keyword argument 'metadata_errors') | `av` 19 removed that argument. Re-pin it: `.venv/bin/pip install "av<19"` |
| "The virtual environment was not created successfully because ensurepip is not available" | Arch/CachyOS Python builds do not bundle pip. Install it: `sudo pacman -S --needed python-pip`, then re-run Step 3. If `python3.13` still cannot find it, create the venv without pip and bootstrap it: `python3.13 -m venv --without-pip .venv && curl -L -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && .venv/bin/python /tmp/get-pip.py` then re-run the install part of Step 3. |
| tkinter import error / "Python is not built with Tk support" | Arch/CachyOS: `sudo pacman -S --needed tk` (it is the optional dependency that enables tkinter for the `python313` package). Fedora: `sudo dnf install -y python3.13-tkinter` (and make sure you're using the venv's python, which inherits it). |
| TTS button says "model files not found" | Step 4 — both files must be in `~/Deskpilot/kokoro_models/`. |
| Dictation stuck on "Loading local Whisper model…" | First use needs internet — one-time download: **~1.5 GB** (GPU) or **~140 MB** (CPU). Big download, so the first mic click can take a minute. |
| Mic status says **(CPU)** but you have an NVIDIA GPU | Run the Step 7 diagnostic. If it prints `CUDA GPUs visible: 0`, the driver is missing or is CUDA 11-era — ctranslate2 needs **CUDA 12** support, so update the NVIDIA driver package (`nvidia` / `nvidia-driver`). If it prints ≥1 but the app still says CPU, the pip GPU libraries are missing — re-run Step 3 including `nvidia-cublas-cu12 nvidia-cudnn-cu12`. |
| `RuntimeError: Library libcublas.so.12 is not found or cannot be loaded` | Step 3's GPU libraries are missing or were dropped. Re-run the Step 3 pip line with `nvidia-cublas-cu12 nvidia-cudnn-cu12` included. |
| GPU dictation errors with a cuDNN message | ctranslate2 4.5+ needs **cuDNN 9** (which `nvidia-cudnn-cu12` provides). If something else in the venv pulled in cuDNN 8, run `.venv/bin/pip install --force-reinstall nvidia-cudnn-cu12`. |
| No NVIDIA GPU at all | Remove `nvidia-cublas-cu12 nvidia-cudnn-cu12` from Step 3. Everything works; dictation runs on the CPU (status shows **(CPU)**). |
| Screen-capture tool errors | Pillow's `ImageGrab` tries X11 first, then falls back to **`gnome-screenshot`, `grim` or `spectacle`** if one is installed — so on KDE Plasma (Wayland or X11) it normally works out of the box, since Fedora KDE ships `spectacle`. If it still fails, install one: Fedora `sudo dnf install -y spectacle` (already present on the KDE spin) or `sudo dnf install -y grim`; Arch `sudo pacman -S --needed grim`. On a headless/remote session with no display at all it cannot capture anything — everything else works fine. (Multi-monitor: the fallback tools grab one screen, so `all_screens` stitching may not apply there.) |
| Settings/chats not persisting | On Linux the data lives **next to the script** (`~/Deskpilot/`) — `deskpilot_settings.json`, `deskpilot_chats.json`, `generated_images/`, plus `kokoro_models/` and `whisper_models/`. The startup banner prints the exact paths; check that folder is writable (if you cloned into a root-owned place, re-clone into your home as Step 2 does). |

**Linux vs Windows differences:** no exe installer or exe compilation on Linux (the `DeskPilot-Setup-*.exe` is Windows-only) — you run it through Python directly, so there is no cuBLAS-next-to-the-exe step: the venv's own libraries are found automatically. The "already running" guard does not refuse a second launch on Linux (its process-liveness check is Windows-only), so **two instances will share and overwrite the same data files — use `--multi` for a deliberate second window**: `~/Deskpilot/.venv/bin/python ~/Deskpilot/deskpilot.py --multi`. That gets its own throwaway `scratch_<pid>` profile seeded from your current settings/history, so nothing it does touches your real data, and it reuses the same downloaded models instead of re-fetching them. `.bak` backups work the same on both.

---

## Where your data lives

On Linux everything Deskpilot writes sits **next to the script**, in `~/Deskpilot/`:

```
deskpilot_settings.json     deskpilot_chats.json      generated_images/
whisper_models/             kokoro_models/            checkpoints/  ledger/
```

⚠️ **Deleting `~/Deskpilot` deletes your chat history** — there is no separate app-data folder to fall back on, unlike Windows (`%LOCALAPPDATA%\Deskpilot`). That is also the folder to copy if you move machines. The startup banner prints the exact paths.
