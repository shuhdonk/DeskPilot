# Deskpilot — Linux Install (copy & paste)

_Clean machine → working app. Do the steps **in order** for your distro._

**Pick your section:** [Arch / CachyOS](#arch--cachyos) · [Fedora (KDE Plasma)](#fedora-kde-plasma)

**Also needed (separately):** an OpenAI-compatible LLM server (e.g. LM Studio + a model).
Deskpilot is the client — it needs a server to talk to.

**Pasting tip (fish / Konsole):** select only the code line(s) themselves. fish runs every pasted line immediately, so a paste that carries extra blank lines makes it execute a flood of empty commands - you will see endless `❯` prompts. If that happens: Ctrl+C, then re-paste just the single command (or type it). Also note: fish does not support heredocs (`<<`) - every command in this doc is a single line for that reason.

---

## Arch / CachyOS

### Step 1 — System packages (terminal)

Open a **terminal** and paste:

```bash
sudo pacman -S --needed python tk portaudio nodejs curl git && python -c "import tkinter; print('tkinter OK:', tkinter.TkVersion)"
```

(Python — tkinter is built into the `python` package, and `tk` provides the Tk library it links against (Arch has no separate `python-tk` package); plus the mic library PyAudio builds against, Node.js for the `run_javascript` tool, curl for Step 4, and git for Step 2. The second part verifies the tkinter import — it should print "tkinter OK: …".)

### Step 2 — Get the app onto this machine (terminal)

Easiest: clone it from GitHub:

```bash
git clone https://github.com/shuhdonk/DeskPilot.git ~/Deskpilot
```

(No internet? Copy `deskpilot.py` over via USB / scp, then run `mkdir -p ~/Deskpilot && cp /tmp/deskpilot.py ~/Deskpilot/` instead.)

### Step 3 — Python packages (terminal, one line)

```bash
cd ~/Deskpilot && python -m venv .venv && .venv/bin/pip install --upgrade pip && .venv/bin/pip install openai pypdf Pillow kokoro-onnx sounddevice SpeechRecognition PyAudio faster-whisper numpy
```

(Creates an isolated environment and installs everything the app can use. Takes a few minutes. No `source .../activate` needed - the commands call the venv's own pip directly, so this works in bash, zsh **and fish** (CachyOS's default shell). If you'd rather activate the env for your prompt: `source .venv/bin/activate.fish` in fish, or `source .venv/bin/activate` in bash/zsh.)

### Step 4 — Add the TTS model files (terminal, one line)

```bash
mkdir -p ~/Deskpilot/kokoro_models && cd ~/Deskpilot/kokoro_models && curl -L -O https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx && curl -L -O https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

(Downloads the two TTS files, ~116 MB total, from the official kokoro-onnx GitHub release. If you'd rather use a browser: open <https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0>, download `kokoro-v1.0.int8.onnx` and `voices-v1.0.bin`, and put them in `~/Deskpilot/kokoro_models/`.)

(The dictation Whisper model downloads itself automatically on first use — nothing to do.)

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

---

## Fedora (KDE Plasma)

> ⚠️ **Read this first:** on current Fedora (43+) the default `python3` is **Python 3.14**, which Deskpilot does NOT support (PyAudio has no 3.14 wheels, kokoro-onnx caps at <3.14). This section therefore uses the **`python3.13`** binary explicitly in every step - never bare `python3`. Also, tkinter is a *separate* package on Fedora (`python3.13-tkinter`), and PyAudio compiles from source, so a few build packages are needed up front.

### Step 1 — System packages (terminal)

Open a **terminal** (Konsole) and paste:

```bash
sudo dnf install -y python3.13 python3.13-tkinter python3.13-devel portaudio-devel gcc nodejs curl git && python3.13 -c "import tkinter; print('tkinter OK:', tkinter.TkVersion)"
```

(`python3.13` = the supported interpreter, `python3.13-tkinter` = the Tk GUI library it links against, `python3.13-devel` + `portaudio-devel` + `gcc` = what PyAudio needs to compile its mic bindings, plus Node.js for the `run_javascript` tool, curl for Step 4 and git for Step 2. The last part verifies tkinter — it should print "tkinter OK: …".)

### Step 2 — Get the app onto this machine (terminal)

Easiest: clone it from GitHub:

```bash
git clone https://github.com/shuhdonk/DeskPilot.git ~/Deskpilot
```

(No internet? Copy `deskpilot.py` over via USB / scp, then run `mkdir -p ~/Deskpilot && cp /tmp/deskpilot.py ~/Deskpilot/` instead.)

### Step 3 — Python packages (terminal, one line)

```bash
cd ~/Deskpilot && python3.13 -m venv .venv && .venv/bin/pip install --upgrade pip && .venv/bin/pip install openai pypdf Pillow kokoro-onnx sounddevice SpeechRecognition PyAudio faster-whisper numpy
```

(Creates an isolated 3.13 environment and installs everything the app can use. Takes a few minutes — PyAudio builds from source here, so expect a minute of compiler output; that's normal. No `source .../activate` needed - the commands call the venv's own pip directly, which works in bash (Fedora's default) and fish alike.)

### Step 4 — Add the TTS model files (terminal, one line)

```bash
mkdir -p ~/Deskpilot/kokoro_models && cd ~/Deskpilot/kokoro_models && curl -L -O https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx && curl -L -O https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

(Downloads the two TTS files, ~116 MB total, from the official kokoro-onnx GitHub release. If you'd rather use a browser: open <https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0>, download `kokoro-v1.0.int8.onnx` and `voices-v1.0.bin`, and put them in `~/Deskpilot/kokoro_models/`.)

(The dictation Whisper model downloads itself automatically on first use — nothing to do.)

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

---

## Notes & if something goes wrong (both distros)

| Problem | Fix |
|-|-|
| `.venv/bin/pip` / `.venv/bin/python` says "no such file" | You're in the wrong folder — run `cd ~/Deskpilot` first, or use the full paths from Step 6. |
| Sourcing `.venv/bin/activate` errors with '"case" builtin not inside of switch block' | You're in fish - it can't read the sh-style activate script. Use `source .venv/bin/activate.fish` instead (or skip activation; the steps above don't need it). |
| Endless empty `❯` prompts after pasting | The paste carried extra newlines and fish ran each one as an empty command. Ctrl+C, check for a partial clone (`ls ~/Deskpilot`, `rm -rf` it if incomplete), then re-paste just the single command line. |
| PyAudio build fails in Step 3 (portaudio.h not found) | Arch: `sudo pacman -S --needed portaudio`. Fedora: `sudo dnf install -y portaudio-devel gcc python3.13-devel`. Then re-run Step 3's pip line. |
| tkinter import error / "Python is not built with Tk support" | Arch: `sudo pacman -S --needed tk`. Fedora: `sudo dnf install -y python3.13-tkinter` (and make sure you're using the venv's python, which inherits it). |
| TTS button says "model files not found" | Step 4 — both files must be in `~/Deskpilot/kokoro_models/`. |
| Dictation stuck on "Loading local Whisper model…" | First use needs internet (one-time ~150 MB download). |
| Screen-capture tool errors | It needs an X11 display; on a pure Wayland session it can't grab the screen (use an X11 session, or take screenshots manually with `grim`). Everything else works fine. |
| Settings/chats not persisting | Data lives next to the script (`~/Deskpilot/`) — the startup banner prints the exact paths; check that folder is writable. |

**Linux vs Windows differences:** no exe compilation needed (run via Python directly); the "already running" guard doesn't refuse on Linux (its liveness check is Windows-only) — `--multi` and `.bak` backups work the same.
