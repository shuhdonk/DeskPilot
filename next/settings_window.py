"""DeskPilot Next — Settings bridge (WebView2 window over the REAL settings store).

Design
------
* Runs as a CHILD PROCESS (`--settings-window`), not a thread. pywebview owns the
  main thread, so it cannot coexist with Tk's; and a child process means the UI
  needs no HTTP port and no auth token.
* It reuses the REAL save path (`save_settings` -> `_atomic_write`), so the rolling
  .bak, the fsync-before-swap rule and the File Workspace sync all behave exactly
  as they do in the Tk app. Nothing about persistence is reimplemented here.
* Validation is INLINE (per-field errors returned to the UI), never a modal. That
  was the main complaint about the old dialog.

Data dir: uses the SAME resolution as the app (resolve_data_dir), but with
APP_NAME "DeskpilotNext" so the Next build and the installed app can run
side by side without sharing an instance lock or a settings file.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── make the stable app importable, but DO NOT run its main() ───────────────
_HERE = Path(__file__).resolve().parent
_APP_DIR = _HERE.parent                      # deskpilot_data/
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))



def _app_module():
    """The running app as a module. Never re-runs its main().

    Source form: deskpilot.py sits in _APP_DIR and imports as a normal module.
    Frozen form: the entry script is loaded as __main__ and there is no module
    named "deskpilot" in the PYZ, so fall back to sys.modules. Verified both ways
    against a real onedir build (the entry module also imports by its own name).
    """
    try:
        import deskpilot as m
        return m
    except ImportError:
        main_mod = sys.modules.get("__main__")
        if main_mod is not None and hasattr(main_mod, "DEFAULT_SETTINGS"):
            return main_mod
        raise


app = _app_module()

from settings_schema import (                # noqa: E402
    SECTIONS, build_payload, apply_values, SAMPLING_SECTION,
)

NEXT_APP_NAME = "DeskpilotNext"
DEFAULT_SETTINGS_FILE = _APP_DIR / "deskpilot_settings.json"
GEOMETRY_NAME = "settings_window_geometry.json"


def next_data_dir() -> Path:
    """A writable dir for the NEXT build, separate from the stable one."""
    local = os.environ.get("LOCALAPPDATA")
    if local:
        return Path(local) / NEXT_APP_NAME
    return _APP_DIR / NEXT_APP_NAME


def live_data_dir() -> Path:
    """The STABLE app's live data dir (same resolution the app uses)."""
    local = os.environ.get("LOCALAPPDATA")
    if local:
        return Path(local) / "Deskpilot"
    return _APP_DIR


def read_path() -> Path:
    """Where to READ current values from. The live store is authoritative."""
    nd = next_data_dir() / "deskpilot_settings.json"
    if nd.exists():
        return nd
    live = live_data_dir() / "deskpilot_settings.json"
    if live.exists():
        return live
    snapshot = _APP_DIR / "deskpilot_settings.json"
    if snapshot.exists():
        return snapshot
    return nd


def write_path() -> Path:
    """Where to WRITE. Always the Next store.

    ⚠ Read and write are deliberately DIFFERENT paths. The first version returned
    one path for both, which meant opening the window (reading the live store)
    and then saving would have rewritten the STABLE app's settings file - the
    exact thing the side-by-side requirement forbids.
    """
    nd = next_data_dir()
    nd.mkdir(parents=True, exist_ok=True)
    return nd / "deskpilot_settings.json"


def load_settings(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return dict(getattr(app, "DEFAULT_SETTINGS", {}))


# ── geometry (remembered across opens) ──────────────────────────────────────
def geometry_path(base: Optional[Path] = None) -> Path:
    """Where the remembered size/position lives - beside the store in use.

    Takes the base dir rather than always using the Next profile: in the shipped
    app the window must remember its geometry in the app's own data dir, and the
    two builds must not fight over one file.
    """
    return (base or next_data_dir()) / GEOMETRY_NAME


def load_geometry(base: Optional[Path] = None) -> Dict[str, int]:
    """Last window position/size, or {} when absent/corrupt. Never raises."""
    try:
        data = json.loads(geometry_path(base).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {k: int(data[k]) for k in ("x", "y", "width", "height")
                    if isinstance(data.get(k), (int, float))}
    except Exception:
        pass
    return {}


def save_geometry(geo: Dict[str, int], base: Optional[Path] = None) -> None:
    try:
        # base must be threaded through: writing to the Next profile while reading
        # from the app's own store means the remembered position never applies.
        path = geometry_path(base)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(geo, indent=2), encoding="utf-8")
    except Exception:
        pass


# ── releasing the always-on-top pin ──────────────────────────────────────────
# Kept at module level and dependency-free so it can be tested with no window.
_GWL_EXSTYLE = -20
_WS_EX_TOPMOST = 0x00000008
_HWND_NOTOPMOST = -2
_SWP_NOSIZE = 0x0001
_SWP_NOMOVE = 0x0002
_SWP_NOACTIVATE = 0x0010


def find_own_window(title: str) -> int:
    """HWND of a visible window with this exact title belonging to OUR process.

    Zero when there is none. The pid filter is not paranoia: without it a stray
    "DeskPilot Settings" window in another program would be the one we restyle.
    """
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPVOID, ctypes.c_int]
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.IsWindowVisible.restype = wintypes.BOOL

    me = ctypes.windll.kernel32.GetCurrentProcessId()
    found = {"hwnd": 0}

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _lparam):
        if found["hwnd"]:
            return False                      # stop enumerating
        if not user32.IsWindowVisible(hwnd):
            return True
        buf = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(hwnd, buf, 256)
        if buf.value != title:
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value != me:
            return True                       # someone else's window - never ours
        found["hwnd"] = hwnd
        return False

    user32.EnumWindows(cb, 0)
    return int(found["hwnd"] or 0)


def is_topmost(hwnd: int) -> bool:
    """WS_EX_TOPMOST on this HWND. False on any failure - never raises."""
    if not hwnd:
        return False
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        user32.GetWindowLongW.restype = ctypes.c_long
        user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
        return bool(user32.GetWindowLongW(hwnd, _GWL_EXSTYLE) & _WS_EX_TOPMOST)
    except Exception:
        return False


def unpin_topmost(title: str, attempts: int = 30, delay: float = 0.1) -> bool:
    """Clear WS_EX_TOPMOST on our own window. True if it is unpinned afterwards.

    Safe from any thread: SetWindowPos is a Win32 call the OS marshals to the
    owning thread, unlike win.on_top which writes a WinForms property directly and
    must stay on the UI thread. That distinction is the whole reason the first
    attempt at this crashed the window.

    Retries because the caller runs as soon as the window is shown and the form may
    still be finishing construction; gives up quietly, since a window that stays
    pinned is the pre-existing annoyance, not a new failure.
    """
    import ctypes
    import time
    from ctypes import wintypes

    for _ in range(max(1, attempts)):
        hwnd = find_own_window(title)
        if hwnd:
            try:
                user32 = ctypes.windll.user32
                user32.SetWindowPos.restype = wintypes.BOOL
                user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND,
                                                ctypes.c_int, ctypes.c_int,
                                                ctypes.c_int, ctypes.c_int,
                                                ctypes.c_uint]
                user32.SetWindowPos(hwnd, ctypes.c_void_p(_HWND_NOTOPMOST), 0, 0, 0, 0,
                                    _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOACTIVATE)
            except Exception:
                return False
            if not is_topmost(hwnd):
                return True
        time.sleep(delay)
    return False


def use_next_profile() -> Path:
    """Point the app's PROMPT-FILE resolution at the Next profile.

    ⚠ This existed because the child had no profile at all. app.DATA_DIR is set
    only by app.main(), which the settings child deliberately never calls, so
    system_prompt_file() fell through to resolve_data_dir() with APP_NAME still
    "Deskpilot" and returned the LIVE app's SYSTEMPROMPT.md. The window would have
    read and written the stable app's prompt - the exact side-by-side violation the
    read/write split in write_path() was built to prevent. DATA_DIR has exactly one
    reader (system_prompt_file), so this is the whole blast radius.

    Seeds the prompt file from the live store when absent, with the launcher's
    rule: copy, never move.
    """
    nd = next_data_dir()
    nd.mkdir(parents=True, exist_ok=True)
    app.APP_NAME = NEXT_APP_NAME
    app.DATA_DIR = nd
    mine = nd / getattr(app, "SYSTEMPROMPT_NAME", "SYSTEMPROMPT.md")
    if not mine.exists():
        live = live_data_dir() / getattr(app, "SYSTEMPROMPT_NAME", "SYSTEMPROMPT.md")
        if live.exists():
            try:
                mine.write_bytes(live.read_bytes())
            except Exception:
                pass
    return nd


class Api:
    """The JS-facing surface. Every method is synchronous and returns plain data."""

    def __init__(self, window, settings: Dict[str, Any], path: Path):
        self._win = window
        self._settings = settings
        self._path = path
        self._lock = threading.Lock()
        # Raw entries from the last /models reply, for Save-time name resolution.
        # Empty until a refresh runs, and normalization is skipped while empty -
        # better to save what was typed than to guess without evidence.
        self._model_entries: List[dict] = []
        # default_workspace_root() reads this module global. Without the sync, a
        # File Workspace set in the store would be ignored and skills would be
        # installed into the script folder instead of the user's workspace.
        app._sync_active_file_workspace(self._settings)

    # ── persistence (one path: the real save_settings/_atomic_write) ────────
    def _persist(self) -> bool:
        """Write self._settings through the real path, restoring the global after."""
        old_settings_file = app.SETTINGS_FILE
        try:
            app.SETTINGS_FILE = self._path
            self._path.parent.mkdir(parents=True, exist_ok=True)
            ok = bool(app.save_settings(self._settings))
        except Exception:
            return False
        finally:
            app.SETTINGS_FILE = old_settings_file
        if ok:
            self._touch_sentinel()
        return ok

    def _touch_sentinel(self) -> None:
        """Tell the parent process 'the store just changed', so it can re-read it.

        There is no channel between the two processes - stdout is a log file and the
        pywebview bridge is in-process to the child only - so the signal is a file
        beside the store whose mtime the parent polls. Best-effort: if it cannot be
        written the app still works, it just refreshes on window close as before.
        """
        try:
            name = getattr(app, "SETTINGS_SENTINEL_NAME", "settings_window.saved")
            Path(self._path).with_name(name).write_text(
                "saved\n", encoding="utf-8")
        except Exception:
            pass

    # ── save ────────────────────────────────────────────────────────────────
    def save(self, values: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            updated, errors = apply_values(dict(self._settings), values or {})
            if errors:
                return {"ok": False, "errors": errors}
            notes = self._normalize_models(updated)

            self._settings = updated
            if not self._persist():
                return {"ok": False, "errors": {"_": f"Could not write {self._path}"}}
            app._sync_active_file_workspace(self._settings)
            # Echo back what was actually stored, not what was typed: normalisation
            # may have rewritten the model name, and the form must show the truth.
            # Only keys that are really present - the UI spreads this over its own
            # state, so a null here would BLANK a field that was never touched.
            echo = {k: updated[k] for k in ("model_name", "model_history",
                                            "expert_model", "expert_model_history")
                    if k in updated}
            return {"ok": True, "errors": {}, "notes": notes, "values": echo}

    def _normalize_models(self, updated: Dict[str, Any]) -> List[str]:
        """Resolve the typed model names to exact server IDs, and keep the history.

        Ported from the Tk dialog's Save handler, which did this inline. Two things
        ride on it:
          * a display name, or an id missing its namespace prefix, would otherwise
            save fine and then 404 at CHAT time - far from where it was typed;
          * model_history / expert_model_history are what the dropdown offers when
            the server is down, and nothing else writes them.
        Returns human-readable notes for the status line.
        """
        notes: List[str] = []
        cap = getattr(app, "MODEL_HISTORY_MAX", 20)

        name = str(updated.get("model_name") or "").strip()
        if name and self._model_entries:
            try:
                canon, status = app.normalize_model_name(name, self._model_entries)
            except Exception:
                canon, status = name, "error"
            if status == "resolved" and canon != name:
                notes.append(f"Model Name corrected to the server ID: {canon}")
                name = canon
            elif status == "ambiguous":
                notes.append("Model Name matches several server models - used as typed")
            updated["model_name"] = name
        if name:
            hist = [h for h in (updated.get("model_history") or [])
                    if isinstance(h, str) and h != name]
            hist.insert(0, name)
            updated["model_history"] = hist[:cap]

        exp = str(updated.get("expert_model") or "").strip()
        if exp:
            eh = [h for h in (updated.get("expert_model_history") or [])
                  if isinstance(h, str) and h != exp]
            eh.insert(0, exp)
            updated["expert_model_history"] = eh[:cap]
        return notes

    # ── model lists ──────────────────────────────────────────────────────
    def refresh_models(self, url: str, key: str) -> Dict[str, Any]:
        """GET {url}/models for the PRIMARY server.

        Returns the ids to offer, which ones the server reports as LOADED (the exact
        strings a chat request must use), and a note explaining the situation. The
        raw entries are cached so Save can normalize a typed name against them.
        """
        if not (url or "").strip():
            return {"ok": False, "message": "Set the Server URL first."}
        try:
            info = app.fetch_model_info(url or "", key or "", timeout=8)
        except Exception as e:
            self._model_entries = []
            return {"ok": False, "message": f"Failed: {e}", "ids": [], "loaded": []}
        self._model_entries = info or []
        ids = [str(m.get("id")).strip() for m in self._model_entries if m.get("id")]
        try:
            loaded = list(app.loaded_model_ids(self._model_entries))
        except Exception:
            loaded = []
        if loaded:
            note = "Loaded on server (use these exact IDs): " + ", ".join(loaded)
        elif ids:
            note = "Server lists models but reports none loaded - pick one to load."
        else:
            note = "No model list from the server - saved history shown below."
        return {"ok": bool(ids), "ids": ids, "loaded": loaded, "note": note[:200],
                "message": (f"Model list updated: {len(ids)} model(s) from server"
                            if ids else
                            "Model list unavailable - showing saved history")}

    def refresh_expert_models(self, url: str, key: str) -> Dict[str, Any]:
        """GET {url}/models for the CONSULTANT endpoint - its own URL and key.

        The expert list is fetched from the expert endpoint because guessing an ID
        produced a name that is not in Google's list at all. Gemini returns
        "models/<name>" while its own chat examples use the bare "<name>"; offer the
        bare form, since _tool_ask_expert retries the other form if the endpoint
        disagrees.
        """
        if not (url or "").strip() or not (key or "").strip():
            return {"ok": False,
                    "message": "Set Consultant URL and key first.", "ids": []}
        try:
            info = app.fetch_model_info(url, key, timeout=20)
        except Exception as e:
            return {"ok": False, "message": f"Failed: {e}", "ids": []}
        ids: List[str] = []
        for m in info or []:
            mid = str(m.get("id") or "").strip()
            if not mid:
                continue
            bare = mid[7:] if mid.startswith("models/") else mid
            if bare not in ids:
                ids.append(bare)
        if ids:
            pro = [i for i in ids if "pro" in i.lower()]
            msg = f"{len(ids)} models from the expert endpoint"
            if pro:
                msg += " - Pro: " + ", ".join(pro[:4])
        else:
            msg = ("No model list from that endpoint (check the URL/key) - "
                   "type the ID.")
        return {"ok": bool(ids), "ids": ids, "loaded": [], "note": msg[:200],
                "message": msg}

    # ── skills ─────────────────────────────────────────────────────────────
    def skills_list(self) -> Dict[str, Any]:
        """Installed skills (all of them) with their enabled flag."""
        try:
            found = app._discover_skills(self._settings)
        except Exception as e:
            return {"ok": False, "message": f"Scan failed: {e}", "skills": []}
        enabled = {n.strip().lower() for n in app._enabled_skill_names(self._settings)}
        rows = [{
            "name": s.get("name", ""),
            "description": s.get("description", ""),
            "dir": str(s.get("dir", "")),
            "platforms": s.get("platforms", ""),
            "supported": bool(s.get("supported_here", True)),
            "enabled": str(s.get("name", "")).strip().lower() in enabled,
        } for s in found]
        return {"ok": True, "skills": rows,
                "folders": [str(p) for p in app._skill_dirs(self._settings)]}

    def skills_catalog(self, live: bool = False) -> Dict[str, Any]:
        """The offline bundled catalog, or the live GitHub one when asked."""
        records = None
        if live:
            records = app._refresh_catalog_from_github()
        if not records:
            records = app._catalog_records()
        rows = [{
            "name": r.get("name", ""),
            "description": r.get("description", ""),
            "category": r.get("category", ""),
            "folder": app._skill_folder_key(r),
        } for r in records]
        return {"ok": True, "records": rows, "live": bool(live and records)}

    def skills_install(self, names: List[str]) -> Dict[str, Any]:
        """Download the selected catalog folders into the skills dir.

        Installed skills stay DISABLED (the app's own default): adding a skill
        must not silently inject it into every prompt.
        """
        by_name = {r.get("name", ""): r for r in app._catalog_records()}
        results = []
        ok_n = 0
        for name in names or []:
            rec = by_name.get(name)
            if not rec:
                results.append({"name": name, "ok": False, "message": "not in catalog"})
                continue
            folder = app._skill_folder_key(rec)
            dest = app._skill_install_dir(folder)
            if dest is None:
                results.append({"name": name, "ok": False, "message": "unsafe path blocked"})
                continue
            ok, msg = app._skill_folder_download(folder, dest)
            ok_n += 1 if ok else 0
            results.append({"name": name, "ok": bool(ok), "message": msg})
        return {"ok": True, "installed": ok_n, "results": results}

    def skills_toggle(self, name: str, enabled: bool) -> Dict[str, Any]:
        """Add/remove one skill from the opt-in enabled list."""
        with self._lock:
            cur = [n for n in app._enabled_skill_names(self._settings)
                   if n.strip().lower() != (name or "").strip().lower()]
            if enabled and (name or "").strip():
                cur.append(name.strip())
            self._settings["skills_enabled"] = cur
            if not self._persist():
                return {"ok": False, "message": "Could not save"}
            return {"ok": True, "enabled": cur}

    # ── MCP ────────────────────────────────────────────────────────────────
    def _mcp_rows(self) -> List[dict]:
        # Reuse the app's own sanitizer rather than duplicating its rules. It is a
        # plain method that reads only self.settings, so a throwaway holder works.
        holder = type("_Cfg", (), {})()
        holder.settings = self._settings
        return app.DeskpilotApp._mcp_servers_cfg(holder)

    def mcp_list(self) -> Dict[str, Any]:
        return {"ok": True, "servers": self._mcp_rows(),
                "max": getattr(app, "MCP_MAX_SERVERS", 8)}

    def mcp_add(self, name: str, command: str, args: str, env: str) -> Dict[str, Any]:
        name = (name or "").strip()
        command = (command or "").strip()
        if not name or not command:
            return {"ok": False, "errors": {"name": "Name and Command are both required."}}
        with self._lock:
            servers = self._mcp_rows()
            if any(s["name"] == name for s in servers):
                return {"ok": False, "errors": {"name": f"A server named '{name}' already exists."}}
            limit = getattr(app, "MCP_MAX_SERVERS", 8)
            if len(servers) >= limit:
                return {"ok": False, "errors": {"name": f"Maximum of {limit} servers."}}
            servers.append({
                "name": name,
                "command": command,
                "args": app._parse_args_list(args or ""),
                "env": app._parse_env_pairs(env or ""),
            })
            self._settings["mcp_servers"] = servers
            if not self._persist():
                return {"ok": False, "errors": {"_": "Could not save"}}
            return {"ok": True, "servers": servers}

    def mcp_remove(self, name: str) -> Dict[str, Any]:
        with self._lock:
            servers = [s for s in self._mcp_rows() if s["name"] != name]
            self._settings["mcp_servers"] = servers
            if not self._persist():
                return {"ok": False, "message": "Could not save"}
            return {"ok": True, "servers": servers}

    # ── system prompt (the ⇌ editor) ───────────────────────────────────────
    def prompt_get(self) -> Dict[str, Any]:
        """The editable SYSTEMPROMPT.md, plus the assembled "as sent" view.

        Two separate texts on purpose. The FILE is what Save writes; "as sent" is
        what the model receives. They are NOT the same thing - sync regenerates
        every managed block and the assembler injects more - so the UI must render
        the second read-only: editing it and saving would bake a timestamp and the
        app-owned blocks into the file.
        """
        name = getattr(app, "SYSTEMPROMPT_NAME", "SYSTEMPROMPT.md")
        try:
            path = app.system_prompt_file()
            text = app.system_prompt_text()
            exists = path.is_file()
        except Exception as e:
            return {"ok": False, "message": "Could not read %s: %s" % (name, e)}
        try:
            blocks = sorted({k for k, _s, _e in app._sp_managed_spans(text)})
        except Exception:
            blocks = []
        try:
            unknown = list(app._unknown_prompt_tokens())
        except Exception:
            unknown = []
        sent, sent_err = "", ""
        try:
            # studio=False: this window has no chat, so it shows what a normal
            # chat sends. Studio modes and the active project's memory block are
            # per-chat and are NOT in this text - said out loud in sent_note
            # rather than left for the user to discover.
            msg = app.build_system_message(str(self._settings.get("file_workspace") or ""),
                                           studio=False)
            sent = "" if msg is None else str(msg.get("content") or "")
        except Exception as e:
            sent_err = str(e)
        return {"ok": True, "path": str(path), "exists": exists, "name": name,
                "text": text, "sent": sent, "sent_err": sent_err,
                "sent_note": ("Assembled for a chat with Studio off. A chat's own "
                              "project memory block, its Studio mode, and the date "
                              "stamp the app appends to your latest message are not "
                              "shown here."),
                "enabled": bool(self._settings.get("custom_system_prompt_enabled", True)),
                "blocks": blocks, "unknown": unknown,
                "cap": int(getattr(app, "CUSTOM_PROMPT_MAX", 0) or 0),
                "size": len(text)}

    def _persist_for_prompt(self, updated: Dict[str, Any]) -> bool:
        # save_prompt_text() may write the persona back into the settings dict it
        # was handed; adopt that dict so the Next store gets it, and keep the
        # window's own copy in sync for the next prompt_get().
        self._settings = updated
        return self._persist()

    def prompt_save(self, text: str, baseline: str) -> Dict[str, Any]:
        """Save through the ONE contract (v1.2.27) - no guard is reimplemented here.

        `baseline` must be what prompt_get() returned, normalised. The lost-update
        guard compares it against the disk, so a Settings save or a skill toggle in
        another window refuses this write instead of being silently overwritten.
        """
        with self._lock:
            try:
                res = app.save_prompt_text(text, self._settings,
                                           app._sp_norm(baseline or ""),
                                           persist=self._persist_for_prompt)
            except Exception as e:
                return {"ok": False, "message": "Save failed: %s" % e, "warn": True,
                        "text": "", "touched": [], "unknown": [], "refused": ""}
            if res.get("ok"):
                res["enabled"] = bool(self._settings.get("custom_system_prompt_enabled",
                                                         True))
            return res

    def prompt_release(self, kind: str, text: str) -> Dict[str, Any]:
        """Cut one managed block out of the buffer. Pure - the user still presses Save,
        so the lost-update guard applies to a released block like any other edit."""
        try:
            return app.release_prompt_block(text, (kind or "").strip())
        except Exception as e:
            return {"ok": False, "message": "Could not release that block: %s" % e,
                    "text": text}

    def prompt_toggle(self, enabled: bool) -> Dict[str, Any]:
        """Gate the prompt WITHOUT discarding it (the setting keeps the text)."""
        with self._lock:
            self._settings["custom_system_prompt_enabled"] = bool(enabled)
            # Same order as the Tk toggle (deskpilot.py _on_prompt_toggled): persist
            # first, THEN sync - the persona lives in the file, so the block must be
            # rewritten from the setting that is now on disk.
            if not self._persist():
                return {"ok": False, "message": "Could not save"}
            try:
                app.sync_system_prompt(self._settings, "prompt-toggle")
            except Exception:
                pass
            return self.prompt_get()

    # ── browse ──────────────────────────────────────────────────────────────
    def pick_folder(self, label: str, current: str) -> str:
        import webview
        try:
            res = self._win.create_file_dialog(
                webview.FOLDER_DIALOG, directory=current or str(Path.home()))
        except Exception:
            return ""
        if not res:
            return ""
        return res[0] if isinstance(res, (list, tuple)) else str(res)

    # ── test connection ────────────────────────────────────────────────────
    def test_connection(self, url: str, key: str) -> Dict[str, Any]:
        try:
            models = app.fetch_model_info(url or "", key or "", timeout=8)
        except Exception as e:
            return {"ok": False, "message": f"Failed: {e}"}
        if not models:
            return {"ok": False,
                    "message": "No reply from the server (check URL, key and that it is running)."}
        ids = [str(m.get("id", "")) for m in models if m.get("id")]
        shown = ", ".join(ids[:4]) + (" …" if len(ids) > 4 else "")
        return {"ok": True, "message": f"Connected — {len(ids)} model(s): {shown}"}

    # ── cancel ─────────────────────────────────────────────────────────────
    def cancel(self) -> None:
        try:
            self._win.destroy()
        except Exception:
            pass


def resolve_profiles(isolated: bool, settings_path: Optional[Path] = None,
                     write_override: Optional[Path] = None):
    """(read_path, write_path, geometry_base) for this window's profile mode.

    isolated=True  (the Next launcher) - READ the live store so the fields show real
                     values, WRITE the Next store. That deliberate asymmetry is the
                     side-by-side requirement: the stable app must never be touched.
    isolated=False (the shipped app)   - READ and WRITE the same store, the one the
                     running app is using. Applied in the shipped app the split above
                     would write to a profile nothing reads, so Save would look like
                     it did nothing at all.

    Pure on purpose: run() starts pywebview's loop and cannot be called headlessly,
    and this is the decision that fails silently if wrong.
    """
    if isolated:
        read = settings_path or read_path()
        write = write_override or write_path()
        return read, write, next_data_dir()
    read = settings_path or app.SETTINGS_FILE
    write = write_override or read
    return read, write, Path(read).parent


def run(settings_path: Optional[Path] = None, *, isolated: bool = True,
        write_override: Optional[Path] = None) -> int:
    """Open the settings window.

    isolated=True  (the Next launcher) - READ the live store so the fields show
                     real values, WRITE the Next store, and point DATA_DIR at the
                     Next profile so the prompt file resolves there too. The whole
                     point is that the stable app is never touched.
    isolated=False (the shipped app)   - READ and WRITE the app's own store, and do
                     NOT retarget DATA_DIR. The split above is the isolation
                     requirement; used here it would write to a profile the running
                     app never reads, so Settings would look like it did nothing.

    Both call sites state the mode explicitly. Nothing infers it from the
    environment, because guessing wrong here fails silently.
    """
    import webview

    if isolated:
        # Must happen BEFORE anything touches the prompt file, and before Api is
        # built: without it the child edits the LIVE app's SYSTEMPROMPT.md.
        use_next_profile()
    read, write, geo_base = resolve_profiles(isolated, settings_path, write_override)
    settings = load_settings(read)

    html_path = _HERE / "settings_ui.html"
    html = html_path.read_text(encoding="utf-8")

    payload = build_payload(settings)
    html = html.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))

    api = Api(None, settings, write)
    geo = load_geometry(geo_base)
    win = webview.create_window(
        "DeskPilot Settings",
        html=html,
        x=geo.get("x"), y=geo.get("y"),
        width=geo.get("width", 980), height=geo.get("height", 660),
        min_size=(760, 520),
        background_color="#16181d",
        # A child process has no Tk parent to stay above the main window, so the
        # first build appeared BEHIND it. focus=True + on_top=True is pywebview's
        # own answer - no Win32 SetForegroundWindow hacks needed.
        focus=True,
        # on_top=True is only for the OPENING moment: a child process has no Tk
        # parent, so without it the window appeared BEHIND the main app. It is
        # cleared on `shown` below, which is why the window is not permanently
        # unlosable any more.
        on_top=True,
        js_api=api,
    )
    api._win = win

    def _unpin(_e=None) -> None:
        """Front on open, normal window afterwards.

        unpin_topmost() goes through Win32 SetWindowPos, which the OS marshals to
        the UI thread, so running it on the thread pywebview uses for unlocked
        events is fine. Do NOT replace this with `win.on_top = False`: that writes
        a WinForms property from this thread and crashed the window (blank client
        area, "(Not Responding)", crash on close) on 2026-11-07.
        """
        try:
            unpin_topmost("DeskPilot Settings")
        except Exception:
            pass    # a window that stays pinned is the old annoyance, not a crash

    win.events.shown += _unpin

    def _remember_geometry() -> None:
        try:
            save_geometry({"x": win.x, "y": win.y,
                           "width": win.width, "height": win.height}, geo_base)
        except Exception:
            pass

    win.events.closing += _remember_geometry

    webview.start(debug=False)
    return 0


if __name__ == "__main__":
    # Running this file directly is the NEXT entry point, so it is the isolated
    # mode by definition. The shipped app does not go through here - it dispatches
    # --settings-window from deskpilot.main() and calls run(isolated=False).
    sys.exit(run(isolated=True))
