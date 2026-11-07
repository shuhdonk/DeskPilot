"""DeskPilot Next — settings SCHEMA (single source of truth).

Why this file exists
--------------------
The Tk settings dialog builds itself as a flat `rows = [...]` list: label text,
help text, type, and validation live in three different places, and every
layout test hard-codes a widget census. Adding a setting meant editing the
dialog, the tests, and the docs.

Here a setting is ONE record. The web UI renders from it; the future Tk dialog
renders from it; tests assert against it. Adding a setting is one entry.

VERIFIED AGAINST the working-copy deskpilot.py (DEFAULT_SETTINGS + SAMPLING_SCHEMA).
No version stamp here on purpose - test_settings_schema.py re-checks every key and
range against the live app on every run, so this list cannot rot:
  * sampler fields are BLANK STRINGS by default ("omit = server default"), not
    numbers. A blank must round-trip as blank or the "Sent:" note changes meaning.
  * off-values suppress the key: top_k 0, min_p 0, repetition_penalty 1.0,
    presence/frequency 0. seed is blank (None) because 0 is a VALID seed.
  * ranges come from SAMPLING_SCHEMA: top_p 0-1, top_k 0-400, min_p 0-1,
    repetition 1-2, presence/frequency -2..2, seed 0-2147483647.
  * turn_time_limit default is 0 (no limit), chat_render_window 120,
    tool_keep_verbatim 12, tool_nudge_every 20, project_memory_chars 8000.

Field record:
    key      - exact key in deskpilot_settings.json
    label    - short human name
    help     - ONE sentence under the field (never crammed into the label)
    type     - text | password | int | float | sampler | bool | path
    default  - value used when the key is absent
    unit     - optional suffix
    min/max  - numeric bounds; validated INLINE, never with a modal
    advanced - hidden behind "Show advanced"

Deliberately EXCLUDED (own panels / pure view state):
    window_geometry, sidebar_width, sidebar_collapsed, model_history,
    prompt_preview_geometry, mcp_bar_expanded, spellcheck_words,
    expert_model_history, tool_permissions.
    skills_enabled and mcp_servers are LISTS, not scalar fields: they are edited
    by the Skills / MCP panels (`kind` sections) and written by the bridge, so
    they never appear as form fields.
"""

from __future__ import annotations

SAMPLING_SECTION = "sampling_defaults"


def _s(key, label, help_text, default="", lo=None, hi=None, is_int=False):
    return dict(key=key, label=label, help=help_text, type="sampler",
                default=default, min=lo, max=hi, int=is_int, unit=None,
                store="sampling")


SECTIONS = [
    {
        "id": "connection",
        "title": "Connection",
        "subtitle": "Where DeskPilot talks to the model server.",
        "fields": [
            dict(key="server_url", label="Server URL", type="text",
                 default="http://localhost:11434/v1",
                 help="OpenAI-compatible base URL, ending in /v1."),
            dict(key="api_key", label="API Key", type="password",
                 default="gemma-local",
                 help="Sent as the bearer token. Local servers usually accept any value."),
            dict(key="model_name", label="Model Name", type="text",
                 default="gemma3", refresh="models", suggest="model_history",
                 help="Exact model id the server advertises."),
            dict(key="context_window_override", label="Context window", type="int",
                 default=0, min=0,
                 help="0 = auto-detect from the server. Set it only if detection is wrong."),
            dict(key="max_tokens", label="Max tokens per reply", type="int",
                 default=0, min=0,
                 help="0 = server default."),
            dict(key="turn_time_limit", label="Turn time limit", type="int",
                 default=0, min=0, unit="s",
                 help="0 = no limit; otherwise the turn is cancelled when it runs over."),
        ],
    },
    {
        "id": "sampling",
        "title": "Sampling",
        "subtitle": "Applied to every chat unless a chat overrides it. Blank = server default.",
        "fields": [
            _s("top_p", "Top P", "Nucleus sampling. Blank = server default.", "", 0.0, 1.0),
            _s("top_k", "Top K", "0 = off. Blank = server default.", "", 0, 400, True),
            _s("min_p", "Min P", "0 = off. Blank = server default.", "", 0.0, 1.0),
            _s("repetition_penalty", "Repetition penalty", "1.0 = off.", "", 1.0, 2.0),
            _s("presence_penalty", "Presence penalty", "-2 to 2. 0 = off.", "", -2.0, 2.0),
            _s("frequency_penalty", "Frequency penalty", "-2 to 2. 0 = off.", "", -2.0, 2.0),
            _s("seed", "Seed", "Blank = random. Set it to make a run repeatable.",
               "", 0, 2147483647, True),
            # ⚠ These three are TOP-LEVEL settings, NOT entries in sampling_defaults.
            # The app reads them with settings.get("show_sampling_note", True) etc.
            # Putting them in the sampling section must NOT route them into the
            # sampler dict - that is why `store` is explicit per field.
            dict(key="show_sampling_note", label="Show what was sent", type="bool", default=True,
                 store="top",
                 help="One note per prompt listing the sampling values actually sent."),
            dict(key="studio_tools_off", label="Studio withholds tools", type="bool", default=True,
                 store="top",
                 help="Studio chats get no tool schemas; per-tool grants can re-admit one."),
            dict(key="studio_sampling", label="Studio loosens sampling", type="bool", default=True,
                 store="top",
                 help="Studio chats use a looser top_k/temperature for more varied text."),
        ],
    },
    {
        "id": "conversation",
        "title": "Conversation",
        "subtitle": "When context is summarised, handed off, or trimmed.",
        "fields": [
            dict(key="handoff_threshold_pct", label="Handoff threshold", type="int",
                 default=75, min=0, max=100, unit="%",
                 help="Summarise into a handoff note at this context usage. 0 = off."),
            dict(key="handoff_rearm_pct", label="Handoff re-arm", type="int",
                 default=10, min=0, max=100, unit="%",
                 help="Gauge must grow this much again before another handoff. 0 = every step."),
            dict(key="handoff_notes_dir", label="Handoff notes folder", type="path", default="",
                 help="Blank = deskpilot_data/handoff_notes next to the app."),
            dict(key="compaction_threshold", label="Compaction threshold", type="int",
                 default=70, min=0, max=100, unit="%",
                 help="Fold old turns into a summary at this usage. 0 = off."),
            dict(key="compaction_keep_recent", label="Keep recent messages", type="int",
                 default=30, min=0,
                 help="Most-recent messages kept verbatim when compacting."),
            dict(key="chat_render_window", label="Render window", type="int",
                 default=120, min=0,
                 help="Messages drawn per chat view. 0 = all (slow on long chats)."),
            dict(key="tool_keep_verbatim", label="Tool results in full", type="int",
                 default=12, min=0,
                 help="Newest tool results sent in full; older ones are stubbed. 0 = off."),
            dict(key="tool_nudge_every", label="Tool nudge interval", type="int",
                 default=20, min=0,
                 help="Advisory efficiency note every N tool steps. Never ends a turn. 0 = off."),
            dict(key="project_memory_chars", label="Project memory budget", type="int",
                 default=8000, min=0,
                 help="Characters of the workspace MEMORY.md injected per request. 0 = off."),
            dict(key="project_memory_file", label="Project memory file", type="path", default="",
                 help="Blank = <workspace>/MEMORY.md."),
        ],
    },
    {
        "id": "workspace",
        "title": "Workspace",
        "subtitle": "Where files and skills are read from.",
        "fields": [
            dict(key="file_workspace", label="File workspace", type="path", default="",
                 help="Confines read/write_local_file to this folder. Blank = unrestricted."),
            dict(key="skills_dir", label="Skills folder", type="path", default="",
                 help="Extra folder scanned for Agent Skills. Blank = defaults only."),
        ],
    },
    {
        "id": "prompt",
        "title": "System prompt",
        "subtitle": "The editable SYSTEMPROMPT.md, and what the model actually receives",
        "kind": "prompt",
        # Not a form: a full editor, driven by the bridge's
        # prompt_get/prompt_save/prompt_release/prompt_toggle, which delegate to the
        # app's own save_prompt_text() contract. This section owns
        # custom_system_prompt and custom_system_prompt_enabled outright - they are
        # deliberately absent from Appearance so one setting has one control.
        "fields": [],
    },
    {
        "id": "skills",
        "title": "Skills",
        "subtitle": "Install Agent Skills and choose which ones the model may use.",
        # `kind` sections carry NO fields. The web UI renders a custom panel and
        # talks to the bridge; the same schema still drives the nav and the tests.
        "kind": "skills",
        "fields": [],
    },
    {
        "id": "mcp",
        "title": "MCP servers",
        "subtitle": "External stdio servers whose tools join the permission bar.",
        "kind": "mcp",
        "fields": [],
    },
    {
        "id": "tools",
        "title": "Tools & keys",
        "subtitle": "Search, scraping and the network guard.",
        "fields": [
            dict(key="searxng_url", label="SearXNG URL", type="text", default="",
                 help="Blank = derive from the server host on port 8080."),
            dict(key="exa_api_key", label="Exa API key", type="password", default="",
                 help="Blank disables the exa_search tool."),
            dict(key="firecrawl_api_key", label="Firecrawl API key", type="password", default="",
                 help="Blank disables the firecrawl_scrape tool."),
            dict(key="allow_local_network", label="Allow LAN addresses", type="bool", default=False,
                 help="Lets fetch_url reach localhost/LAN. Off blocks SSRF-style targets."),
            dict(key="spellcheck_enabled", label="Spellcheck input", type="bool", default=False,
                 help="Underline suspected misspellings in the input box (needs pyspellchecker)."),
        ],
    },
    {
        "id": "consultant",
        "title": "Consultant",
        "subtitle": "The optional second-opinion model (ask_expert).",
        "fields": [
            dict(key="expert_server_url", label="Consultant URL", type="text", default="",
                 help="Any OpenAI-compatible endpoint. Blank disables ask_expert."),
            dict(key="expert_api_key", label="Consultant key", type="password", default="",
                 help="Never reuses the primary API key."),
            dict(key="expert_model", label="Consultant model", type="text", default="",
                 refresh="expert_models", suggest="expert_model_history",
                 help="Use the model list, do not guess. Blank disables ask_expert."),
        ],
    },
    {
        "id": "appearance",
        "title": "Appearance",
        "subtitle": "Font size and voice.",
        "fields": [
            dict(key="ui_font_size", label="UI font size", type="int",
                 default=12, min=8, max=20, unit="px",
                 help="Base size for the whole interface."),
            # custom_system_prompt / _enabled are NOT fields here. They belong to the
            # System prompt panel, which edits the real file and writes the persona
            # back through save_prompt_text(). Leaving a checkbox and a raw textarea
            # here put two controls on one setting in the same window - they can
            # disagree on screen and nothing says which one wins.
            dict(key="tts_enabled", label="Text-to-speech", type="bool", default=False,
                 help="Speak replies aloud with the local Kokoro voice."),
        ],
    },
]


# ── flat lookup helpers ──────────────────────────────────────────────────────
FIELDS = {}
for _sec in SECTIONS:
    for _f in _sec["fields"]:
        _f["section"] = _sec["id"]
        # `store` decides WHERE the value lives: "sampling" = inside the
        # sampling_defaults dict; "top" = a top-level settings key. Section
        # membership alone is NOT enough - the sampling section contains both
        # sampler entries and top-level toggles (show_sampling_note, studio_*).
        _f.setdefault("store", "top")
        FIELDS[_f["key"]] = _f


def _num(raw):
    """Parse a numeric field, tolerating blanks and junk. Returns (value|None, ok)."""
    if raw is None:
        return None, True
    if isinstance(raw, str):
        raw = raw.strip()
        if raw == "":
            return None, True
    try:
        return float(raw), True
    except (TypeError, ValueError):
        return None, False


def coerce(key, value):
    """Turn a UI-supplied value into the type the settings dict expects.

    Returns (value, error_or_None). Errors are rendered INLINE next to the
    field, not in a modal - that was the main complaint about the old dialog.
    """
    f = FIELDS.get(key)
    if f is None:
        return value, None

    t = f["type"]
    if t == "bool":
        if isinstance(value, bool):
            return value, None
        return str(value).strip().lower() in ("1", "true", "yes", "on"), None

    if t in ("int", "float", "sampler"):
        raw = "" if value is None else str(value).strip()
        if raw == "":
            # Blank is meaningful for samplers (omit = server default) and for seed.
            return ("" if t == "sampler" else (None if f.get("default") is None else f["default"])), None
        num, ok = _num(raw)
        if not ok:
            return None, f"{f['label']} must be a number."
        if t == "int":
            if num != int(num):
                return None, f"{f['label']} must be a whole number."
            num = int(num)
        elif f.get("int"):
            num = int(round(num))
        lo, hi = f.get("min"), f.get("max")
        if lo is not None and num < lo:
            return None, f"{f['label']} must be at least {lo:g}."
        if hi is not None and num > hi:
            return None, f"{f['label']} must be at most {hi:g}."
        return num, None

    raw = "" if value is None else str(value)
    if not f.get("multiline"):
        raw = raw.strip()
    return raw, None


def build_payload(settings):
    """What the web UI receives: schema + current values. No secrets are redacted
    here - the payload never leaves the process; it is injected into the HTML."""
    values = {}
    sampling = settings.get(SAMPLING_SECTION) if isinstance(settings.get(SAMPLING_SECTION), dict) else {}
    for sec in SECTIONS:
        for f in sec["fields"]:
            k = f["key"]
            if f.get("store") == "sampling":
                v = sampling.get(k, f["default"])
            else:
                v = settings.get(k, f["default"])
            values[k] = "" if v is None else v
    # Suggestion lists for refreshable fields (previously-used model names). These
    # are deliberately NOT form fields - the user never edits a history list - but the
    # UI needs them so the model dropdown still offers something when the server is
    # down, which is what the Tk combobox did by merging history into its values.
    suggest = {}
    for sec in SECTIONS:
        for f in sec["fields"]:
            skey = f.get("suggest")
            if skey:
                lst = settings.get(skey)
                suggest[f["key"]] = [h for h in (lst or []) if isinstance(h, str)]
    return {"sections": SECTIONS, "values": values, "suggest": suggest,
            "sampling_section": SAMPLING_SECTION}


def apply_values(settings, incoming):
    """Write validated values back. Returns (settings, errors_by_key)."""
    errors = {}
    sampling = settings.get(SAMPLING_SECTION)
    if not isinstance(sampling, dict):
        sampling = {}
        settings[SAMPLING_SECTION] = sampling

    for key, raw in incoming.items():
        f = FIELDS.get(key)
        if f is None:
            continue
        val, err = coerce(key, raw)
        if err:
            errors[key] = err
            continue
        if f.get("store") == "sampling":
            sampling[key] = val
        else:
            settings[key] = val
    return settings, errors
