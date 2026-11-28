"""config.toml loader with mtime-based hot reload.

The config file is re-read only when its mtime changes, so a config edit
takes effect on the next tool call — no server restart, no reload endpoint.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

_DEFAULTS: dict[str, Any] = {
    "embedding": {
        "model": "BAAI/bge-base-en-v1.5",
        "cache_path": ".fastembed_cache",
        # Non-empty routes embedding R/W to OpenRouter (OpenAI SDK, base_url
        # openrouter.ai); empty falls back to fastembed. The operator owns
        # this string — it is sent as-is, never validated, never substituted.
        "openrouter_model": "",
    },
    "server": {
        "host": "127.0.0.1",
        "port": 8765,
    },
    "logging": {
        "level": "INFO",
        "format": "json",
        "sink": "stderr",
        "file": "openmind.log",
        "max_bytes": 10_485_760,
        "backup_count": 5,
    },
    "write": {
        "max_document_write_attempts": 3,
        "retry_backoff_seconds": 1,
    },
}

_path = Path("config.toml")
_cached: dict[str, Any] | None = None
_cached_mtime: float | None = None


def _merge(defaults: dict[str, Any], loaded: dict[str, Any]) -> dict[str, Any]:
    """Shallow-merge `loaded` over `defaults`, one level deep (section dicts)."""
    out = dict(defaults)
    for key, value in loaded.items():
        base = out.get(key)
        if isinstance(base, dict) and isinstance(value, dict):
            out[key] = {**base, **value}
        else:
            out[key] = value
    return out


def _load() -> dict[str, Any]:
    global _cached, _cached_mtime
    try:
        mtime = _path.stat().st_mtime_ns
    except FileNotFoundError:
        # No config file: fall back to defaults; retry when one appears.
        _cached, _cached_mtime = dict(_DEFAULTS), None
        return _cached

    if _cached is not None and mtime == _cached_mtime:
        return _cached

    try:
        with _path.open("rb") as fh:
            loaded = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        # Unreadable or malformed config: keep the last good copy.
        if _cached is not None:
            return _cached
        loaded = {}

    _cached = _merge(_DEFAULTS, loaded)
    _cached_mtime = mtime
    return _cached


def get(section: str, key: str | None = None) -> Any:
    """Read a config value. Re-checks the file's mtime on every call."""
    cfg = _load()
    value = cfg.get(section)
    if key is None:
        return value
    if not isinstance(value, dict):
        return None
    return value.get(key)


# --- .env chain ------------------------------------------------------------

ENV_FILE = ".env"
GLOBAL_ENV_FILE = "../.env"


def _env_file(path: str) -> dict[str, str]:
    """Parse one KEY=VALUE env file. Unreadable or malformed lines are
    skipped, never guessed: a broken line must not become a wrong secret."""
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError:
        return {}
    out: dict[str, str] = {}
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        out[key.strip()] = value.strip().strip("'\"")
    return out


def env(name: str) -> str | None:
    """Read one secret: colocated `.env`, then the global `../.env`, then the
    process environment. The file chain is the documented home for tokens
    (config.toml stays credential-free, RULES.md A02); the process env is the
    final fallback so a container can inject the same variable.

    Every call re-reads the files: `config.py`'s mtime cache does not apply
    here, so a rotated token takes effect on the next call, not the next
    restart.
    """
    for path in (ENV_FILE, GLOBAL_ENV_FILE):
        value = _env_file(path).get(name)
        if value:
            return value
    value = os.environ.get(name)
    return value or None
