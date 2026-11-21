"""config.toml loader with mtime-based hot reload.

The config file is re-read only when its mtime changes, so a config edit
takes effect on the next tool call — no server restart, no reload endpoint.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

_DEFAULTS: dict[str, Any] = {
    "embedding": {
        "model": "BAAI/bge-small-en-v1.5",
        "cache_path": ".fastembed_cache",
    },
    "server": {
        "host": "127.0.0.1",
        "port": 8765,
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
