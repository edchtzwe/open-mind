"""Process telemetry — JSON lines, stdlib only, one handler.

Two different logs live in this project and must never share a path. The root
`log` document is knowledge: it is written inside the same SQLite transaction
as the document it describes, so a failed write erases it. This module is the
opposite — it leaves the process on its own, so a rolled-back write still
leaves evidence behind.

Stdlib only. RULES.md A06 pins the dependency set at three; Datadog,
CloudWatch and a local collector all ingest JSON lines, so shipping is the
platform's job, not this library's. Changing sinks means changing one handler,
never the call sites.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from typing import Any

from openmind.config import get

LOGGER_NAME = "openmind"

# LogRecord attributes that `extra=` must not shadow; shadowing raises.
_RESERVED = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
        "asctime",
    }
)


class JsonFormatter(logging.Formatter):
    """One line per record with a fixed key set — no free-form prose."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
        }
        # Every extra the caller supplied, minus the reserved record fields.
        for key, value in record.__dict__.items():
            if key in _RESERVED or key in payload or key.startswith("_"):
                continue
            if value is not None:
                payload[key] = value
        if "event" not in payload:
            payload["event"] = record.getMessage()
        if record.exc_info:
            payload["error_type"] = record.exc_info[0].__name__
            payload["error_message"] = str(record.exc_info[1])
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def get_logger() -> logging.Logger:
    """The one logger. Handlers are installed by `configure()`, not here."""
    return logging.getLogger(LOGGER_NAME)


def configure() -> None:
    """Install the single handler. Called once from `create_app()`."""
    section = get("logging") or {}
    level = str(section.get("level", "INFO")).upper()
    sink = str(section.get("sink", "stderr"))
    fmt = str(section.get("format", "json"))

    logger = get_logger()
    logger.setLevel(level)
    # Own the output: uvicorn's root handlers must not duplicate every line.
    logger.propagate = False
    for existing in list(logger.handlers):
        logger.removeHandler(existing)
        existing.close()

    if sink == "file":
        handler: logging.Handler = logging.handlers.RotatingFileHandler(
            str(section.get("file", "openmind.log")),
            maxBytes=int(section.get("max_bytes", 10_485_760)),
            backupCount=int(section.get("backup_count", 5)),
        )
    else:
        handler = logging.StreamHandler(sys.stderr)

    handler.setFormatter(
        JsonFormatter()
        if fmt == "json"
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(handler)


def log_event(event: str, *, level: int = logging.INFO, **fields: Any) -> None:
    """Emit one structured event. None fields are dropped, not serialised."""
    extra = {k: v for k, v in fields.items() if k not in _RESERVED and v is not None}
    extra["event"] = event
    get_logger().log(level, event, extra=extra)
