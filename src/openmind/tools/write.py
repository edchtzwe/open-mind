"""Write-domain tools: thin MCP adapters over `openmind.write`.

Signatures and docstrings are the model's contract; the transactional
logic lives in the domain module. Exceptions map to JSON-RPC errors here
and nowhere else.
"""

from __future__ import annotations

import logging

from openmind import telemetry
from openmind.tools import mcp
from openmind.write import WriteFailed, WriteRejected
from openmind.write import create_wiki as commit_wiki
from openmind.write import write_document as commit_document


@mcp.tool()
async def create_wiki(name: str) -> dict:
    """Create a wiki before writing its first document.

    `name` is the wiki's permanent identity: casing is preserved as given,
    lookup everywhere else is case-insensitive, so send it back exactly as
    returned. One transaction creates the wiki and appends the root log
    entry. A duplicate name or a failed write leaves nothing behind.
    """
    try:
        return await commit_wiki(name)
    except WriteRejected as exc:
        telemetry.log_event(
            "create_wiki.rejected",
            level=logging.WARNING,
            tool="create_wiki",
            error_type="WriteRejected",
            error_message=str(exc),
        )
        raise ValueError(f"{exc.field}: {exc.reason}") from exc
    except WriteFailed as exc:
        raise RuntimeError(str(exc)) from exc


@mcp.tool()
async def write_document(
    wiki: str | dict,
    sections: list[dict],
    log_entry: str,
    progress: dict,
    title: str | None = None,
    document_id: int | None = None,
    references: list[dict] | None = None,
    mode: str = "replace_all",
) -> dict:
    """Write one document with its progress record and log entry, atomically.

    `wiki` is a name string, or {"id", "title"}: pass the id returned by a
    prior write to target an existing wiki, or {"id": null, "title": "..."}
    to create the wiki in this same transaction — a duplicate title
    rejects, so send the id back once you have it.
    Give `title` to create, `document_id` to edit an existing document in the
    same wiki. `sections` is [{"position", "heading", "body"}], positions 1-based
    and unique; `mode` is "replace_all" (delete the document's sections first) or
    "replace_positions" (replace only the positions supplied).
    `log_entry` is one sentence; the server appends it to the root log.
    `progress` is {"slug", "sections"} — the server numbers the title.
    `references` lists {"wiki", "title"} of documents this one links to.
    Progress is insert-only, the root log is system-only, and a root document
    cannot be written or deleted. Every failure rolls the whole write back.
    """
    try:
        return await commit_document(
            {
                "wiki": wiki,
                "title": title,
                "document_id": document_id,
                "sections": sections,
                "mode": mode,
                "log_entry": log_entry,
                "progress": progress,
                "references": references,
            }
        )
    except WriteRejected as exc:
        telemetry.log_event(
            "write_document.rejected",
            level=logging.WARNING,
            tool="write_document",
            error_type="WriteRejected",
            error_message=str(exc),
        )
        raise ValueError(f"{exc.field}: {exc.reason}") from exc
    except WriteFailed as exc:
        raise RuntimeError(str(exc)) from exc
