"""open-mind MCP server — Starlette + uvicorn, streamable-HTTP transport.

RBAC note: identity checks belong at the HTTP gate (the middleware below),
before any MCP message reaches a tool handler. The slot is a no-op today;
enabling RBAC later means filling this one class in, nothing else changes.
"""

from __future__ import annotations

import logging
import socket

import uvicorn
from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from openmind import __version__, telemetry
from openmind.config import get
from openmind.write import WriteFailed, WriteRejected
from openmind.write import write_document as commit_document

mcp = MCPServer("open-mind", version=__version__)


# --- tool stubs -----------------------------------------------------------
# Signatures and JSON Schemas are TBD; the SDK derives the schema from the
# type hints, so finalising signatures finalises the advertised schemas.


@mcp.tool()
def list_documents(wiki: str) -> dict:
    """List documents in a wiki so the agent can identify an edit target."""
    raise NotImplementedError("stub — signatures still under review")


@mcp.tool()
def doc_read(document_id: int) -> dict:
    """Return a document's sections before an informed edit."""
    raise NotImplementedError("stub — signatures still under review")


@mcp.tool()
async def write_document(
    wiki: str,
    sections: list[dict],
    log_entry: str,
    progress: dict,
    title: str | None = None,
    document_id: int | None = None,
    references: list[dict] | None = None,
    mode: str = "replace_all",
) -> dict:
    """Write one document with its progress record and log entry, atomically.

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
            wiki=wiki,
            error_type="WriteRejected",
            error_message=str(exc),
        )
        raise ValueError(f"{exc.field}: {exc.reason}") from exc
    except WriteFailed as exc:
        raise RuntimeError(str(exc)) from exc


@mcp.tool()
def log_recall(since: str | None = None) -> dict:
    """Scan recent log entries, optionally from a date. Session's first read."""
    raise NotImplementedError("stub — signatures still under review")


@mcp.tool()
def semantic_search(query: str, wiki: str | None = None) -> list[dict]:
    """Vector match over section embeddings — the default search."""
    raise NotImplementedError("stub — signatures still under review")


@mcp.tool()
def fts_search(query: str, wiki: str | None = None) -> list[dict]:
    """Exact-token match over section heading+body via FTS5."""
    raise NotImplementedError("stub — signatures still under review")


@mcp.tool()
def table_search(wiki_id: int, contains: str | None = None) -> list[dict]:
    """Crawl wiki → documents → sections as plain rows; optional substring filter."""
    raise NotImplementedError("stub — signatures still under review")


# --- HTTP gate -------------------------------------------------------------


class AccessControl(BaseHTTPMiddleware):
    """RBAC slot. No-op until resource_access lands; then it runs here,
    before the request reaches MCP dispatch."""

    async def dispatch(self, request: Request, call_next):
        return await call_next(request)


def _security_settings(host: str, port: int) -> TransportSecuritySettings:
    """Allow the configured bind host plus common local names — otherwise the
    SDK's DNS-rebinding protection answers 421 to every off-localhost call."""
    allowed = {
        f"{host}:{port}",
        f"localhost:{port}",
        f"127.0.0.1:{port}",
        f"[::1]:{port}",
    }
    # Binding 0.0.0.0: peers reach us by the handles the network gave us —
    # this container's hostname (podman network DNS) and its real IPs.
    # Without them the DNS-rebinding guard 421s every non-loopback request.
    allowed.add(f"{socket.gethostname()}:{port}")

    for info in socket.getaddrinfo(socket.gethostname(), port, socket.AF_INET):
        allowed.add(f"{info[4][0]}:{port}")
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=sorted(allowed),
        allowed_origins=sorted(
            f"http://{h}:{port}" for h in (host, "localhost", "127.0.0.1")
        ),
    )


def create_app():
    """The Starlette app: SDK streamable-HTTP endpoint behind the HTTP gate."""
    host = get("server", "host") or "127.0.0.1"
    port = get("server", "port") or 8765
    telemetry.configure()
    telemetry.log_event(
        "server.starting",
        host=host,
        port=port,
        version=__version__,
        mcp_path="/mcp",
    )
    app = mcp.streamable_http_app(
        streamable_http_path="/mcp",
        transport_security=_security_settings(host, port),
    )
    app.add_middleware(AccessControl)
    return app


def main() -> None:
    host = get("server", "host") or "127.0.0.1"
    port = get("server", "port") or 8765
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
