"""open-mind MCP server — Starlette + uvicorn, streamable-HTTP transport.

RBAC note: identity checks belong at the HTTP gate (the middleware below),
before any MCP message reaches a tool handler. The slot is a no-op today;
enabling RBAC later means filling this one class in, nothing else changes.
"""

from __future__ import annotations

import socket

import uvicorn
from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from openmind import __version__
from openmind.config import get

mcp = MCPServer("open-mind")


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
def write_document(**kwargs) -> dict:
    """Write a document plus its progress record and log entry, atomically."""
    raise NotImplementedError("stub — validation rules still under review")


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
