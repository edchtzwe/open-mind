"""open-mind MCP server — Starlette + uvicorn, streamable-HTTP transport.

Tools are not defined here. `openmind.tools` owns the listing; this module
only wires the app, the security settings and the HTTP gate.

RBAC note: identity checks belong at the HTTP gate (the middleware below),
before any MCP message reaches a tool handler. The slot is a no-op today;
enabling RBAC later means filling this one class in, nothing else changes.
"""

from __future__ import annotations

import socket

import uvicorn
from mcp.server.transport_security import TransportSecuritySettings
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from openmind import __version__, telemetry
from openmind.config import ENV_FILE, GLOBAL_ENV_FILE, get
from openmind.embedding import openrouter_key
from openmind.tools import mcp


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


def check_openrouter_config() -> None:
    """Fail-fast at startup: `[embedding] openrouter_model` without a token
    anywhere is a configuration error, not something to discover on the first
    embed. Called once from `server.create_app()`; the .env chain is checked
    first, the process environment last, so a container can inject the key.
    """
    openrouter_model = (get("embedding", "openrouter_model") or "").strip()
    if openrouter_model and not openrouter_key():
        raise RuntimeError(
            f"embedding.openrouter_model is set to {openrouter_model!r} but no "
            f"OPENROUTER_API_KEY was found.\n"
            f"  Provide it in {ENV_FILE} (beside config.toml), {GLOBAL_ENV_FILE} "
            "(one level up), or the process environment — or clear openrouter_model "
            "to fall back to fastembed."
        )


def create_app():
    """The Starlette app: SDK streamable-HTTP endpoint behind the HTTP gate."""
    host = get("server", "host") or "127.0.0.1"
    port = get("server", "port") or 8765
    check_openrouter_config()  # refuse to boot with a flag and no token
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
