"""MCP tool surface: every tool the server advertises lives here, never in
`server.py`.

One module per domain (`write`, `read`); each registers its tools with the
shared `MCPServer` at import. `server.py` imports this package and wires the
resulting app — a controller lists routes, not kernels.
"""

from __future__ import annotations

from mcp.server import MCPServer

from openmind import __version__

mcp = MCPServer("open-mind", version=__version__)

# Importing the modules runs their @mcp.tool() registrations against `mcp`
# above — explicit, order-independent, no decorator magic across packages.
from openmind.tools import read as _read  # noqa: E402, F401
from openmind.tools import write as _write  # noqa: E402, F401

__all__ = ["mcp"]
