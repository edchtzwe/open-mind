"""Read and search tools: still stubs while signatures are under review.

Moving them out of `server.py` now keeps the controller free of listings
even while the bodies are `NotImplementedError`.
"""

from __future__ import annotations

from openmind.tools import mcp


@mcp.tool()
def list_documents(wiki: str) -> dict:
    """List documents in a wiki so the agent can identify an edit target."""
    raise NotImplementedError("stub — signatures still under review")


@mcp.tool()
def doc_read(document_id: int) -> dict:
    """Return a document's sections before an informed edit."""
    raise NotImplementedError("stub — signatures still under review")


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
