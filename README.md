# open-mind

The LLM wiki (see obsidian encyclopedia project), built around semantic search.
MCP server, streamable-HTTP transport, SQLite store.

## Start the server (inside the container)

`config.toml` is read from the current directory, so start from the repo root:

```bash
cd /zeroclaw-data/.zeroclaw/agents/clawz/workspace/open-mind
.venv/bin/open-mind
```

Equivalent forms:

```bash
.venv/bin/python -m openmind.server
.venv/bin/uvicorn openmind.server:create_app --factory --host 0.0.0.0 --port 8765
```

`config.toml` is gitignored; copy `config.toml.example` and edit `[server] host` /
`port`. Edits are picked up on the next tool call — no restart.

## URLs

| Reaching it from | URL |
| :--- | :--- |
| Inside this container | `http://127.0.0.1:8765/mcp` |
| Another container, by IP | `http://10.89.0.7:8765/mcp` |
| Another container, by container name | `http://816fbac0e5aa:8765/mcp` |

`10.89.0.7` is this container's current lease and `816fbac0e5aa` is its current
hostname — the container ID, since none was set. Both change when the container is
recreated; the container **name** is the only stable handle. Give it one on the
host, then peers use that name instead of an ID:

```bash
podman run --name open-mind --hostname open-mind ...
# peers: http://open-mind:8765/mcp
```

All three forms require `[server] host = "0.0.0.0"` in `config.toml`. The shipped
example keeps `127.0.0.1`, which serves loopback only — deliberate, because the
server has no authentication yet.

## Calling it

`/mcp` is a single endpoint; every call is `POST` with two headers:

```
Content-Type: application/json
Accept: application/json, text/event-stream
```

`initialize` returns an `Mcp-Session-Id` header — repeat it on every later request.
Sessions are held in memory and idle ones are dropped after a timeout.

```bash
curl -i -X POST http://127.0.0.1:8765/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{
        "protocolVersion":"2025-06-18","capabilities":{},
        "clientInfo":{"name":"curl","version":"0"}}}'
```

## Transport security

The SDK's DNS-rebinding guard compares the request `Host` header against an
allowlist built in `_security_settings()`: the configured host, `localhost`,
`127.0.0.1`, `[::1]`, this container's hostname, and its resolved IPs. A request
carrying any other `Host` is rejected with `421`. That is why peers must use the
container name or IP exactly as listed above, not a DNS alias you invent.

## Tools

Nine tools. The read surface and the write path are implemented; the three
search tools are still stubs.

| Tool | Purpose |
| :--- | :--- |
| `list_wikis` | Live wikis with document counts, to resolve a wiki name |
| `list_documents` | Documents, optionally scoped to one wiki, to pick an edit target |
| `read_document` | Full document: metadata, every section, references (`sections: null` unless requested) |
| `create_wiki` | Create a wiki before writing its first document |
| `write_document` | Document + progress + log, atomically |
| `log_recall` | Log entries oldest first, each linked to its progress document |
| `semantic_search` | *stub* — vector match over section embeddings (default) |
| `fts_search` | *stub* — exact-token match over heading + body |
| `table_search` | *stub* — wiki → documents → sections as plain rows |

Machine-readable schemas are whatever `tools/list` advertises — the SDK derives
them from the handler type hints, so they are final only when signatures are. No
OpenAPI document is generated: `/mcp` is one JSON-RPC endpoint, and an OpenAPI file
would describe a single POST with a union body while saying nothing about the
tools.

## No authentication yet

`AccessControl` is a no-op middleware slot and no route requires identity. Bind
`0.0.0.0` only on a private podman network, never a public interface.
