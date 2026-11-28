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

Nine tools. The read surface, the write path and `semantic_search` are
implemented; `fts_search` and `table_search` remain stubs.

| Tool | Purpose |
| :--- | :--- |
| `list_wikis` | Live wikis with document counts, to resolve a wiki name |
| `list_documents` | Documents, optionally scoped to one wiki, to pick an edit target |
| `read_document` | Full document: metadata, every section, references (`sections: null` unless requested) |
| `create_wiki` | Create a wiki before writing its first document |
| `write_document` | Document + progress + log, atomically |
| `log_recall` | Log entries oldest first, each linked to its progress document |
| `semantic_search` | Vector match over section embeddings (default) — health-gated, see below |
| `fts_search` | *stub* — exact-token match over heading + body |
| `table_search` | *stub* — wiki → documents → sections as plain rows |

Machine-readable schemas are whatever `tools/list` advertises — the SDK derives
them from the handler type hints, so they are final only when signatures are. No
OpenAPI document is generated: `/mcp` is one JSON-RPC endpoint, and an OpenAPI file
would describe a single POST with a union body while saying nothing about the
tools.

## Embedding: deferred, provider-pluggable

`write_document` never talks to a model. Sections land with NULL vectors inside
the write transaction; after COMMIT the ids are enqueued to a Huey queue (its own
`huey.sqlite3`, never the store's file) and a worker process backfills them. A
slow or dead provider therefore cannot hold up or roll back a write; a crash
between COMMIT and enqueue leaves pending sections, which is exactly what the
doctor counts and reconciles.

Provider selection is one flag in `config.toml`:

```toml
[embedding]
openrouter_model = ""              # non-empty → OpenRouter; empty → fastembed
```

- **Non-empty**: embeddings route through OpenRouter's OpenAI-compatible
  endpoint. The model string is sent as-is — if it is wrong, the error is the
  operator's to fix. The token comes from `.env` (beside `config.toml`), then
  `../.env`, then the process environment — never `config.toml`. With the flag
  set and no token anywhere, **the server refuses to boot** rather than silently
  embedding with a different model.
- **Empty**: fastembed downloads the model on first run into `.fastembed_cache`
  (setup step, not a repo artefact).

### Worker, doctor, health gate

```bash
.venv/bin/python -m openmind.worker     # consumer: one blocking HTTP call per batch
.venv/bin/python -m openmind.doctor     # wake report: pending / stale / broken
.venv/bin/python -m openmind.doctor --reconcile   # enqueue every pending section
```

The doctor suggests, never auto-runs, and exits 1 on anything unhealthy.
`semantic_search` runs the same three counts on every call:

- **pending** (mid-embed) → allowed; in-flight sections are simply absent.
- **stale** (vectors from another model — drift) → refuses loudly: reembed,
  clear `openrouter_model`, or don't use semantic search.
- **broken** (all-or-nothing CHECK violated) → refuses loudly; unsafe.

Cosine similarity runs in Python over every live vector — at document-edit
volume brute force beats an index, and the store stays free of extensions.

## No authentication yet

`AccessControl` is a no-op middleware slot and no route requires identity. Bind
`0.0.0.0` only on a private podman network, never a public interface.
