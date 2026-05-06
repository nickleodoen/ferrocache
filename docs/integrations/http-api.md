# HTTP API

FerroCache speaks plain HTTP/JSON. Any language with an HTTP client can use it. This is the language-agnostic reference.

## Base URL

`http://<host>:<port>` — default port `3000`. With auth on, every request needs `Authorization: Bearer <token>`. `/health` and `/metrics` stay open.

## Endpoint summary

| Method | Path | Purpose |
|---|---|---|
| POST | `/insert` | Insert an entry |
| POST | `/query` | Lookup |
| DELETE | `/entry/:uuid` | Delete a specific entry (cluster fan-out) |
| GET | `/health` | Health check |
| GET | `/stats` | Counters + per-namespace breakdown |
| GET | `/metrics` | Prometheus text exposition |
| GET | `/cluster/status` | Ring membership, peer phi, dead-node list |
| POST | `/admin/compact` | Trigger WAL compaction + snapshot |
| POST | `/admin/invalidate` | Radius-delete by similarity |
| GET | `/admin/entry-stats` | Top-10 most-accessed entries per namespace |

`POST /insert`, `POST /query`, and `POST /admin/invalidate` accept `?local=true` to skip ring routing — used internally by inter-node forwards and by tests.

## `POST /insert`

Insert a `(embedding, response)` pair.

**Request:**

```json
{
  "embedding": [0.1, 0.2, "..."],
  "response": "the cached answer",
  "query_text": "the prompt",
  "model_id": "all-MiniLM-L6-v2::384",
  "ttl_seconds": 3600,
  "cache_scope": "tenant_abc",
  "conversation_id": "conv_xyz"
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `embedding` | `f32[]` | yes | Any dimension; consistent within a `model_id` |
| `response` | string | yes | Stored verbatim |
| `query_text` | string | yes | Used for the exact-match pre-filter (M27) |
| `model_id` | string | yes | Convention: `name::dimension` |
| `ttl_seconds` | u64 | no | Per-entry TTL; missing = no expiry |
| `cache_scope` | string | no | Tenant/scope key. `conv_` prefix is reserved |
| `conversation_id` | string | no | Routes to a conversation namespace |

**Response (200):**

```json
{ "id": "<uuid>", "status": "ok" }
```

**Errors:** `400` bad input, `502` peer unreachable (cluster mode), `500` local failure.

## `POST /query`

Lookup by embedding.

**Request:**

```json
{
  "embedding": [0.1, 0.2, "..."],
  "threshold": 0.92,
  "model_id": "all-MiniLM-L6-v2::384",
  "query_text": "the prompt",
  "cache_scope": "tenant_abc",
  "conversation_id": "conv_xyz"
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `embedding` | `f32[]` | yes | Same dimension as inserts |
| `threshold` | f32 | yes | Cosine cutoff |
| `model_id` | string | yes | |
| `query_text` | string | no | Triggers exact-match pre-filter |
| `cache_scope` | string | no | |
| `conversation_id` | string | no | Two-level lookup: conversation → global |

**Response (200, hit):**

```json
{
  "hit": true,
  "id": "<uuid>",
  "response": "...",
  "similarity": 0.97,
  "exact_match": false,
  "scope": "conversation"
}
```

`exact_match` is `true` when the M27 pre-filter fired. `scope` is only set on conversation-scoped queries: `"conversation"` or `"global"`.

**Response (200, miss):**

```json
{ "hit": false }
```

## `DELETE /entry/:uuid`

Delete a specific entry. In cluster mode the request fans out to every live peer — 404 from a peer is treated as idempotent success.

**Response (200):**

```json
{ "status": "deleted" }
```

## `POST /admin/invalidate`

Radius-delete by embedding similarity.

**Request:**

```json
{
  "embedding": [0.1, 0.2, "..."],
  "threshold": 0.95,
  "model_id": "all-MiniLM-L6-v2::384",
  "cache_scope": "tenant_abc"
}
```

Each replica computes its own match set against the same `(embedding, threshold)` — no UUID list ships across the wire.

**Response (200):**

```json
{ "deleted_count": 3 }
```

## `GET /health`

```json
{ "status": "ok", "node_id": "...", "entry_count": 42 }
```

Always open (no auth required).

## `GET /stats`

```json
{
  "namespaces": {
    "all-MiniLM-L6-v2::384": { "entry_count": 500, "access_stats": "..." },
    "all-MiniLM-L6-v2::384::tenant_abc": { "entry_count": 200 }
  },
  "counters": {
    "queries": 12345,
    "hits": 9876,
    "misses": 2469
  }
}
```

## `GET /metrics`

Prometheus text exposition. See [Observability](../production/observability.md) for the full metric list.

## `GET /cluster/status`

```json
{
  "node_id": "...",
  "peers": [
    { "node_id": "...", "api_addr": "node2:3000", "phi": 0.4, "status": "Alive" }
  ],
  "dead_nodes": [],
  "ring_size": 192,
  "read_repair_enabled": true
}
```

## `POST /admin/compact`

Trigger WAL compaction + snapshot now (normally runs every `compact_interval_inserts`).

**Response (200):**

```json
{ "status": "ok" }
```

## `GET /admin/entry-stats`

Top-10 most-accessed entries per namespace.

```json
{
  "namespaces": {
    "all-MiniLM-L6-v2::384": [
      { "uuid": "...", "access_count": 142, "last_accessed_at": 1714857600 }
    ]
  }
}
```

## `curl` cookbook

```bash
# Insert
curl -X POST http://localhost:3000/insert \
  -H 'Content-Type: application/json' \
  -d '{"embedding":[1.0,0.0,0.0,0.0],"response":"42","query_text":"meaning of life","model_id":"my-model::4"}'

# Query
curl -X POST http://localhost:3000/query \
  -H 'Content-Type: application/json' \
  -d '{"embedding":[1.0,0.0,0.0,0.0],"threshold":0.9,"model_id":"my-model::4"}'

# Delete by UUID
curl -X DELETE http://localhost:3000/entry/3a7b...

# Radius invalidate
curl -X POST http://localhost:3000/admin/invalidate \
  -H 'Content-Type: application/json' \
  -d '{"embedding":[1.0,0.0,0.0,0.0],"threshold":0.95,"model_id":"my-model::4"}'

# Status
curl http://localhost:3000/health
curl http://localhost:3000/stats
curl http://localhost:3000/cluster/status
curl http://localhost:3000/metrics
```

## Auth header

```bash
TOKEN=$(cat /etc/ferrocache/auth-token)
curl -H "Authorization: Bearer $TOKEN" http://localhost:3000/stats
```
