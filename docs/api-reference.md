# API Reference

Complete HTTP API reference. This is the same content as the [HTTP API integration page](integrations/http-api.md), reorganised as a comprehensive reference with a table of contents.

## Contents

- [Authentication](#authentication)
- [Endpoint summary](#endpoint-summary)
- [`POST /insert`](#post-insert)
- [`POST /query`](#post-query)
- [`DELETE /entry/:uuid`](#delete-entryuuid)
- [`POST /admin/invalidate`](#post-admininvalidate)
- [`POST /admin/compact`](#post-admincompact)
- [`GET /admin/entry-stats`](#get-adminentry-stats)
- [`GET /health`](#get-health)
- [`GET /stats`](#get-stats)
- [`GET /cluster/status`](#get-clusterstatus)
- [`GET /metrics`](#get-metrics)
- [Error codes](#error-codes)

## Authentication

If `FERROCACHE_AUTH_TOKEN` is set on the server, every data-plane request must carry:

```
Authorization: Bearer <token>
```

`/health` and `/metrics` are exempt — load balancers and Prometheus scrape unauthenticated.

## Endpoint summary

| Method | Path | Auth required when on |
|---|---|---|
| POST | `/insert` | yes |
| POST | `/query` | yes |
| DELETE | `/entry/:uuid` | yes |
| POST | `/admin/invalidate` | yes |
| POST | `/admin/compact` | yes |
| GET | `/admin/entry-stats` | yes |
| GET | `/health` | no |
| GET | `/stats` | yes |
| GET | `/cluster/status` | yes |
| GET | `/metrics` | no |

`POST /insert`, `POST /query`, and `POST /admin/invalidate` accept `?local=true` to skip ring routing — used internally by inter-node forwards.

## `POST /insert`

Insert a `(embedding, response)` pair.

**Request body:**

| Field | Type | Required |
|---|---|---|
| `embedding` | `f32[]` | yes |
| `response` | string | yes |
| `query_text` | string | yes |
| `model_id` | string | yes |
| `ttl_seconds` | u64 | no |
| `cache_scope` | string | no (`conv_` prefix reserved) |
| `conversation_id` | string | no |

**Example:**

```bash
curl -X POST http://localhost:3000/insert \
  -H 'Content-Type: application/json' \
  -d '{
    "embedding": [0.1, 0.2, 0.3, 0.4],
    "response": "the cached answer",
    "query_text": "the prompt",
    "model_id": "all-MiniLM-L6-v2::384",
    "ttl_seconds": 3600
  }'
# {"id": "<uuid>", "status": "ok"}
```

## `POST /query`

Lookup by embedding.

**Request body:**

| Field | Type | Required |
|---|---|---|
| `embedding` | `f32[]` | yes |
| `threshold` | f32 | yes |
| `model_id` | string | yes |
| `query_text` | string | no (enables exact-match pre-filter) |
| `cache_scope` | string | no |
| `conversation_id` | string | no |

**Response (hit):**

| Field | Type |
|---|---|
| `hit` | `true` |
| `id` | string (UUID) |
| `response` | string |
| `similarity` | f32 |
| `exact_match` | bool |
| `scope` | `"conversation"` \| `"global"` (only when `conversation_id` was set) |

**Response (miss):**

```json
{ "hit": false }
```

## `DELETE /entry/:uuid`

Delete a specific entry. In cluster mode, fans out to every live peer; 404 from a peer is treated as idempotent success.

```bash
curl -X DELETE http://localhost:3000/entry/3a7b...
# {"status": "deleted"}
```

## `POST /admin/invalidate`

Radius-delete by similarity.

| Field | Type | Required |
|---|---|---|
| `embedding` | `f32[]` | yes |
| `threshold` | f32 | yes |
| `model_id` | string | yes |
| `cache_scope` | string | no |

```bash
curl -X POST http://localhost:3000/admin/invalidate \
  -H 'Content-Type: application/json' \
  -d '{
    "embedding": [0.1, 0.2, 0.3, 0.4],
    "threshold": 0.95,
    "model_id": "all-MiniLM-L6-v2::384"
  }'
# {"deleted_count": 3}
```

## `POST /admin/compact`

Trigger WAL compaction + snapshot now (normally runs every `compact_interval_inserts`).

```bash
curl -X POST http://localhost:3000/admin/compact
# {"status": "ok"}
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

## `GET /health`

```json
{ "status": "ok", "node_id": "...", "entry_count": 42 }
```

Always open.

## `GET /stats`

```json
{
  "namespaces": {
    "all-MiniLM-L6-v2::384": { "entry_count": 500 },
    "all-MiniLM-L6-v2::384::tenant_abc": { "entry_count": 200 }
  },
  "counters": { "queries": 12345, "hits": 9876, "misses": 2469 }
}
```

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

## `GET /metrics`

Prometheus text exposition. See [Observability](production/observability.md) for the full metric list.

## Error codes

| Status | Meaning |
|---|---|
| 200 | OK (also for cache miss — `{"hit": false}`) |
| 400 | Bad input (malformed JSON, missing required field, dimension mismatch) |
| 401 | Missing or invalid `Authorization` header (auth on) |
| 404 | Entry not found (only on `DELETE /entry/:uuid`) |
| 500 | Local failure (WAL write error, panic) |
| 502 | Cluster mode: peer unreachable for forwarded request |
