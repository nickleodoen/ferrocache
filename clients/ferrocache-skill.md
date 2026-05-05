# ferrocache-skill — LLM agent reference (v1.0)

Reference card for LLM coding agents (Claude Code, etc.) that need to operate ferrocache. Describes the **complete v1.0 HTTP API**, configuration, runtime behaviour, and common operational tasks.

This file is the source of truth for what ferrocache exposes today. If something here disagrees with older docs, this file wins.

---

## What ferrocache is

A distributed semantic cache for LLM applications. Stores `(embedding, response)` pairs and serves them by cosine-similarity nearest-neighbour lookup, so a paraphrased prompt can hit a cached answer instead of re-calling the LLM.

- Single-binary Rust server. Default port `3000`.
- Embedding-model-agnostic: the **client** computes embeddings and sends float vectors. ferrocache never calls an LLM or embedding model.
- Multi-namespace: each `model_id` (composed with optional `cache_scope` and `conversation_id`) is an isolated HNSW index + side-table.
- WAL-durable with group-commit; restart-safe.
- Optional clustering (consistent hashing + chitchat gossip + synchronous replication + read repair).
- Optional bearer-token auth and cluster mTLS.

---

## Endpoints

| Method | Path                  | Description                                                                                       |
|--------|-----------------------|---------------------------------------------------------------------------------------------------|
| POST   | `/insert`             | Insert an entry into the cache.                                                                   |
| POST   | `/query`              | Look up the nearest neighbour above a threshold.                                                  |
| DELETE | `/entry/:uuid`        | Delete a specific entry by UUID; cluster fan-out to all live peers.                               |
| GET    | `/health`             | Liveness + entry count.                                                                           |
| GET    | `/stats`              | Per-namespace breakdown + counters JSON.                                                          |
| GET    | `/metrics`            | Prometheus text exposition.                                                                       |
| GET    | `/cluster/status`     | Cluster membership, peer phi, dead nodes.                                                         |
| POST   | `/admin/compact`      | Trigger WAL compaction + snapshot.                                                                |
| POST   | `/admin/invalidate`   | Radius-delete entries by cosine similarity ≥ threshold.                                           |
| GET    | `/admin/entry-stats`  | Top-10 most-accessed entries per namespace.                                                       |

`POST /insert`, `POST /query`, and `POST /admin/invalidate` accept `?local=true` to skip ring routing — the request is processed only on the receiving node. Forwarded requests use this internally; it's also useful for diagnostics.

`/health` and `/metrics` are auth-exempt (load balancers, Prometheus). Every other route requires `Authorization: Bearer <token>` when `FERROCACHE_AUTH_TOKEN` is set.

---

## Insert

```http
POST /insert
Content-Type: application/json
{
  "embedding":        [0.1, 0.2, ...],     // required, 1..=4096 dims
  "response":         "answer text",       // required, ≤ 102 400 bytes
  "query_text":       "user prompt",       // required for the M27 exact-match index
  "model_id":         "name::dim",         // required (e.g. "all-MiniLM-L6-v2::384")

  "uuid":             "<optional>",        // forwarded inserts only — coordinator-stamped
  "ttl_seconds":      3600,                // optional (M26)
  "cache_scope":      "tenant_abc",        // optional (M28)
  "conversation_id":  "conv_xyz"           // optional (M29)
}

→ 200 { "id": "<uuid>", "status": "ok" }
→ 400 bad input  (dim out of range, missing model_id, etc.)
→ 500 persistence/index failure
→ 502 replica unreachable
```

### Field semantics

- **`model_id`**: required since M14. Convention is `"<model_name>::<dim>"` (e.g. `"all-MiniLM-L6-v2::384"`). Cross-namespace queries are impossible by construction.
- **`ttl_seconds`** (M26): per-entry TTL. `expires_at = inserted_at + ttl_seconds` is stored in WAL/snapshot. Expired entries return miss inline on query and are reaped in the background. Explicit `ttl_seconds` always wins over `conversation_ttl_seconds`.
- **`cache_scope`** (M28): tenant/scope isolation. Effective namespace becomes `"{model_id}::{cache_scope}"`. Empty/whitespace is treated as absent.
- **`conversation_id`** (M29): inserts go to the conversation namespace `"{base}::conv_{conversation_id}"` **only**. The base namespace is *not* dual-written. The application chooses: context-dependent answer → use a `conversation_id`; general factual answer → omit it.
- **`uuid`**: only set by a coordinator forwarding to replicas with `?local=true` so all replicas store the same id. Clients should **not** set this.

### Auto-TTL for conversations

If `FERROCACHE_CONVERSATION_TTL_SECONDS` is set and the insert has a `conversation_id` but no explicit `ttl_seconds`, the server stamps the auto-TTL. Empty conversation namespaces are auto-pruned by the reaper.

---

## Query

```http
POST /query
Content-Type: application/json
{
  "embedding":        [0.1, 0.2, ...],     // required, must match namespace dim
  "threshold":        0.92,                // required, 0.0..=1.0 (cosine similarity)
  "model_id":         "name::dim",         // required

  "query_text":       "user prompt",       // optional — enables M27 exact-match pre-filter
  "cache_scope":      "tenant_abc",        // optional (M28)
  "conversation_id":  "conv_xyz"           // optional (M29) — triggers two-level lookup
}

→ 200 {
  "hit":          true,
  "id":           "<uuid>",
  "response":     "...",
  "similarity":   0.97,
  "exact_match":  false,                   // true if M27 pre-filter fired
  "scope":        "conversation"           // "conversation" | "global" — only when conversation_id is set
}
→ 200 { "hit": false }
→ 400 bad input
→ 502 owning peer unreachable
```

### Exact-match pre-filter (M27)

When `query_text` is present, ferrocache first does an **O(1) HashMap lookup** against the namespace's normalized `query_text → uuid` map. Hit reports `similarity: 1.0`, `exact_match: true`. Miss falls through silently to the HNSW ANN search; ANN hits report `exact_match: false`.

Normalization is `lowercase + trim + whitespace-collapse` only — no stemming, no Unicode normalization. Two queries that differ only in casing/spacing match exactly; small typos go through HNSW.

### Two-level lookup for conversations (M29)

When `conversation_id` is present:

1. **Level 1**: search the conversation namespace `"{base}::conv_{conversation_id}"`. On hit → return with `scope: "conversation"`.
2. **Level 2** (only on level-1 miss): search the base namespace `"{model_id}::{cache_scope?}"`. On hit → return with `scope: "global"`.
3. Both miss → `{"hit": false}`.

Priority: **conversation > global**. Queries **without** `conversation_id` never see conversation entries; this preserves context safety.

`scope` is omitted from the response when `conversation_id` was not provided (backward-compatible).

---

## Delete

```http
DELETE /entry/:uuid
→ 200 { "deleted": true }
→ 404 entry not found  (idempotent — already deleted)
→ 400 empty UUID
```

Cluster fan-out: the receiving node fans out to **all live peers** (not ring-based — there's no embedding to hash from a UUID). Each peer's 404 is treated as success.

A deleted entry's tombstone is durable in the WAL; restart never re-materialises it.

---

## Invalidate (radius delete)

```http
POST /admin/invalidate
Content-Type: application/json
{
  "embedding":   [0.1, 0.2, ...],          // required
  "threshold":   0.95,                     // required
  "model_id":    "name::dim",              // required
  "cache_scope": "tenant_abc"              // optional (M28)
}

→ 200 { "invalidated_count": 3, "uuids": ["...","...","..."] }
→ 400 bad input
```

Each replica computes its own match set against `(embedding, threshold)` — no UUID list shipped on the wire. The contract is "compute, not copy"; assuming replicas are in sync, they delete the same entries.

`invalidated_count` reports **local** evictions on the receiving node. Cluster-wide impact is observable via `ferrocache_invalidations_total` per node.

---

## Stats and metrics

### `GET /stats`

```json
{
  "entry_count": 850,
  "wal_path": "./ferrocache.wal",
  "hnsw": { "max_nb_connection": 16, "ef_construction": 200, "ef_search": 32, "dimension": 384 },
  "namespaces": {
    "all-MiniLM-L6-v2::384": {
      "entry_count": 500, "dimension": 384,
      "oldest_entry_ts": 1714000000, "newest_entry_ts": 1714003600,
      "total_accesses": 1234, "evicted_ghost_count": 12
    },
    "all-MiniLM-L6-v2::384::tenant_abc": { "...": "..." },
    "all-MiniLM-L6-v2::384::conv_xyz":   { "...": "..." }
  },
  "counters": {
    "queries_total": 9001, "queries_hit": 8500, "queries_miss": 501,
    "hit_rate": 0.944,
    "inserts_total": 850,
    "replication_forwards": 0, "replication_failures": 0, "replication_retries": 0,
    "compactions": 0,
    "read_repairs": 0, "read_repair_failures": 0,
    "evictions_total": 0, "index_rebuilds": 0,
    "expirations_total": 0, "deletions_total": 0, "invalidations_total": 0,
    "exact_match_hits_total": 1421
  }
}
```

### `GET /metrics`

Prometheus text exposition. Top-level counters are all `_total`-suffixed; per-namespace metrics carry a `namespace=` label. Notable series:

- `ferrocache_queries_total{namespace=...}`, `ferrocache_queries_hit{namespace=...}`, `ferrocache_queries_miss{namespace=...}`
- `ferrocache_inserts_total{namespace=...}`
- `ferrocache_evictions_total`, `ferrocache_expirations_total`, `ferrocache_deletions_total`, `ferrocache_invalidations_total` (all top-level + per-namespace)
- `ferrocache_exact_match_hits_total` (M27 — additive subset of `ferrocache_queries_hit`)
- `ferrocache_index_rebuilds_total` (M25)
- `ferrocache_replication_forwards`, `ferrocache_replication_failures`, `ferrocache_replication_retries`
- `ferrocache_read_repairs_total`, `ferrocache_read_repair_failures_total`
- `ferrocache_query_latency_seconds_bucket{le="..."}`, `ferrocache_insert_latency_seconds_bucket{le="..."}` (16 fixed buckets, 100µs–10s)
- `ferrocache_peer_phi{peer=...}`, `ferrocache_peers_suspected`, `ferrocache_peers_dead`
- `ferrocache_ring_changes_total`, `ferrocache_ring_members`

### `GET /admin/entry-stats`

```json
{
  "namespaces": {
    "all-MiniLM-L6-v2::384": {
      "top_entries": [
        { "uuid": "...", "access_count": 47, "last_accessed_at": 1714003500, "query_text_preview": "What is HNSW?" }
      ]
    }
  }
}
```

Top-10 per namespace, sorted by `access_count` desc. O(n log n) per namespace under the index read lock; admin-tier endpoint.

### `GET /cluster/status`

```json
{
  "mode": "cluster",
  "self_node_id": "<uuid>",
  "gossip_addr": "0.0.0.0:4000",
  "nodes": ["node-1", "node-2", "node-3"],
  "node_count": 3,
  "peer_health": {
    "node-2": { "status": "Alive",     "phi": 0.7 },
    "node-3": { "status": "Suspected", "phi": 9.1 }
  },
  "dead_nodes": [],
  "read_repair_enabled": true
}
```

`peer_health.status` is `Alive | Suspected | Dead` from the phi-accrual detector. `dead_nodes` lists peers the reconciler has removed from the ring.

---

## Cache lifecycle (eviction and expiry)

Entries die through four paths; **all of them write durable WAL tombstones** so a restart never re-materialises them.

| Path                              | Trigger                                                                       |
|-----------------------------------|-------------------------------------------------------------------------------|
| **LRU eviction (capacity)**       | `hnsw.max_entries_per_namespace`. Flush task evicts LRU after each batch.     |
| **TTL expiry (age)**              | `ttl_seconds` per entry, or `conversation_ttl_seconds` for conv-scoped inserts. Reaper sweeps every `expire_scan_interval_secs`. |
| **Explicit deletion**             | `DELETE /entry/:uuid`.                                                        |
| **Semantic invalidation (radius)**| `POST /admin/invalidate`.                                                     |

HNSW has no deletion API — removed entries become **ghosts** until rebuild. The query path filters ghost ids before applying the threshold; namespaces rebuild when the ghost ratio crosses 20%.

The reaper sequence is `collect_expired → rebuild_dirty_namespaces → prune_empty_namespaces`. Conversation namespaces (key contains `::conv_`) whose entries map and ghost set are both empty are dropped from the namespace map to free memory.

### LRU tie-break

Eviction order is `(last_accessed_at ASC, inserted_at ASC, internal_id ASC)`. This gives deterministic FIFO-on-tie when a sub-second insert burst produces identical timestamps.

### Access tracking fields

- **`inserted_at`** — Unix seconds, set once on insert, **persisted in WAL**. Survives restart.
- **`last_accessed_at`** — bumped on every query hit, **persisted only in snapshots** (in-memory soft state). Crash loses the last interval since the prior snapshot.
- **`access_count`** — same persistence as `last_accessed_at`.

A query hit takes a brief index write lock to bump `last_accessed_at` and `access_count`. Misses skip this entirely.

---

## Tenant isolation (`cache_scope`)

`cache_scope` is an opaque user-defined string. It composes with `model_id`:

```
effective_namespace(model_id, cache_scope)
  = "{model_id}::{cache_scope}"   if cache_scope is non-empty after trim
  = "{model_id}"                  otherwise
```

Empty or whitespace-only `cache_scope` is treated as absent (defensive against `cache_scope: ""`).

Common scope values: tenant ID, user ID, model temperature, system prompt version, or any combination thereof.

`max_entries_per_namespace` applies **per scoped namespace** — each tenant gets its own cap. Resource isolation comes free with the namespace partitioning.

---

## Conversation scoping (`conversation_id`)

`conversation_id` adds a third namespace segment with a hardcoded `conv_` prefix:

```
conversation_namespace(model_id, cache_scope, conversation_id)
  = "{effective_namespace(model_id, cache_scope)}::conv_{conversation_id}"
```

The `conv_` prefix prevents collision between conversation IDs and cache scopes — `cache_scope="abc"` is `"m::abc"`, but `conversation_id="abc"` is `"m::conv_abc"`. **Reserved**: applications must avoid `cache_scope` values starting with `conv_`.

Inserts with `conversation_id` go to the conversation namespace **only**. Queries with `conversation_id` do a **two-level lookup** — conversation namespace first, base namespace fallback. Queries without `conversation_id` never see conversation entries.

---

## Configuration reference

### Top-level

| Env var                                 | Default              | Notes                                                  |
|-----------------------------------------|----------------------|--------------------------------------------------------|
| `FERROCACHE_PORT`                       | `3000`               | HTTP listen port                                       |
| `FERROCACHE_NODE_ID`                    | random UUID          | Stable node identity for the ring                      |
| `FERROCACHE_WAL_PATH`                   | `./ferrocache.wal`   | WAL file; mount a persistent volume for prod           |
| `FERROCACHE_AUTH_TOKEN`                 | unset (auth off)     | Set to a 32-byte+ random hex string                    |
| `FERROCACHE_WAL_BATCH_SIZE`             | `256`                | Group-commit batch size (1 = pre-M20 per-insert fsync) |
| `FERROCACHE_WAL_BATCH_TIMEOUT_MS`       | `1`                  | Max wait for additional inserts to join a batch        |
| `FERROCACHE_COMPACT_INTERVAL_INSERTS`   | `10000`              | Auto-compact every N inserts; `0` disables auto        |
| `FERROCACHE_EXPIRE_SCAN_INTERVAL_SECS`  | `60`                 | TTL reaper interval; `0` disables                      |
| `FERROCACHE_CONVERSATION_TTL_SECONDS`   | unset (no auto-TTL)  | Auto-TTL applied to conversation-scoped inserts        |

### HNSW

| Env var                                            | Default         | Notes                                       |
|----------------------------------------------------|-----------------|---------------------------------------------|
| `FERROCACHE_HNSW__MAX_NB_CONNECTION`               | `16`            | M parameter (graph connectivity)            |
| `FERROCACHE_HNSW__MAX_ELEMENTS`                    | `100000`        | Pre-allocated capacity per index            |
| `FERROCACHE_HNSW__EF_CONSTRUCTION`                 | `200`           | Build-time search depth                     |
| `FERROCACHE_HNSW__EF_SEARCH`                       | `32`            | Query-time search depth                     |
| `FERROCACHE_HNSW__DEFAULT_THRESHOLD`               | `0.92`          | Default cosine similarity threshold         |
| `FERROCACHE_HNSW__MAX_ENTRIES_PER_NAMESPACE`       | unset           | LRU cap per namespace; unset = unlimited    |

### Cluster

| Env var                                                | Default          | Notes                                                |
|--------------------------------------------------------|------------------|------------------------------------------------------|
| `FERROCACHE_CLUSTER__ENABLED`                          | `false`          | Single-node mode when false                          |
| `FERROCACHE_CLUSTER__GOSSIP_ADDR`                      | `0.0.0.0:4000`   | chitchat UDP bind                                    |
| `FERROCACHE_CLUSTER__API_ADDR`                         | `0.0.0.0:3000`   | Advertised API address                               |
| `FERROCACHE_CLUSTER__SEED_NODES`                       | `[]`             | Comma-separated `host:port` list                     |
| `FERROCACHE_CLUSTER__VIRTUAL_NODES`                    | `64`             | Virtual nodes per physical node                      |
| `FERROCACHE_CLUSTER__REPLICATION_FACTOR`               | `2`              | Replicas per key                                     |
| `FERROCACHE_CLUSTER__MAX_REPLICATION_RETRIES`          | `3`              | Retries on connect/timeout/5xx (never 4xx)           |
| `FERROCACHE_CLUSTER__PHI_THRESHOLD`                    | `8.0`            | `Suspected` at φ ≥ this; `Dead` at 2× this           |
| `FERROCACHE_CLUSTER__DEAD_NODE_REMOVAL_ENABLED`        | `true`           | Set false for monitoring-only mode                   |
| `FERROCACHE_CLUSTER__READ_REPAIR_ENABLED`              | `true`           | Replica fan-out on coordinator miss                  |
| `FERROCACHE_CLUSTER__TLS__ENABLED`                     | `false`          | Inter-node mTLS                                      |
| `FERROCACHE_CLUSTER__TLS__CA_CERT_PATH`                | unset            | Required when TLS enabled in production              |
| `FERROCACHE_CLUSTER__TLS__NODE_CERT_PATH`              | unset            | Required when TLS enabled in production              |
| `FERROCACHE_CLUSTER__TLS__NODE_KEY_PATH`               | unset            | Required when TLS enabled in production              |
| `FERROCACHE_CLUSTER__TLS__INTERNAL_PORT`               | `port + 1000`    | Second listener for cluster traffic                  |

---

## Production checklist

1. **Authentication**: set `FERROCACHE_AUTH_TOKEN` to a 32-byte hex string. `/health` and `/metrics` stay open; everything else requires `Authorization: Bearer <token>`.
2. **Memory cap**: set `FERROCACHE_HNSW__MAX_ENTRIES_PER_NAMESPACE` to prevent unbounded growth. LRU eviction kicks in automatically.
3. **Conversation TTL**: if you use `conversation_id`, set `FERROCACHE_CONVERSATION_TTL_SECONDS`. Without it, dead conversation namespaces accumulate forever.
4. **Reaper interval**: `FERROCACHE_EXPIRE_SCAN_INTERVAL_SECS=60` is fine for most workloads. Lower for tight TTL deployments (cost: more lock contention); `0` disables the reaper (inline expiry check on query still runs).
5. **Cluster**: enable mTLS (`FERROCACHE_CLUSTER__TLS__ENABLED=true` + cert paths) for any deployment whose nodes don't share a private network.
6. **Persistence**: mount a volume at `wal_path`. The WAL + snapshot together restore full state on restart.
7. **Monitoring**: scrape `/metrics`. Critical alerts:
   - `ferrocache_replication_failures_total` increasing → peer connectivity issue.
   - `ferrocache_peers_dead > 0` → ring is degraded.
   - `ferrocache_evictions_total` rate climbing → consider raising the cap or reviewing access patterns.
   - `ferrocache_expirations_total` rate aligning with insert rate → TTL is working as configured.
   - `ferrocache_query_latency_seconds_bucket` p99 spike → likely an index rebuild under write lock; tune `max_entries_per_namespace`.

---

## Common operations

### Insert + query (Python)

```python
import os
from ferrocache import FerrocacheClient

c = FerrocacheClient("http://localhost:3000",
                     auth_token=os.environ.get("FERROCACHE_AUTH_TOKEN"))

c.insert(
    embedding=embed("What is HNSW?"),
    response="Hierarchical Navigable Small World — a graph-based ANN index.",
    query_text="What is HNSW?",
    model_id="all-MiniLM-L6-v2::384",
    ttl_seconds=86400,
)

hit = c.query(
    embedding=embed("Tell me about HNSW"),
    threshold=0.85,
    model_id="all-MiniLM-L6-v2::384",
    query_text="Tell me about HNSW",     # enables exact-match pre-filter
)
if hit["hit"]:
    print(hit["response"], "similarity:", hit["similarity"])
```

### Delete an entry

```python
c.delete_entry("<uuid>")  # 200 deleted, or 404 idempotent
```

### Semantic invalidation

```python
# Drop every entry in tenant_abc whose cosine similarity to v >= 0.95.
c.invalidate(
    embedding=v,
    threshold=0.95,
    model_id="all-MiniLM-L6-v2::384",
    cache_scope="tenant_abc",
)
```

### OpenAI middleware

```python
from openai import OpenAI
from ferrocache.middleware import wrap_openai

# All chat.completions.create calls now check ferrocache first.
oai = wrap_openai(OpenAI(),
                  cache_scope="tenant_abc",
                  conversation_id="conv_xyz")

resp = oai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's our Q3 plan?"}],
)
print(resp._ferrocache_hit, resp.choices[0].message.content)
```

---

## Troubleshooting

### Entries returning after restart

- Check the WAL has tombstone entries for the deleted UUIDs (look for `"tombstone":true` in the NDJSON file).
- Pre-M30 versions had a latent bug where the runtime tombstone path through the WAL channel re-materialised phantom entries on flush. Fixed in M29 — `replay_entry` now branches on `tombstone`. If you see this on v1.0+, file a bug.
- Verify `FERROCACHE_WAL_PATH` is consistent across restarts.

### High eviction rate (`ferrocache_evictions_total` climbing fast)

- Increase `FERROCACHE_HNSW__MAX_ENTRIES_PER_NAMESPACE`.
- Investigate access patterns via `/admin/entry-stats` — top entries should be high-traffic; if everything has access_count=1, the LRU is just churning through cold inserts.
- Consider sharding via `cache_scope` so noisy workloads don't evict quiet ones.

### Conversation namespace memory growth

- Set `FERROCACHE_CONVERSATION_TTL_SECONDS`. Without it, every `conversation_id` accumulates forever.
- Check `/stats` for namespaces matching `*::conv_*`. The reaper auto-prunes empty conv namespaces; if dead conversations linger, the reaper interval might be too long for your insert rate.

### Exact-match pre-filter not firing

- Confirm the client is sending `query_text` on `/query` (not just `/insert`).
- The match is normalized: lowercase + trim + whitespace-collapse. Two queries with the same content but different unicode (NFC vs NFD, smart quotes vs ASCII) will NOT match — pre-normalize on the client if you need it.
- The pre-filter is logged at `INFO` with `exact_match=true` on hit. Check the `ferrocache_exact_match_hits_total` counter.

### `502 upstream replica unavailable`

- Replica forward failed after `cluster.max_replication_retries` attempts.
- Check `/cluster/status.peer_health` — the failing peer's `phi` value indicates connectivity. If `Dead`, the reconciler will remove it from the ring within ~2s; subsequent inserts degrade gracefully (warn log naming `effective_replicas`).
- For chronic failures: check the gossip seed list, firewall rules on the gossip UDP port, and (if mTLS is on) certificate expiry.

### `400 dimension mismatch` on insert

- Each namespace locks to the dimension of its first inserted vector. Subsequent inserts with a different dim are rejected.
- Fix: check the embedding model is consistent across callers, or send to a different `model_id` namespace.

### Tenant data leaking across `cache_scope`

- Should be impossible by construction; namespaces are isolated HashMap entries. If you see it: confirm `cache_scope` is being sent on **both** insert and query (omitting it routes to the unscoped base namespace, which is a separate scope).
- The unscoped namespace IS a distinct scope. Inserting without scope and querying with scope is a miss; this is correct behaviour.

### Conversation answers leaking to non-conversation queries

- Should be impossible. Queries without `conversation_id` never search conversation namespaces.
- If you see it: verify the application code path correctly threads `conversation_id` through the cache call. The two-level lookup is opt-in via `conversation_id` on the query.

---

## Quickstart for a new agent

```bash
# 1. Start a single-node ferrocache.
cargo run --release          # or: docker run -p 3000:3000 ghcr.io/.../ferrocache

# 2. Smoke test.
curl localhost:3000/health
# → {"status":"ok","node_id":"...","entry_count":0}

# 3. Insert + query.
curl -s -X POST localhost:3000/insert -H 'Content-Type: application/json' \
  -d '{"embedding":[1,0,0,0],"response":"42","query_text":"meaning","model_id":"demo::4"}'
curl -s -X POST localhost:3000/query -H 'Content-Type: application/json' \
  -d '{"embedding":[1,0,0,0],"threshold":0.9,"model_id":"demo::4"}'
# → {"hit":true,"id":"...","response":"42","similarity":1.0,"exact_match":false}

# 4. Inspect.
curl localhost:3000/stats | jq '.namespaces'
curl localhost:3000/metrics | grep ferrocache_queries_total
```

For cluster mode, scope/conversation features, and security: see the README and the per-mission decision log in `Claude.md`.
