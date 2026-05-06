# Eviction & TTL

Entries can leave the cache through four distinct paths. All of them write durable WAL tombstones, so a restarted node never re-materialises a removed entry.

| Path | Trigger |
|---|---|
| **LRU eviction (capacity)** | `max_entries_per_namespace` cap; flush task evicts after each insert batch |
| **TTL expiry (age)** | Per-entry `ttl_seconds`; reaper sweeps every `expire_scan_interval_secs` |
| **Explicit deletion (UUID)** | `DELETE /entry/:uuid`; cluster fan-out to all live peers |
| **Semantic invalidation (radius)** | `POST /admin/invalidate {embedding, threshold, ...}` |

## LRU eviction

Set `max_entries_per_namespace` to cap a namespace's size:

```bash
export FERROCACHE_HNSW__MAX_ENTRIES_PER_NAMESPACE=10000
```

After each insert batch, the flush task checks each touched namespace. If it's over the cap, the least-recently-accessed entries are evicted until the namespace fits.

**Tie-break order:** `(last_accessed_at, inserted_at, internal_id)` — deterministic FIFO when access timestamps are identical.

**Per-namespace, not global.** Each scoped namespace (tenant, conversation) gets its own cap — a noisy tenant cannot evict a quiet tenant's entries.

### Access metadata is soft state

`last_accessed_at` and `access_count` are **not** WAL-durable. They live in memory only and rebuild from traffic after a restart. Writing them to the WAL on every hit would defeat the cache (one fsync per read).

### HNSW lazy deletion (ghosts)

HNSW has no native deletion API. When you evict an entry:

1. Its UUID is removed from the side-table.
2. Its internal HNSW id goes into a per-namespace `evicted_ids: HashSet<usize>` (a "ghost" set).
3. A tombstone is written to the WAL.

Queries oversample by `1 + min(ghosts, 8)` and filter ghost IDs out before applying the threshold.

When `ghost_ratio = ghosts / total > 0.20`, the namespace is rebuilt from scratch on the next flush. Rebuild is synchronous and runs under the index write lock — see the benchmark page for the throughput trade-off.

## TTL expiry

Set `ttl_seconds` on insert:

```python
client.insert(
    embedding=emb,
    response=ans,
    query_text=q,
    model_id="...",
    ttl_seconds=3600,   # 1 hour
)
```

**Inline expiry on read.** The query path checks `expires_at` and treats expired entries as misses without mutating state. The actual cleanup happens in the reaper.

**Reaper loop.** Every `expire_scan_interval_secs` (default 60s):

1. `collect_expired` — scan every namespace for entries past `expires_at`.
2. Write tombstones to the WAL (one per expired entry).
3. After the WAL flushes, `replay_entry` removes the entries from the in-memory index.
4. `rebuild_dirty_namespaces` rebuilds any namespace whose ghost ratio crossed 20%.
5. `prune_empty_namespaces` drops conversation namespaces (those with `::conv_` in the name) that are now empty. Base namespaces survive empty.

### Auto-TTL for conversations

```bash
export FERROCACHE_CONVERSATION_TTL_SECONDS=86400  # 24 hours
```

When a conversation-scoped insert (`conversation_id=...`) has no explicit `ttl_seconds`, FerroCache stamps the auto-TTL on it. Explicit `ttl_seconds` always wins.

Without auto-TTL, abandoned conversation namespaces would linger forever. Set this for any conversation-scoping deployment.

## Explicit deletion

```bash
curl -X DELETE http://localhost:3000/entry/3a7b...
```

In cluster mode, the request fans out to every live peer. 404 from a peer is treated as idempotent success — if the entry never existed there, deleting it is a no-op.

The reverse `uuid → namespace` map handles namespace resolution server-side, so you don't need to pass `model_id` or `cache_scope`.

## Semantic invalidation (radius delete)

Delete every entry within a similarity radius of a target embedding:

```bash
curl -X POST http://localhost:3000/admin/invalidate \
  -H 'Content-Type: application/json' \
  -d '{
    "embedding": [/* ... */],
    "threshold": 0.95,
    "model_id": "all-MiniLM-L6-v2::384",
    "cache_scope": "tenant_abc"
  }'
# {"deleted_count": 7}
```

Use cases:

- A factual answer changed — invalidate everything semantically close to "what is the refund policy?".
- A document was retracted — invalidate everything semantically close to its embedding.

**"Compute, not copy."** In cluster mode, each replica computes its own match set against the same `(embedding, threshold)`. No UUID list ships across the wire — replicas might temporarily diverge on which entries match (e.g., due to ghost-set drift), but the next read repair healing pass fixes that.

## Durability guarantee

Every removal path writes a WAL tombstone before mutating the index. On restart:

- `replay_entry` branches on `entry.tombstone`.
- Tombstones call `remove_by_uuid` instead of inserting.
- Phantom entries can never come back — even if the original insert and its tombstone are both in the WAL, replay processes them in order and ends with the correct state.

This is the M29 fix: previously the runtime tombstone path (reaper → WAL → flush → `replay_entry`) was re-materialising phantom entries because `replay_entry` had no tombstone branch. Now the startup and runtime paths share one code path.

## Observability

| Metric | Meaning |
|---|---|
| `ferrocache_evictions_total` | LRU evictions |
| `ferrocache_expirations_total` | TTL expiries |
| `ferrocache_deletions_total` | Explicit DELETE fan-outs |
| `ferrocache_invalidations_total` | Radius-invalidate counts |
| `ferrocache_index_rebuilds_total` | HNSW rebuilds (ghost ratio threshold) |

All five exist top-level and per-namespace.
