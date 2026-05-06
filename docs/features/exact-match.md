# Exact-Match Pre-filter

Verbatim repeat queries skip HNSW entirely. An O(1) HashMap lookup checks `query_text` against a per-namespace index; on a hit, it returns in <0.4ms with `similarity: 1.0` and `exact_match: true`.

## How it works

Every `/insert` records a normalised version of `query_text` in a per-namespace `HashMap<normalized_text, uuid>`. Every `/query` with a `query_text` field checks that map **before** running HNSW.

```python
# Insert
client.insert(
    embedding=emb,
    response="Refunds take 7 days.",
    query_text="What is the refund policy?",
    model_id="all-MiniLM-L6-v2::384",
)

# Query — same text, exact-match pre-filter fires
hit = client.query(
    embedding=emb,
    threshold=0.85,
    model_id="all-MiniLM-L6-v2::384",
    query_text="What is the refund policy?",
)
# hit == {"hit": True, "id": "...", "response": "Refunds take 7 days.",
#         "similarity": 1.0, "exact_match": True}
```

Without `query_text`, the pre-filter is skipped — the embedding still matches via HNSW, but at HNSW latency.

## Normalisation rules

The `normalize_query_text` function does exactly:

```rust
text.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase()
```

That's it. No stemming, no Unicode normalisation, no punctuation stripping. Deterministic and dirt-cheap.

What this means in practice:

- `"What is the refund policy?"` and `"  what is the   refund policy?  "` collapse to the same key.
- `"What is the refund policy?"` and `"what is the refund policy"` (no `?`) are **different** keys — punctuation matters.
- `"renforcer"` and `"Renforcer"` are the same key.
- `"café"` and `"cafe"` are **different** keys.

If you need fuzzy normalisation, do it client-side before passing `query_text`.

## When the pre-filter fires

| `query_text` on insert | `query_text` on query | Pre-filter? |
|---|---|---|
| Set | Set | ✅ tries pre-filter; falls back to HNSW on miss |
| Set | Not set | ❌ HNSW only |
| Not set | Set | ❌ pre-filter map has no entry |
| Not set | Not set | ❌ HNSW only |

`query_text` is **required** on insert — there's no way to get an entry that lacks it. So the only failure mode is "query forgot to pass `query_text`."

## Cleanup on delete / evict

When an entry leaves the cache (LRU, TTL, DELETE, invalidate), its `query_text` key is removed from the pre-filter map. The cleanup uses a UUID ownership check:

```rust
if exact_match_index[normalized_text] == evicted_uuid {
    exact_match_index.remove(normalized_text);
}
```

This is what makes delete-then-reinsert safe: if the same `query_text` is reinserted under a different UUID, the old delete won't accidentally remove the new entry's pre-filter key.

## Performance

| | p50 | p95 | p99 |
|---|---|---|---|
| Exact-match pre-filter | 0.38ms | — | — |
| HNSW query (1k entries) | 0.44ms | 0.51ms | 0.54ms |

The numbers are HTTP round-trip; the pre-filter is barely faster than HNSW because the network round-trip dominates. The real win is on a large index (>10k entries), where the HashMap stays O(1) while HNSW's `ef_search` tax grows.

## Response field

Hit via pre-filter:

```json
{
  "hit": true,
  "id": "...",
  "response": "...",
  "similarity": 1.0,
  "exact_match": true
}
```

Hit via HNSW (semantic match):

```json
{
  "hit": true,
  "id": "...",
  "response": "...",
  "similarity": 0.91,
  "exact_match": false
}
```

Use `exact_match` to telemetry-distinguish "byte-for-byte cache hit" from "semantic neighbour cache hit" — useful for tuning `threshold`.
