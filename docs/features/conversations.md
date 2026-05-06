# Conversation Scoping

`conversation_id` adds a third namespace segment for context-dependent answers. Inserts go to the conversation namespace **only**. Queries do a **two-level lookup** — conversation namespace first, base namespace as fallback.

## Why two-level lookup

Two kinds of cached answer have to coexist:

- **Context-dependent** ("what did we decide?", "summarize what I just said") — must NOT leak across conversations.
- **General factual** ("what is HNSW?", "list the months") — SHOULD be shared.

The application picks which by setting `conversation_id` on insert.

## Insert: one namespace only

```python
# Context-dependent — goes to conversation namespace
client.insert(
    embedding=embed("We decided on Option B."),
    response="Per our discussion, Option B was selected.",
    query_text="we decided on Option B",
    model_id="all-MiniLM-L6-v2::384",
    conversation_id="conv_2026_05_01",
)

# Generic factual — goes to base namespace (no conversation_id)
client.insert(
    embedding=embed("What is HNSW?"),
    response="Hierarchical Navigable Small World — an ANN graph index.",
    query_text="What is HNSW?",
    model_id="all-MiniLM-L6-v2::384",
)
```

There is **no dual-write**. An insert with `conversation_id` does not also populate the base namespace. Promote-to-global is an explicit, application-level operation.

## Query: conversation first, then global

```python
# Within the conversation:
hit = client.query(
    embedding=embed("what did we decide?"),
    threshold=0.85,
    model_id="all-MiniLM-L6-v2::384",
    conversation_id="conv_2026_05_01",
)
# hit["scope"] == "conversation" — context-specific answer found

hit = client.query(
    embedding=embed("HNSW algorithm"),
    threshold=0.85,
    model_id="all-MiniLM-L6-v2::384",
    conversation_id="conv_2026_05_01",
)
# hit["scope"] == "global" — fell back to base namespace; factual answer is shared
```

A query **without** `conversation_id` never sees conversation-namespace entries — base only, and `scope` is omitted from the response.

## Why not unified search

A naïve "search both namespaces and pick the best similarity" would let a global entry beat a conversation-specific entry on similarity score. That's wrong: within a conversation, the conversation entry is the right answer even if a global entry is slightly more similar.

Two-level lookup respects priority: **conversation > global**.

## Auto-TTL

Conversations don't naturally end — they just get abandoned. Without TTL, conversation namespaces would accumulate forever.

```bash
export FERROCACHE_CONVERSATION_TTL_SECONDS=86400   # 24 hours
```

When a conversation-scoped insert has no explicit `ttl_seconds`, FerroCache stamps the auto-TTL on it. Explicit `ttl_seconds` always wins.

The reaper sweeps expired conversation entries every `expire_scan_interval_secs`. When a conversation namespace ends up empty, it's pruned (only namespaces with `::conv_` in the name are pruned — base namespaces survive empty).

## Composing with `cache_scope`

Conversation scoping stacks on top of tenant scoping:

```python
client.insert(
    embedding=emb,
    response=ans,
    query_text=q,
    model_id="all-MiniLM-L6-v2::384",
    cache_scope="tenant_abc",
    conversation_id="2026_05_01",
)
# stored under "all-MiniLM-L6-v2::384::tenant_abc::conv_2026_05_01"
```

Two-level lookup falls back to the **scoped base** namespace, not the global one:

- Conversation namespace: `model_id::tenant_abc::conv_2026_05_01`
- Fallback: `model_id::tenant_abc`
- Never: `model_id` (base, ignoring the scope)

Tenant isolation is preserved across the fallback.

## The `scope` field

Query responses include `scope` only when `conversation_id` was set:

```json
{
  "hit": true,
  "id": "...",
  "response": "...",
  "similarity": 0.91,
  "scope": "conversation"   // or "global" on fallback
}
```

For queries without `conversation_id`, `scope` is absent.

## Reserved prefix: `conv_`

Don't use a `cache_scope` that starts with `conv_`. FerroCache adds the `conv_` prefix to conversation IDs internally — reserving the prefix on the user-facing `cache_scope` field prevents collisions.

The reservation is documented but **not enforced**. If you set `cache_scope="conv_foo"`, you'll get a namespace that looks like a conversation but isn't. Don't.
