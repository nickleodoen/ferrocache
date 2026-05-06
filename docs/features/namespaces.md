# Namespaces & Isolation

FerroCache partitions every entry by **namespace**. Cross-namespace queries are impossible by construction — there is no API that returns entries from a namespace other than the one being queried.

The namespace string is composed from three input fields:

```
effective_namespace = "{model_id}"                  # base
                    | "{model_id}::{cache_scope}"   # scoped
                    | "{model_id}::{cache_scope}::conv_{conversation_id}"  # conversation-scoped
                    | "{model_id}::conv_{conversation_id}"                 # conv without scope
```

## Field-by-field

### `model_id` (required)

Identifies the embedding model. Convention: `name::dimension`, e.g. `all-MiniLM-L6-v2::384` or `text-embedding-3-small::1536`. FerroCache treats it as an opaque string — the format is a discipline, not a parser.

Two embeddings from different models will sit in different namespaces and never compare to each other. This is essential: cosine similarity is only meaningful between vectors from the same embedding space.

### `cache_scope` (optional)

A user-chosen segment for tenant / user / config isolation. Common values:

- Tenant ID: `cache_scope="tenant_abc"`
- User ID: `cache_scope="user_42"`
- LLM config hash: `cache_scope="gpt-4o-mini::temp=0.7::sysv=v3"`
- Composition: `cache_scope=f"{tenant}:{model_temp}"`

The `conv_` prefix is **reserved** — don't use a `cache_scope` that starts with `conv_`. It's how FerroCache distinguishes tenant scopes from conversation scopes (see below). The reservation is documented, not enforced.

### `conversation_id` (optional)

Routes the entry into a per-conversation namespace. The hardcoded prefix `conv_` is added by FerroCache:

```python
client.insert(..., conversation_id="2026_05_01")
# stored under "<model_id>::conv_2026_05_01"
# (or "<model_id>::<scope>::conv_2026_05_01" if cache_scope is also set)
```

Queries with a `conversation_id` do a [two-level lookup](conversations.md): conversation namespace first, then base.

## Why this composition

One `HashMap<String, NamespacedIndex>` powers all three forms of isolation:

- **Model isolation** — different `model_id`, different namespace.
- **Tenant isolation** — different `cache_scope`, different namespace.
- **Conversation scoping** — different `conversation_id`, different namespace (with two-level fallback).

There's no per-tenant table, no per-conversation table, no special code paths. Adding a new dimension of isolation would mean adding another segment to the namespace string.

## Multi-tenant patterns

### Per-tenant cache

```python
client.insert(embedding=emb, response=ans, query_text=q,
              model_id="...", cache_scope="tenant_abc")
client.query(embedding=emb, threshold=0.92, model_id="...",
             cache_scope="tenant_abc")  # hits
client.query(embedding=emb, threshold=0.92, model_id="...",
             cache_scope="tenant_xyz")  # miss — different namespace
```

### Per-(tenant, model-config) cache

```python
import hashlib
def llm_config_hash(model: str, temperature: float, system_prompt: str) -> str:
    h = hashlib.sha256(f"{model}|{temperature}|{system_prompt}".encode()).hexdigest()[:8]
    return h

scope = f"tenant_{tenant_id}:cfg_{llm_config_hash('gpt-4o-mini', 0.7, sys)}"
client.query(embedding=emb, threshold=0.92, model_id="...", cache_scope=scope)
```

## Resource isolation comes for free

`max_entries_per_namespace` applies **per scoped namespace** — a noisy tenant cannot evict a quiet tenant's entries. LRU is bounded by namespace, not globally.

```toml
[hnsw]
max_entries_per_namespace = 10000  # each tenant gets 10k slots
```

## Inspecting namespaces

`/stats` reports every namespace as a first-class entry:

```json
{
  "namespaces": {
    "all-MiniLM-L6-v2::384":              { "entry_count": 500 },
    "all-MiniLM-L6-v2::384::tenant_abc":  { "entry_count": 200 },
    "all-MiniLM-L6-v2::384::tenant_xyz":  { "entry_count": 150 },
    "all-MiniLM-L6-v2::384::tenant_abc::conv_2026_05_01": { "entry_count": 12 }
  }
}
```

## Limits & caveats

- **Cross-namespace queries are impossible.** If you need to search across tenants, run separate queries — there is no fan-out endpoint.
- **`conv_` is reserved** at the start of `cache_scope`. Conversation IDs themselves can be anything (they're prefixed automatically).
- **Empty conversation namespaces are auto-pruned.** When the reaper sweeps a conversation namespace and finds zero entries, the namespace itself is deleted to keep the map clean. Base namespaces survive empty.
- **Namespace strings are case-sensitive.** `tenant_abc` and `Tenant_abc` are different namespaces.
