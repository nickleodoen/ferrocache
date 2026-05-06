# Python Client

Zero-dependency stdlib-only client. Uses `urllib` and `json`. Distributed on PyPI as `ferrocache`.

## Install

```bash
pip install ferrocache
```

## `FerrocacheClient`

```python
from ferrocache import FerrocacheClient

client = FerrocacheClient(
    base_url="http://localhost:3000",
    timeout=10.0,                  # optional, seconds
    auth_token=None,               # optional, falls back to FERROCACHE_AUTH_TOKEN env var
)
```

If `auth_token` is `None`, the client reads `FERROCACHE_AUTH_TOKEN` from the environment. An empty string disables auth even when the env var is set.

## Methods

### `insert(...)`

```python
client.insert(
    embedding: list[float],
    response: str,
    query_text: str,
    model_id: str,
    ttl_seconds: int | None = None,
    cache_scope: str | None = None,
    conversation_id: str | None = None,
) -> dict
# {"id": "<uuid>", "status": "ok"}
```

`embedding`, `response`, `query_text`, and `model_id` are required. `ttl_seconds`, `cache_scope`, and `conversation_id` are optional.

### `query(...)`

```python
client.query(
    embedding: list[float],
    threshold: float = 0.92,
    model_id: str = ...,                # required
    query_text: str | None = None,      # enables exact-match pre-filter
    cache_scope: str | None = None,
    conversation_id: str | None = None,
) -> dict
# {"hit": True, "id": "...", "response": "...", "similarity": 0.97,
#  "exact_match": False, "scope": "conversation"}
# {"hit": False}
```

`scope` is only present when `conversation_id` was passed: `"conversation"` or `"global"`.

### `delete_entry(uuid)`

```python
client.delete_entry("3a7b...") -> dict
# {"status": "deleted"}  (also on cluster fan-out 404 — idempotent)
```

### `invalidate(...)`

```python
client.invalidate(
    embedding: list[float],
    threshold: float,
    model_id: str,
    cache_scope: str | None = None,
) -> dict
# {"deleted_count": <int>}
```

Radius-delete: removes every entry with cosine similarity `>= threshold` to `embedding`.

### `health()` / `stats()` / `cluster_status()`

```python
client.health()          # {"status": "ok", "node_id": "...", "entry_count": ...}
client.stats()           # entry counts + per-namespace breakdown + counters
client.cluster_status()  # ring membership, peer phi, dead-node list
```

## Error handling

```python
from ferrocache import FerrocacheClient, FerrocacheError

client = FerrocacheClient("http://localhost:3000")
try:
    hit = client.query(embedding=[...], threshold=0.92, model_id="...")
except FerrocacheError as e:
    print(f"Cache request failed: {e}")
    # Fall back to your LLM call
```

`FerrocacheError` is raised on non-2xx responses or transport errors (connection refused, timeout, etc.).

## Async client?

The current client is synchronous. An async client built on `httpx.AsyncClient` is on the roadmap — see [Contributing](../contributing.md). Today, run the sync client in a thread pool from async code:

```python
import asyncio
from ferrocache import FerrocacheClient

client = FerrocacheClient("http://localhost:3000")

async def query_async(embedding, model_id):
    return await asyncio.to_thread(
        client.query, embedding=embedding, threshold=0.92, model_id=model_id
    )
```

## Full example

```python
from ferrocache import FerrocacheClient
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = FerrocacheClient("http://localhost:3000")

def cached_answer(question: str, expensive_call) -> str:
    emb = model.encode(question).tolist()
    hit = client.query(
        embedding=emb,
        threshold=0.90,
        model_id="all-MiniLM-L6-v2::384",
        query_text=question,
    )
    if hit["hit"]:
        return hit["response"]

    answer = expensive_call(question)
    client.insert(
        embedding=emb,
        response=answer,
        query_text=question,
        model_id="all-MiniLM-L6-v2::384",
    )
    return answer
```
