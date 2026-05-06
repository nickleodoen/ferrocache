# OpenAI

Drop-in wrapper for the OpenAI Python SDK. One line of code adds semantic caching to any existing script.

## Install

```bash
pip install ferrocache[openai]
```

## Usage

```python
from openai import OpenAI
from ferrocache.middleware import wrap_openai

client = wrap_openai(OpenAI())

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
print(resp.choices[0].message.content)
print(resp._ferrocache_hit)         # True on cache hit, False on miss, None on fail-open
print(resp._ferrocache_similarity)  # cosine similarity, only set on a hit
```

## How it works

`wrap_openai` returns a transparent proxy. Attribute access is forwarded to the underlying client unchanged — only `chat.completions.create` is intercepted. The wrapper:

1. Embeds the user message locally (default: `sentence-transformers all-MiniLM-L6-v2`).
2. Calls FerroCache `/query`.
3. On a hit: returns a synthetic `ChatCompletion` with the cached content.
4. On a miss: calls the real OpenAI API, then writes the result back via `/insert`.

If FerroCache is unreachable, the wrapper falls through to the real API call — your app keeps working (`_ferrocache_hit = None`).

## Constructor kwargs

```python
wrap_openai(
    inner,                          # the real OpenAI() client
    cache_url: str = ...,           # default: env FERROCACHE_URL or http://localhost:3000
    threshold: float = 0.92,        # cosine similarity cutoff
    auth_token: str | None = None,  # bearer token; defaults to FERROCACHE_AUTH_TOKEN
    cache_scope: str | None = None,
    conversation_id: str | None = None,
    embed_fn: Callable | None = None,  # custom embedding function
    fail_open: bool = True,         # cache outage falls through to API
)
```

| Argument | Default | Env var |
|---|---|---|
| `cache_url` | `http://localhost:3000` | `FERROCACHE_URL` |
| `threshold` | `0.92` | `FERROCACHE_THRESHOLD` |
| `auth_token` | `None` | `FERROCACHE_AUTH_TOKEN` |
| `cache_scope` | `None` | — |
| `conversation_id` | `None` | — |
| `embed_fn` | `sentence-transformers all-MiniLM-L6-v2` | — |
| `fail_open` | `True` | — |

## Multi-tenant pattern

```python
from openai import OpenAI
from ferrocache.middleware import wrap_openai

# Each tenant gets an isolated cache namespace.
def client_for(tenant_id: str):
    return wrap_openai(OpenAI(), cache_scope=tenant_id)

resp_a = client_for("tenant_abc").chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Summarize last quarter's report."}],
)
resp_b = client_for("tenant_xyz").chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Summarize last quarter's report."}],
)
# resp_a and resp_b never share cache entries — by construction.
```

## Conversation-scoped pattern

```python
client = wrap_openai(OpenAI(), conversation_id="conv_2026_05_01")

# Generic factual answer — would normally come from the global cache.
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is HNSW?"}],
)
# A subsequent context-dependent query in the same conversation_id
# can hit a per-conversation entry without leaking across conversations.
```

See [Conversation Scoping](../features/conversations.md) for the two-level lookup semantics.

## Custom embedding function

```python
from openai import OpenAI
from ferrocache.middleware import wrap_openai

def voyage_embed(text: str) -> list[float]:
    # ... call your embedding API
    return [...]

client = wrap_openai(OpenAI(), embed_fn=voyage_embed)
```

`embed_fn` must return a `list[float]`. The dimension determines the namespace key — make sure your `model_id` (auto-derived as `{embed_model_name}::{dim}` by default) doesn't collide with another embedding's namespace.

## Response fields

On a cache hit, the synthetic `ChatCompletion` carries:

- `_ferrocache_hit = True`
- `_ferrocache_similarity = <float>` — cosine similarity score
- `choices[0].message.content` — cached response text

On a miss:

- `_ferrocache_hit = False`
- All other fields come from the real OpenAI response

On fail-open (cache unreachable):

- `_ferrocache_hit = None`
