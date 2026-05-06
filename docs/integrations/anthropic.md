# Anthropic

Drop-in wrapper for the Anthropic Python SDK.

## Install

```bash
pip install ferrocache[anthropic]
```

## Usage

```python
from anthropic import Anthropic
from ferrocache.middleware import wrap_anthropic

client = wrap_anthropic(Anthropic())

resp = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=512,
    messages=[{"role": "user", "content": "Briefly: what is HNSW?"}],
)
print(resp.content[0].text)
print(resp._ferrocache_hit)         # True | False | None (fail-open)
print(resp._ferrocache_similarity)  # cosine similarity, only on hit
```

## How it works

The wrapper returns a transparent proxy. Only `messages.create` is intercepted. Each call:

1. Embeds the user message locally.
2. Looks up FerroCache `/query`.
3. Hit → returns a synthetic `Message` carrying the cached text.
4. Miss → calls Anthropic, writes back to `/insert`.

Fail-open semantics match the OpenAI wrapper: cache outage doesn't break your app.

## Constructor kwargs

```python
wrap_anthropic(
    inner,                          # the real Anthropic() client
    cache_url: str = ...,           # default: env FERROCACHE_URL or http://localhost:3000
    threshold: float = 0.92,
    auth_token: str | None = None,
    cache_scope: str | None = None,
    conversation_id: str | None = None,
    embed_fn: Callable | None = None,
    fail_open: bool = True,
)
```

Same kwargs as `wrap_openai`. See the [OpenAI page](openai.md) for a detailed argument table.

## Multi-tenant pattern

```python
from anthropic import Anthropic
from ferrocache.middleware import wrap_anthropic

client_a = wrap_anthropic(Anthropic(), cache_scope="tenant_abc")
client_b = wrap_anthropic(Anthropic(), cache_scope="tenant_xyz")
```

## Response fields

| Field | Hit | Miss | Fail-open |
|---|---|---|---|
| `_ferrocache_hit` | `True` | `False` | `None` |
| `_ferrocache_similarity` | float | unset | unset |
| `content[0].text` | cached | from Anthropic | from Anthropic |
| `usage` | synthetic zeros | real | real |

## Streaming

The wrapper currently intercepts non-streaming `messages.create` calls. Streaming requests pass through to Anthropic untouched. Streaming cache support is on the roadmap — see [Contributing](../contributing.md).
