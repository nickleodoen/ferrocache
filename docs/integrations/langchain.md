# LangChain

`FerrocacheCache` is a LangChain `BaseCache` implementation backed by FerroCache. Register it once, every LLM call in your LangChain app gets cached.

## Install

```bash
pip install ferrocache[langchain]
```

## Usage

```python
from langchain.globals import set_llm_cache
from ferrocache.langchain import FerrocacheCache

set_llm_cache(FerrocacheCache(cache_scope="tenant_abc"))

# That's it. Use any LangChain LLM as normal:
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")
print(llm.invoke("What is the capital of France?"))
```

## Constructor kwargs

```python
FerrocacheCache(
    cache_url: str = ...,           # default: env FERROCACHE_URL or http://localhost:3000
    threshold: float = 0.92,
    auth_token: str | None = None,
    cache_scope: str | None = None,
    conversation_id: str | None = None,
    embed_fn: Callable | None = None,
    fail_open: bool = True,
)
```

Same kwarg shape as the SDK wrappers — see the [OpenAI page](openai.md) for the full table.

## How lookup / update work

LangChain's `BaseCache` interface is:

- `lookup(prompt, llm_string)` → returns the cached `Generation` list, or `None` for a miss.
- `update(prompt, llm_string, return_val)` → store a fresh result.

`FerrocacheCache` embeds `prompt` (using `embed_fn`), then calls FerroCache `/query` for lookup and `/insert` for update. The `llm_string` (which encodes model, temperature, etc.) is folded into `cache_scope` so different LLM configurations don't share cache entries.

## Async note

`alookup` and `aupdate` are implemented by running the sync client in a thread pool (`asyncio.to_thread`). A native async backend is on the roadmap.

## Limitations

- **`clear()` is a no-op.** LangChain `BaseCache` exposes `clear()` for nuking the cache. FerroCache currently has no "drop everything" endpoint — use `DELETE /entry/:uuid` per entry, or `POST /admin/invalidate` for radius deletes. Or restart the server with the WAL truncated.
- **Streaming responses** are not currently cached (LangChain streams aren't replayable from a single `Generation`).

## Multi-tenant pattern

Per-request cache scope:

```python
from langchain_core.runnables import RunnableConfig
from langchain.globals import set_llm_cache
from ferrocache.langchain import FerrocacheCache

# Set a default cache; per-request overrides come from a contextvar.
set_llm_cache(FerrocacheCache())

# In your request handler:
def handle_request(tenant_id: str, prompt: str):
    # ... rebuild the cache with the tenant scope
    set_llm_cache(FerrocacheCache(cache_scope=tenant_id))
    return llm.invoke(prompt)
```

For request-scoped cache instances in concurrent code, prefer instantiating `FerrocacheCache` per-request rather than mutating the global.
