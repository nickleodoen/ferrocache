# LlamaIndex

`FerrocacheLLM` wraps any LlamaIndex-compatible LLM. Calls go through FerroCache first; misses fall through to the wrapped LLM.

## Install

```bash
pip install ferrocache[llamaindex]
```

## Usage

```python
from llama_index.llms.openai import OpenAI
from ferrocache.llamaindex import FerrocacheLLM

llm = FerrocacheLLM(
    inner=OpenAI(model="gpt-4o-mini"),
    cache_scope="tenant_abc",
)

# Use it anywhere LlamaIndex expects an LLM:
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(llm=llm)
print(query_engine.query("What's the company refund policy?"))
```

## Constructor kwargs

```python
FerrocacheLLM(
    inner: BaseLLM,                 # the wrapped LlamaIndex LLM
    cache_url: str = ...,
    threshold: float = 0.92,
    auth_token: str | None = None,
    cache_scope: str | None = None,
    conversation_id: str | None = None,
    embed_fn: Callable | None = None,
    fail_open: bool = True,
)
```

Same kwargs as the other backends. See the [OpenAI page](openai.md) for argument details.

## Coverage

- ✅ `complete(prompt)` and `chat(messages)` are intercepted.
- ❌ `stream_complete` / `stream_chat` pass through unchanged (streaming is not currently cached).
- ✅ Async variants are supported via thread-pool delegation.

## Query engine integration

Pass `FerrocacheLLM` anywhere a LlamaIndex LLM is accepted — it implements the `LLM` protocol:

```python
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from ferrocache.llamaindex import FerrocacheLLM

Settings.llm = FerrocacheLLM(inner=OpenAI(model="gpt-4o-mini"))
```

## Multi-tenant + conversation pattern

```python
def llm_for(tenant_id: str, conversation_id: str | None = None):
    return FerrocacheLLM(
        inner=OpenAI(model="gpt-4o-mini"),
        cache_scope=tenant_id,
        conversation_id=conversation_id,
    )
```
