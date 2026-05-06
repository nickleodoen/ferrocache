# Your First Cache

This walkthrough takes you from "FerroCache is running" to "I just saw a cache hit." About 5 minutes.

## 1. Start FerroCache

```bash
docker run -p 3000:3000 ghcr.io/nickleodoen/ferrocache:latest
```

Verify:

```bash
curl http://localhost:3000/health
# {"status":"ok","node_id":"...","entry_count":0}
```

## 2. Compute an embedding

FerroCache is embedding-agnostic — it stores and compares the float vectors you give it. Use any embedding model that emits a fixed-dimension float array. For this walkthrough we'll use `sentence-transformers`.

```bash
pip install sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim
embedding = model.encode("What is the company refund policy?").tolist()
```

The convention for `model_id` is `name::dimension` — here `all-MiniLM-L6-v2::384`. FerroCache treats it as an opaque string but the format makes namespace browsing easier.

## 3. Insert an entry

Using the Python client:

```python
from ferrocache import FerrocacheClient

client = FerrocacheClient("http://localhost:3000")
result = client.insert(
    embedding=embedding,
    response="Refunds are processed within 7 business days of approval.",
    query_text="What is the company refund policy?",
    model_id="all-MiniLM-L6-v2::384",
)
print(result)
# {"id": "<uuid>", "status": "ok"}
```

Using `curl`:

```bash
curl -X POST http://localhost:3000/insert \
  -H 'Content-Type: application/json' \
  -d '{
    "embedding": [/* 384 floats */],
    "response": "Refunds are processed within 7 business days of approval.",
    "query_text": "What is the company refund policy?",
    "model_id": "all-MiniLM-L6-v2::384"
  }'
```

## 4. Query with a paraphrase

```python
paraphrase = model.encode("How long do refunds take?").tolist()

hit = client.query(
    embedding=paraphrase,
    threshold=0.85,
    model_id="all-MiniLM-L6-v2::384",
)
print(hit)
# {"hit": True, "id": "...", "response": "Refunds are processed within 7 business days of approval.",
#  "similarity": 0.91, "exact_match": False}
```

The query was different text ("How long do refunds take?") but semantically close — FerroCache returned the cached answer instead of forcing you to call your LLM.

Using `curl`:

```bash
curl -X POST http://localhost:3000/query \
  -H 'Content-Type: application/json' \
  -d '{
    "embedding": [/* 384 floats */],
    "threshold": 0.85,
    "model_id": "all-MiniLM-L6-v2::384"
  }'
```

## 5. Watch for the exact-match pre-filter

If you query the *exact* original text, FerroCache short-circuits HNSW with an O(1) HashMap lookup:

```python
hit = client.query(
    embedding=embedding,
    threshold=0.85,
    model_id="all-MiniLM-L6-v2::384",
    query_text="What is the company refund policy?",   # enables pre-filter
)
print(hit["exact_match"], hit["similarity"])
# True 1.0
```

Without `query_text`, the pre-filter is skipped (the embedding still matches via HNSW).

## 6. Check `/stats`

```bash
curl http://localhost:3000/stats | python3 -m json.tool
```

You'll see your namespace listed with its entry count and access counters.

## Next steps

- [Python Client](../integrations/python.md) — full method reference.
- [OpenAI wrapper](../integrations/openai.md) — one-line drop-in replacement.
- [Tenant isolation](../features/namespaces.md) — `cache_scope` for multi-tenant SaaS.
- [Conversation scoping](../features/conversations.md) — context-dependent answers.
