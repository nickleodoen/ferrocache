# ferrocache — Skill File

## What ferrocache is

ferrocache is a distributed semantic cache for LLM applications, written in
Rust. It intercepts a query *before* it reaches an expensive LLM/API call,
returns the cached response when an embedding-similarity search finds a hit
above a configurable threshold, and otherwise forwards the call and stores
the result for next time. It runs as a single binary (or a multi-node cluster
via consistent hashing + gossip), keeps a write-ahead log + periodic
snapshots for durability, and ships HTTP, Python, MCP, LangChain, and
LlamaIndex clients.

## Running ferrocache

### Single node (binary)

```bash
cargo run --release
# or
docker run -p 3000:3000 ghcr.io/nickleodoen/ferrocache:latest
```

The server listens on `:3000` plain HTTP. No auth, no TLS by default —
appropriate for local development only.

### 3-node cluster

```bash
docker compose up -d --build  # 3 nodes on :3001, :3002, :3003 (host)
```

Each node listens on `:3000` inside its container; the host ports are 3001/3002/3003.
Ring + gossip + replication are wired by default. Cluster status:
`curl http://localhost:3001/cluster/status`.

### With bearer token auth

```bash
export FERROCACHE_AUTH_TOKEN="$(openssl rand -hex 32)"
cargo run --release
```

All `/query`, `/insert`, `/stats`, `/cluster/status`, `/admin/compact` calls
require `Authorization: Bearer <token>`. `/health` and `/metrics` remain open.

### With cluster mTLS

```bash
cargo run --bin gen_certs node1 node2 node3
# Distribute ca.pem + each node's cert/key to each container, then:
export FERROCACHE_CLUSTER__TLS__ENABLED=true
export FERROCACHE_CLUSTER__TLS__CA_CERT_PATH=/certs/ca.pem
export FERROCACHE_CLUSTER__TLS__NODE_CERT_PATH=/certs/node1/cert.pem
export FERROCACHE_CLUSTER__TLS__NODE_KEY_PATH=/certs/node1/key.pem
export FERROCACHE_CLUSTER__TLS__INTERNAL_PORT=4443
```

A second listener binds on `internal_port` (default `port + 1000`) and
requires a client cert chained to the cluster CA. The public port stays
plain HTTP. See `docs/security.md` for the deployment recipe.

## API Reference

All bodies are JSON. Routes that mutate or read cached data require
`Authorization: Bearer <token>` when `FERROCACHE_AUTH_TOKEN` is set.

### POST /insert

Store a (query, embedding, response) tuple in the namespace identified by `model_id`.

Request:
```json
{
  "embedding": [0.1, 0.2, ...],
  "response": "the answer to cache",
  "query_text": "the original question",
  "model_id": "all-MiniLM-L6-v2::384"
}
```

Response (200):
```json
{ "id": "uuid-v4-string", "status": "ok" }
```

Errors: `400` for missing `model_id`, empty/oversized embedding (>4096 dims), oversized response (>100 KB), or dimension mismatch within the namespace. `401` if auth is enabled and the token is wrong/missing. `502` if a peer fails to replicate (after retries are exhausted).

```bash
curl -X POST http://localhost:3000/insert \
  -H 'Authorization: Bearer my-token' \
  -H 'Content-Type: application/json' \
  -d '{"embedding":[0.1,0.2,0.3],"response":"hi","query_text":"q","model_id":"m::3"}'
```

### POST /query

Look up the nearest entry in the namespace; return it if cosine similarity ≥ `threshold`.

Request:
```json
{
  "embedding": [0.1, 0.2, ...],
  "threshold": 0.92,
  "model_id": "all-MiniLM-L6-v2::384"
}
```

Hit response (200):
```json
{ "hit": true, "id": "uuid", "response": "the cached answer", "similarity": 0.97 }
```

Miss response (200):
```json
{ "hit": false }
```

Errors: same set as `/insert`, plus `400` if `threshold` is outside `[0.0, 1.0]`.

```bash
curl -X POST http://localhost:3000/query \
  -H 'Authorization: Bearer my-token' \
  -H 'Content-Type: application/json' \
  -d '{"embedding":[0.1,0.2,0.3],"threshold":0.9,"model_id":"m::3"}'
```

### GET /health

Always unauthenticated.

```json
{ "status": "ok", "node_id": "n1", "entry_count": 42 }
```

### GET /stats

Returns entry counts, HNSW config, per-namespace stats, and the same counters surfaced via `/metrics`.

```json
{
  "entry_count": 42,
  "wal_path": "./ferrocache.wal",
  "hnsw": { "max_nb_connection": 16, "ef_construction": 200, "ef_search": 32, "dimension": 384 },
  "namespaces": { "all-MiniLM-L6-v2::384": { "entry_count": 42, "dimension": 384 } },
  "counters": {
    "queries_total": 100, "queries_hit": 73, "queries_miss": 27, "hit_rate": 0.73,
    "inserts_total": 50, "replication_forwards": 0, "replication_failures": 0,
    "replication_retries": 0, "compactions": 0
  }
}
```

### GET /metrics

Prometheus text exposition. Always unauthenticated. Key metric families:
`ferrocache_queries_total`, `ferrocache_queries_hit_total`, `ferrocache_queries_miss_total`,
`ferrocache_hit_rate`, `ferrocache_inserts_total`, `ferrocache_query_duration_seconds_bucket`,
`ferrocache_insert_duration_seconds_bucket`, `ferrocache_replication_{forwards,failures,retries}_total`,
`ferrocache_compactions_total`, `ferrocache_namespace_entries{namespace="..."}`,
`ferrocache_cluster_nodes`.

### GET /cluster/status

```json
{
  "mode": "clustered",
  "self_node_id": "node1",
  "gossip_addr": "0.0.0.0:4000",
  "nodes": ["node1", "node2", "node3"],
  "node_count": 3
}
```

`mode: "single"` when cluster is disabled.

### POST /admin/compact

Forces a snapshot + WAL truncation. Useful before a planned restart so the next startup loads the snapshot instead of replaying the full WAL.

```json
{ "status": "ok", "entries_snapshotted": 12345, "wal_sequence": 12345 }
```

## Authentication

Set `FERROCACHE_AUTH_TOKEN` on the server side. With it set:

- All data routes require `Authorization: Bearer <token>` → 401 otherwise
- `/health` and `/metrics` are exempt — load balancers and Prometheus don't need the token
- Token comparison is constant-time (`subtle::ConstantTimeEq`)
- The token value is never logged at any level

When mTLS is also enabled in cluster mode, inter-node forwards carry the
*same* bearer token in addition to the client cert. All nodes share one token.

```bash
# 401 without the header
curl -X POST localhost:3000/query -H 'Content-Type: application/json' -d '...'

# 200 with it
curl -X POST localhost:3000/query \
  -H 'Authorization: Bearer my-token' \
  -H 'Content-Type: application/json' -d '...'
```

In Python, the client picks up `FERROCACHE_AUTH_TOKEN` from the environment
automatically, or accepts an explicit `auth_token=` kwarg.

## Integration Patterns

### Pattern 1 — Direct HTTP client (Python, stdlib only)

```python
from ferrocache import FerrocacheClient

client = FerrocacheClient("http://localhost:3000", auth_token="my-token")
client.insert(
    embedding=[0.1, 0.2, 0.3, ...],
    response="The answer",
    query_text="What is X?",
    model_id="all-MiniLM-L6-v2::384",
)

hit = client.query(
    embedding=[0.1, 0.2, 0.3, ...],
    threshold=0.92,
    model_id="all-MiniLM-L6-v2::384",
)
if hit["hit"]:
    print(hit["response"], hit["similarity"])
```

### Pattern 2 — OpenAI middleware (one import swap)

```python
from openai import OpenAI
from ferrocache.middleware import wrap_openai

client = wrap_openai(OpenAI(), auth_token="my-token")
# All chat.completions.create calls now check ferrocache first.
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
# resp._ferrocache_hit is True / False / None (None = ferrocache was unreachable; fail-open)
```

### Pattern 3 — Anthropic middleware

```python
from anthropic import Anthropic
from ferrocache.middleware import wrap_anthropic

client = wrap_anthropic(Anthropic(), auth_token="my-token")
resp = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=512,
    messages=[{"role": "user", "content": "Summarize this..."}],
)
```

### Pattern 4 — LangChain cache backend

```python
from langchain.globals import set_llm_cache
from ferrocache.langchain import FerrocacheCache

set_llm_cache(FerrocacheCache(auth_token="my-token"))
# Every LLM call in any LangChain chain now consults ferrocache.
```

### Pattern 5 — LlamaIndex LLM wrapper

```python
from llama_index.llms.openai import OpenAI
from ferrocache.llamaindex import FerrocacheLLM

llm = FerrocacheLLM(inner=OpenAI(model="gpt-4o-mini"), auth_token="my-token")
# Use `llm` anywhere LlamaIndex expects an LLM.
```

### Pattern 6 — MCP server (Claude Desktop / Claude Code)

```bash
pip install ferrocache[mcp]
python -m ferrocache.mcp_server  # speaks JSON-RPC over stdio
```

Three tools: `semantic_cache_lookup`, `semantic_cache_store`, `cache_status`.
The server reads `FERROCACHE_URL`, `FERROCACHE_THRESHOLD`, `FERROCACHE_EMBED_MODEL`,
and `FERROCACHE_AUTH_TOKEN` from the environment. See `docs/mcp-setup.md` for
the Claude Desktop / Claude Code config JSON.

## Threshold Selection Guide

Cosine similarity 0.0–1.0; higher = stricter match. The right threshold
depends on how much paraphrasing your inputs tolerate:

| Use Case                | Threshold   | Why |
|-------------------------|-------------|-----|
| FAQ / knowledge base    | 0.88–0.92   | High variation in phrasing; moderate false-hit risk |
| Code documentation      | 0.92–0.95   | Technical queries need precise matching |
| Customer support        | 0.90–0.93   | Balance hit rate vs. accuracy |
| Translation cache       | 0.95–0.98   | Slight wording changes can change meaning |
| Exact-dedup only        | 0.99+       | Only near-identical queries hit |

If you can run an offline simulation, `tests/simulate.py` (in the repo)
shows a 100% hit / 0 false-positive workload at threshold 0.90 on an FAQ corpus.

## Namespace / model_id Rules

- `model_id` is **required** on every `/insert` and `/query` (since M14)
- Format convention: `model_name::dimension`, e.g. `all-MiniLM-L6-v2::384`, `text-embedding-3-small::1536`
- Vectors from different `model_id`s are **never** compared — each namespace has its own HNSW index. Cross-model false hits are impossible by construction.
- The Python middleware (default `embed_fn`) auto-derives `model_id` from the embed model name + dimension. Custom `embed_fn`s require an explicit `model_id` argument — ferrocache will refuse to guess.
- Switching embedding models means a new `model_id`. Old entries stay in their old namespace but become unreachable until you query with their original `model_id`.
- Pre-M14 entries (no `model_id` in the WAL) load into the `legacy::unknown` namespace.

## Production Checklist

- [ ] `FERROCACHE_AUTH_TOKEN` set to a long random string (`openssl rand -hex 32`)
- [ ] Cluster deployments: `FERROCACHE_CLUSTER__TLS__ENABLED=true` with certs from your PKI
- [ ] Public-port TLS terminated at a reverse proxy (nginx, caddy, ALB)
- [ ] Firewall: public port open; internal port + gossip UDP cluster-only
- [ ] Disk encryption on the WAL volume (LUKS, FileVault, EBS)
- [ ] Prometheus scraping `/metrics`; Grafana dashboard imported from `monitoring/grafana/dashboards/ferrocache.json`
- [ ] Alerts on `ferrocache_replication_failures_total > 0` (peer issues)
- [ ] Alerts on `ferrocache_replication_retries_total` rate (flapping peers)
- [ ] Alerts on `ferrocache_hit_rate < 0.5` (cache not helping — wrong threshold or wrong `model_id`)
- [ ] `compact_interval_inserts` reviewed (default 10K is fine for most workloads)
- [ ] `max_replication_retries` reviewed (default 3 is fine; raise for noisy networks)
- [ ] Cert rotation runbook documented (rolling restart with new PEMs)

## Troubleshooting

### "model_id is required" — 400

Every `/insert` and `/query` must include `model_id`. With the Python
middleware it's auto-derived; with raw HTTP, add `"model_id": "your-model::dim"`
to the body.

### Cache hit rate is 0%

Check that insert and query use the **same** `model_id`. Different model_ids
have separate HNSW indexes — no entries are visible across them. Also
double-check the embedding is not all zeros (cosine on a zero vector is
undefined; HNSW will return weird results).

### 502 on `/insert` in cluster mode

A peer node is unreachable. Check `ferrocache_replication_retries_total` — if
it's climbing, peers are flaky and retries are kicking in. If a forward fails
*after* `max_replication_retries`, ferrocache returns 502. Verify gossip
connectivity via `/cluster/status` (every node should see the same
`node_count`).

### Slow startup (full WAL replay)

Each insert appends a line to the WAL. On restart, the WAL replays from the
start unless a snapshot is present. Run `POST /admin/compact` to write a
snapshot and truncate the WAL — next restart will be near-instant. Or set
`compact_interval_inserts` to a sensible value so auto-compaction triggers
periodically.

### "unauthorized" — 401

`FERROCACHE_AUTH_TOKEN` is set on the server but the request didn't carry a
matching `Authorization: Bearer <token>` header. Either include the header or
unset the env var on the server.

### TLS handshake failures in cluster mode

All nodes must share the same cluster CA. If a node is configured against a
different CA, peers will reject its cert and the handshake fails before any
HTTP framing. Verify with `openssl x509 -in node.pem -noout -issuer` — the
issuer DN must match the cluster CA's subject DN on every node.

### `/metrics` shows `ferrocache_replication_retries_total` climbing

A peer is flaky (intermittent connection refused, timeouts, or 5xx). Check
the peer's logs and `/health`. The retry budget is `max_replication_retries`
(default 3) per forward; retries that succeed don't count as failures.
