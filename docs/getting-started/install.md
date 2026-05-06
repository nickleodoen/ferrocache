# Quick Install

FerroCache is two pieces: a Rust **server** (the cache) and optional **clients** for your language. You always need the server. The Python client is the most common companion, but any HTTP client works.

## Docker (recommended)

Single-node, ephemeral:

```bash
docker run -p 3000:3000 ghcr.io/nickleodoen/ferrocache:latest
```

With a persistent WAL volume:

```bash
docker run -d \
  -p 3000:3000 \
  -v $(pwd)/ferrocache-data:/data \
  -e FERROCACHE_WAL_PATH=/data/ferrocache.wal \
  --name ferrocache \
  ghcr.io/nickleodoen/ferrocache:latest
```

Verify it's up:

```bash
curl http://localhost:3000/health
# {"status":"ok","node_id":"...","entry_count":0}
```

## 3-node cluster (docker-compose)

```bash
git clone https://github.com/nickleodoen/ferrocache
cd ferrocache
docker compose up -d --build
sleep 5
curl http://localhost:3001/cluster/status
```

External ports `3001` / `3002` / `3003` map to the three nodes.

## Python client

```bash
pip install ferrocache              # zero dependencies
pip install ferrocache[openai]      # + OpenAI middleware
pip install ferrocache[anthropic]   # + Anthropic middleware
pip install ferrocache[langchain]   # + LangChain backend
pip install ferrocache[llamaindex]  # + LlamaIndex backend
pip install ferrocache[mcp]         # + MCP server for Claude Desktop / Code
pip install ferrocache[all]         # everything
```

Smoke test:

```python
from ferrocache import FerrocacheClient
print(FerrocacheClient("http://localhost:3000").health())
```

## Build from source (Rust required)

```bash
git clone https://github.com/nickleodoen/ferrocache
cd ferrocache
cargo build --release
./target/release/ferrocache
```

Requires Rust 1.75+ (`rustup` recommended). The `gen_certs` binary is built alongside but is not shipped in the production Docker image.

## Environment variables (quick reference)

| Variable | Purpose |
|---|---|
| `FERROCACHE_PORT` | Public HTTP port (default `3000`) |
| `FERROCACHE_WAL_PATH` | WAL file path (default `./ferrocache.wal`) |
| `FERROCACHE_AUTH_TOKEN` | Bearer token for the public API (auth off when unset) |
| `FERROCACHE_HNSW__DEFAULT_THRESHOLD` | Cosine similarity cutoff (default `0.92`) |
| `FERROCACHE_HNSW__MAX_ENTRIES_PER_NAMESPACE` | LRU cap per namespace |
| `FERROCACHE_CLUSTER__ENABLED` | `true` to join a cluster |
| `FERROCACHE_CLUSTER__SEED_NODES` | Comma-separated seed list |
| `FERROCACHE_CLUSTER__TLS__ENABLED` | `true` to enable inter-node mTLS |

The full list lives in [Configuration](configuration.md).

## Next steps

- [Your First Cache](first-cache.md) — insert and query an entry end-to-end.
- [HTTP API](../integrations/http-api.md) — language-agnostic reference.
- [Cluster Setup](../production/cluster.md) — production multi-node deployment.
