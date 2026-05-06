## v1.0.0 — 2026-05-06

First stable release. 30 missions across 7 phases:

- Distributed semantic cache: HNSW vector search, consistent hashing,
  gossip-based cluster, synchronous replication
- Durability: write-ahead log with fsync, atomic snapshots, WAL compaction
- Resilience: phi accrual failure detection, automatic ring reassignment,
  read repair anti-entropy
- Security: bearer token auth, mutual TLS between cluster nodes
- Performance: group-commit WAL batching (2,600+ inserts/sec at c=50),
  Prometheus metrics, Grafana dashboard
- Cache features: LRU eviction, TTL per entry, exact-match pre-filter,
  targeted deletion, semantic radius invalidation
- Isolation: tenant namespaces (cache_scope), conversation scoping
  with two-level fallback
- Integrations: Python client, OpenAI/Anthropic middleware, LangChain,
  LlamaIndex, MCP server for Claude Desktop/Code
- Distribution: PyPI package, GHCR Docker image, tagged release CI
