# Benchmarks

> **The right comparison for FerroCache is Redis, not GPTCache.**
>
> GPTCache is a Python library — it runs inside your process and its "latency" is a function call, not a network call. FerroCache is a service — like Redis, it has a network boundary by design, which is what lets it be shared across your entire fleet.

## Single-operation latency

Measured on Apple M4 Pro, release build, 384-dim unit vectors, 1k pre-populated entries where applicable. Reproduce with `cargo bench`.

| Benchmark | p50 | p95 | p99 |
|---|---|---|---|
| Index insert (in-memory only) | 21.6 µs | — | — |
| Query hit (HTTP round-trip) | 0.44ms | 0.51ms | 0.54ms |
| Query miss (HTTP round-trip) | 0.42ms | 0.48ms | 0.50ms |
| Insert (includes WAL fsync) | 7.95ms | 8.36ms | 8.71ms |
| Exact-match pre-filter | 0.38ms | — | — |

## Throughput under concurrency

Apple Silicon, `cargo run --release`, 384-dim embeddings, 5s per cell. Reproduce with `make bench-concurrent`.

The pre-M20 path took the WAL mutex per insert → `fsync(2)` serialised every writer. Group-commit coalesces concurrent inserts into a single batched write + one fsync.

**Insert throughput** (per-insert fsync vs default group-commit):

| Mode | 1 client | 10 clients | 50 clients | 100 clients |
|---|---:|---:|---:|---:|
| Per-insert fsync | 169/s | 183/s | 167/s | 183/s |
| Group-commit (256) | 122/s | 682/s | 1057/s | **1900/s** |
| **Speedup** | 0.7× | 3.7× | 6.3× | 10.4× |

p99 insert latency at concurrency 100: **675ms (no group-commit) → 70ms (group-commit)**.

**Query throughput** (read path — no fsync):

| Workload | 1 client | 10 clients | 50 clients | 100 clients |
|---|---:|---:|---:|---:|
| Query (hit) | 1568/s | 3529/s | 3513/s | 3491/s |
| Query (miss) | 2632/s | 3498/s | 3513/s | 3579/s |

**Eviction overhead** (concurrency 100, 5K entry cap forcing rebuild on every batch):

| Setting | Insert ops/s | p99 insert |
|---|---:|---:|
| `max_entries_per_namespace = None` | 1900/s | 70ms |
| `max_entries_per_namespace = 5000` | 1313/s | 1824ms |

LRU eviction adds ~30% throughput overhead under sustained pressure. The p99 spike comes from periodic HNSW rebuilds (20% ghost-ratio threshold) running under the index write lock — a known trade-off of inline rebuild for graph quality. See [Eviction & TTL](../features/eviction-ttl.md).

## Comparison vs GPTCache

Same workload (200 inserts, 600 expected-hit queries, 50 unrelated; 384-dim embeddings via sentence-transformers `all-MiniLM-L6-v2`; threshold 0.90; Apple Silicon). Reproduce with `make benchmark-vs-gptcache`.

| Metric | FerroCache | GPTCache | Notes |
|---|---|---|---|
| Hit rate (threshold 0.90) | 99.8% | N/A † | Same embedding model |
| False hits on unrelated | 0 | N/A † | |
| Query latency p50 | 0.44ms | N/A † | FerroCache: HTTP round-trip |
| Query latency p99 | 0.84ms | N/A † | |
| Insert latency p50 | 8.05ms | N/A † | FerroCache: WAL fsync |
| Insert latency p99 | 10.88ms | N/A † | |
| RSS after 200-entry seed | 14.1 MB | N/A † | Resident set size |
| Insert throughput (concurrency 50) | 2476/s | N/A | GPTCache is single-threaded |

> † GPTCache requires `faiss-cpu` + `onnxruntime`, which currently lack wheels for Python 3.13+. The benchmark harness runs GPTCache in a child process so a native segfault doesn't take down the parent — on this machine (Python 3.14) the child SIGSEGVs at faiss init. The script reports N/A and continues; on Python ≤ 3.12 it produces real comparison numbers.

## Feature comparison vs GPTCache

| | FerroCache | GPTCache |
|---|---|---|
| Architecture | Service (HTTP) | Library (in-process) |
| Multi-node cluster | ✅ | ❌ |
| Shared across fleet | ✅ | ❌ (per-process) |
| WAL durability | ✅ (fsync) | ❌ (in-memory) |
| Survives app restart | ✅ | ❌ |
| Tenant isolation | ✅ `cache_scope` | ❌ |
| Conversation scoping | ✅ | ❌ |
| Exact-match pre-filter | ✅ | ❌ |
| TTL per entry | ✅ | ⚠️ partial |
| LRU eviction | ✅ | ✅ |
| Any language client | ✅ | ❌ (Python only) |
| Prometheus metrics | ✅ | ❌ |

## Reproducing benchmarks

```bash
# Criterion microbenchmarks
cargo bench

# Concurrent HTTP throughput
make bench-concurrent

# vs GPTCache (requires Python ≤ 3.12 for working faiss/onnxruntime)
make benchmark-vs-gptcache

# 44-assertion cluster integration suite (sanity, not throughput)
make cluster-test
```
