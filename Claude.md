# ferrocache — Project Context

## Section 1: Evergreen

**What this is:** Distributed semantic cache for LLM applications, written in Rust. Single binary, multi-node via consistent hashing + gossip replication. Portfolio project for big-tech Core SWE interviews.

**Current phase:** All phases complete (M1–M8). Project is portfolio-ready.

**Completed work:**
- M1: axum scaffold (3 routes, tracing, env-based port)
- M2: hnsw_rs integration (DistCosine, side-table keyed by usize, dim lock)
- M3: WAL (NDJSON, fsync-per-insert, replay on startup, corrupt-line skip; WAL-first insert path; UUIDs stable across restarts)
- M4: config crate (TOML + env merge), request validation (4096 dim, 100KB resp, threshold range), /stats endpoint, dead-code cleanup
- M5: consistent hash ring (FNV-1a, 64 vnodes default) + chitchat gossip discovery, /cluster/status endpoint, single-node fallback
- M6: cluster-aware /query routing + /insert synchronous replication via reqwest, `?local=true` loop-prevention, coordinator UUID stamping, get_n_nodes replica walk
- M7: Dockerfile, docker-compose 3-node cluster, bash integration tests; fixed env-var prefix-separator bug (FERROCACHE_PORT was being skipped silently)
- M8: README + Mermaid diagram, criterion benchmarks (real numbers), stdlib-only Python client, GitHub Actions CI (check + integration), library/binary split for bench imports

**Module map:**
- lib.rs — re-exports modules so benches and external consumers can import
- main.rs — entry, tracing init, config load, WAL replay, optional cluster init, serve
- server.rs — router, handlers, request validation, routing/replication coordination, tests
- index.rs — SemanticIndex (hnsw_rs wrapper + HashMap side-table), replay_entry
- wal.rs — Wal (append w/ fsync, mkdir -p parent), WalEntry, replay
- models.rs — request/response DTOs (InsertRequest carries optional `uuid`)
- config.rs — FerrocacheConfig, HnswConfig, ClusterConfig (gossip_addr, api_addr, replication_factor, seed_nodes)
- state.rs — AppState { node_id, index, wal, wal_path, hnsw_config, cluster, router, replication_factor }
- ring.rs — HashRing (BTreeMap<u64, node_id>, FNV-1a, virtual nodes, get_n_nodes for replica walk)
- cluster.rs — ClusterState wrapping chitchat; reconciler syncs ring + node_id→api_addr map from gossip KV
- router.rs — ClusterRouter (reqwest client) — forward_query / forward_insert to peers with `?local=true`
- benches/cache_bench.rs — criterion: insert / query_hit / query_miss / insert_with_wal
- clients/python/ferrocache.py — stdlib-only Python client (urllib + json)

**Architecture decisions (append-only, one line each):**
- Client computes embeddings externally; ferrocache stores/compares f32 vectors
- Crates: tokio, axum 0.7, hnsw_rs, chitchat 0.10, tracing, serde_json, config, reqwest
- WAL format: newline-delimited JSON, replayed on startup, fsync per insert
- Consistent hashing key: u64 from first 8 bytes of embedding (first 2 f32 values, big-endian)
- Cosine distance: hnsw_rs DistCosine returns 1-similarity; we convert
- Side-table: HashMap<usize, CacheEntry> keyed by hnsw internal id; UUID inside CacheEntry
- WAL is source of truth; production insert path is WAL-first then replay_entry
- insert() (UUID-generating) is #[cfg(test)] only; production uses replay_entry
- Config priority: env vars (FERROCACHE_ prefix, `_` prefix-sep, `__` section-sep, `,` list-sep) > ferrocache.toml > defaults
- tokio::sync::RwLock for index, tokio::sync::Mutex for WAL
- Validation limits (server.rs constants): MAX_EMBEDDING_DIM=4096, MAX_RESPONSE_BYTES=102_400
- Threshold accepted range on /query: 0.0..=1.0
- Hash ring: FNV-1a (cross-process deterministic), 64 virtual nodes per physical node by default
- Cluster mode is opt-in via `cluster.enabled`; `false` keeps Phase 1 behavior bit-identical
- Ring reconciler runs every 2s in a background task; chitchat gossip interval = 1s
- Each node advertises its API addr via chitchat KV under key `api_addr`; reconciler reads peers' addrs
- Routing: `?local=true` skips ring lookup (used by forwarded requests + debugging); coordinator stamps UUID before fan-out
- Replication: synchronous, primary-or-coordinator + (replication_factor-1) replicas; any replica failure → 502
- Forward failures return 502 (BAD_GATEWAY) to distinguish "peer failed" from local 500
- Docker runtime image needs `libgomp1` (hnsw_rs → rayon → OpenMP)
- `seed_nodes` is parsed from a comma-separated env string via config-rs `list_separator(",")`+`with_list_parse_key`
- `Wal::open` mkdir's parent dir so containers can mount an empty `/data` volume
- Crate is library + binary (lib.rs re-exports modules); `cargo build --release --bin ferrocache` for prod images
- Benchmarks via criterion; Python client uses only stdlib (urllib+json)

**Non-negotiable constraints:**
- No auth/TLS until Phase 3
- No UI
- No direct OpenAI/Anthropic API calls inside ferrocache
- Use tokio, not async-std
- Every new module gets unit tests before moving on

**Session log rules:**
- Keep only the last 2 session logs below
- When adding a new log, delete the oldest if there are already 2
- Summarize deleted sessions as one-line entries under "Completed work" above

## Section 2: Rolling Session Log (last 2 sessions only)

### 2026-05-01 — Mission 7: Docker + docker-compose 3-node cluster + integration tests
**Built:** Multi-stage `Dockerfile` (Rust 1.94-bookworm builder → debian:bookworm-slim runtime with `ca-certificates`, `libgomp1`, `curl`); `mkdir -p /data` in the runtime image. `docker-compose.yml` with `node1`/`node2`/`node3`, host ports 3001/3002/3003, named volumes for WAL persistence, full-mesh seed nodes via Docker DNS (`nodeX:4000`). `tests/cluster_integration.sh` exercises 18 assertions via curl+jq: convergence (3/3 ring), cross-node read-after-write, routing-aware queries, health, `local=true` scoping. Added `Makefile` and `.dockerignore`. `Wal::open` now mkdir's the parent dir so empty `/data` volumes work. **Outcome:** 40 unit tests pass, all 18 integration assertions pass, cluster converges in 1 second.

**Key decisions:**
- Comma-separated `seed_nodes` (not JSON array) — cleaner under config-rs's env-var parsing.
- Used Rust 1.94-bookworm to match host (edition=2024 needs ≥1.85). `libgomp1` for hnsw_rs/rayon/OpenMP.
- WAL mkdir-p logic stays in the WAL module (single-responsibility, also helps bare-metal first-run).

**Deviations:** **Real bug found and fixed:** `config` crate's env source defaults `prefix_separator` to the same value as `separator`. With `separator("__")`, the prefix pattern became `ferrocache__`, which `FERROCACHE_PORT` doesn't match — every env var was silently dropped. Fix: `.prefix_separator("_")` + `.try_parsing(true)`. Latent since M4. Brief suggested JSON-array env vars and Rust 1.78; switched to CSV and 1.94.

**Open:** No node-failure / partition test. WAL still uncompacted. No metrics endpoint. `cluster_id` hardcoded.

### 2026-05-01 — Mission 8: README + benchmarks + Python client + CI
**Built:** Restructured the crate as `lib.rs` (re-exports all modules) + `main.rs` so benches and the Python tooling can import internals. `benches/cache_bench.rs` (criterion) with 4 benches using deterministic LCG-generated 384-dim unit vectors: `insert_384d`, `query_hit_1k_384d`, `query_miss_1k_384d`, `insert_wal_fsync_384d`. Wrote a full README (~150 lines) with quickstart for single-node + 3-node Docker, API table, config table, Mermaid architecture diagram (request-flow with replicas), design decisions, real benchmark numbers, Python usage snippet, dev/CI commands. `clients/python/ferrocache.py` is a zero-dependency stdlib client (`urllib.request` + `json`) with `insert/query/health/stats/cluster_status` and a `FerrocacheError` exception. `clients/python/example_usage.py` demos a roundtrip — verified live against a running node (insert + exact-match query + cluster_status). `.github/workflows/ci.yml` has two jobs: `check` (fmt → check → test → clippy with rust-cache) and `integration` (depends on check; `docker compose up -d --build` → `sleep 10` → `cluster_integration.sh` → tear down with `if: always()`). Updated `.gitignore` (target, *.wal, __pycache__). Updated `Dockerfile` to also `COPY benches/` and use `--bin ferrocache` so the bench manifest validates without being built.

**Key decisions:**
- Lib + bin crate split — required so `benches/cache_bench.rs` can `use ferrocache::index::SemanticIndex` etc. Same modules, just exposed via `pub mod` in `lib.rs`. Zero behavior change.
- Benchmarks use the production `replay_entry` path (since `insert` is `#[cfg(test)]`-gated). Gives realistic numbers.
- Deterministic LCG-based vectors (no `rand` dep) — stable across runs and machines. Vectors are normalized so cosine distance is meaningful.
- Python client deliberately stdlib-only — no `pip install` step before the example works. Type hints + `from __future__ import annotations` for 3.9+ compatibility.
- README's Mermaid diagram uses `flowchart LR` with a subgraph for the cluster + dotted lines to distinguish primary writes from replica fans-out. Renders natively on GitHub.
- CI runs `fmt --check` first (fastest fail signal), then `check`, then `test`, then `clippy`. The integration job only fires after `check` passes — saves Docker build minutes on a broken commit.
- Dockerfile now `--bin ferrocache` so cargo doesn't try to validate / build benches in production images.

**Benchmark results captured (Apple Silicon, release):**
- `insert_384d` ~21.6 µs median (~46k ops/sec, in-memory only)
- `query_hit_1k_384d` ~152 µs median
- `query_miss_1k_384d` ~154 µs median
- `insert_wal_fsync_384d` ~5.1 ms median (fsync-bound on APFS)

**Deviations:**
- Added `--bin ferrocache` to the Dockerfile build step (M7's `cargo build --release` started failing once `[[bench]]` was declared, because cargo eagerly validates target manifests).
- Bench file uses `tokio::runtime::Builder::new_current_thread()` rather than spinning up a multi-threaded runtime per iter — single-threaded is enough for `Wal::append + fsync` and avoids spawning thread pools in the hot path.

**Next session (Phase 3 / M9 if pursued):** node-failure resilience tests (kill a replica mid-write, expect 502), WAL compaction/snapshotting, /metrics endpoint (Prometheus or plain JSON counters), TLS+auth, full-text query mode (passing `query_text` through a separate exact-match index). All non-blocking.

**Open:** Coordinator currently fans out to replicas serially — could parallelize with `futures::join_all` for tail latency. No retry on transient peer failures. Benchmarks don't cover concurrent load; criterion handles latency-per-op well but for throughput-under-load we'd want a separate harness (wrk, vegeta).
