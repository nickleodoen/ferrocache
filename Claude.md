# ferrocache — Project Context

## Section 1: Evergreen

**What this is:** Distributed semantic cache for LLM applications, written in Rust. Single binary, multi-node via consistent hashing + gossip replication. Portfolio project for big-tech Core SWE interviews.

**Current phase:** Phase 2 complete. Next: Phase 3 — polish (README, benchmarks, Python client, CI).

**Completed work:**
- M1: axum scaffold (3 routes, tracing, env-based port)
- M2: hnsw_rs integration (DistCosine, side-table keyed by usize, dim lock)
- M3: WAL (NDJSON, fsync-per-insert, replay on startup, corrupt-line skip; WAL-first insert path; UUIDs stable across restarts)
- M4: config crate (TOML + env merge), request validation (4096 dim, 100KB resp, threshold range), /stats endpoint, dead-code cleanup
- M5: consistent hash ring (FNV-1a, 64 vnodes default) + chitchat gossip discovery, /cluster/status endpoint, single-node fallback
- M6: cluster-aware /query routing + /insert synchronous replication, `?local=true` loop-prevention param, UUID stamping by coordinator
- M7: Dockerfile, docker-compose 3-node cluster, bash integration tests; fixed env-var prefix-separator bug (FERROCACHE_PORT was being skipped silently)

**Module map:**
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

### 2026-05-01 — Mission 6: Routing + replication
**Built:** New `src/router.rs`: `ClusterRouter { client: reqwest::Client }` (5s timeout, reusable). `forward_query(addr, &QueryRequest)` POSTs `http://{addr}/query?local=true` and decodes; non-2xx → error. `forward_insert(addr, &InsertRequest)` symmetric. Tests spin up an in-process axum mock on `127.0.0.1:0` and assert `?local=true` is forwarded + `uuid` round-trips. `HashRing::get_n_nodes(key, n)` walks BTreeMap clockwise (chained `range(..key)` for wrap), dedupes via `HashSet`. `ClusterState` tracks `addrs: HashMap<node_id, api_addr>`; each node advertises its API addr by passing `vec![("api_addr", config.api_addr)]` as `initial_key_values` to `spawn_chitchat`; reconciler reads peers via `cc.node_state(chat_id).get("api_addr")`. New `get_target_addr` and `get_replica_addrs(emb, n)` return `(node_id, api_addr)` tuples. `ClusterConfig` gained `api_addr` (default `0.0.0.0:3000`) and `replication_factor` (default 2). `InsertRequest` gained `uuid: Option<String>`. Server `/insert` and `/query` now take `Query<LocalParam>`: route to owner via `forward_query` when not local, or fan out to replicas synchronously after stamping UUID on the request body. 40/40 tests pass.

**Key decisions:**
- Coordinator stamps the UUID before fan-out so all replicas store the same id (cluster-wide UUID stability mirrors WAL UUID stability across restarts).
- Synchronous all-or-nothing replication: any replica failure → 502. Anti-entropy is a separate problem.
- `local=true` serves both loop-prevention (forwarded) and diagnostic (curl a specific node) cases via the same code path.
- 502 vs 500 distinguishes "peer down" from "I'm broken" — operationally important.
- If self is in the replica set, skip the network hop for our own write (avoid spending a request on `localhost`).

**Deviations:** `HashRing::get_n_nodes` takes `u64` (not embedding); thin `_for_embedding` wrapper. `local_insert_inner` returns `Result<String, Response>` so coordinator can read the UUID without re-parsing. `#[allow(clippy::too_many_arguments)]` on `AppState::new` (8 args) — explicit DI is clearer than a builder.

**Open:** No retry/timeout tuning beyond reqwest's 5s default. Replication > live nodes silently degrades.

### 2026-05-01 — Mission 7: Docker + docker-compose 3-node cluster + integration tests
**Built:** Multi-stage `Dockerfile` (Rust 1.94-bookworm builder → debian:bookworm-slim runtime with `ca-certificates`, `libgomp1`, `curl`); `mkdir -p /data` in the runtime image. `docker-compose.yml` with `node1`/`node2`/`node3`, each on host ports 3001/3002/3003, named volumes for WAL persistence, full-mesh seed nodes via Docker DNS (`nodeX:4000`). `tests/cluster_integration.sh` exercises 18 assertions via `curl`+`jq`: convergence (3/3 ring), cross-node read-after-write, routing-aware queries from node3 for an embedding inserted on node2, health, and `local=true` scoping (insert local-only on node1, hit on node1, miss on node2). Added `Makefile` with `cluster-up`/`cluster-down`/`cluster-test`/`clean` targets, `.dockerignore`. `Wal::open` now mkdir's the parent dir so empty `/data` volumes work without a Dockerfile-side `touch`. **Outcome:** 40 unit tests pass, `docker compose build` succeeds, all 18 integration assertions pass with cluster converging in 1 second.

**Key decisions:**
- Comma-separated `seed_nodes` (not JSON array) — cleaner under config-rs's env-var parsing than fighting JSON encoding inside an env string.
- Used `Rust 1.94-bookworm` to match the host's compiler version (also needed: edition=2024 needs ≥1.85). `libgomp1` in the runtime image because hnsw_rs pulls in rayon → OpenMP.
- `Wal::open` mkdir-p logic stays in the WAL module (single-responsibility) rather than dockerfile RUN — also makes the binary friendlier on bare-metal first-run.
- `.dockerignore` excludes `target/` (massive), `*.wal`, `tests/`, `Makefile`, etc. — keeps build context tiny and reproducible.

**Deviations:**
- **Real bug found and fixed:** `config` crate's env source defaults `prefix_separator` to the same value as `separator`. We had `separator("__")`, so the prefix pattern became `ferrocache__`, which `FERROCACHE_PORT` (single underscore) doesn't match — every single env var was silently dropped. First docker-compose run came up in single-node mode with random UUIDs and `wal_path=./ferrocache.wal` despite all the `FERROCACHE_*` env vars. Fix: `.prefix_separator("_")` explicitly. Also added `.try_parsing(true)` (required for bool/int coercion AND for list_separator to fire). This was a latent bug present since M4; the unit-test `set_default` chain hid it because defaults never went through the env source.
- Brief suggested JSON-array env vars (`'["node2:4000"]'`); switched to comma-separated.
- Brief suggested Rust 1.78; used 1.94 for edition=2024 support.

**Next session (Phase 3 / M8):** README with quickstart + architecture diagram, criterion benchmarks (insert+query latency, p50/p99 under load), Python client (sync + async), GitHub Actions CI (cargo test + cargo clippy + cluster-integration on a runner with Docker).

**Open:** Integration script doesn't exercise node-failure / partition / re-join. WAL grows unbounded — no compaction or snapshot yet. No metrics endpoint; logs only. `cluster_id` still hardcoded to "ferrocache". Container healthchecks not wired (compose just starts and we sleep; real prod would use HEALTHCHECK on `/health`).
