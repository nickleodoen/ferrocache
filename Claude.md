# ferrocache — Project Context

## Section 1: Evergreen

**What this is:** Distributed semantic cache for LLM applications, written in Rust. Single binary, multi-node via consistent hashing + gossip replication. Portfolio project for big-tech Core SWE interviews.

**Current phase:** Phase 2 in progress. M5 done (hash ring + gossip discovery). Next: M6 — request routing + replication.

**Completed work:**
- M1: axum scaffold (3 routes, tracing, env-based port)
- M2: hnsw_rs integration (DistCosine, side-table keyed by usize, dim lock)
- M3: WAL (NDJSON, fsync-per-insert, replay on startup, corrupt-line skip; WAL-first insert path; UUIDs stable across restarts)
- M4: config crate (TOML + env merge), request validation, /stats endpoint, dead-code cleanup
- M5: consistent hash ring (FNV-1a, 64 vnodes default) + chitchat gossip discovery, /cluster/status endpoint, single-node fallback

**Module map:**
- main.rs — entry, tracing init, config load, WAL replay, optional cluster init, serve
- server.rs — router, handlers, request validation, tests
- index.rs — SemanticIndex (hnsw_rs wrapper + HashMap side-table), replay_entry
- wal.rs — Wal (append w/ fsync), WalEntry, replay
- models.rs — request/response DTOs
- config.rs — FerrocacheConfig, HnswConfig, ClusterConfig, load()
- state.rs — AppState { node_id, index, wal, wal_path, hnsw_config, cluster: Option<Arc<ClusterState>> }
- ring.rs — HashRing (BTreeMap<u64, node_id>, FNV-1a, virtual nodes)
- cluster.rs — ClusterState wrapping chitchat; background reconciler syncs ring to live members

**Architecture decisions (append-only, one line each):**
- Client computes embeddings externally; ferrocache stores/compares f32 vectors
- Crates: tokio, axum 0.7, hnsw_rs, chitchat 0.10, tracing, serde_json, config, reqwest (for M6)
- WAL format: newline-delimited JSON, replayed on startup, fsync per insert
- Consistent hashing key: u64 from first 8 bytes of embedding (first 2 f32 values, big-endian)
- Cosine distance: hnsw_rs DistCosine returns 1-similarity; we convert
- Side-table: HashMap<usize, CacheEntry> keyed by hnsw internal id; UUID inside CacheEntry
- WAL is source of truth; production insert path is WAL-first then replay_entry
- insert() (UUID-generating) is #[cfg(test)] only; production uses replay_entry
- Config priority: env vars (FERROCACHE_ prefix, `__` separator) > ferrocache.toml > defaults
- tokio::sync::RwLock for index, tokio::sync::Mutex for WAL
- Validation limits (server.rs constants): MAX_EMBEDDING_DIM=4096, MAX_RESPONSE_BYTES=102_400
- Threshold accepted range on /query: 0.0..=1.0
- Hash ring: FNV-1a (cross-process deterministic), 64 virtual nodes per physical node by default
- Cluster mode is opt-in via `cluster.enabled`; `false` keeps Phase 1 behavior bit-identical
- Ring reconciler runs every 2s in a background task; chitchat gossip interval = 1s

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

### 2026-05-01 — Mission 4: Config + validation + /stats + Phase 1 polish
**Built:** New `src/config.rs` with `FerrocacheConfig { port, node_id?, wal_path, hnsw }` + `HnswConfig`. `load()` uses `config` crate's builder: defaults → `ferrocache.toml` (optional) → env vars (`FERROCACHE_` prefix, `__` separator). Created `ferrocache.toml` with all keys commented. `SemanticIndex::new` now takes `&HnswConfig`. `AppState` carries `wal_path` + `hnsw_config` for /stats. `main.rs` calls `FerrocacheConfig::load()`; logs `?config`. Validation in handlers: empty/oversized embedding (>4096), oversized response (>100KB), threshold out of [0,1] → 400 via shared `bad_request` helper. New `GET /stats` returns `{entry_count, wal_path, hnsw{...,dimension}}`. `/query` now also returns hit `id`.

**Key decisions:**
- Centralized validation helpers (`validate_embedding`, `bad_request`) so /query and /insert share rejection paths.
- Stored a *copy* of `HnswConfig` in AppState (vs pointer) — tiny, no extra locking for /stats.
- Restructured Claude.md into evergreen + 2-session rolling window.
- Dead-code cleanup: dropped `CacheEntry::embedding`/`query_text` (WAL is source of truth) and `Wal::path` field. `QueryHit::id` now actually returned in `/query`.

**Deviations:** `Wal::open` now takes `impl AsRef<Path>` after dropping `path` field. `AppState::new` takes 5 args explicitly.

**Open:** Replay still single-threaded; WAL still uncompacted; HNSW `max_elements` is a soft hint (no eviction); `node_id` is per-process unless set in config.

### 2026-05-01 — Mission 5: Hash ring + gossip discovery
**Built:** New `src/ring.rs`: `HashRing { ring: BTreeMap<u64, String>, virtual_nodes }` with FNV-1a hashing. `add_node`/`remove_node` insert/remove `virtual_nodes` virtual positions per physical node. `get_node(key)` does the standard "first ≥ key, wrap around" lookup. `embedding_to_key` reads first 2 f32s big-endian into a `u64`. New `src/cluster.rs`: `ClusterState` wraps `chitchat::ChitchatHandle`, holds `Arc<RwLock<HashRing>>`. `new()` builds `ChitchatId` (node_id, unix-time generation, gossip addr) + `ChitchatConfig` and calls `spawn_chitchat(cfg, [], &UdpTransport)`. Background `tokio::spawn` reconciler ticks every 2s: reads chitchat live nodes via `handle.with_chitchat(|cc| cc.live_nodes()...)`, diffs against current ring, applies adds/removes, logs changes. `ClusterConfig { enabled, gossip_addr, seed_nodes, virtual_nodes }` added to `FerrocacheConfig` with `enabled=false` default — single-node mode untouched. New `GET /cluster/status` returns clustered or single-mode JSON. `AppState` gained `cluster: Option<Arc<ClusterState>>`. 31/31 tests pass; clippy + fmt clean.

**Key decisions:**
- FNV-1a (not SipHash/DefaultHasher) so hash positions are identical across processes/machines — required for consistent hashing.
- Self always added to local ring up-front; reconciler never removes self even if chitchat momentarily forgets it.
- Reconciler only writes the ring when something actually changed (cheap idle path).
- `chitchat_handle` kept in `Arc<ChitchatHandle>` and held in `ClusterState` so dropping the state shuts gossip down cleanly. Marked `#[allow(dead_code)]` because the field is "kept alive for side effects."
- Ring exposes `get_node`/`get_node_for_embedding`/`is_local`/`get_target_node` ahead of M6 routing; marked `#[allow(dead_code)]` with comments pointing to M6 (covered by tests).
- Used `format!("{node_id}-{i}")` for virtual node labels — simple, debuggable, deterministic.

**Deviations:**
- Spec said `chitchat = "0.8"`. Latest is **0.10.1** — used that. API still has `spawn_chitchat`, `ChitchatHandle`, `ChitchatConfig`, `ChitchatId::new(node_id, generation, gossip_addr)`, `Chitchat::live_nodes()`, `&UdpTransport`. The handle exposes `with_chitchat<F, T>(|&mut Chitchat| -> T) -> T` which is what the reconciler calls.
- `ChitchatConfig` requires more fields than the brief implied (`cluster_id`, `marked_for_deletion_grace_period`, `failure_detector_config`, `catchup_callback`, `extra_liveness_predicate`). Filled with sane defaults; `cluster_id="ferrocache"` so foreign clusters don't merge.
- Initial `test_two_nodes_distribute` failed: hashing `0u64.to_be_bytes()..99u64.to_be_bytes()` clusters because only the LSB varies, so all keys landed in B's arc. Switched to `format!("key-{k}")` inputs so FNV-1a actually avalanches — the same change applied to `test_remove_node` and `test_consistency_after_add`. The hash function is fine; the test was using degenerate inputs.
- `reqwest` added now for M6 even though unused this mission, per brief — avoids Cargo.toml churn.

**Next session (M6):** route /insert and /query to the owning node via reqwest (returning the proxied response transparently); add a replication factor (write to N owners along the ring); add a `?local=true` query param for diagnostics that bypasses routing.

**Open:** No real cluster integration test yet — chitchat needs UDP and two processes; tests cover ring + single-mode handler. Multi-process tests probably want an `assert_cmd`-style harness in M6/M7. Reconciler logs but doesn't expose a metrics surface; revisit when adding /metrics. `cluster_id` is hard-coded to "ferrocache" — make it config-driven once we run multi-tenant.
