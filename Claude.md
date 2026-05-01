# ferrocache — Project Context

## Section 1: Evergreen

**What this is:** Distributed semantic cache for LLM applications, written in Rust. Single binary, multi-node via consistent hashing + gossip replication. Portfolio project for big-tech Core SWE interviews.

**Current phase:** Phase 2 in progress. M5 + M6 done (ring + gossip + routing + replication). Next: M7 — Docker / multi-node integration testing.

**Completed work:**
- M1: axum scaffold (3 routes, tracing, env-based port)
- M2: hnsw_rs integration (DistCosine, side-table keyed by usize, dim lock)
- M3: WAL (NDJSON, fsync-per-insert, replay on startup, corrupt-line skip; WAL-first insert path; UUIDs stable across restarts)
- M4: config crate (TOML + env merge), request validation (4096 dim, 100KB resp, threshold range), /stats endpoint, dead-code cleanup
- M5: consistent hash ring (FNV-1a, 64 vnodes default) + chitchat gossip discovery, /cluster/status endpoint, single-node fallback
- M6: cluster-aware /query routing + /insert synchronous replication, `?local=true` loop-prevention param, UUID stamping by coordinator

**Module map:**
- main.rs — entry, tracing init, config load, WAL replay, optional cluster init, serve
- server.rs — router, handlers, request validation, routing/replication coordination, tests
- index.rs — SemanticIndex (hnsw_rs wrapper + HashMap side-table), replay_entry
- wal.rs — Wal (append w/ fsync), WalEntry, replay
- models.rs — request/response DTOs (InsertRequest carries optional `uuid`)
- config.rs — FerrocacheConfig, HnswConfig, ClusterConfig (gossip_addr, api_addr, replication_factor)
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
- Config priority: env vars (FERROCACHE_ prefix, `__` separator) > ferrocache.toml > defaults
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

### 2026-05-01 — Mission 5: Hash ring + gossip discovery
**Built:** New `src/ring.rs`: `HashRing { ring: BTreeMap<u64, String>, virtual_nodes }` with FNV-1a hashing. `add_node`/`remove_node` insert/remove `virtual_nodes` virtual positions per physical node. `get_node(key)` does the standard "first ≥ key, wrap around" lookup. `embedding_to_key` reads first 2 f32s big-endian into a `u64`. New `src/cluster.rs`: `ClusterState` wraps `chitchat::ChitchatHandle`, holds `Arc<RwLock<HashRing>>`. `new()` builds `ChitchatId` (node_id, unix-time generation, gossip addr) + `ChitchatConfig` and calls `spawn_chitchat(cfg, [], &UdpTransport)`. Background `tokio::spawn` reconciler ticks every 2s: reads chitchat live nodes via `handle.with_chitchat(|cc| cc.live_nodes()...)`, diffs against current ring, applies adds/removes, logs changes. `ClusterConfig { enabled, gossip_addr, seed_nodes, virtual_nodes }` added to `FerrocacheConfig` with `enabled=false` default — single-node mode untouched. New `GET /cluster/status` returns clustered or single-mode JSON. `AppState` gained `cluster: Option<Arc<ClusterState>>`. 31/31 tests pass; clippy + fmt clean.

**Key decisions:**
- FNV-1a (not SipHash/DefaultHasher) so hash positions are identical across processes/machines.
- Self always added to local ring up-front; reconciler never removes self.
- Reconciler only writes the ring when something actually changed (cheap idle path).
- `chitchat_handle` kept in `Arc<ChitchatHandle>` and held in `ClusterState` so dropping shuts gossip down cleanly.

**Deviations:** chitchat is at **0.10.1**, not 0.8 as briefed. `ChitchatConfig` requires more fields than the brief implied (`cluster_id`, `marked_for_deletion_grace_period`, etc.); filled with sane defaults. Initial `test_two_nodes_distribute` failed because `0u64.to_be_bytes()..99u64.to_be_bytes()` clusters under FNV (only LSB varies); switched to `format!("key-{k}")` inputs.

**Open:** No real cluster integration test yet — chitchat needs UDP and two processes; tests cover ring + single-mode handler. `cluster_id` is hard-coded.

### 2026-05-01 — Mission 6: Routing + replication
**Built:** New `src/router.rs`: `ClusterRouter { client: reqwest::Client }` (5s timeout, reusable). `forward_query(addr, &QueryRequest)` POSTs `http://{addr}/query?local=true` and decodes; non-2xx becomes an error. `forward_insert(addr, &InsertRequest)` symmetric. New tests spin up an in-process axum mock on `127.0.0.1:0`, assert the `?local=true` query param is forwarded, assert `uuid` round-trips through `InsertRequest`. `HashRing::get_n_nodes(key, n)` walks BTreeMap clockwise (with wrap via chained `range(..key)`), collects up to `n` distinct physical nodes via a `HashSet` dedupe — used for replica selection. `ClusterState` now also tracks `addrs: Arc<RwLock<HashMap<node_id, api_addr>>>`. Each node advertises its API addr via chitchat by passing `vec![("api_addr", config.api_addr)]` as the `initial_key_values` arg to `spawn_chitchat`. Reconciler reads peers' KV via `cc.node_state(chat_id).get("api_addr")` and updates the map alongside the ring. `ClusterState::get_target_addr` and `get_replica_addrs(emb, n)` expose `(node_id, api_addr)` tuples. `ClusterConfig` gained `api_addr` (default `0.0.0.0:3000`) and `replication_factor` (default 2). `InsertRequest` gained `uuid: Option<String>` (skipped when `None`); when present on a `local=true` insert, that UUID is used instead of generating a new one. Server handlers now take `Query<LocalParam>`: `/query` routes to the embedding's owner (or local if self/disabled/`local=true`), `/insert` fetches the replica set, locally inserts if self is a replica, then synchronously forwards to each non-self replica via `ClusterRouter`; any forward failure → 502. `AppState::new` grew `router: Option<Arc<ClusterRouter>>` and `replication_factor: usize`. 40/40 tests pass; clippy + fmt clean.

**Key decisions:**
- Coordinator stamps the UUID before fan-out (mutates `req.uuid = Some(...)`) so replica WALs all store the same id — UUID stability across the cluster matches the WAL's UUID stability across restarts.
- Synchronous all-or-nothing replication: any replica failure → 502 to client. No partial-write recovery; that's an anti-entropy/M-later problem.
- `local=true` is read both for the loop-prevention case (forwarded) and the diagnostic case (curl a specific node). Same code path either way.
- 502 on forward failure (vs 500 for local errors): distinguishes "peer down" from "I'm broken" — operationally important.
- Replica set comes from `get_n_nodes_for_embedding(emb, factor)`. If self is in the set, we skip the network hop for our own write — avoids spending a request on `localhost`.
- Cluster-disabled path (`cluster.is_none()`) takes the same `process_insert_locally` codepath that `local=true` does. Phase 1 behavior unchanged.

**Deviations:**
- `HashRing::get_n_nodes` takes a raw `u64` key (not an embedding); a thin `get_n_nodes_for_embedding` wraps it. Mirrors the `get_node`/`get_node_for_embedding` split from M5 — the raw-key form is more testable and the embedding form is what callers want.
- `local_insert_inner` returns `Result<String, Response>` rather than producing a full `Response` — the coordinator path needs the UUID (for the client-facing reply), the single-node path just unwraps it. Avoids a pointless re-parse.
- Added `#[allow(clippy::too_many_arguments)]` on `AppState::new` (8 args) — explicit DI is clearer than wrapping in a builder for what is effectively config.
- Type aliases (`SharedRing`, `SharedAddrs`) added to silence `clippy::type_complexity` in cluster tests.

**Next session (M7):** docker-compose with 3 ferrocache nodes, seed-node bootstrapping, an integration test harness that spins up the cluster and verifies (a) writes replicate to N nodes, (b) reads route to the right shard, (c) killing a node and querying through a survivor still works.

**Open:** No retry/timeout tuning beyond the 5s reqwest default. No backpressure if a slow replica blocks the coordinator. `replication_factor > number_of_live_nodes` silently degrades to fewer replicas (`get_n_nodes` returns what it can) — could surface a warning when the ring is short. Chitchat KV propagation latency (~1–2s) means newly-joined nodes are briefly addressless and will be skipped until their `api_addr` propagates; acceptable for now.
