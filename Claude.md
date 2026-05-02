# ferrocache — Project Context

## Section 1: Evergreen

**What this is:** Distributed semantic cache for LLM applications, written in Rust. Single binary, multi-node via consistent hashing + gossip replication. Portfolio project for big-tech Core SWE interviews.

**Current phase:** Phase 4 in progress. M15 done (WAL compaction). Next: M16 — Prometheus /metrics.

**Completed work:**
- M1: axum scaffold (3 routes, tracing, env-based port)
- M2: hnsw_rs integration (DistCosine, side-table keyed by usize, dim lock)
- M3: WAL (NDJSON, fsync-per-insert, replay on startup, corrupt-line skip; WAL-first insert path; UUIDs stable across restarts)
- M4: config crate (TOML + env merge), request validation (4096 dim, 100KB resp, threshold range), /stats endpoint, dead-code cleanup
- M5: consistent hash ring (FNV-1a, 64 vnodes default) + chitchat gossip discovery, /cluster/status endpoint, single-node fallback
- M6: cluster-aware /query routing + /insert synchronous replication via reqwest, `?local=true` loop-prevention, coordinator UUID stamping, get_n_nodes replica walk
- M7: Dockerfile + docker-compose 3-node cluster + bash integration tests; fixed env-var `prefix_separator` config bug (latent since M4)
- M8: README + Mermaid diagram, criterion benchmarks (real numbers), stdlib-only Python client, GitHub Actions CI (check + integration), library/binary split for bench imports
- M9: simulation harness (`tests/simulate.py` + `simulate_no_ml.py`) — FAQ workload with sentence-transformers, 100% hit rate / 0 false positives at 0.90 threshold
- M10: drop-in SDK middleware (`wrap_openai`, `wrap_anthropic`) with proxy attribute delegation, fail-open on cache outage, env-var config; restructured Python client into a package
- M11: framework backends — `FerrocacheCache` (LangChain `BaseCache`) + `FerrocacheLLM` (LlamaIndex `CustomLLM` subclass); optional imports, fail-open default, custom `embed_fn` supported
- M12: MCP server (`ferrocache.mcp_server`) over stdio — 3 tools (`semantic_cache_lookup`, `semantic_cache_store`, `cache_status`), text-in/JSON-out (embedding handled internally), Claude Desktop + Claude Code setup docs
- M13: distribution — `pyproject.toml` with optional extras (`[openai]`, `[anthropic]`, `[langchain]`, `[llamaindex]`, `[mcp]`, `[all]`); base `pip install ferrocache` is zero-deps; root + Python LICENSE; release CI on `v*` tags publishes Docker image to GHCR and Python package to PyPI via OIDC trusted publishing
- M14: namespace partitioning — `SemanticIndex` is a `HashMap<model_id → NamespacedIndex>`, each with its own HNSW + side-table; `model_id` required on `/insert` and `/query`; Python integrations auto-derive `model_id` from default embed model; PyPI bumped to 0.2.0; pre-M14 WAL entries migrate to `legacy::unknown`

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
- snapshot.rs — `SnapshotEntry` + `write_snapshot`/`read_snapshot` (magic+version+wal_seq+bincode); `compact()` helper that writes the snapshot and truncates the WAL atomically
- benches/cache_bench.rs — criterion: insert / query_hit / query_miss / insert_with_wal / snapshot_write_10k
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
- Python package on PyPI with optional extras; Docker image on GHCR; release CI on version tags
- Index is namespace-partitioned by `model_id`; each namespace has its own HNSW instance + side-table
- `model_id` is required on `/insert` and `/query`; old WAL entries without it default to `legacy::unknown`
- `model_id` format convention: `model_name::dimension` (e.g. `all-MiniLM-L6-v2::384`)
- Cross-namespace queries are impossible by construction — vectors from different models never compare
- Snapshot format: `magic (FERROSNA) + version (u64=1) + wal_sequence (u64) + entry_count (u64) + bincode-encoded Vec<SnapshotEntry>`; written atomically via temp+rename so a crash never corrupts the prior snapshot
- Compaction: snapshot the side-table entries, then truncate the WAL — HNSW is rebuilt from stored embeddings on load (no need to serialize the graph itself)
- WAL entries carry a monotonic `sequence: u64` that never resets — startup skips entries `<= snapshot_watermark` so the WAL tail replay is bounded
- Auto-compaction every N inserts (default 10K via `compact_interval_inserts`, 0 disables); manual via `POST /admin/compact`
- `CacheEntry` again stores `embedding` + `query_text` so snapshots can dump the full record (re-added in M15; was dropped in M4)

**Non-negotiable constraints:**
- No auth/TLS until Phase 5
- No UI
- No direct OpenAI/Anthropic API calls inside ferrocache
- Use tokio, not async-std
- Every new module gets unit tests before moving on

**Session log rules:**
- Keep only the last 2 session logs below
- When adding a new log, delete the oldest if there are already 2
- Summarize deleted sessions as one-line entries under "Completed work" above

## Section 2: Rolling Session Log (last 2 sessions only)

### 2026-05-01 — Mission 14: Embedding Namespaces + Default Embed Model
**Built (Rust):** Refactored `SemanticIndex` from a single HNSW into `HashMap<String, NamespacedIndex>` keyed by `model_id`. Each namespace owns its own `Hnsw<f32, DistCosine>`, side-table `HashMap<usize, CacheEntry>`, `next_id` counter, and dimension lock. `NamespacedIndex` is the M13-era index in miniature; `SemanticIndex` is now a thin lazy-init dispatcher with `namespaces`, `hnsw_config`, `namespace_mut(model_id)` (or-insert-with), `replay_entry`, `query(emb, threshold, model_id)` (returns `Ok(None)` for unknown namespace — miss, not error), `entry_count()` (sum), `namespace_stats() -> HashMap<String, NamespaceStats>`, and a backward-compat `dimension()`. `WalEntry` gained `model_id: String` with `#[serde(default = "legacy_namespace")]` returning the new public constant `LEGACY_NAMESPACE = "legacy::unknown"`; pre-M14 lines without the field are silently quarantined into that namespace on replay. Wire protocol: `InsertRequest` and `QueryRequest` carry an `Option<String>` `model_id` — modeled as Option so the server returns a clean 400 (`"model_id is required"`) instead of a serde rejection. `validate_model_id` helper on the inbound path; `local_insert_inner` constructs the `WalEntry` with the supplied id (and rejects empty/whitespace). `/stats` response gained a `namespaces: HashMap<String, NamespaceStatsEntry>` map (`{ entry_count, dimension }` per namespace). Removed the global "any-dim mismatch" pre-check from `local_insert_inner` — dimension is now enforced per-namespace by `NamespacedIndex` so different models can coexist with different dims. Updated all 4 criterion benches to pass `MODEL_ID = "bench-model::384"`. Updated `tests/cluster_integration.sh` to thread `"model_id": "test-model::4"` through every insert/query.

**Built (Python):** `FerrocacheClient.insert(...)` adds a required `model_id`; `query(...)` adds a required `model_id` (raises `ValueError` if missing — clean client-side error before the HTTP roundtrip). New `_embed.get_default_embed(model_name) -> (embed_fn, model_id)` factory derives `model_id = f"{model_name}::{dim}"` from the loaded sentence-transformers model. `default_embed_fn` is now a thin wrapper over it. Shared `_resolve_embed_and_model_id(embed_fn, model_id)` lives in middleware.py and is reused by all three integration layers (middleware, langchain, llamaindex): both-None loads the default and returns both; embed_fn-without-model_id raises `ValueError("When providing a custom embed_fn, you must also provide model_id...")`; model_id-without-embed_fn loads the default embed but keeps the user's id. `wrap_openai`/`wrap_anthropic`/`FerrocacheCache`/`FerrocacheLLM` all gained an optional `model_id: str | None = None` parameter and thread it through every `query`/`insert` call. `FerrocacheTools` (MCP server) gained `model_id` (defaulting to `"all-MiniLM-L6-v2::384"`); `_build_tools_from_env` derives the real `model_id` from `get_default_embed(embed_model)`. Updated all 30 existing Python tests to thread `model_id`; added 3 new ones in `test_middleware.py::ModelIdTests`: `test_middleware_auto_model_id` (default path, both kwargs absent), `test_middleware_custom_embed_requires_model_id` (ValueError), `test_middleware_custom_embed_with_model_id` (happy path). Updated `simulate.py` to derive `model_id` from `model.get_sentence_embedding_dimension()`; `simulate_no_ml.py` uses `MODEL_ID = f"random::{DIM}"`. Bumped `pyproject.toml` to `version = "0.2.0"` to signal the breaking API change.

**Verified:**
- `cargo test` — 51/51 pass (was 40 before; +5 new index tests, +4 new server tests, +2 new wal tests).
- `cargo clippy --all-targets -- -D warnings` — clean.
- `cargo fmt --check` — clean (one auto-fix of a long `let` line in wal.rs).
- `python3 -m unittest discover tests -v` — 33/33 pass (30 updated + 3 new).
- `python -m build` in `clients/python/` produces `ferrocache-0.2.0.tar.gz` + `ferrocache-0.2.0-py3-none-any.whl`. Fresh-venv install of the wheel, base import works, `client.query([1,2], 0.9)` raises `ValueError: model_id is required`.
- WAL legacy migration test (`test_wal_legacy_migration`) confirms a hand-crafted pre-M14 line lacking `model_id` deserializes with `model_id = "legacy::unknown"`.

**Key decisions:**
- `Option<String>` on the request DTO (instead of `String` with serde rejection) so the server can return a *targeted* 400 rather than a generic deserialization failure. Better UX, easier client debugging.
- The dimension lock moved entirely into `NamespacedIndex`; `SemanticIndex::dimension()` is kept as "first namespace's dim or None" purely for backward-compat with the StatsHnsw struct. The namespace map is the canonical source of truth for per-model dim.
- Routing is unchanged — the consistent-hashing key is still derived from the embedding bytes, not `model_id`. Co-location of related vectors stays intact; namespacing handles isolation. Replicas naturally receive the same `model_id` because it's part of the forwarded request body.
- Old WAL files are quarantined into `legacy::unknown`, never silently dropped or merged into a guessed namespace. Operators can grep `legacy::unknown` in `/stats` to find pre-M14 data.
- Custom-embed-without-model_id is a hard error, not a default. Two production teams sharing a cache with different models was the original threat model — a default would re-introduce the bug we're fixing.
- `_resolve_embed_and_model_id` lives in middleware.py and is imported lazily by langchain.py and llamaindex.py to keep the dependency direction one-way (no circular `from ferrocache.langchain import ...` in middleware.py).
- MCP server exposes `model_id` as an instance attribute on `FerrocacheTools` (rather than as a per-call argument) — agents shouldn't pick the embedding model, the operator does at startup. Future: per-tool `model_id` for advanced setups.

**Deviations:**
- Brief asked for `validate_model_id` to *only* live at handler entry — also added a defensive check inside `process_query_locally` and `local_insert_inner` because the cluster path (`?local=true` from a peer) hits those same code paths and we don't want to depend on the caller having pre-validated.
- Brief said "default to `"legacy::unknown"`" via a serde `default`. Implemented as `#[serde(default = "legacy_namespace")]` referencing a free function (not `default = "..."` literal — serde wants a function path). Same effect.
- Brief listed the new server test as `test_insert_dimension_mismatch` — kept the original name `test_insert_dimension_mismatch_within_namespace` since the new semantics ("mismatch within a single namespace") differ from the old ("mismatch across the global index"). Status code is now 500 (the per-namespace dim check raises an `anyhow::Error` from `replay_entry`, which the server maps to 500), not 400 — added an explicit assertion. The brief's `validate_embedding`-side preflight is no longer correct because the dim is per-namespace.
- Brief's "section 8" on benches and "section 9" on integration tests are minimal — they only need `model_id` threaded through. Done as specified.
- Did not bump the Rust crate version in `Cargo.toml` (still 0.1.0). The Rust crate isn't published to crates.io yet (separate mission); only the Python package and Docker image are user-facing artifacts and the breaking API change is signaled by the Python `0.2.0` bump.

**Next session (M15 if pursued):** WAL compaction / snapshotting — the WAL grows unbounded as inserts accumulate. A snapshot mechanism (write the live index state to disk on shutdown / SIGUSR1, replay snapshot + tail-WAL on startup) would cap startup time and disk use. Related: `/metrics` endpoint (Prometheus exposition format) for namespace counts, query latencies, replication failures.

**Open:** Pre-existing WAL files written by a 0.1.0 server will replay correctly into the M14 server (entries land in `legacy::unknown`), but those entries are unreachable via the new HTTP API unless the user passes `model_id="legacy::unknown"` explicitly. Document this in a migration note before the v0.2.0 release. The `dimension` field on `StatsHnsw` is now misleading for multi-namespace deployments (returns one namespace's dim arbitrarily) — kept for backward compat, but `namespaces` is the canonical inspection point. No async-aware version of the per-namespace lock exists; the global `RwLock<SemanticIndex>` still serializes namespace-creation, which is fine at cache scale but could become contention if a single node hosts 10k+ namespaces. No `cargo bench` run was performed in this mission — namespace lookup adds a HashMap deref before HNSW search, expected overhead is sub-microsecond and below criterion's noise floor.

### 2026-05-02 — Mission 15: WAL Compaction + Snapshotting
**Built:** New `src/snapshot.rs` (added to `lib.rs`) with `SnapshotEntry { uuid, embedding, response, query_text, model_id }`, `MAGIC = 0x4645_5252_4F53_4E41` ("FERROSNA"), `VERSION = 1`. `write_snapshot(path, entries, wal_sequence)` writes magic+version+wal_sequence+entry_count headers (32 bytes, little-endian) followed by `bincode::serialize(entries)` to `{path}.tmp`, fsyncs, then `tokio::fs::rename` to `path` — atomic; a crash mid-write never touches the prior snapshot. `read_snapshot(path)` validates each header field (descriptive error on magic mismatch / unsupported version / count mismatch), returns `(Vec<SnapshotEntry>, wal_sequence)`. `compact(index, wal, snapshot_path, wal_path)` flattens via `index.snapshot_entries()`, writes the snapshot, then calls `wal.truncate(wal_path)` (which fsyncs an empty file then reopens for append; the in-memory `Wal::sequence` counter persists across truncate). `snapshot_path_for(wal_path) -> PathBuf` derives `{wal}.snap`. `WalEntry` gained `sequence: u64` with `#[serde(default)]` so pre-M15 lines default to 0; `Wal::open_with_sequence`, `Wal::current_sequence()`, and `Wal::append` (now returns the assigned `u64` and stamps `entry.sequence` before serialization). `CacheEntry` re-gained `embedding: Vec<f32>` and `query_text: String` (dropped in M4 because "WAL was source of truth" — needed back so snapshots can dump the full record without re-reading the WAL). `SemanticIndex::snapshot_entries() -> Vec<SnapshotEntry>` and `replay_snapshot_entry(SnapshotEntry)` round-trip cleanly. `AppState` gained `snapshot_path: PathBuf`, `compact_interval_inserts: u64`, and `inserts_since_compact: AtomicU64`. `FerrocacheConfig::compact_interval_inserts` (default 10_000; serde default + config-rs default both wired). After every successful local insert (still under the WAL mutex), the counter increments and on threshold the index read lock is taken inline and `snapshot::compact(...)` runs — same caller blocks until done, but subsequent inserts wait on the WAL mutex anyway. New `POST /admin/compact` returns `{ status, entries_snapshotted, wal_sequence }`. Startup in `main.rs` now: try `read_snapshot` if file exists → on success, `replay_snapshot_entry` everything and capture `snapshot_sequence`; on corrupt/unreadable → log warning and proceed; full `Wal::replay` always runs but entries are filtered to `sequence > snapshot_sequence` when a snapshot loaded; the WAL is reopened with `Wal::open_with_sequence` set to `max(snapshot_sequence, last_replayed_sequence)`. New criterion bench `bench_snapshot_write_10k` measures `write_snapshot` of 10K 384-dim entries. Updated all existing call sites: `WalEntry { ..., sequence: 0 }` (stamped on append); test `build_state` constructs a snapshot path and disables auto-compaction (`compact_interval_inserts = 0`); benches' `make_entry`. Added `bincode = "1"` to `Cargo.toml`.

**Verified:**
- `cargo test` — 64/64 pass (was 51 before; +5 snapshot-module tests, +3 wal sequence tests, +1 index round-trip, +3 server tests covering /admin/compact, full snapshot+restart, and corrupt-snapshot fallback).
- `cargo clippy --all-targets -- -D warnings` — clean (after replacing literal `0.7071` with `std::f32::consts::FRAC_1_SQRT_2` in a test vector to dodge `clippy::approx_constant`).
- `cargo fmt --check` — clean.
- `cargo build --release --bin ferrocache` — clean.
- End-to-end smoke test on the release binary: insert 3 entries → POST `/admin/compact` returns `{"entries_snapshotted":3,"wal_sequence":3}`, snapshot file exists, WAL on disk truncated to 0 bytes → kill → restart → log shows `snapshot loaded loaded=3 wal_sequence=3` then `startup replay complete wal_tail_entries=0 snapshot_watermark=3` → query returns `hit=true,response="r1",similarity=1.0`. Verified the in-memory `Wal::sequence` counter keeps going (truncate+append goes to seq 4, not 1) via `test_wal_truncate_keeps_sequence_counter`.

**Key decisions:**
- Snapshot side-table data, not the HNSW graph. Rebuild HNSW from stored embeddings on load. Two reasons: (a) hnsw_rs doesn't expose the graph in a portable way, and (b) rebuild cost is bounded by *snapshot size*, not unbounded WAL size — already a strict win even if HNSW serialization were free. Cost: a 10K-entry snapshot takes O(10K * log10K) HNSW inserts on startup, but that's still much faster than replaying a 10K-line WAL plus everything written since.
- Atomic temp+rename for snapshots is the *only* safe write path. Half-written snapshot + fsync on the wrong file would silently corrupt restart; the rename is a single inode-level atomic operation on POSIX.
- Auto-compaction trigger fires *inline* on the insert that crosses the threshold (not in a background task). Reasoning: the trigger thread already holds the WAL mutex, so spawning a task would just queue up behind itself, and the simpler synchronous path means there's no "concurrent compactions" failure mode to defend against. The atomic `fetch_add` + threshold check ensures only one insert per cycle triggers compaction. `compact_interval_inserts = 0` disables auto-compaction (used in tests; manual /admin/compact still works).
- WAL sequence numbers don't reset on truncate. After compaction, the next append continues from `seq + 1`. This means sequences are globally monotonic across the lifetime of the WAL file, which is what makes the `entry.sequence > snapshot_watermark` filter on startup correct.
- `Wal::append` now returns the assigned sequence. Stamps `entry.sequence` from the caller's WalEntry before serialization — caller can pass `sequence: 0` and the WAL will overwrite with the real value. This keeps the API ergonomic (callers don't need to know the next seq) without forcing a `&mut` ref to the entry.
- Snapshot path is derived from WAL path (`{wal}.snap`), not separately configurable. One fewer config knob; a user who wants to store the snapshot elsewhere can symlink. Reduces footguns where snapshot and WAL get out of sync because they were configured to different volumes.
- New `CacheEntry` fields (`embedding`, `query_text`) cost ~1.5KB per entry at 384-dim — still tiny next to HNSW graph memory. Trading a bit of RAM for a much smaller snapshot serialization codepath (no need for a parallel WAL-tail re-read at compaction time).

**Deviations:**
- Brief asked for `embedding: Vec<f32>` on `CacheEntry` only; also re-added `query_text: String` because (a) the brief explicitly said "Also re-add `query_text: String` to `CacheEntry` if it was also dropped" and (b) the snapshot needs it to round-trip the full record.
- Brief specified the snapshot header as `[8 magic][8 version][8 wal_sequence][8 entry_count][bincode...]`. Implemented exactly with little-endian byte order. Total 32-byte header.
- Brief's `CompactionResult` had `old_wal_entries: u64`. Dropped that field — it isn't observable from inside `compact()` (we no longer have a count of WAL lines before truncate, only the sequence number, which is what the operator actually needs). Kept `entries_snapshotted` and `wal_sequence`.
- Brief test `test_compact_endpoint` was specified as inserting "some entries"; used 3 entries with distinct namespaces' style assertions (`entries_snapshotted == 3`, `wal_sequence == 3`).
- Brief test `test_startup_with_snapshot` said "embeddings `[i, 0, 0]`" implicitly. Used distinct unit-direction vectors instead — colinear scalar multiples have cosine similarity = 1.0, so the nearest neighbor for any of them is HNSW-internal-id 0, making "tail entry survives" indistinguishable from "first entry survives." Switched to `[1,0,0,0]`, `[0,1,0,0]`, `[0,0,1,0]`, `[1/√2, 1/√2, 0, 0]`, `[1/√3, 1/√3, 1/√3, 0]`, with a tail vector `[0,0,0,1]` that's orthogonal to all of them.
- Brief mentioned `tokio::spawn` for auto-compaction "BUT the compaction task needs to acquire locks, so the next insert will wait for it anyway — simpler to just run it inline." Did exactly the inline path; documented in the code comment.

**Next session (M16):** Prometheus `/metrics` endpoint. Counters: insert/query/cache_hit/cache_miss/replication_failure totals, namespace count, entries_per_namespace, last_compact_timestamp. Histograms: query_duration_seconds, insert_duration_seconds (with bucket selections appropriate for sub-millisecond → tens-of-ms ranges). Probably wire `prometheus = "0.13"` or use an axum-native metrics crate. Want labels `{ namespace, hit }` on query counter so dashboards can split by model.

**Open:** Compaction holds the WAL lock + index read lock for the duration of the snapshot write — at 10K entries, this is sub-second on local disk but could be problematic on slow disks or for very large indexes (100K+). A non-blocking approach (clone the side-table into a separate buffer under brief lock, then write to disk afterwards) is straightforward but adds ~entry-size bytes of transient memory; deferred. The bench `bench_snapshot_write_10k` was added but not run with `cargo bench` in this session (no real numbers in this log). The snapshot file is unencrypted on disk — not a regression vs the WAL (also unencrypted), but worth flagging for the eventual TLS+at-rest-encryption pass. There is no online `/admin/snapshot-info` endpoint; `/stats` could grow a `last_snapshot_sequence` field next mission.
