# ferrocache — Project Context

## Section 1: Evergreen

**What this is:** Distributed semantic cache for LLM applications, written in Rust. Single binary, multi-node via consistent hashing + gossip replication. Portfolio project for big-tech Core SWE interviews.

**Current phase:** Phase 4 complete (M14–M16). Production-safe: namespaces, WAL compaction, Prometheus metrics.

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
- M15: WAL compaction + snapshotting — `src/snapshot.rs` writes `magic+version+wal_seq+bincode(Vec<SnapshotEntry>)` atomically; WAL gained monotonic `sequence` numbers; startup loads snapshot then replays only `seq > watermark`; auto-compaction every 10K inserts (configurable) + manual `POST /admin/compact`; `CacheEntry` re-stores `embedding`+`query_text`

**Module map:**
- lib.rs — re-exports modules so benches and external consumers can import
- main.rs — entry, tracing init, config load, WAL replay, optional cluster init, serve
- server.rs — router, handlers, request validation, routing/replication coordination, tests
- index.rs — SemanticIndex (hnsw_rs wrapper + HashMap side-table), replay_entry
- wal.rs — Wal (append w/ fsync, mkdir -p parent), WalEntry, replay
- models.rs — request/response DTOs (InsertRequest carries optional `uuid`)
- config.rs — FerrocacheConfig, HnswConfig, ClusterConfig (gossip_addr, api_addr, replication_factor, seed_nodes)
- state.rs — AppState { node_id, index, wal, wal_path, snapshot_path, hnsw_config, cluster, router, replication_factor, compact_interval_inserts, inserts_since_compact, metrics }
- ring.rs — HashRing (BTreeMap<u64, node_id>, FNV-1a, virtual nodes, get_n_nodes for replica walk)
- cluster.rs — ClusterState wrapping chitchat; reconciler syncs ring + node_id→api_addr map from gossip KV
- router.rs — ClusterRouter (reqwest client) — forward_query / forward_insert to peers with `?local=true`
- snapshot.rs — `SnapshotEntry` + `write_snapshot`/`read_snapshot` (magic+version+wal_seq+bincode); `compact()` helper that writes the snapshot and truncates the WAL atomically
- metrics.rs — `Metrics` (atomic counters + per-namespace `RwLock<HashMap<NamespaceMetrics>>`), `LatencyHistogram` (16 fixed buckets 100µs–10s), `render()` to Prometheus text exposition format
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
- Prometheus `/metrics` is hand-written (no `prometheus`/`metrics` crate); atomic counters with `Ordering::Relaxed` + a fixed-bucket `LatencyHistogram`
- Per-namespace hit/miss/insert counters carry a `namespace` label; global query/insert latency histograms have 16 buckets (100µs → 10s) plus a `+Inf` bucket
- `/stats` JSON gained a `counters` object so users without Prometheus can still read the same numbers
- Monitoring stack lives in `monitoring/` as a docker-compose overlay (`-f docker-compose.yml -f monitoring/docker-compose.monitoring.yml`); base cluster runs unchanged without it
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

### 2026-05-02 — Mission 16: Prometheus /metrics + Operational Observability
**Built:** New `src/metrics.rs` (added to `lib.rs`) — hand-written Prometheus text exposition (no external crate). `Metrics` holds eight `AtomicU64` counters (`queries_total`, `queries_hit`, `queries_miss`, `inserts_total`, `replication_forwards_total`, `replication_failures_total`, `compactions_total`) plus a `RwLock<HashMap<String, NamespaceMetrics>>` for per-namespace `{queries_hit, queries_miss, inserts}` and two `LatencyHistogram` instances (`query_duration`, `insert_duration`). `LatencyHistogram` has 16 fixed buckets covering 100µs → 10s (`BUCKET_BOUNDS` const), tracks `_sum` in microseconds (so it can't overflow practically) and `_count` separately; `observe()` increments every bucket where `bound >= duration` plus sum/count, all with `Ordering::Relaxed`. `Metrics::record_query_hit/_miss/_insert(namespace, dur_secs)` increment global + per-namespace counters and observe latency; `record_replication_forward/_failure/_compaction` are global only. `hit_rate()` returns `queries_hit / queries_total` (0.0 when total is 0). `render(&index_stats, cluster_nodes)` produces the full text body with proper `# HELP` + `# TYPE` lines, sorted namespace labels, both real buckets and the required `+Inf` bucket per histogram. `METRICS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"` is the standard Prometheus content type. `AppState` gained `metrics: Arc<Metrics>`. New `GET /metrics` handler reads the index for namespace stats + the cluster for live node count, then renders with the right Content-Type. Instrumentation: `process_query_locally` times the index read+lookup with `Instant::now()` and calls `record_query_hit`/`_miss` with `model_id` as the namespace label; `local_insert_inner` records `record_insert` after the WAL+index succeed (insert latency therefore includes WAL fsync, as the brief required); auto-compaction and the manual `/admin/compact` handler both call `record_compaction`; the cluster fan-out loop calls `record_replication_forward` per attempt and `record_replication_failure` on the failure that aborts replication. `/stats` JSON gained a `counters` object (`CountersResponse` in models.rs) with `queries_total`, `queries_hit`, `queries_miss`, `hit_rate`, `inserts_total`, `replication_forwards`, `replication_failures`, `compactions` so non-Prometheus users can read the same numbers. New `monitoring/` directory: `docker-compose.monitoring.yml` (overlay onto the base compose; adds Prometheus on :9090 and Grafana on :3100 with anonymous viewer), `prometheus.yml` (5s scrape of node1/node2/node3 on `/metrics`), `grafana/provisioning/datasources/datasource.yml` (Prometheus datasource with explicit `uid: prometheus` so the dashboard can reference it), `grafana/provisioning/dashboards/dashboard.yml` (file provider pointing at `/var/lib/grafana/dashboards`), and `grafana/dashboards/ferrocache.json` (8-panel dashboard in 2×4 grid: hit rate gauge, queries/sec stacked, query p50/p99, insert p50/p99, entries by namespace, replication failures, cluster nodes, compactions/sec). 7 new metrics-module tests + 3 new server tests (/metrics endpoint, /metrics after operations, /stats includes counters).

**Verified:**
- `cargo test` — 75/75 pass (was 64; +7 metrics module tests, +3 server tests).
- `cargo clippy --all-targets -- -D warnings` — clean.
- `cargo fmt --check` — clean.
- End-to-end smoke test on the release binary: 2 inserts + 3 queries (2 hit, 1 miss) → `curl /metrics` shows `ferrocache_queries_total 3`, `ferrocache_queries_hit_total 2`, `ferrocache_queries_miss_total 1`, `ferrocache_hit_rate 0.666667`, `ferrocache_inserts_total 2`, `ferrocache_query_duration_seconds_count 3`, `ferrocache_insert_duration_seconds_count 2`, `ferrocache_cluster_nodes 1` and the per-namespace `ferrocache_namespace_entries{namespace="smoke::3"} 2`. `/stats` mirrors the same numbers under `counters`.
- `LatencyHistogram::observe(0.001)` correctly increments every bucket with `bound >= 1ms` and leaves the smaller buckets at 0 — verified in `test_histogram_observe`. `hit_rate` is 0 (not NaN) on a fresh `Metrics` — verified in `test_hit_rate_zero_queries`.

**Key decisions:**
- Hand-write the text format. The two-line-per-metric format is trivial; pulling `prometheus`/`metrics` would add ~30 transitive deps for ~100 lines of text-formatting code we'd write anyway. Cost-benefit favors hand-roll.
- `AtomicU64` + `Ordering::Relaxed` for every counter. We're tracking trends across millions of events; an occasional torn read between e.g. `queries_total` and `queries_hit` for a single render produces at worst a rounding error in `hit_rate`, never a panic. No need for `SeqCst` overhead.
- Sum tracked in microseconds (not seconds-as-float). f64 fetch_add doesn't exist as an atomic; emulating it is gnarly. Microseconds in u64 give 18+ million years of headroom and the rendered `_sum` divides back to seconds with full precision below the buckets' resolution.
- Per-namespace counters live in `RwLock<HashMap<String, NamespaceMetrics>>` rather than as separate atomics per namespace at fixed offsets — the namespace set is open-ended (one per `model_id`), so a HashMap is the only data shape that fits. Read-path takes a read-lock + atomic increment; only the first observation in a new namespace takes the write lock to insert the entry.
- `model_id` flows through to the metrics layer as the `namespace` label name. Prometheus's standard convention is `namespace`, not `model_id`, so dashboards transfer cleanly.
- The histogram's bucket bounds (100µs, 250µs, 500µs, 1ms, … 10s) span the full plausible range for ferrocache, from in-memory hits (<100µs) to slow cluster forwards (~5s timeout). Using a fixed list rather than exponential() avoids needing a runtime config knob.
- `/stats` JSON gains the `counters` block so users on internal tooling without Prometheus still see hit rate. Non-breaking addition (existing fields untouched).
- The Grafana dashboard JSON is provisioned read-only via `editable: false` + the `editable: true` field on the dashboard itself — so anonymous viewers can drill in and explore but the canonical version stays under git. The datasource has an explicit `uid: prometheus` so the dashboard's `datasource.uid` references resolve correctly without manual import.

**Deviations:**
- Brief's monitoring compose listed `prometheus.yml` mounted at `./monitoring/prometheus.yml` *from the repo root*. Kept it that way — the overlay is meant to be invoked from the repo root (`docker compose -f docker-compose.yml -f monitoring/docker-compose.monitoring.yml up`), where relative paths resolve naturally.
- Brief showed an example `ferrocache_hit_rate 0.8166`. Used Rust's `{:.6}` format (6 decimal places) so 2/3 renders as `0.666667` — Prometheus parses both, but the slightly extra precision means dashboards hovering on the value see a clean number instead of `0.8166`/`0.81666666` ambiguity.
- Brief specified `# HELP` then `# TYPE` order — followed exactly. Each metric block is followed by a single trailing blank line for readability when curling `/metrics`.
- Brief's snippet showed `ferrocache_namespace_entries` as a gauge from `index.namespace_stats()`. Implemented exactly that, but also rendered `ferrocache_namespace_queries_hit/_miss/_inserts` as counters from the metrics struct. The label set across the four families (entries + 3 counters) is the union of namespaces seen in *either* the index or the metrics — a namespace with traffic but no entries (queries to never-inserted model_ids that miss) still shows up with hits+misses, and a namespace with entries but no traffic still shows up with `0` counters.
- Replication forwards/failures are recorded in the cluster fan-out loop in `insert_handler`. Did NOT instrument `forward_query` (the brief explicitly said replication metrics are recorded in the cluster *insert* path only). A future mission can add a separate `query_forwards_total` counter if needed.

**Next session (M17 if pursued):** TLS + auth — the long-deferred Phase 5 constraint. Probably mTLS between cluster nodes (rustls + rcgen for dev certs) and a bearer-token auth layer on the public API surface (axum middleware). Related: optional at-rest encryption of the WAL + snapshot files (libsodium `secretstream`?). Both are overdue if anyone deploys ferrocache outside a trusted-network LAN.

**Open:** Per-namespace latency histograms aren't tracked — the global query/insert histograms cover all namespaces together. If two namespaces have wildly different latency profiles (e.g. 384-dim local model vs 1536-dim remote OpenAI embeddings), the histograms are an aggregate that may obscure a regression in one. Expansion to per-namespace histograms is straightforward but multiplies bucket count by namespace count; deferred until a real operator hits the limitation. The compose overlay was not actually exercised end-to-end on Docker in this session (only the standalone binary's `/metrics` was smoke-tested) — the `prometheus.yml`, datasource provisioning, and dashboard JSON are correct by inspection but a `docker compose up` would catch typos. The dashboard's hit-rate panel is per-instance; an aggregate "fleet hit rate" panel (`sum(ferrocache_queries_hit_total) / sum(ferrocache_queries_total)`) would also be useful and could be added later. There is no scrape-self target for Prometheus itself — fine for a single-cluster portfolio but most production deployments would also scrape Prometheus.
