# ferrocache — Project Context

## Section 1: Evergreen

**What this is:** Distributed semantic cache for LLM applications, written in Rust. Single binary, multi-node via consistent hashing + gossip replication. Portfolio project for big-tech Core SWE interviews.

**Current phase:** Phase 5 in progress. M17 done (bearer token auth). Next: M18 — mTLS between cluster nodes.

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
- M16: Prometheus `/metrics` (hand-written text exposition, no `prometheus` crate) — atomic counters with per-namespace `RwLock<HashMap>`, fixed 16-bucket `LatencyHistogram` (100µs → 10s), `/stats` JSON gains `counters` block, monitoring compose overlay (Prometheus + Grafana on :9090/:3100) with provisioned datasource and 8-panel ferrocache dashboard

**Module map:**
- lib.rs — re-exports modules so benches and external consumers can import
- main.rs — entry, tracing init, config load, WAL replay, optional cluster init, serve
- server.rs — router, handlers, request validation, routing/replication coordination, tests
- index.rs — SemanticIndex (hnsw_rs wrapper + HashMap side-table), replay_entry
- wal.rs — Wal (append w/ fsync, mkdir -p parent), WalEntry, replay
- models.rs — request/response DTOs (InsertRequest carries optional `uuid`)
- config.rs — FerrocacheConfig, HnswConfig, ClusterConfig (gossip_addr, api_addr, replication_factor, seed_nodes)
- state.rs — AppState { node_id, index, wal, wal_path, snapshot_path, hnsw_config, cluster, router, replication_factor, compact_interval_inserts, inserts_since_compact, metrics, auth_token }
- auth.rs — `auth_middleware` (axum `from_fn_with_state`); constant-time `subtle::ConstantTimeEq`; `/health` + `/metrics` exempt; only installed when `auth_token` is non-empty
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
- Bearer token auth via `FERROCACHE_AUTH_TOKEN` env var; disabled when unset (opt-in, backward compatible)
- Auth exempts `/health` and `/metrics` (load balancers and Prometheus need unauthenticated access)
- Token comparison uses `subtle::ConstantTimeEq` to prevent timing attacks; the token value is never logged at any level
- Inter-node replication forwards include the same bearer token (shared-secret cluster model — all nodes carry the same `FERROCACHE_AUTH_TOKEN`)

**Non-negotiable constraints:**
- Auth is opt-in via `FERROCACHE_AUTH_TOKEN` (M17); TLS deferred to M18
- No UI
- No direct OpenAI/Anthropic API calls inside ferrocache
- Use tokio, not async-std
- Every new module gets unit tests before moving on

**Session log rules:**
- Keep only the last 2 session logs below
- When adding a new log, delete the oldest if there are already 2
- Summarize deleted sessions as one-line entries under "Completed work" above

## Section 2: Rolling Session Log (last 2 sessions only)

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

### 2026-05-02 — Mission 17: Bearer Token Auth on the Public HTTP API
**Built:** New `src/auth.rs` (added to `lib.rs`) — `AuthToken { value: String }` + `auth_middleware` axum `from_fn_with_state` handler. Auth path: exempt `/health` and `/metrics` early; pull the `authorization` header; require `"Bearer "` prefix; constant-time `subtle::ConstantTimeEq` against the configured token (length-checked first because `ct_eq` requires equal-length inputs and token length isn't secret); 401 + `{"error":"unauthorized"}` JSON on every failure path. `subtle = "2"` added to `Cargo.toml`. `FerrocacheConfig` gained `auth_token: Option<String>` (`#[serde(default)]`, env-driven, no builder default — relies on serde default like the existing `node_id`). `AppState` gained `auth_token: Option<String>` plumbed through `AppState::new`. `server::build_router` now reads `state.auth_token` and, only when it's `Some(non-empty)`, layers the middleware via `axum::middleware::from_fn_with_state(Arc::new(AuthToken { value }), auth_middleware)`. `ClusterRouter::new` now takes `auth_token: Option<String>` and stamps `Authorization: Bearer <token>` on every `forward_query` / `forward_insert` reqwest call (helper `with_auth(RequestBuilder)`). `main.rs` swapped the `?config` debug log for explicit field-by-field tracing so the token never lands in logs even at trace level, and emits exactly one of `"bearer token auth enabled"` / `"bearer token auth disabled (FERROCACHE_AUTH_TOKEN not set)"`. Python: `FerrocacheClient.__init__` accepts `auth_token=None` and falls back to `os.environ["FERROCACHE_AUTH_TOKEN"]`; `_headers()` helper centralizes the bearer-stamping logic for both `_get` and `_post`. `wrap_openai`, `wrap_anthropic`, `FerrocacheCache`, `FerrocacheLLM` all gained `auth_token: str | None = None` kwargs that thread through to `FerrocacheClient(...)`. `FerrocacheTools` (MCP) accepts `auth_token` and pokes it onto an existing client only if that client doesn't already carry one (lets the MCP server inherit `FERROCACHE_AUTH_TOKEN` via the client's env-fallback path automatically). Docs: `ferrocache.toml` gained a commented `auth_token =` line; `docker-compose.yml` gained a commented `FERROCACHE_AUTH_TOKEN` env entry on each of the 3 nodes; README gained a `## Security` section after `## Configuration` with curl examples and a Python snippet; `tests/cluster_integration.sh` gained a "Test 6" sanity check that `/health` and `/metrics` always work without auth (proper auth-on integration testing requires a separate compose run with the env exported on every node — flagged as follow-up).

**Verified:**
- `cargo test` — 88/88 pass (was 75; +12 auth tests covering the 9 brief scenarios + insert/stats/admin-compact require-auth, +1 router test that the bearer header lands on the wire when configured).
- `cargo clippy --all-targets -- -D warnings` — clean.
- `cargo fmt --check` — clean (after one `let _ = … is_some_and(...)` formatting pass).
- `cargo build --release --bin ferrocache` — clean.
- `python3 -m unittest discover tests` — 39/39 pass (was 33; +6 auth tests covering header on/off, env fallback, POST headers, middleware flows token, MCP flows token; +2 existing config tests touched up to assert `auth_token=None` in the new constructor signature).
- End-to-end smoke test on the release binary (`FERROCACHE_AUTH_TOKEN=secret`): `/health` 200 unauthenticated, `/metrics` 200 unauthenticated, `/query` no header → 401, `/query` with `Bearer secret` → 200, `/query` with `Bearer wrong` → 401, `/stats` no header → 401, `/stats` with token → 200. `grep -i secret <log>` returns nothing — confirmed the token is never logged.
- Disabled-mode smoke (no env var): `/query` with no header → 200, log line `"bearer token auth disabled (FERROCACHE_AUTH_TOKEN not set)"`.

**Key decisions:**
- `subtle::ConstantTimeEq` only — never `==`. The crate is 30 lines of public API and exists for exactly this case. The length-check before `ct_eq` is required (the API panics on unequal lengths) and is fine: token length is not secret. The token *value* is.
- `from_fn_with_state(Arc<AuthToken>, ...)` rather than tower-http's `ValidateRequestHeaderLayer` or a hand-rolled `tower::Layer`. Reasoning: the route exemption logic (`/health`, `/metrics`) is the only branch we need, and it's three lines inline; pulling another crate or implementing `Service`/`Layer` boilerplate is pure overhead. The closure-style middleware is the idiomatic axum 0.7 pattern.
- Auth is layered *after* `with_state(state)` so the data handlers continue to receive the original `AppState`, while the middleware gets its own `Arc<AuthToken>` state. The two states are independent; there's no need to thread the token through `AppState`'s state machinery (it's stored there only so `build_router` can read it once).
- Empty-string token treated as disabled. Both `config.auth_token == None` and `config.auth_token == Some("")` skip the middleware install. Reason: env vars in `docker-compose` are easy to accidentally set to `""` (e.g., `FERROCACHE_AUTH_TOKEN: ""`), and silently disabling auth on an empty string is more useful than failing at startup or, worse, requiring an empty bearer token.
- `ClusterRouter::new(auth_token: Option<String>)` — the cluster's inter-node calls go through the same public API surface as external clients. There is no separate "cluster-internal token"; all nodes share the same `FERROCACHE_AUTH_TOKEN`. This keeps the threat model clean: anyone who can present the token can do anything.
- Replaced `tracing::info!(?config, ...)` (Debug-derived) with explicit field-by-field logging. The Debug impl is auto-derived from `#[derive(Debug)]` on `FerrocacheConfig`; trying to skip-Debug a single field would mean either a custom `Debug` impl (boilerplate) or wrapping the whole struct in `secrecy::Secret<...>`. Field-by-field logging is one line longer and provably leak-free.
- Python: `auth_token=None` fallback to `os.environ.get("FERROCACHE_AUTH_TOKEN")` lives *in the client*, not in every wrapper. So `wrap_openai(...)` without an explicit `auth_token=` and without `FERROCACHE_AUTH_TOKEN` set behaves exactly like before; setting the env var or passing the kwarg both light up auth at the SDK boundary. The `_resolve_*` helpers in `middleware.py` weren't extended for auth because the env-fallback already lives one layer down.
- `FerrocacheTools.__init__` mutates `client.auth_token` only when the client doesn't already carry one. Reason: in production the typical wiring is `_build_tools_from_env() → FerrocacheClient(url) → FerrocacheTools(client, ...)` and the *client* picks up the env var automatically, so the explicit `auth_token=` kwarg is redundant. But for tests / direct construction we still want a way to set it on the client retroactively without rebuilding. The "don't overwrite" guard avoids surprise when both are passed.

**Deviations:**
- Brief showed `from typing import Any` and a `State(token): State<Arc<AuthToken>>` extractor in the middleware. Implemented exactly that. Brief's `unauthorized_response()` returned `(StatusCode::UNAUTHORIZED, Json(ErrorResponse { error: ... })).into_response()` — used `Json(ErrorResponse { ... })` (the existing one in `models.rs`) rather than introducing a new type.
- Brief said "401 (not 403) on missing or wrong token" — followed exactly. All four failure paths (no header, bad utf-8, missing `Bearer ` prefix, wrong/short token) return the same `unauthorized()`; differentiating them in the response would leak which step failed and is also pointless for the caller (they fix the same thing in all cases).
- Brief wrote `if expected.len() != provided_bytes.len() || expected.ct_eq(provided_bytes).unwrap_u8() != 1`. Implemented exactly with the short-circuit OR; the length check happens first because `ct_eq` panics on unequal lengths.
- Brief's bearer prefix check used `strip_prefix("Bearer ").unwrap_or("")`. Used a `match` instead so a missing prefix returns 401 directly rather than treating the whole header as the token. Net effect identical — `Basic abc123` becomes the empty token, which never compares equal to a non-empty configured token — but the explicit branch is easier to read and the test `test_auth_rejects_non_bearer_scheme` documents the behavior.
- Brief specified `FerrocacheClient(base_url, auth_token=None)` and `if auth_token: headers["Authorization"] = ...`. Refactored the per-request header construction into a `_headers(content_type)` helper because the client has both GET and POST paths and duplicating the bearer-stamp inline twice was uglier than the helper.
- Brief's Python integration wrappers all had to thread `auth_token` through. Implemented for all four (`wrap_openai`, `wrap_anthropic`, `FerrocacheCache`, `FerrocacheLLM`) but for `FerrocacheTools` (MCP), routed it through the underlying client's `auth_token` attribute rather than adding a parallel field on `FerrocacheTools` itself — keeps the contract "the bearer token lives on the client" consistent across the codebase.
- Brief's `tests/cluster_integration.sh` "Test 6" was specified as a single `/health` check. Added a second assertion that `/metrics` also returns the proper Content-Type unauthenticated, since that's the other always-public route and is just as important to prove.

**Next session (M18 if pursued):** mTLS between cluster nodes. `rustls` + `rcgen` for dev cert generation; `axum-server` for TLS termination on the listener; `reqwest::Client::builder().add_root_certificate(...)` on the cluster router. Probably `cluster.tls_enabled`, `cluster.cert_path`, `cluster.key_path`, `cluster.ca_path` config keys. Same opt-in pattern as M17 — when `tls_enabled = false`, behavior is identical to today. Per-node certs verified against a shared CA (so adding a node = signing a new cert, not redeploying everyone). Public API TLS stays out of scope (handled by reverse proxy in production).

**Open:** Auth-on integration testing against the Docker compose cluster requires exporting `FERROCACHE_AUTH_TOKEN` on every node and re-running `tests/cluster_integration.sh` against an authenticated cluster — the current script only sanity-checks that `/health` and `/metrics` stay public. A second compose file (e.g. `docker-compose.auth.yml` overlay) plus a parallel integration test script would close the gap; deferred. The `subtle` crate is the only new transitive dep (~1 file, 0 transitive deps of its own) — checked; clean. There's no rate-limiting on failed auth attempts: a constant-time-comparison-safe response still lets an attacker grind tokens at line-rate. Real production deployments behind a reverse proxy would terminate that at the proxy (nginx `limit_req`) — flagged but out of scope. The token is loaded into memory at startup and never zeroized; if process memory is dumped, the token leaks. `secrecy::Secret<String>` would help but adds wrapper noise everywhere the token is read; deferred until there's a concrete threat model that demands it. There is no per-route auth scope — `/admin/compact` and `/query` carry the same authority. A "read-only token" / "admin token" split is a reasonable next step but multiplies the config surface; the brief explicitly excluded this and we honored that.
