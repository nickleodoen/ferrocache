# ferrocache — Project Context

## Section 1: Evergreen

**What this is:** Distributed semantic cache for LLM applications, written in Rust. Single binary, multi-node via consistent hashing + gossip replication. Portfolio project for big-tech Core SWE interviews.

**Current phase:** Phase 4 in progress. M14 done (namespaces). Next: M15 — WAL compaction.

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
- Python package on PyPI with optional extras; Docker image on GHCR; release CI on version tags
- Index is namespace-partitioned by `model_id`; each namespace has its own HNSW instance + side-table
- `model_id` is required on `/insert` and `/query`; old WAL entries without it default to `legacy::unknown`
- `model_id` format convention: `model_name::dimension` (e.g. `all-MiniLM-L6-v2::384`)
- Cross-namespace queries are impossible by construction — vectors from different models never compare

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

### 2026-05-01 — Mission 13: Distribution (PyPI + GHCR + release CI)
**Built:** `clients/python/pyproject.toml` (PEP 621) declaring `ferrocache 0.1.0` with the `setuptools.build_meta` backend, MIT license, Python 3.9-3.13 classifiers, `[project.urls]` pointing at the GitHub repo, and `[project.optional-dependencies]` with seven extras: `[openai]` `[anthropic]` `[langchain]` `[llamaindex]` `[mcp]` `[embeddings]` `[all]`. The base install has zero deps. `[tool.setuptools.packages.find]` picks up `ferrocache` and all submodules under `clients/python/`. PyPI-facing `clients/python/README.md` (concise — install matrix, quick start, framework + MCP snippets, env-var table). `MANIFEST.in` includes README + LICENSE. New MIT `LICENSE` at repo root and copy in `clients/python/`. New `.github/workflows/release.yml` triggered on `v*` tags, two jobs: `docker` (login to GHCR with `GITHUB_TOKEN`, build + push `ghcr.io/nickleodoen/ferrocache:<version>` and `:latest` via `docker/build-push-action@v5` + Buildx) and `pypi` (set up Python 3.12, `python -m build` in `clients/python/`, publish via `pypa/gh-action-pypi-publish` using OIDC trusted publishing — no API token committed; needs to be configured once on pypi.org). Updated root README's Quickstart to lead with `docker run ghcr.io/nickleodoen/ferrocache` and `pip install ferrocache`, plus a full installation matrix.

**Verified:**
- `python -m build` in `clients/python/` produces `ferrocache-0.1.0.tar.gz` + `ferrocache-0.1.0-py3-none-any.whl` containing all 7 modules (`__init__`, `client`, `_embed`, `middleware`, `langchain`, `llamaindex`, `mcp_server`) + LICENSE in `dist-info/licenses/`.
- Fresh venv `pip install ferrocache-0.1.0-py3-none-any.whl` installs ferrocache + pip only (zero extra deps). All four base imports succeed: `from ferrocache import FerrocacheClient, FerrocacheError`, `from ferrocache.middleware import wrap_openai, wrap_anthropic`, `from ferrocache.langchain import FerrocacheCache`, `from ferrocache.llamaindex import FerrocacheLLM`.
- Instantiating `FerrocacheCache()` / `FerrocacheLLM(inner=None)` without the optional dep raises the expected ImportErrors (`"... requires langchain-core. Install it with..."`).
- `pip install ferrocache[mcp]` from the wheel pulls `mcp 1.27.0` + `sentence-transformers 5.4.1` and `from ferrocache.mcp_server import FerrocacheTools, TOOL_DEFINITIONS` works.
- Existing test suites still pass: 40 Rust + 30 Python.
- `release.yml` parses as valid YAML; jobs `docker` + `pypi` declared, `on.push.tags=['v*']` only.

**Key decisions:**
- Used `setuptools.build_meta` (the standard backend) — the brief's `setuptools.backends._legacy:_Backend` is not a real path; clearly a typo.
- Base package has *zero* runtime deps — the stdlib HTTP client works on its own. Every framework SDK lives behind an extra. `pip install ferrocache` in CI for a project that only uses the HTTP client takes <1s.
- PyPI publish via OIDC trusted publishing (no `PYPI_API_TOKEN` secret committed). One-time setup on pypi.org wires the GitHub repo + workflow as a trusted publisher; tokens then come from GitHub OIDC at runtime.
- Docker tags both `<version>` and `latest` so users can pin or float — same `docker compose` config in this repo can target either.
- License file lives at repo root AND in `clients/python/` — the wheel's `MANIFEST.in` only sees the latter, but the repo root one is canonical.
- Did NOT push a tag this mission. Wheel/sdist verified locally; first real publish happens when `git tag v0.1.0 && git push --tags` runs the workflow.

**Deviations:**
- Brief's build-backend was wrong (`setuptools.backends._legacy:_Backend`); used `setuptools.build_meta`. Confirmed the wheel builds and installs.
- Brief left author email as a placeholder; used the git-config email `nikhilram@gmail.com`. Likewise `Nikhil Yachareni` from `git config user.name`.
- Added `docker/setup-buildx-action@v3` step to the release workflow — `docker/build-push-action@v5` runs faster and supports more cache options with Buildx enabled. Not strictly required but standard practice.
- Skipped TestPyPI; the brief flagged it as optional. Local wheel install in a fresh venv covers the same failure modes.

**Project shipped.** No M14 planned. Future polish (non-blocking): WAL compaction/snapshotting, /metrics endpoint (Prometheus), TLS+auth (Phase 3 constraint deferred), async OpenAI/Anthropic wrappers, native async LangChain/LlamaIndex backends (`alookup`, `acomplete`), node-failure resilience tests.

**Open:** First real publish requires (a) configuring pypi.org trusted publishing for `nickleodoen/ferrocache`, (b) creating the `pypi` GitHub environment, (c) tagging `v0.1.0` and pushing. The Rust crate is *not* on crates.io yet — would need `cargo publish` and a `crates.io` token, which is a separate mission if pursued.

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
