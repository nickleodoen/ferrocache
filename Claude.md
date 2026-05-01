# ferrocache — Project Context

## Section 1: Evergreen

**What this is:** Distributed semantic cache for LLM applications, written in Rust. Single binary, multi-node via consistent hashing + gossip replication. Portfolio project for big-tech Core SWE interviews.

**Current phase:** All phases complete (M1–M13). Project is shipped.

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

### 2026-05-01 — Mission 12: MCP server (Claude Desktop / Claude Code)
**Built:** `clients/python/ferrocache/mcp_server.py` — MCP server speaking JSON-RPC over stdio via the official `mcp` SDK (1.27.0 detected locally). Three tools registered: `semantic_cache_lookup` (text + optional threshold), `semantic_cache_store` (text + response), `cache_status` (no args). Tool dispatch lives in a standalone `FerrocacheTools` class with three async methods backed by `FerrocacheClient` + an `embed_fn`; the MCP `@server.list_tools()` and `@server.call_tool()` decorators are thin shims that translate to/from this class. Errors are caught and turned into `{"error": "..."}` dict payloads — the server never crashes on a bad tool call. Sync `FerrocacheClient` is wrapped in `asyncio.to_thread` to avoid blocking the event loop. Embedding is handled inside the MCP layer: tools accept text, the server lazily loads sentence-transformers (configurable via `FERROCACHE_EMBED_MODEL`) and produces a 384-dim vector. New `clients/python/mcp_requirements.txt` pinning `mcp>=1.0.0` + `sentence-transformers`. New `docs/mcp-setup.md` with copy-pasteable Claude Desktop JSON config (mac/win/linux paths) and Claude Code `claude mcp add` command. README gained an "MCP Server" section linking to the doc. New `tests/test_mcp_server.py`: 8 cases under `unittest.IsolatedAsyncioTestCase` covering hit, miss, store success, status, ferrocache-unreachable on lookup AND store, dispatch-of-unknown-tool, and tool-catalog shape. All 38 Python tests pass (10 middleware + 7 langchain + 5 llamaindex + 8 mcp + 8 catalog/dispatch). Smoke-tested: server starts via `python3 -m ferrocache.mcp_server`, loads sentence-transformers, waits on stdio cleanly.

**Key decisions:**
- Tool dispatch extracted into `FerrocacheTools` so unit tests can hit the methods directly without a JSON-RPC harness — the MCP decorators are a 5-line shim around it.
- `asyncio.to_thread` for the sync client calls — small overhead, keeps the event loop responsive if multiple tool calls arrive concurrently.
- Errors return `{"error": "..."}` rather than raising — MCP clients (and Claude) handle structured error payloads gracefully, but a server crash kills the conversation.
- Embedding model loaded once at startup (inside `_build_tools_from_env`); subsequent tool calls reuse it. First-call cold start is the model-load time only.
- `mcp_server.py` is *not* auto-imported in `__init__.py` — the `mcp` SDK stays optional. Other ferrocache imports work fine without it.
- Tool descriptions are written for an LLM reader: when to use, what comes back, when to call which one. Lookup explicitly says "BEFORE expensive call", store says "AFTER".

**Deviations:**
- Brief sketched the SDK as `from mcp.server import Server`; the actual class lives at `mcp.server.lowlevel.server.Server` but `mcp.server.Server` re-exports it. Used the public path.
- Added an 8th test (`test_dispatch_unknown_tool`) on top of the brief's 7 — unknown tool names return a clean error rather than 500ing the server, worth covering.
- The brief's `mcp_requirements.txt` pin of `mcp>=1.0.0` works fine; locally it resolved to 1.27.0.

**Next session (M13 if pursued):** distribution polish — publish the Rust crate to crates.io, the Python client to PyPI (`pip install ferrocache`), the Docker image to Docker Hub. Includes `pyproject.toml`, GitHub Actions release workflow, and a CHANGELOG.

**Open:** Streaming tool results are not used — every tool returns a single JSON blob (fine for cache lookups, less ideal for any future tool that fans out). No tests exercise the actual MCP transport (decorators + stdio_server) — relies on the SDK's tested behavior. Embedding model is downloaded on first launch (~80MB); subsequent runs use HF cache.

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
