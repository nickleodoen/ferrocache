# ferrocache — Project Context

## Section 1: Evergreen

**What this is:** Distributed semantic cache for LLM applications, written in Rust. Single binary, multi-node via consistent hashing + gossip replication. Portfolio project for big-tech Core SWE interviews.

**Current phase:** Phase 3 in progress. M11 done (framework backends). Next: M12 — MCP server.

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

### 2026-05-01 — Mission 10: SDK middleware (OpenAI + Anthropic)
**Built:** Restructured `clients/python/` from a single `ferrocache.py` into a package: `ferrocache/__init__.py` (re-exports `FerrocacheClient`, `FerrocacheError`), `ferrocache/client.py` (the existing HTTP client, unchanged), `ferrocache/_embed.py` (lazy sentence-transformers default, raises with a clear install hint), `ferrocache/middleware.py` (the new wrappers). `wrap_openai` and `wrap_anthropic` proxy attribute access — only `chat.completions.create` / `messages.create` are intercepted; everything else (`models.list`, `embeddings.create`, etc.) delegates to the real client via `__getattr__`. Coordinator logic factored into a single `_intercept` function shared by both providers via a `_ProviderHooks` adapter. Synthetic cached responses are `SimpleNamespace`-based and structurally compatible with each SDK (`response.choices[0].message.content` for OpenAI, `response.content[0].text` for Anthropic). Real-API responses get `_ferrocache_hit` set on them (with a fallback to `__dict__` for pydantic models that reject unknown attrs). Tests in `tests/test_middleware.py`: 10 unittest cases using `unittest.mock` — hit / miss / fail-open / passthrough for both providers, env-var resolution, explicit-kwarg override, custom embed_fn. All pass; `python3 -m unittest tests.test_middleware` shows 10 OK in 4ms. Existing scripts (`example_usage.py`, `simulate.py`, `simulate_no_ml.py`) work unchanged because the package directory has the same import name as the old single file. README gained an "SDK Middleware" section after Python client.

**Key decisions:**
- Proxy via `__getattr__` (not subclass) — lets us avoid touching the real SDK class hierarchy (which is pydantic-derived and unfriendly to inherit from). Only the methods we intercept have explicit wrapper classes.
- Single `_intercept` function with `_ProviderHooks` (extract + build) instead of two parallel implementations — keeps both wrappers honest about behavior parity.
- `_set_attr_safely` falls back to `obj.__dict__` for SDK responses that may be pydantic models (real `ChatCompletion` is — direct `setattr` may raise). Best-effort: never break the user's response.
- `fail_open=True` is the default and the failure mode is explicit: log a warning, set `_ferrocache_hit=None`, return the real API response. Cache outage must NEVER break the application.
- `default_embed_fn` is constructed lazily inside `wrap_*` (not at module import) so users with a custom `embed_fn` never need sentence-transformers installed.
- `unittest.mock.patch("ferrocache.middleware.FerrocacheClient")` patches the symbol the middleware imports, not the real client module — standard pattern but worth noting for future test additions.

**Deviations:**
- Used stdlib `unittest` (with classes) rather than `pytest`-style assertion functions — runs under both `python3 -m unittest` and `python3 -m pytest`, and adds zero dev deps.
- Brief mentioned `client.embeddings.create` as one of the passthrough cases; I tested `client.models.list` instead because it's simpler to mock and proves the same pattern.
- The proxy classes set `self._real` in `__init__` (regular attribute) — `__getattr__` only fires when normal lookup misses, so `_real` access doesn't recurse. Worth flagging for anyone adding new hooks.

**Next session (M11 if pursued):** LangChain `BaseChatModel` subclass + LlamaIndex `LLM` subclass that wrap the same caching path. Likely lives in `clients/python/ferrocache/langchain.py` and `ferrocache/llamaindex.py`, optional imports.

**Open:** Wrappers are sync-only; OpenAI/Anthropic SDKs both ship async clients (`AsyncOpenAI`, `AsyncAnthropic`) that we don't cover yet. Streaming responses (`stream=True`) bypass the cache — they return immediately and the iterator isn't intercepted. Embedding happens on every call even on hits-likely paths; could short-circuit with a cheap exact-match check on `query_text` before paying for the embedding.

### 2026-05-01 — Mission 11: LangChain + LlamaIndex backends
**Built:** `clients/python/ferrocache/langchain.py` — `FerrocacheCache(BaseCache)` implementing the `lookup`/`update`/`clear` contract from `langchain_core.caches`. `lookup` embeds the prompt, queries ferrocache, returns `[Generation(text=...)]` on hit / `None` on miss. `update` embeds the prompt and inserts the cached `Generation[0].text`. `clear` is a logged no-op (ferrocache is append-only). `clients/python/ferrocache/llamaindex.py` — `FerrocacheLLM(CustomLLM)` using `pydantic.PrivateAttr` for inner-LLM and config state. Implements `complete`, `chat`, `stream_complete` (streaming bypasses the cache), and `metadata` (delegates to inner). Both backends gate `langchain_core` / `llama_index.core` imports behind a `try/except ImportError` so the modules load fine on machines without those frameworks; instantiating without the dep raises a clear `ImportError("Install ...")`. Both share the M10 `_resolve_url` / `_resolve_threshold` helpers and the lazy `default_embed_fn`. New tests: 7 `test_langchain.py` (hit, miss, update, lookup-fail-open, update-fail-open, clear noop, custom embed) + 5 `test_llamaindex.py` (complete-hit, complete-miss + insert, fail-open, custom embed, chat-hit). All 22 Python tests pass under `python3 -m unittest`. README gained a "Framework Integration" section with 2-line snippets for each. Examples added: `example_langchain.py` (uses `set_llm_cache`) and `example_llamaindex.py` (wraps `OpenAI(...)`). `__init__.py` *deliberately* does not auto-import these modules — frameworks remain optional deps.

**Key decisions:**
- LangChain side picked the official `BaseCache` interface — drops in via `set_llm_cache` and the user's chain just works.
- LlamaIndex side subclasses `CustomLLM` rather than implementing a separate cache abstraction — LlamaIndex doesn't have a clean cache-protocol equivalent, and an LLM wrapper composes naturally with query engines, agents, etc.
- LlamaIndex inner state stored via `PrivateAttr` to avoid pydantic's strict-field rules on the LLM base class — keeping it private also stops it from leaking into model serialization.
- Streaming bypasses the cache (`stream_complete` delegates raw) — caching a stream means buffering the full response, which defeats streaming's whole point. Documented in the file.
- Both backends import `_resolve_url` / `_resolve_threshold` from `middleware.py` rather than duplicating the env-var resolution logic — single source of truth.
- Module-level imports of `langchain_core` / `llama_index.core` use `try/except` with a sentinel; `__init__` raises a clear ImportError if you instantiate without the dep. Skipping the framework module entirely also works (other ferrocache imports keep working).
- Examples require real API keys — they're demos in `clients/python/`, not tests. Tests use `unittest.mock` exclusively.

**Deviations:**
- Brief sketched `FerrocacheLLM(OpenAI())` (positional inner). Used `FerrocacheLLM(inner=OpenAI(...))` keyword form instead — pydantic init behaves better and it's clearer at the call-site.
- LlamaIndex's `ChatMessage.content` accessor varies by version (some releases make it a property that synthesizes from `blocks`); `_last_user_text` defensively handles both `str` and callable forms.
- Detected `langchain_core 1.3.2` and `llama-index-core 0.14.21` in the local env. Public APIs match the brief's expectations; no shape surprises beyond `ChatMessage.content` accessor.

**Next session (M12 if pursued):** MCP (Model Context Protocol) server exposing ferrocache as a tool to MCP clients (Claude Desktop, IDEs). Likely a thin Python `mcp` server that wraps the existing client.

**Open:** No async backend versions yet (LangChain has `alookup`/`aupdate`; LlamaIndex has `acomplete`). Both currently fall back to LangChain's default-impl-runs-sync-in-executor behavior, which is fine for the cache-lookup hot path but adds an executor hop. Adding native async backends is a one-evening follow-up if needed.
