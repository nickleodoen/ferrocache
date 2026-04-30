# ferrocache — Persistent Project Context

## What this project is
Distributed semantic cache for LLM applications, written in Rust.
Single binary, multi-node via consistent hashing + gossip replication.
Target: portfolio project for big-tech Core SWE interviews.

## Current phase
Phase 1 — Single node. Goal: axum HTTP server + hnsw_rs index + WAL + unit tests.

## Architecture decisions (append-only, never delete)
- Embedding-agnostic: client computes embeddings, semcache stores/compares float vectors
- Client computes embeddings externally, passes f32 vectors via HTTP
- Crate choices: tokio, axum, hnsw_rs, chitchat, tracing, serde_json
- WAL format: newline-delimited JSON, replayed on startup
- Consistent hashing: hash first 8 bytes of embedding vector as u64 key

## What Claude Code must do at the END of every session
Append a section to this file under "## Session Log" with:
- Date
- What was built
- Key decisions made and why
- What the next session should start with
- Any open questions or blockers

## Non-negotiable constraints
- No auth/TLS until Phase 3
- No UI
- No direct OpenAI/Anthropic API calls inside semcache
- Use tokio, not async-std
- Every new module gets unit tests before moving on

## Session Log

### 2026-04-30 — Mission 1: scaffold + axum HTTP skeleton
**Built:**
- `Cargo.toml` populated with Phase 1 deps (tokio, axum 0.7, serde, serde_json, uuid, tracing, tracing-subscriber, anyhow); dev-deps tower + http-body-util.
- Module split: `main.rs` (entry + tracing init + bind), `server.rs` (router + handlers + tests), `models.rs` (request/response DTOs), `state.rs` (AppState).
- Three routes: `POST /query` (stubbed `{hit:false}`), `POST /insert` (UUID + AtomicU64 increment), `GET /health` (status/node_id/entry_count).
- `AppState` shared as `Arc<AppState>` via axum `State`; `entry_count` is `Arc<AtomicU64>` (SeqCst); `node_id` generated once at startup.
- Tracing via `tracing_subscriber::fmt` with `EnvFilter` (default `info`, `RUST_LOG` overrides).
- Bind `0.0.0.0:3000`, override via `FERROCACHE_PORT`. Bind addr logged at startup.
- 4 unit tests in `src/server.rs` using `tower::ServiceExt::oneshot` against the `Router` directly — no TCP. All pass; `cargo clippy --all-targets -- -D warnings` clean.

**Key decisions:**
- `#[allow(dead_code)]` on `InsertRequest` — `response`/`query_text` are part of the wire contract for Missions 2/3 (HNSW + WAL) but unread today; suppression beats fake reads.
- `Arc<AtomicU64>` inside `AppState` (rather than just `AtomicU64`) so Mission 3's WAL writer can hold its own clone without going through the state struct.
- Used `axum::serve` + `tokio::net::TcpListener` (axum 0.7 idiom), not the older `Server::bind` API.
- Hashed first-8-bytes-of-embedding key (per architecture decisions) is deferred — query handler currently ignores embedding contents.

**Next session (Mission 2) should start with:**
- Add `hnsw_rs` to Cargo.toml.
- Build an in-memory `Index` module wrapping HNSW with insert/query, wired into `AppState` behind an `RwLock` (or HNSW's internal sync if available).
- Wire `/insert` to actually add the vector + store the response payload (decide: separate `HashMap<id, (response, query_text)>` keyed by HNSW id or external UUID).
- Wire `/query` to nearest-neighbor lookup, applying `threshold` on cosine similarity; on hit return `{hit:true, response, similarity}`.
- Extend tests: insert→query roundtrip hit; threshold miss; dimension-mismatch error.

**Open questions / blockers:**
- `hnsw_rs` API ergonomics around id mapping — may need a side-table from internal HNSW id → UUID. Resolve in Mission 2.
- Distance metric: cosine vs L2. Default to cosine (typical for embeddings); confirm before coding.
- No persistence yet — entries are lost on restart until Mission 3 (WAL).

### 2026-04-30 — Mission 2: HNSW index integration
**Built:**
- New `src/index.rs` module: `SemanticIndex` wrapping `hnsw_rs::Hnsw<'static, f32, DistCosine>` with a `HashMap<usize, CacheEntry>` side-table, a monotonic `next_id: usize`, and a locked-on-first-insert `dimension: Option<usize>`.
- HNSW params: M=16, max_elements=100_000, max_layer=16, ef_construction=200, ef_search=32 (named consts at top of file).
- `insert(embedding, response, query_text) -> Result<String>`: enforces dimension, picks `next_id`, generates UUID, calls `hnsw.insert((&embedding, id))`, stores `CacheEntry { uuid, embedding, response, query_text }`, returns UUID.
- `query(&[f32], threshold) -> Result<Option<QueryHit>>`: dimension check; calls `hnsw.search(emb, 1, ef_search=32)`; converts cosine distance to similarity via `1.0 - distance`; threshold-gates and looks up the entry by `n.get_origin_id()` in the side-table.
- `entry_count() -> usize` reports `entries.len()`.
- Returns `QueryHit { id, response, similarity }` on hit.
- `state.rs`: replaced `Arc<AtomicU64>` with `Arc<tokio::sync::RwLock<SemanticIndex>>`. Handlers acquire `.read()` for /query and /health, `.write()` for /insert.
- `models.rs`: `QueryResponse` now has optional `response`/`similarity` (skipped when `None`); added `ErrorResponse { error }`.
- `server.rs`: handlers return `Response` (not `Json<...>`) to allow 400 on dimension mismatch with `ErrorResponse` body. Health pulls count from index lock. Tracing logs hit/similarity/dim.
- Tests: 5 unit tests in `index.rs` (hit, miss-below-threshold, insert-dim-mismatch, query-dim-mismatch, entry_count) + 8 server tests (including new `test_insert_then_query_hit`, `test_query_miss_different_vector`, `test_insert_dimension_mismatch`). 12/12 pass; clippy clean.

**Key decisions:**
- `Hnsw<'static, ...>` — owned data in our `Vec<f32>` is copied internally by hnsw_rs; static is the safe parameterization.
- `tokio::sync::RwLock` (not `std::sync::RwLock`) so handlers don't block the runtime; aligns with axum's async handlers.
- Side-table keyed by our own monotonic `next_id` (not by external UUID) — hnsw_rs requires `usize` ids and returns them via `Neighbour::get_origin_id()`. UUID is stored inside the `CacheEntry` for client-facing identity. This sets up cleanly for Mission 3 WAL replay (the `usize` is never persisted; on restart we replay inserts and re-issue ids).
- Cosine distance confirmed via `anndists::dist::distances::DistCosine`: `dist = 1 - cos_sim` (line 217 of distances.rs), so `similarity = 1.0 - distance`. Matches spec.
- 400 on dimension mismatch (both insert and query) — clean client signal, doesn't poison the index.
- `#[allow(dead_code)]` on `QueryHit::id` and `CacheEntry` fields used only by Mission 3 (WAL) — these are part of the data model contract, not stale code.

**API deviations from the brief:**
- Spec said `Hnsw::new(M, max_elements, max_layer, ef_construction, dist)` — confirmed match. The order was `(max_nb_connection, max_elements, max_layer, ef_construction, f)`.
- `hnsw.insert` takes `&self` (interior mutability), not `&mut self` — but our wrapper's `insert` is still `&mut self` because `entries`/`next_id`/`dimension` need exclusive access. Equivalent at the lock granularity we have.
- `Neighbour` uses `get_distance()`/`get_origin_id()` accessors rather than direct field access — used the accessors.

**Next session (Mission 3) should start with:**
- WAL: newline-delimited JSON, one line per insert. Format: `{"id": "<uuid>", "embedding": [...], "response": "...", "query_text": "..."}`.
- Append synchronously inside `/insert` *before* returning success. Use `tokio::fs::OpenOptions` with `append(true)`, fsync after each line.
- On startup: replay file line-by-line into `SemanticIndex` before binding the HTTP listener.
- Configurable WAL path (env var `FERROCACHE_WAL_PATH`, default `./ferrocache.wal`).
- Tests: round-trip (insert → drop → reload → query hits); corrupt-line tolerance (skip with warn); empty/missing file is fine.

**Open questions / blockers:**
- WAL fsync cost vs throughput — for Phase 1 we fsync per insert (correctness > throughput). Revisit when we benchmark.
- HNSW does not natively persist — we rebuild from WAL on every startup. For very large WALs we'll need periodic snapshots (Phase 2 or later).
- HNSW `max_elements=100_000` is a hint; doesn't hard-cap. We'll need an eviction policy before this matters.

### 2026-04-30 — Mission 3: Write-ahead log
**Built:**
- New `src/wal.rs`: `Wal { file: tokio::fs::File, path: PathBuf }` + `WalEntry { uuid, embedding, response, query_text }`. `Wal::open` uses `OpenOptions::new().create(true).append(true)`. `Wal::append` serializes JSON, writes line + `\n`, then `file.sync_data().await` (fsync per insert). `Wal::replay` streams the file via `BufReader::lines()`, parses each line, **skips** corrupt lines with `tracing::warn!` (line number + error), returns empty vec for missing/empty file.
- `index.rs` refactor: factored core insert into private `insert_with_uuid(uuid, embedding, response, query_text)`; public `insert` now lives behind `#[cfg(test)]` and just generates a UUID + delegates; new `replay_entry(WalEntry)` takes the persisted UUID (so UUIDs are stable across restarts). Added `dimension() -> Option<usize>` getter so the handler can validate before WAL append.
- `state.rs`: `AppState { node_id, index: Arc<RwLock<SemanticIndex>>, wal: Arc<Mutex<Wal>> }`. Used `tokio::sync::Mutex` for the WAL — appends are exclusive; brief hold makes a full Mutex appropriate.
- `main.rs` startup sequence: read `FERROCACHE_WAL_PATH` (default `./ferrocache.wal`), `Wal::replay`, build fresh `SemanticIndex`, fold replayed entries via `index.replay_entry`, log replayed count. Then `Wal::open` for appends. Then build router and serve.
- `server.rs` `/insert`: acquire WAL Mutex → peek dim via index read lock → if mismatch, 400 with `{error}` (no WAL pollution) → generate UUID → build `WalEntry` → `wal.append` (fsync) → `index.write().await.replay_entry(entry)` → 200. WAL Mutex serializes inserters end-to-end so dim cannot change between peek and write. WAL append failure → 500, no index mutation.
- 17/17 tests pass (5 index, 4 WAL, 8 server). New WAL tests: append→replay roundtrip; corrupt-line skip (mixed valid/garbage); missing file → empty; empty file → empty. New server test `test_insert_persists_via_wal`: HTTP insert → drop AppState → `Wal::replay` → fresh index → `replay_entry` → query hits with original response.
- Smoke test passed end-to-end: insert via curl → kill process → restart → replay logs `count=1` → /query returns `{hit:true, response:"smoked", similarity:1.0}` and /health reports `entry_count:1`.

**Key decisions:**
- Validate dimension *before* WAL append, not after — keeps the WAL free of lines that would only get rejected on replay. Done under the WAL Mutex so the validation can't race.
- WAL Mutex held for the entire insert critical section. Single-writer Phase 1; revisit when concurrent inserts matter (could move to per-shard WALs in Phase 2).
- `insert` (UUID-generating) on `SemanticIndex` is now `#[cfg(test)]` only — production path is WAL-first via `replay_entry`. Avoids two divergent insert paths in production code.
- `BufReader::lines()` for replay: streams instead of slurping the whole file. Whitespace-only lines are skipped silently; only malformed JSON triggers the warn.
- WAL `path` field is currently unused at runtime (`#[allow(dead_code)]`) but kept for future log lines / snapshot rotation.
- Decision deferred: WAL fsync per write is the conservative correctness choice. Group-commit/batched fsync stays unbuilt until we have a real benchmark.

**API deviations from the brief:**
- `AppState::new()` now takes `(SemanticIndex, Wal)` rather than constructing them internally — tests need to inject a temp WAL path, and main.rs needs to replay before building the state. The `Default` impl was dropped (no longer makes sense without I/O).
- The handler uses `replay_entry` (not the spec's "insert into HNSW") to add to the index after WAL append. Same effect, but via the UUID-preserving path so the in-memory entry's UUID matches the WAL line.
- Spec said "if WAL append fails → 500, do NOT insert into index" — implemented exactly. Also added a 500 path if `replay_entry` somehow fails after WAL append (shouldn't happen due to upfront dim check, but the error handling is there).

**Next session (Mission 4) should start with:**
- Likely targets: config file (TOML or JSON for HNSW params, port, WAL path); structured request validation (max embedding size, max response size); per-route metrics (counter for hits/misses, histogram for query latency).
- If moving toward Phase 2 instead: introduce `chitchat` for gossip, define a node bootstrap config, and decide on the consistent-hashing scheme key (per `Claude.md`: hash first 8 bytes of embedding as u64).
- WAL maintenance: even before clustering, consider snapshot+truncate so WAL doesn't grow unbounded across long-running deployments.

**Open questions / blockers:**
- Replay is single-threaded and rebuilds HNSW from scratch — at large WAL sizes this becomes the dominant startup cost. Snapshotting required before this matters in practice.
- WAL has no checksum per line. A torn write at the tail (partial line during a crash mid-fsync) would currently parse as invalid JSON and be dropped with a warn — acceptable for now since the client wouldn't have seen a 200 for that insert.
- Concurrency: the WAL Mutex serializes inserts but not against /query. /query takes the index read lock and is unaffected. Reads during a write hold are blocked only while we hold the index write lock (very brief).
