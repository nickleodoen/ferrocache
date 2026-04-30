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
