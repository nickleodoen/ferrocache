# ferrocache — Project Context

## Section 1: Evergreen

**What this is:** Distributed semantic cache for LLM applications, written in Rust. Single binary, multi-node via consistent hashing + gossip replication. Portfolio project for big-tech Core SWE interviews.

**Current phase:** Phase 1 complete. Next: Phase 2 — distributed (consistent hashing, gossip, replication).

**Completed work:**
- M1: axum scaffold (3 routes, tracing, env-based port)
- M2: hnsw_rs integration (DistCosine, side-table keyed by usize, dim lock)
- M3: WAL (NDJSON, fsync-per-insert, replay on startup, corrupt-line skip)
- M4: config crate (TOML + env merge), request validation, /stats endpoint, dead-code cleanup

**Module map:**
- main.rs — entry, tracing init, config load, WAL replay, serve
- server.rs — router, handlers, request validation, tests
- index.rs — SemanticIndex (hnsw_rs wrapper + HashMap side-table), replay_entry
- wal.rs — Wal (append w/ fsync), WalEntry, replay
- models.rs — request/response DTOs
- config.rs — FerrocacheConfig, HnswConfig, load()
- state.rs — AppState { node_id, index: RwLock, wal: Mutex, wal_path, hnsw_config }

**Architecture decisions (append-only, one line each):**
- Client computes embeddings externally; ferrocache stores/compares f32 vectors
- Crates: tokio, axum 0.7, hnsw_rs, chitchat (Phase 2), tracing, serde_json, config
- WAL format: newline-delimited JSON, replayed on startup, fsync per insert
- Consistent hashing key: first 8 bytes of embedding as u64 (Phase 2)
- Cosine distance: hnsw_rs DistCosine returns 1-similarity; we convert
- Side-table: HashMap<usize, CacheEntry> keyed by hnsw internal id; UUID inside CacheEntry
- WAL is source of truth; production insert path is WAL-first then replay_entry
- insert() (UUID-generating) is #[cfg(test)] only; production uses replay_entry
- Config priority: env vars (FERROCACHE_ prefix, `__` separator) > ferrocache.toml > defaults
- tokio::sync::RwLock for index, tokio::sync::Mutex for WAL
- Validation limits (server.rs constants): MAX_EMBEDDING_DIM=4096, MAX_RESPONSE_BYTES=102_400
- Threshold accepted range on /query: 0.0..=1.0

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

### 2026-04-30 — Mission 3: Write-ahead log
**Built:** `src/wal.rs` (`Wal::open`/`append`/`replay`, fsync per insert, NDJSON, corrupt-line skip with warn). `index.rs` factored core insert into private `insert_with_uuid`; new `replay_entry(WalEntry)` reuses persisted UUID; added `dimension()` getter; `insert` is now `#[cfg(test)]`. `state.rs`: added `wal: Arc<Mutex<Wal>>`. `main.rs`: `FERROCACHE_WAL_PATH` env (default `./ferrocache.wal`), replay → open → serve. `/insert`: WAL-mutex held end-to-end; dim peek via read lock → 400 on mismatch; WAL append → index.replay_entry. 17/17 tests pass; smoke test verified insert→kill→restart→query hit.

**Key decisions:**
- Validate dim before WAL append (prevents persisting lines we'd reject on replay anyway).
- WAL Mutex serializes inserts so dim cannot race between peek and write.
- BufReader::lines() for streaming replay; whitespace-only lines silently skipped.

**Deviations:** `AppState::new` now takes `(SemanticIndex, Wal)` so tests can inject a temp WAL path; `Default` impl dropped.

**Open:** No snapshots → full replay per startup; WAL grows unbounded; no per-line checksum (torn tail writes parse-fail and drop with warn).

### 2026-05-01 — Mission 4: Config + validation + /stats + Phase 1 polish
**Built:** New `src/config.rs` with `FerrocacheConfig { port, node_id?, wal_path, hnsw }` + `HnswConfig { max_nb_connection, max_elements, max_layer, ef_construction, ef_search, default_threshold }`. `load()` uses `config` crate's builder: defaults → `ferrocache.toml` (optional) → env vars (`FERROCACHE_` prefix, `__` separator for nested keys). Created `ferrocache.toml` with all keys commented out. `SemanticIndex::new` now takes `&HnswConfig` (HNSW consts removed). `AppState` carries `wal_path` + `hnsw_config` for /stats. `main.rs` calls `FerrocacheConfig::load()` first; replaced env-var lookups; logs `?config` at startup. Added validation in handlers: empty/oversized embedding (>4096), oversized response (>100KB), threshold out of [0,1] → 400 via shared `bad_request` helper. New `GET /stats` returns `{entry_count, wal_path, hnsw{...,dimension}}` (`dimension` null pre-first-insert). `/query` now also returns hit `id` (was previously discarded).

**Key decisions:**
- Centralized validation helpers (`validate_embedding`, `bad_request`) so /query and /insert share the same rejection paths.
- Stored a *copy* of `HnswConfig` in AppState (vs pointer) — it's tiny (Copy-able primitives) and lets /stats read without locking the index for config fields.
- Restructured Claude.md into evergreen + 2-session rolling window per mission spec.
- Dead-code cleanup: dropped `CacheEntry::embedding`/`query_text` (WAL is source of truth, in-memory copies were unused) and `Wal::path` field (never read). `QueryHit::id` is now actually used (returned in `/query` response).

**Deviations:**
- `Wal::open` signature changed from `impl Into<PathBuf>` to `impl AsRef<Path>` after dropping the stored `path` field — same callsites, simpler bound.
- `AppState::new` now takes 5 args (`node_id, index, wal, wal_path, hnsw_config`) — explicit injection beats hidden state for testability.
- `insert_with_uuid` keeps `_query_text` parameter prefixed-underscore: it's part of the WAL contract but the in-memory entry doesn't need it. Kept the param so the call site stays symmetric with `WalEntry` field order.

**Next session (Mission 5 / Phase 2 kickoff):** introduce `chitchat` for gossip; define cluster bootstrap config (seed nodes); design consistent-hash ring keyed by `u64::from_be_bytes(embedding[0..8])`; decide replication factor and read/write quorums.

**Open:** Replay still single-threaded; WAL still uncompacted; HNSW `max_elements` is a soft hint (no eviction); `node_id` is per-process (no stable identity across restarts unless set in config).
