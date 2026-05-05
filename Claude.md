# ferrocache — Project Context

## Section 1: Evergreen

**What this is:** Distributed semantic cache for LLM applications, written in Rust. Single binary, multi-node via consistent hashing + gossip replication. Portfolio project for big-tech Core SWE interviews.

**Current phase:** Phase 7 complete. v1.0 ready. 30 missions across 7 phases.

**Completed work:**
- Phase 1 (M1-M4): axum scaffold, hnsw_rs HNSW index, WAL durability (NDJSON, fsync, replay), config + request validation + /stats
- Phase 2 (M5-M8): consistent hash ring (FNV-1a, 64 vnodes) + chitchat gossip discovery, cluster-aware /query routing + synchronous /insert replication, Dockerfile + 3-node docker-compose + integration tests, README + criterion benchmarks + Python client + CI
- Phase 3 (M9-M13): simulation harness with sentence-transformers, drop-in SDK middleware (`wrap_openai`/`wrap_anthropic`), LangChain `FerrocacheCache` + LlamaIndex `FerrocacheLLM`, MCP stdio server, PyPI distribution + GHCR Docker image + tagged release CI
- Phase 4 (M14-M16): namespace partitioning by `model_id`, WAL compaction + bincode snapshots with monotonic sequence numbers, hand-written Prometheus `/metrics` + Grafana dashboard overlay
- Phase 5 (M17-M20): bearer-token auth, cluster mTLS (rustls 0.23 + rcgen on `port+1000`), replication retry with exponential backoff + jitter, WAL group-commit (mpsc → batched fsync; 7.5× throughput at concurrency 100)
- Phase 6 (M21-M23): phi accrual failure detector + peer health metrics, ring reassignment driven exclusively by Dead status (zero data movement), read repair via `/internal/entry/{uuid}` fan-out
- Phase 7 (M24-M30): per-entry access tracking + `/admin/entry-stats`, LRU eviction with HNSW lazy deletion + WAL tombstones, TTL + `DELETE /entry/:uuid` + `POST /admin/invalidate`, exact-match pre-filter, `cache_scope` tenant isolation, `conversation_id` with two-level namespace lookup, v1.0 integration tests + GPTCache benchmark + docs

**Module map:**
- lib.rs — re-exports modules so benches and external consumers can import
- main.rs — entry: tracing, config, snapshot+WAL replay, optional cluster init, public + internal listeners, reaper spawn
- server.rs — axum router, handlers, request validation, routing/replication coordination, tests
- index.rs — `SemanticIndex`: per-namespace HNSW + side-table + `exact_match_index` + `evicted_ids` ghosts; replay paths; eviction/rebuild/expiry; namespace utilities (`effective_namespace`/`conversation_namespace`/`prune_empty_namespaces`); centralised `drop_entry` removal helper
- wal.rs — `Wal` (append, fsync, sequence, truncate, replay) + `GroupCommitWal` (mpsc-fed flush task, single fsync per batch, sole WAL writer); `WalCommand::{Insert,Compact}`
- models.rs — request/response DTOs (InsertRequest, QueryRequest, QueryResponse, etc.)
- config.rs — FerrocacheConfig + HnswConfig + ClusterConfig (TOML + env merge via the `config` crate)
- state.rs — AppState wiring index/wal/cluster/router/metrics/auth/conversation_ttl
- auth.rs — bearer-token axum middleware, constant-time compare, `/health` + `/metrics` exempt
- tls.rs — TlsBundle, rcgen CA + leaf generation, rustls mTLS server config + reqwest client identity, idempotent aws-lc-rs install
- bin/gen_certs.rs — dev cert generator binary; not built into the production image
- ring.rs — HashRing (BTreeMap, FNV-1a, virtual nodes, `get_n_nodes` replica walk)
- cluster.rs — ClusterState wrapping chitchat; `reconcile_step` drives ring membership from phi accrual + chitchat snapshots; `dead_nodes: HashSet<String>`
- failure_detector.rs — PhiAccrualDetector (sliding window, normal CDF, phi=-log10(1-F)); `PeerStatus::{Alive,Suspected,Dead}`; injectable clock for tests
- router.rs — ClusterRouter (reqwest); `forward_query`/`forward_insert`/`forward_get_entry`; `forward_with_retry` helper with jitter and 4xx/5xx classification
- snapshot.rs — bincode snapshot writer/reader (magic+version+wal_seq+entries), atomic temp+rename, `compact()` helper
- metrics.rs — Metrics (atomic counters + per-namespace `RwLock<HashMap>`), LatencyHistogram (16 fixed buckets), Prometheus text exposition `render()`
- benches/cache_bench.rs — criterion: insert / query_hit / query_miss / insert_with_wal / snapshot_write_10k
- tests/bench_concurrent.py — concurrent HTTP throughput harness (`make bench-concurrent`)
- clients/python/ferrocache.py — stdlib-only Python client (urllib + json)

**Architecture decisions (append-only, one line each):**
- Client computes embeddings externally; ferrocache stores/compares f32 vectors
- WAL is the source of truth; production insert path is WAL-first then `replay_entry` — the UUID-generating `insert()` is `#[cfg(test)]` only
- The flush task is the SOLE writer to the WAL; `compact` is also a `WalCommand` so it serialises FIFO with insert flushes; `wal_batch_size = 1` degrades to per-insert fsync (test default)
- WAL group-commit batches up to `wal_batch_size` requests within `wal_batch_timeout_ms` (defaults 256/1ms): one write+fsync per batch, then index write lock taken once
- Config priority: env vars > ferrocache.toml > defaults; `FERROCACHE_` prefix, `_` prefix-sep, `__` section-sep, `,` list-sep
- Cluster mode is opt-in via `cluster.enabled`; false keeps single-node behavior bit-identical
- Routing: `?local=true` skips ring lookup (used by forwarded requests + tests); coordinator stamps UUID before fan-out; forward failures return 502 to distinguish "peer failed" from local 500
- Index is namespace-partitioned by `model_id`; cross-namespace queries are impossible by construction; `model_id` format convention: `model_name::dimension`
- `effective_namespace(model_id, cache_scope)` returns `"{model_id}::{scope}"` or just `model_id` when scope is empty/whitespace; the WAL `model_id` field IS the effective namespace (no schema migration)
- `conversation_namespace` adds the hardcoded `::conv_` prefix on top of effective_namespace; `conv_` is reserved as a forbidden `cache_scope` prefix (documented, not enforced) so conversation IDs and scopes can never collide
- Two-level conversation lookup: query the conversation namespace first; on miss, fall back to the base namespace; `QueryResponse.scope` reports `"conversation"` or `"global"`. Inserts go to the conversation namespace ONLY (no dual-write)
- Auto-TTL: `conversation_ttl_seconds` config stamps TTL on conversation-scoped inserts when no explicit `ttl_seconds`; explicit always wins
- HNSW lazy deletion: evicted internal IDs tracked in per-namespace `evicted_ids: HashSet<usize>`; queries oversample by `1 + min(ghosts, 8)` and filter ghosts before the threshold check; rebuild from scratch when `ghost_ratio > 0.20`
- WAL tombstones (`{tombstone:true,uuid,model_id,sequence}`) prevent evicted/deleted entries from returning after restart; `replay_entry` branches on `entry.tombstone` and calls `remove_by_uuid` (M29 fix — runtime tombstone path was previously re-materialising phantom entries)
- `NamespacedIndex::drop_entry(internal_id)` is the SOLE removal path; every eviction/delete/expire route funnels through it so exact-match cleanup, `evicted_ids` tracking, and side-table mutation stay consistent
- LRU tie-break: `(last_accessed_at, inserted_at, internal_id)` — deterministic FIFO-on-tie. Access metadata is soft state (`last_accessed_at`/`access_count` snapshot-only, lost on crash, rebuilt via traffic) — a per-hit WAL write would defeat the cache
- Exact-match pre-filter: per-namespace `HashMap<normalized_text, uuid>` checked BEFORE HNSW when `query_text` is on `/query`; hit reports `similarity:1.0, exact_match:true`. Cleanup uses a UUID ownership check to survive delete-then-reinsert
- `normalize_query_text`: `split_whitespace + to_lowercase + join(" ")` only — no stemming, no Unicode normalisation; deterministic
- Reaper sequence: `collect_expired → rebuild_dirty_namespaces → prune_empty_namespaces` under one write lock; only `::conv_`-prefixed namespaces are pruned (base namespaces survive empty)
- `DELETE /entry/:uuid` fans out to ALL non-dead peers (no embedding to ring-hash); 404 from a peer is idempotent success. `POST /admin/invalidate` ships `(embedding, threshold)` to each replica which computes its own matches — "compute, not copy"
- Ring removals are driven exclusively by phi-accrual `Dead` status, NEVER by chitchat liveness directly; `Suspected` peers stay in the ring; `dead_node_removal_enabled` config supports a monitoring-only canary mode
- Re-joining peers are added back to the ring AND removed from `dead_nodes` on the same reconcile tick; ring reassignment is zero-data-movement (clockwise successor is already a replica via `replication_factor=2`)
- Dead peers are skipped: query → immediate miss instead of retry-then-502; insert fan-out → degraded replication with warn log naming `dead_peers` and `effective_replicas`
- Read repair on miss: coordinator fans out to non-dead replicas in parallel, returns first hit, spawns background `/internal/entry/{uuid}` fetch + WAL re-insert; `read_repair_enabled` config gates the entire fan-out; no Merkle trees, no anti-entropy
- Replication retry: exponential backoff (50ms × 2^attempt, ±20% jitter, capped 5s, max 3 attempts); retries on connect errors / timeouts / 5xx; NEVER on 4xx (deterministic); `replication_retries_total` is distinct from `replication_failures_total`
- Auth (bearer token, opt-in via `FERROCACHE_AUTH_TOKEN`): constant-time compare via `subtle::ConstantTimeEq`; `/health` and `/metrics` are exempt; inter-node forwards carry the same shared-secret token
- Cluster mTLS opt-in via `cluster.tls.enabled`: second TLS listener on `internal_port` (default `port+1000`); `WebPkiClientVerifier` requires a client cert chained to the cluster CA; public port stays plain HTTP (reverse proxy handles TLS); gossip UDP remains unencrypted (only ring metadata)

**Non-negotiable constraints:**
- Auth (bearer token) and cluster mTLS are both opt-in; public-port TLS handled by reverse proxy
- No UI
- No direct OpenAI/Anthropic API calls inside ferrocache
- Use tokio, not async-std
- Every new module gets unit tests before moving on

**Session log rules:**
- Keep only the last 2 session logs below
- When adding a new log, delete the oldest if there are already 2
- Summarize deleted sessions as one-line entries under "Completed work" above

## Section 2: Rolling Session Log (last 2 sessions only)


### 2026-05-04 — Mission 29: Multi-Turn Conversation Scoping
**Built:** Two-level conversation lookup composed on top of M28 scope namespaces. Added `conversation_id: Option<String>` to insert/query, `scope: Option<String>` ("conversation"/"global"/None) to QueryResponse, and `conversation_namespace = "{base}::conv_{id}"` with hardcoded `conv_` prefix to prevent collision with user-chosen cache_scopes. Inserts go to ONE namespace (conv if present, else base); queries with conv_id try conv namespace first, fall back to base on miss — never a unified search (would let global similarity beat conv priority). Added `conversation_ttl_seconds` config (auto-TTL applied only when no explicit ttl_seconds AND conv_id present), `prune_empty_namespaces()` (only `::conv_`-prefixed; base namespaces never pruned), and reaper sequence `collect_expired → rebuild_dirty_namespaces → prune_empty_namespaces` under one write lock. Fixed latent M26 bug: runtime tombstone path (reaper → WAL → flush → `replay_entry`) was re-materialising phantom entries because `replay_entry` had no tombstone branch — moved branch into `replay_entry` so startup and runtime paths share one code path. Threaded conversation_id through Python client, middleware (`wrap_openai`/`wrap_anthropic`), LangChain, LlamaIndex, and MCP.
**Tests:** 222/222 Rust + 59/59 Python + 32/32 integration pass.

### 2026-05-04 — Mission 30: Phase 7 Integration, Benchmarks, and Documentation
**Built:** v1.0 close-out — validation, benchmarks, docs only. Added 12-assertion Phase 7 integration test block (Test 9) covering M24–M29 features against live 3-node cluster: TTL, scope isolation, conversation lookup with `scope` field, DELETE fan-out, invalidate radius, exact-match pre-filter, TTL expiry at threshold 0.99 (tight to avoid match against test 10's [0.3,0.3,0.3,0.3] entry, cosine ≈ 0.91 to TTL probe), entry-stats endpoint. Used `?local=true` on insert/invalidate/exact-match tests so they don't depend on ring assignment. Added `tests/simulate.py --ttl <secs>`, `tests/benchmark_vs_gptcache.py` with subprocess isolation for GPTCache (faiss/onnx segfault on Python 3.13+; harness reports N/A gracefully), and Makefile `benchmark-vs-gptcache` target. Recorded eviction overhead: 31% throughput drop with `max_entries_per_namespace=5000` due to inline HNSW rebuild under index write lock — documented as M25 design trade-off, not regression. Complete v1.0 rewrite of README and `clients/ferrocache-skill.md` (last touched at M8 and M19 respectively).
**Tests:** 222/222 Rust + 59/59 Python + 44/44 integration pass (was 32/32; +12 Phase 7 assertions, +1 entry-stats smoke).
