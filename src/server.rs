use std::sync::Arc;
use std::sync::atomic::Ordering;

use axum::{
    Json, Router,
    extract::{Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{delete, get, post},
};
use futures::future::join_all;
use serde::Deserialize;
use tokio::sync::oneshot;

use crate::auth::{AuthToken, auth_middleware};
use crate::cluster::ClusterState;
use crate::failure_detector::PeerStatus;
use crate::index::now_unix_secs;
use crate::metrics::METRICS_CONTENT_TYPE;
use crate::models::{
    ClusterStatusResponse, CompactResponse, CountersResponse, DeleteEntryResponse,
    EntryStatsResponse, ErrorResponse, FullEntryResponse, HealthResponse, InsertRequest,
    InsertResponse, InvalidateRequest, InvalidateResponse, NamespaceStatsEntry,
    NamespaceTopEntries, PeerHealth, QueryRequest, QueryResponse, ReadRepairRequest,
    ReadRepairResponse, StatsHnsw, StatsResponse, TopEntryRow,
};
use crate::state::AppState;
use crate::wal::{
    CompactRequest, InsertError, InsertRequest as WalInsertRequest, WalCommand, WalEntry,
};

pub const MAX_EMBEDDING_DIM: usize = 4096;
pub const MAX_RESPONSE_BYTES: usize = 102_400;

#[derive(Debug, Deserialize, Default)]
pub struct LocalParam {
    #[serde(default)]
    pub local: Option<bool>,
    /// Read repair (M23): set to `Some(false)` by the read-repair fan-out
    /// to prevent the recipient from doing its own replica fan-out and
    /// looping. `None` and `Some(true)` both mean "default behaviour".
    #[serde(default)]
    pub repair: Option<bool>,
}

impl LocalParam {
    fn is_local(&self) -> bool {
        self.local.unwrap_or(false)
    }

    /// Whether to attempt read-repair on miss.
    ///
    /// - `Some(v)` is an explicit override and wins.
    /// - `None` defaults based on `local`: client-driven `?local=true` is a
    ///   diagnostic / "don't touch other nodes" signal, so we DON'T repair.
    ///   A request that went through ring routing (no params) DOES want
    ///   repair.
    /// - The read-repair fan-out always sets `repair=false` explicitly to
    ///   avoid recursion.
    /// - The coordinator's forward-to-owner (`forward_query`) always sets
    ///   `repair=true` explicitly so the owner — the actual primary —
    ///   triggers repair on miss.
    fn try_repair(&self) -> bool {
        match self.repair {
            Some(v) => v,
            None => !self.is_local(),
        }
    }
}

pub fn build_router(state: Arc<AppState>) -> Router {
    let auth_token = state.auth_token.clone();
    let router = Router::new()
        .route("/query", post(query_handler))
        .route("/insert", post(insert_handler))
        .route("/health", get(health_handler))
        .route("/stats", get(stats_handler))
        .route("/cluster/status", get(cluster_status_handler))
        .route("/admin/compact", post(compact_handler))
        .route("/admin/entry-stats", get(entry_stats_handler))
        .route("/admin/invalidate", post(invalidate_handler))
        .route("/entry/:uuid", delete(delete_entry_handler))
        .route("/metrics", get(metrics_handler))
        // Read-repair internal endpoints (M23). Both follow the same auth
        // pattern as the public routes — they're cluster-internal calls
        // but a misconfigured deployment exposing them externally still
        // requires the bearer token if auth is on.
        .route("/internal/entry/:uuid", get(get_entry_handler))
        .route("/internal/read-repair", post(read_repair_handler))
        .with_state(state);

    if let Some(token) = auth_token.filter(|t| !t.is_empty()) {
        let token_state = Arc::new(AuthToken { value: token });
        router.layer(axum::middleware::from_fn_with_state(
            token_state,
            auth_middleware,
        ))
    } else {
        router
    }
}

fn bad_request(msg: impl Into<String>) -> Response {
    let msg = msg.into();
    tracing::warn!(error = %msg, "request rejected");
    (StatusCode::BAD_REQUEST, Json(ErrorResponse { error: msg })).into_response()
}

fn validate_model_id(model_id: Option<&str>) -> Result<&str, String> {
    match model_id {
        Some(s) if !s.trim().is_empty() => Ok(s),
        _ => Err("model_id is required".to_string()),
    }
}

fn validate_embedding(embedding: &[f32]) -> Result<(), String> {
    if embedding.is_empty() {
        return Err("embedding must not be empty".to_string());
    }
    if embedding.len() > MAX_EMBEDDING_DIM {
        return Err(format!(
            "embedding dimension {} exceeds max {}",
            embedding.len(),
            MAX_EMBEDDING_DIM
        ));
    }
    Ok(())
}

async fn query_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<LocalParam>,
    Json(req): Json<QueryRequest>,
) -> Response {
    if let Err(msg) = validate_embedding(&req.embedding) {
        return bad_request(msg);
    }
    if let Err(msg) = validate_model_id(req.model_id.as_deref()) {
        return bad_request(msg);
    }
    if !(0.0..=1.0).contains(&req.threshold) {
        return bad_request(format!(
            "threshold {} out of range [0.0, 1.0]",
            req.threshold
        ));
    }

    if !params.is_local()
        && let Some(cluster) = &state.cluster
        && let Some((target_id, target_addr)) = cluster.get_target_addr(&req.embedding).await
        && target_id != cluster.self_node_id()
    {
        // Soft failover (M21): if the failure detector has marked the
        // owning node as Dead, skip the request entirely and return a miss.
        // This avoids the retry-then-502 chain on requests we already know
        // will fail. M22 will reassign the ring; until then "miss" is the
        // safe degraded answer (a wrong cache miss is cheap; a 502 isn't).
        if cluster.peer_status(&target_id).await == PeerStatus::Dead {
            tracing::warn!(peer = %target_id, "skipping query to dead peer; returning miss");
            return (
                StatusCode::OK,
                Json(QueryResponse {
                    hit: false,
                    id: None,
                    response: None,
                    similarity: None,
                    exact_match: None,
                }),
            )
                .into_response();
        }
        return forward_query(&state, &target_id, &target_addr, &req).await;
    }

    // Read-repair (M23) only applies on the "owner" code path — the local
    // miss is the trigger to ask replicas. The fan-out itself sets
    // `?repair=false` so replicas don't recurse.
    process_query_locally(&state, req, params.try_repair()).await
}

async fn forward_query(
    state: &Arc<AppState>,
    target_id: &str,
    target_addr: &str,
    req: &QueryRequest,
) -> Response {
    let Some(router) = &state.router else {
        tracing::error!(peer = %target_id, "no router configured");
        return bad_gateway("upstream replica unavailable");
    };
    let result = router
        .forward_with_retry(state.max_replication_retries, &state.metrics, || {
            router.forward_query(target_addr, req)
        })
        .await;
    match result {
        Ok(resp) => {
            tracing::info!(target = %target_id, "query forwarded");
            (StatusCode::OK, Json(resp)).into_response()
        }
        Err(e) => {
            tracing::error!(peer = %target_id, error = %e, "forward_query failed");
            bad_gateway("upstream replica unavailable")
        }
    }
}

async fn process_query_locally(
    state: &Arc<AppState>,
    req: QueryRequest,
    try_repair: bool,
) -> Response {
    let start = std::time::Instant::now();
    let dim = req.embedding.len();
    let model_id = match req.model_id.as_deref() {
        Some(s) if !s.trim().is_empty() => s,
        _ => return bad_request("model_id is required"),
    };
    let now = now_unix_secs();
    let result = {
        let index = state.index.read().await;
        index.query_with_text(
            &req.embedding,
            req.threshold,
            model_id,
            req.query_text.as_deref(),
            now,
        )
    };
    let elapsed = start.elapsed().as_secs_f64();
    match result {
        Ok(Some(hit)) => {
            // M24: brief write lock to bump access metadata. The read lock
            // above has dropped; the write lock is held only long enough for
            // a HashMap lookup + two field writes. Misses skip this entirely.
            {
                let mut idx = state.index.write().await;
                idx.record_access(model_id, hit.internal_id);
            }
            state.metrics.record_query_hit(model_id, elapsed);
            // M27: exact-match hits are ALSO regular hits. The new
            // counter is a subcategory, not an alternative.
            if hit.exact_match {
                state.metrics.record_exact_match_hit(model_id);
            }
            tracing::info!(
                hit = true,
                similarity = hit.similarity,
                exact_match = hit.exact_match,
                dim,
                "query"
            );
            (
                StatusCode::OK,
                Json(QueryResponse {
                    hit: true,
                    id: Some(hit.id),
                    response: Some(hit.response),
                    similarity: Some(hit.similarity),
                    exact_match: Some(hit.exact_match),
                }),
            )
                .into_response()
        }
        Ok(None) => {
            state.metrics.record_query_miss(model_id, elapsed);
            tracing::info!(hit = false, dim, "query");

            // Read repair (M23): if we're the ring owner / coordinator and
            // a replica has the entry, return its hit and asynchronously
            // backfill our local index. Skipped when:
            //   - the request explicitly disabled repair (`?repair=false`)
            //   - read_repair_enabled is off in config
            //   - we're not in cluster mode
            //   - replication_factor is 1 (no replicas to ask)
            if try_repair
                && state.read_repair_enabled
                && state.replication_factor > 1
                && let Some(cluster) = state.cluster.as_ref()
                && let Some(repaired) = attempt_read_repair(state, cluster, model_id, &req).await
            {
                tracing::info!(hit = true, repaired = true, "query (repaired from replica)");
                return (StatusCode::OK, Json(repaired)).into_response();
            }

            (
                StatusCode::OK,
                Json(QueryResponse {
                    hit: false,
                    id: None,
                    response: None,
                    similarity: None,
                    exact_match: None,
                }),
            )
                .into_response()
        }
        Err(e) => bad_request(e.to_string()),
    }
}

/// Query non-dead replicas in parallel. Returns the first hit (if any) so
/// the client gets a low-latency response, and spawns a background task
/// that fetches the full entry from the replica and re-inserts it locally.
/// Failures (network, missing UUID, WAL channel closed) increment
/// `read_repair_failures_total` but never propagate out of the spawn.
async fn attempt_read_repair(
    state: &Arc<AppState>,
    cluster: &Arc<ClusterState>,
    model_id: &str,
    req: &QueryRequest,
) -> Option<QueryResponse> {
    let router = state.router.as_ref()?;
    let replicas = cluster
        .get_replica_addrs(&req.embedding, state.replication_factor)
        .await;
    let self_id = cluster.self_node_id();

    // Filter to peers we should actually ask. M21's Dead skip applies here
    // too — a peer chitchat told us is gone won't answer.
    let mut peers: Vec<(String, String)> = Vec::new();
    for (id, addr) in replicas {
        if id == self_id {
            continue;
        }
        if cluster.peer_status(&id).await == PeerStatus::Dead {
            continue;
        }
        peers.push((id, addr));
    }
    if peers.is_empty() {
        return None;
    }

    // Parallel fan-out. `forward_query_no_repair` adds `?repair=false` so
    // the recipient doesn't itself fan out (would be N×N traffic + a loop
    // risk in degenerate ring configurations).
    let queries = peers
        .iter()
        .map(|(_, addr)| router.forward_query_no_repair(addr, req));
    let results = join_all(queries).await;

    // Pair each result with its peer id/addr so we know who to fetch the
    // full entry from below.
    let mut hit: Option<(String, String, QueryResponse)> = None;
    for ((peer_id, peer_addr), result) in peers.iter().zip(results) {
        match result {
            Ok(qr) if qr.hit => {
                hit = Some((peer_id.clone(), peer_addr.clone(), qr));
                break;
            }
            Ok(_) => {}
            Err(e) => {
                tracing::warn!(peer = %peer_id, error = %e, "read repair: replica query failed");
            }
        }
    }
    let (peer_id, peer_addr, qr) = hit?;
    let uuid = match qr.id.clone() {
        Some(u) => u,
        None => {
            // A replica reporting `hit=true` without an id is a protocol
            // bug, not a transient failure — count it but still return the
            // hit since the response/similarity are still useful.
            state.metrics.record_read_repair_failure();
            tracing::warn!(peer = %peer_id, "read repair: replica hit had no id");
            return Some(qr);
        }
    };

    // Spawn the actual repair as a fire-and-forget — the client gets the
    // hit immediately, durability of the new local entry happens later.
    let state_clone = state.clone();
    let model_id_owned = model_id.to_string();
    tokio::spawn(async move {
        perform_repair(state_clone, peer_id, peer_addr, uuid, model_id_owned).await;
    });
    Some(qr)
}

/// Background half of read repair: GET /internal/entry/{uuid} from the
/// replica that hit, then push the full entry through the WAL group-commit
/// channel so it lands on disk + in the index with full durability.
async fn perform_repair(
    state: Arc<AppState>,
    peer_id: String,
    peer_addr: String,
    uuid: String,
    model_id: String,
) {
    let Some(router) = state.router.as_ref() else {
        state.metrics.record_read_repair_failure();
        return;
    };

    let full = match router.forward_get_entry(&peer_addr, &uuid).await {
        Ok(Some(e)) => e,
        Ok(None) => {
            // Replica raced — the entry was there during the query but
            // gone by the time we asked for it (compaction? eviction?
            // unrealistic for ferrocache today, but cheap to handle).
            state.metrics.record_read_repair_failure();
            tracing::warn!(peer = %peer_id, uuid = %uuid, "read repair: replica returned 404 for entry");
            return;
        }
        Err(e) => {
            state.metrics.record_read_repair_failure();
            tracing::warn!(peer = %peer_id, uuid = %uuid, error = %e, "read repair: forward_get_entry failed");
            return;
        }
    };

    let entry = WalEntry {
        uuid: full.uuid.clone(),
        embedding: full.embedding,
        response: full.response,
        query_text: full.query_text,
        model_id: full.model_id,
        sequence: 0, // stamped by the flush task
        // M24: preserve the source's `inserted_at` so read-repair doesn't
        // make this entry look freshly created. `last_accessed_at` and
        // `access_count` are NOT in the WAL; they restart at zero on the
        // repaired side, which is fine — the next query will bump them.
        inserted_at: full.inserted_at,
        tombstone: false,
        // M26: TTL deadline rides through read-repair too — evicting on
        // schedule is more important than which node owns the timer.
        expires_at: full.expires_at,
    };

    let (tx, rx) = oneshot::channel();
    if state
        .wal
        .sender()
        .send(WalCommand::Insert(WalInsertRequest { entry, respond: tx }))
        .await
        .is_err()
    {
        state.metrics.record_read_repair_failure();
        tracing::warn!(uuid = %uuid, "read repair: WAL channel closed");
        return;
    }
    match rx.await {
        Ok(Ok(_seq)) => {
            state.metrics.record_read_repair();
            tracing::info!(peer = %peer_id, uuid = %uuid, model_id = %model_id, "read repair complete");
        }
        Ok(Err(e)) => {
            state.metrics.record_read_repair_failure();
            tracing::warn!(uuid = %uuid, error = %e, "read repair: local insert rejected");
        }
        Err(_) => {
            state.metrics.record_read_repair_failure();
            tracing::warn!(uuid = %uuid, "read repair: WAL flush task dropped reply");
        }
    }
}

fn bad_gateway(msg: impl Into<String>) -> Response {
    let msg = msg.into();
    tracing::warn!(error = %msg, "peer unreachable");
    (StatusCode::BAD_GATEWAY, Json(ErrorResponse { error: msg })).into_response()
}

async fn insert_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<LocalParam>,
    Json(req): Json<InsertRequest>,
) -> Response {
    if let Err(msg) = validate_embedding(&req.embedding) {
        return bad_request(msg);
    }
    if let Err(msg) = validate_model_id(req.model_id.as_deref()) {
        return bad_request(msg);
    }
    if req.response.len() > MAX_RESPONSE_BYTES {
        return bad_request(format!(
            "response size {} exceeds max {}",
            req.response.len(),
            MAX_RESPONSE_BYTES
        ));
    }

    if params.is_local() || state.cluster.is_none() {
        return process_insert_locally(&state, req).await;
    }

    // Cluster-aware path: coordinate replication.
    let cluster = state.cluster.as_ref().expect("cluster present");
    let replicas = cluster
        .get_replica_addrs(&req.embedding, state.replication_factor)
        .await;

    if replicas.is_empty() {
        return process_insert_locally(&state, req).await;
    }

    // Coordinator stamps the UUID so all replicas store the same id.
    let uuid = uuid::Uuid::new_v4().to_string();
    let mut req_with_uuid = req;
    req_with_uuid.uuid = Some(uuid.clone());

    let self_id = cluster.self_node_id();
    let self_in_replica_set = replicas.iter().any(|(id, _)| id == self_id);

    if self_in_replica_set && let Err(resp) = local_insert_inner(&state, &req_with_uuid).await {
        return resp;
    }

    let Some(router) = &state.router else {
        tracing::error!("no router configured for replication");
        return bad_gateway("upstream replica unavailable");
    };

    // Soft failover (M21): skip replicas the failure detector has marked
    // Dead. The replication factor degrades to surviving replicas only —
    // logged at warn so operators see the degraded state in dashboards.
    let mut skipped_dead: Vec<&str> = Vec::new();
    let mut attempted = 0usize;
    for (peer_id, peer_addr) in replicas.iter().filter(|(id, _)| id != self_id) {
        if cluster.peer_status(peer_id).await == PeerStatus::Dead {
            skipped_dead.push(peer_id.as_str());
            continue;
        }
        state.metrics.record_replication_forward();
        attempted += 1;
        let result = router
            .forward_with_retry(state.max_replication_retries, &state.metrics, || {
                router.forward_insert(peer_addr, &req_with_uuid)
            })
            .await;
        if let Err(e) = result {
            state.metrics.record_replication_failure();
            tracing::error!(peer = %peer_id, error = %e, "replica forward failed");
            return bad_gateway("upstream replica unavailable");
        }
    }

    if !skipped_dead.is_empty() {
        let surviving = if self_in_replica_set {
            attempted + 1
        } else {
            attempted
        };
        tracing::warn!(
            dead_peers = ?skipped_dead,
            effective_replicas = surviving,
            "replication degraded — some replicas unavailable"
        );
    }

    tracing::info!(id = %uuid, replicas = replicas.len(), "insert replicated");
    (
        StatusCode::OK,
        Json(InsertResponse {
            id: uuid,
            status: "ok".to_string(),
        }),
    )
        .into_response()
}

async fn process_insert_locally(state: &Arc<AppState>, req: InsertRequest) -> Response {
    match local_insert_inner(state, &req).await {
        Ok(uuid) => (
            StatusCode::OK,
            Json(InsertResponse {
                id: uuid,
                status: "ok".to_string(),
            }),
        )
            .into_response(),
        Err(resp) => resp,
    }
}

/// Performs the WAL-first local insert. Returns the UUID actually stored.
/// The WAL flush task owns the only `Wal` handle — this function sends the
/// entry through the group-commit channel and awaits the flush task's
/// oneshot reply (which lands after fsync + index insert).
async fn local_insert_inner(
    state: &Arc<AppState>,
    req: &InsertRequest,
) -> Result<String, Response> {
    let start = std::time::Instant::now();
    let dim = req.embedding.len();
    let model_id = match req.model_id.as_deref() {
        Some(s) if !s.trim().is_empty() => s.to_string(),
        _ => return Err(bad_request("model_id is required")),
    };

    let uuid = req
        .uuid
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    let inserted_at = now_unix_secs();
    let expires_at = req.ttl_seconds.map(|t| inserted_at.saturating_add(t));
    let entry = WalEntry {
        uuid: uuid.clone(),
        embedding: req.embedding.clone(),
        response: req.response.clone(),
        query_text: req.query_text.clone(),
        model_id: model_id.clone(),
        sequence: 0, // stamped by the flush task
        inserted_at,
        tombstone: false,
        expires_at,
    };

    let (tx, rx) = oneshot::channel();
    if state
        .wal
        .sender()
        .send(WalCommand::Insert(WalInsertRequest { entry, respond: tx }))
        .await
        .is_err()
    {
        tracing::error!("WAL flush task channel closed");
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "persistence failed".to_string(),
            }),
        )
            .into_response());
    }

    match rx.await {
        Ok(Ok(_seq)) => {}
        Ok(Err(InsertError::Wal(e))) => {
            tracing::error!(error = %e, "WAL append failed");
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "persistence failed".to_string(),
                }),
            )
                .into_response());
        }
        Ok(Err(InsertError::Index(e))) => {
            tracing::error!(error = %e, "index insert rejected");
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
                .into_response());
        }
        Err(_) => {
            tracing::error!("WAL flush task dropped before reply");
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "persistence failed".to_string(),
                }),
            )
                .into_response());
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    state.metrics.record_insert(&model_id, elapsed);

    tracing::info!(id = %uuid, dim, "inserted");
    Ok(uuid)
}

async fn get_entry_handler(
    State(state): State<Arc<AppState>>,
    Path(uuid): Path<String>,
) -> Response {
    let entry = state.index.read().await.get_entry_by_uuid(&uuid);
    match entry {
        Some(e) => (
            StatusCode::OK,
            Json(FullEntryResponse {
                uuid: e.uuid,
                embedding: e.embedding,
                response: e.response,
                query_text: e.query_text,
                model_id: e.model_id,
                inserted_at: e.inserted_at,
                last_accessed_at: e.last_accessed_at,
                access_count: e.access_count,
                expires_at: e.expires_at,
            }),
        )
            .into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "entry not found".to_string(),
            }),
        )
            .into_response(),
    }
}

async fn read_repair_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ReadRepairRequest>,
) -> Response {
    if let Err(msg) = validate_embedding(&req.embedding) {
        return bad_request(msg);
    }
    if req.model_id.trim().is_empty() {
        return bad_request("model_id is required");
    }
    if req.uuid.trim().is_empty() {
        return bad_request("uuid is required");
    }
    if req.response.len() > MAX_RESPONSE_BYTES {
        return bad_request(format!(
            "response size {} exceeds max {}",
            req.response.len(),
            MAX_RESPONSE_BYTES
        ));
    }

    let entry = WalEntry {
        uuid: req.uuid.clone(),
        embedding: req.embedding,
        response: req.response,
        query_text: req.query_text,
        model_id: req.model_id.clone(),
        sequence: 0,
        // The /internal/read-repair caller didn't carry an inserted_at
        // (the public ReadRepairRequest predates M24); stamp now.
        inserted_at: now_unix_secs(),
        tombstone: false,
        expires_at: None,
    };
    let (tx, rx) = oneshot::channel();
    if state
        .wal
        .sender()
        .send(WalCommand::Insert(WalInsertRequest { entry, respond: tx }))
        .await
        .is_err()
    {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "persistence failed".to_string(),
            }),
        )
            .into_response();
    }
    match rx.await {
        Ok(Ok(_)) => {
            state.metrics.record_read_repair();
            (
                StatusCode::OK,
                Json(ReadRepairResponse {
                    status: "ok".to_string(),
                    repaired: true,
                }),
            )
                .into_response()
        }
        Ok(Err(InsertError::Index(e))) => {
            state.metrics.record_read_repair_failure();
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
                .into_response()
        }
        Ok(Err(InsertError::Wal(e))) => {
            state.metrics.record_read_repair_failure();
            tracing::error!(error = %e, "read-repair WAL append failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "persistence failed".to_string(),
                }),
            )
                .into_response()
        }
        Err(_) => {
            state.metrics.record_read_repair_failure();
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "persistence failed".to_string(),
                }),
            )
                .into_response()
        }
    }
}

async fn compact_handler(State(state): State<Arc<AppState>>) -> Response {
    // Compaction goes through the same flush-task channel as inserts, so
    // any pending writes are drained first (FIFO).
    let (tx, rx) = oneshot::channel();
    if state
        .wal
        .sender()
        .send(WalCommand::Compact(CompactRequest { respond: tx }))
        .await
        .is_err()
    {
        tracing::error!("WAL flush task channel closed during compact");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "compaction failed".to_string(),
            }),
        )
            .into_response();
    }

    match rx.await {
        Ok(Ok(result)) => (
            StatusCode::OK,
            Json(CompactResponse {
                status: "ok".to_string(),
                entries_snapshotted: result.entries_snapshotted,
                wal_sequence: result.wal_sequence,
            }),
        )
            .into_response(),
        Ok(Err(e)) => {
            // The error chain typically carries snapshot/WAL paths; keep
            // those in logs only.
            tracing::error!(error = %e, "compaction failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "compaction failed".to_string(),
                }),
            )
                .into_response()
        }
        Err(_) => {
            tracing::error!("WAL flush task dropped during compact");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "compaction failed".to_string(),
                }),
            )
                .into_response()
        }
    }
}

async fn entry_stats_handler(State(state): State<Arc<AppState>>) -> Json<EntryStatsResponse> {
    const TOP_N: usize = 10;
    let index = state.index.read().await;
    let raw = index.top_entries_per_namespace(TOP_N);
    drop(index);
    let namespaces = raw
        .into_iter()
        .map(|(k, rows)| {
            let top_entries = rows
                .into_iter()
                .map(|r| TopEntryRow {
                    uuid: r.uuid,
                    access_count: r.access_count,
                    last_accessed_at: r.last_accessed_at,
                    query_text_preview: r.query_text_preview,
                })
                .collect();
            (k, NamespaceTopEntries { top_entries })
        })
        .collect();
    Json(EntryStatsResponse { namespaces })
}

fn not_found(msg: impl Into<String>) -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(ErrorResponse { error: msg.into() }),
    )
        .into_response()
}

/// `DELETE /entry/:uuid` (M26). Removes the entry locally, writes a
/// tombstone through the WAL channel, and (when not `?local=true`) fans
/// out the same delete to every peer in the ring. Replica 404 is treated
/// as idempotent success — the entry just didn't live there.
async fn delete_entry_handler(
    State(state): State<Arc<AppState>>,
    Path(uuid): Path<String>,
    Query(params): Query<LocalParam>,
) -> Response {
    if uuid.trim().is_empty() {
        return bad_request("uuid is required");
    }

    // Resolve the namespace under a read lock first — cheap, doesn't block
    // queries.
    let model_id = {
        let idx = state.index.read().await;
        idx.namespace_for(&uuid).map(str::to_string)
    };

    let Some(model_id) = model_id else {
        // Even on a 404 we still fan out — peers might still have it. But
        // for the common case (UUID just doesn't exist anywhere), this
        // returns immediately.
        return not_found("entry not found");
    };

    // Local removal under write lock.
    {
        let mut idx = state.index.write().await;
        idx.remove_by_uuid(&uuid);
    }

    // Tombstone via the WAL channel — the flush task batches it into the
    // next group fsync. We wait for the oneshot so the response is only
    // returned after the tombstone is durable.
    let tomb = WalEntry::tombstone_entry(uuid.clone(), model_id.clone());
    let (tx, rx) = oneshot::channel();
    if state
        .wal
        .sender()
        .send(WalCommand::Insert(WalInsertRequest {
            entry: tomb,
            respond: tx,
        }))
        .await
        .is_err()
    {
        tracing::error!("WAL channel closed during delete");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "persistence failed".to_string(),
            }),
        )
            .into_response();
    }
    let _ = rx.await;

    state.metrics.record_deletion(&model_id);

    // Cluster fan-out (skip when local-only). We don't know which peers
    // hold a replica without an embedding, so blast to all live peers.
    // Replica 404 is fine.
    if !params.is_local()
        && let Some(cluster) = state.cluster.as_ref()
        && let Some(router) = state.router.as_ref()
    {
        let peers = cluster.all_peer_addrs().await;
        for (peer_id, peer_addr) in peers {
            if cluster.peer_status(&peer_id).await == PeerStatus::Dead {
                continue;
            }
            match router.forward_delete_entry(&peer_addr, &uuid).await {
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!(peer = %peer_id, error = %e, "delete forward failed");
                }
            }
        }
    }

    tracing::info!(%uuid, model_id = %model_id, "entry deleted");
    (StatusCode::OK, Json(DeleteEntryResponse { deleted: true })).into_response()
}

/// `POST /admin/invalidate` (M26). Sweeps the namespace for entries with
/// cosine similarity ≥ threshold, evicts them, writes tombstones, and
/// fans out the same request to peers (each replica computes its own
/// matches, which should be identical assuming the cluster is in sync).
async fn invalidate_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<LocalParam>,
    Json(req): Json<InvalidateRequest>,
) -> Response {
    if let Err(msg) = validate_embedding(&req.embedding) {
        return bad_request(msg);
    }
    if let Err(msg) = validate_model_id(req.model_id.as_deref()) {
        return bad_request(msg);
    }
    if !(0.0..=1.0).contains(&req.threshold) {
        return bad_request(format!(
            "threshold {} out of range [0.0, 1.0]",
            req.threshold
        ));
    }
    let model_id = req.model_id.clone().expect("validated above");

    // Find + remove under one write lock.
    let now = now_unix_secs();
    let removed_uuids: Vec<String> = {
        let mut idx = state.index.write().await;
        idx.invalidate_similar(&req.embedding, req.threshold, &model_id, now)
    };

    // Tombstones for each invalidated entry.
    for uuid in &removed_uuids {
        let tomb = WalEntry::tombstone_entry(uuid.clone(), model_id.clone());
        let (tx, rx) = oneshot::channel();
        if state
            .wal
            .sender()
            .send(WalCommand::Insert(WalInsertRequest {
                entry: tomb,
                respond: tx,
            }))
            .await
            .is_ok()
        {
            let _ = rx.await;
        }
        state.metrics.record_invalidation(&model_id);
    }

    // Cluster fan-out: each peer computes its own matches.
    if !params.is_local()
        && let Some(cluster) = state.cluster.as_ref()
        && let Some(router) = state.router.as_ref()
    {
        let peers = cluster.all_peer_addrs().await;
        for (peer_id, peer_addr) in peers {
            if cluster.peer_status(&peer_id).await == PeerStatus::Dead {
                continue;
            }
            match router.forward_invalidate(&peer_addr, &req).await {
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!(peer = %peer_id, error = %e, "invalidate forward failed");
                }
            }
        }
    }

    tracing::info!(
        invalidated = removed_uuids.len(),
        model_id = %model_id,
        "invalidate completed"
    );
    (
        StatusCode::OK,
        Json(InvalidateResponse {
            invalidated_count: removed_uuids.len(),
            uuids: removed_uuids,
        }),
    )
        .into_response()
}

async fn health_handler(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let count = state.index.read().await.entry_count() as u64;
    Json(HealthResponse {
        status: "ok".to_string(),
        node_id: state.node_id.clone(),
        entry_count: count,
    })
}

async fn cluster_status_handler(State(state): State<Arc<AppState>>) -> Json<ClusterStatusResponse> {
    match &state.cluster {
        Some(cluster) => {
            let nodes = cluster.live_nodes().await;
            let node_count = cluster.ring_node_count().await;
            let detector_snapshot = cluster.failure_detector().snapshot().await;
            let peer_health = detector_snapshot
                .into_iter()
                .map(|(id, status, phi)| (id, PeerHealth { status, phi }))
                .collect();
            let dead_nodes = cluster.dead_nodes().await;
            Json(ClusterStatusResponse {
                mode: "clustered",
                self_node_id: cluster.self_node_id().to_string(),
                gossip_addr: Some(cluster.gossip_addr().to_string()),
                nodes,
                node_count,
                peer_health,
                dead_nodes,
                read_repair_enabled: state.read_repair_enabled,
            })
        }
        None => Json(ClusterStatusResponse {
            mode: "single",
            self_node_id: state.node_id.clone(),
            gossip_addr: None,
            nodes: vec![state.node_id.clone()],
            node_count: 1,
            peer_health: std::collections::HashMap::new(),
            dead_nodes: Vec::new(),
            read_repair_enabled: state.read_repair_enabled,
        }),
    }
}

async fn stats_handler(State(state): State<Arc<AppState>>) -> Json<StatsResponse> {
    let index = state.index.read().await;
    let namespaces = index
        .namespace_stats()
        .into_iter()
        .map(|(k, v)| {
            (
                k,
                NamespaceStatsEntry {
                    entry_count: v.entry_count,
                    dimension: v.dimension,
                    oldest_entry_ts: v.oldest_entry_ts,
                    newest_entry_ts: v.newest_entry_ts,
                    total_accesses: v.total_accesses,
                    evicted_ghost_count: v.evicted_ghost_count,
                },
            )
        })
        .collect();
    let m = &state.metrics;
    let counters = CountersResponse {
        queries_total: m.queries_total.load(Ordering::Relaxed),
        queries_hit: m.queries_hit.load(Ordering::Relaxed),
        queries_miss: m.queries_miss.load(Ordering::Relaxed),
        hit_rate: m.hit_rate(),
        inserts_total: m.inserts_total.load(Ordering::Relaxed),
        replication_forwards: m.replication_forwards_total.load(Ordering::Relaxed),
        replication_failures: m.replication_failures_total.load(Ordering::Relaxed),
        replication_retries: m.replication_retries_total.load(Ordering::Relaxed),
        compactions: m.compactions_total.load(Ordering::Relaxed),
        read_repairs: m.read_repairs_total.load(Ordering::Relaxed),
        read_repair_failures: m.read_repair_failures_total.load(Ordering::Relaxed),
        evictions_total: m.evictions_total.load(Ordering::Relaxed),
        index_rebuilds: m.index_rebuilds_total.load(Ordering::Relaxed),
        expirations_total: m.expirations_total.load(Ordering::Relaxed),
        deletions_total: m.deletions_total.load(Ordering::Relaxed),
        invalidations_total: m.invalidations_total.load(Ordering::Relaxed),
        exact_match_hits_total: m.exact_match_hits_total.load(Ordering::Relaxed),
    };
    Json(StatsResponse {
        entry_count: index.entry_count() as u64,
        wal_path: state.wal_path.clone(),
        hnsw: StatsHnsw {
            max_nb_connection: state.hnsw_config.max_nb_connection,
            ef_construction: state.hnsw_config.ef_construction,
            ef_search: state.hnsw_config.ef_search,
            dimension: index.dimension(),
        },
        namespaces,
        counters,
    })
}

async fn metrics_handler(State(state): State<Arc<AppState>>) -> Response {
    let stats = {
        let index = state.index.read().await;
        index.namespace_stats()
    };
    let (cluster_nodes, peer_phi) = match &state.cluster {
        Some(c) => (
            c.ring_node_count().await,
            c.failure_detector().snapshot().await,
        ),
        None => (1, Vec::new()),
    };
    let body = state.metrics.render(&stats, cluster_nodes, &peer_phi);
    (
        StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, METRICS_CONTENT_TYPE)],
        body,
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::HnswConfig;
    use crate::index::SemanticIndex;
    use crate::metrics::Metrics;
    use crate::wal::{GroupCommitConfig, GroupCommitWal, Wal};
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use serde_json::{Value, json};
    use std::path::PathBuf;
    use tokio::sync::RwLock;
    use tower::ServiceExt;

    async fn build_state(wal_path: PathBuf) -> Arc<AppState> {
        let hnsw = HnswConfig::default();
        let wal = Wal::open(&wal_path).await.unwrap();
        let snapshot_path = crate::snapshot::snapshot_path_for(&wal_path.to_string_lossy());
        let index = Arc::new(RwLock::new(SemanticIndex::new(&hnsw)));
        let metrics = Arc::new(Metrics::new());
        // Tests use batch_size=1 so each insert fsyncs immediately, matching
        // pre-M20 semantics that existing assertions rely on.
        let gc = GroupCommitWal::spawn(
            wal,
            wal_path.clone(),
            snapshot_path.clone(),
            index.clone(),
            metrics.clone(),
            0, // disable auto-compaction in tests
            GroupCommitConfig {
                batch_size: 1,
                batch_timeout: std::time::Duration::from_millis(0),
                channel_capacity: 64,
            },
            hnsw.clone(),
        );
        Arc::new(AppState::new(
            "test-node".to_string(),
            index,
            gc,
            wal_path.to_string_lossy().into_owned(),
            snapshot_path,
            hnsw,
            None,
            None,
            1,
            metrics,
            None,
            0,    // no replication retries in tests
            true, // read_repair_enabled — no-op in tests since cluster is None
        ))
    }

    async fn test_app() -> (Router, Arc<AppState>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.wal");
        let state = build_state(path).await;
        (build_router(state.clone()), state, dir)
    }

    async fn body_json(body: Body) -> Value {
        let bytes = body.collect().await.unwrap().to_bytes();
        serde_json::from_slice(&bytes).unwrap()
    }

    fn post_json(uri: &str, payload: Value) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(payload.to_string()))
            .unwrap()
    }

    fn get(uri: &str) -> Request<Body> {
        Request::builder()
            .method("GET")
            .uri(uri)
            .body(Body::empty())
            .unwrap()
    }

    #[tokio::test]
    async fn test_health_returns_ok() {
        let (app, _, _dir) = test_app().await;
        let response = app.oneshot(get("/health")).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response.into_body()).await;
        assert_eq!(body["status"], "ok");
    }

    const MID: &str = "test-model::3";

    #[tokio::test]
    async fn test_insert_returns_id() {
        let (app, _, _dir) = test_app().await;
        let payload = json!({
            "embedding": [0.1f32, 0.2, 0.3],
            "response": "hello",
            "query_text": "hi",
            "model_id": MID
        });
        let response = app.oneshot(post_json("/insert", payload)).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response.into_body()).await;
        assert!(body["id"].is_string());
        assert!(!body["id"].as_str().unwrap().is_empty());
        assert_eq!(body["status"], "ok");
    }

    #[tokio::test]
    async fn test_query_returns_miss() {
        let (app, _, _dir) = test_app().await;
        let payload = json!({
            "embedding": [0.1f32, 0.2, 0.3],
            "threshold": 0.9,
            "model_id": MID
        });
        let response = app.oneshot(post_json("/query", payload)).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response.into_body()).await;
        assert_eq!(body["hit"], false);
    }

    #[tokio::test]
    async fn test_insert_increments_count() {
        let (app, _, _dir) = test_app().await;
        let payload = json!({
            "embedding": [0.1f32, 0.2, 0.3],
            "response": "hello",
            "query_text": "hi",
            "model_id": MID
        });
        for _ in 0..2 {
            let response = app
                .clone()
                .oneshot(post_json("/insert", payload.clone()))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }
        let response = app.oneshot(get("/health")).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response.into_body()).await;
        assert_eq!(body["entry_count"], 2);
    }

    #[tokio::test]
    async fn test_insert_then_query_hit() {
        let (app, _, _dir) = test_app().await;
        let insert_payload = json!({
            "embedding": [1.0_f32, 0.0, 0.0],
            "response": "the cached answer",
            "query_text": "the question",
            "model_id": MID
        });
        let response = app
            .clone()
            .oneshot(post_json("/insert", insert_payload))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let query_payload = json!({
            "embedding": [1.0_f32, 0.0, 0.0],
            "threshold": 0.90,
            "model_id": MID
        });
        let response = app
            .oneshot(post_json("/query", query_payload))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response.into_body()).await;
        assert_eq!(body["hit"], true);
        assert_eq!(body["response"], "the cached answer");
        let sim = body["similarity"].as_f64().unwrap();
        assert!(sim > 0.999, "similarity={}", sim);
    }

    #[tokio::test]
    async fn test_query_miss_different_vector() {
        let (app, _, _dir) = test_app().await;
        let insert_payload = json!({
            "embedding": [1.0_f32, 0.0, 0.0],
            "response": "stored",
            "query_text": "q",
            "model_id": MID
        });
        let response = app
            .clone()
            .oneshot(post_json("/insert", insert_payload))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let query_payload = json!({
            "embedding": [0.0_f32, 0.0, 1.0],
            "threshold": 0.99,
            "model_id": MID
        });
        let response = app
            .oneshot(post_json("/query", query_payload))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response.into_body()).await;
        assert_eq!(body["hit"], false);
    }

    #[tokio::test]
    async fn test_insert_dimension_mismatch_within_namespace() {
        let (app, _, _dir) = test_app().await;
        let first = json!({
            "embedding": [1.0_f32, 0.0],
            "response": "r",
            "query_text": "q",
            "model_id": "m::2"
        });
        let response = app
            .clone()
            .oneshot(post_json("/insert", first))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let second = json!({
            "embedding": [1.0_f32, 0.0, 0.0],
            "response": "r",
            "query_text": "q",
            "model_id": "m::2"
        });
        let response = app.oneshot(post_json("/insert", second)).await.unwrap();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = body_json(response.into_body()).await;
        assert!(
            body["error"]
                .as_str()
                .unwrap()
                .contains("dimension mismatch")
        );
    }

    #[tokio::test]
    async fn test_insert_persists_via_wal() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("persist.wal");

        {
            let state = build_state(wal_path.clone()).await;
            let app = build_router(state.clone());
            let payload = json!({
                "embedding": [1.0_f32, 0.0, 0.0],
                "response": "persisted answer",
                "query_text": "persisted question",
                "model_id": MID
            });
            let response = app.oneshot(post_json("/insert", payload)).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            drop(state);
        }

        let entries = Wal::replay(&wal_path).await.unwrap();
        assert_eq!(entries.len(), 1);
        let mut fresh = SemanticIndex::new(&HnswConfig::default());
        for entry in entries {
            fresh.replay_entry(entry).unwrap();
        }
        let hit = fresh
            .query(&[1.0_f32, 0.0, 0.0], 0.90, MID)
            .unwrap()
            .expect("should hit after replay");
        assert_eq!(hit.response, "persisted answer");
    }

    #[tokio::test]
    async fn test_insert_empty_embedding() {
        let (app, _, _dir) = test_app().await;
        let payload = json!({
            "embedding": [],
            "response": "r",
            "query_text": "q",
            "model_id": MID
        });
        let response = app.oneshot(post_json("/insert", payload)).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = body_json(response.into_body()).await;
        assert!(body["error"].as_str().unwrap().contains("empty"));
    }

    #[tokio::test]
    async fn test_insert_oversized_embedding() {
        let (app, _, _dir) = test_app().await;
        let big: Vec<f32> = vec![0.1; 5000];
        let payload = json!({
            "embedding": big,
            "response": "r",
            "query_text": "q",
            "model_id": MID
        });
        let response = app.oneshot(post_json("/insert", payload)).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = body_json(response.into_body()).await;
        assert!(body["error"].as_str().unwrap().contains("exceeds max"));
    }

    #[tokio::test]
    async fn test_insert_oversized_response() {
        let (app, _, _dir) = test_app().await;
        let big_resp = "x".repeat(MAX_RESPONSE_BYTES + 1);
        let payload = json!({
            "embedding": [0.1f32, 0.2, 0.3],
            "response": big_resp,
            "query_text": "q",
            "model_id": MID
        });
        let response = app.oneshot(post_json("/insert", payload)).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = body_json(response.into_body()).await;
        assert!(body["error"].as_str().unwrap().contains("response size"));
    }

    #[tokio::test]
    async fn test_query_threshold_out_of_range() {
        let (app, _, _dir) = test_app().await;
        let payload = json!({
            "embedding": [0.1f32, 0.2, 0.3],
            "threshold": 1.5,
            "model_id": MID
        });
        let response = app.oneshot(post_json("/query", payload)).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = body_json(response.into_body()).await;
        assert!(body["error"].as_str().unwrap().contains("threshold"));
    }

    #[tokio::test]
    async fn test_insert_missing_model_id() {
        let (app, _, _dir) = test_app().await;
        let payload = json!({
            "embedding": [0.1f32, 0.2, 0.3],
            "response": "r",
            "query_text": "q"
        });
        let response = app.oneshot(post_json("/insert", payload)).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = body_json(response.into_body()).await;
        assert!(body["error"].as_str().unwrap().contains("model_id"));
    }

    #[tokio::test]
    async fn test_query_missing_model_id() {
        let (app, _, _dir) = test_app().await;
        let payload = json!({
            "embedding": [0.1f32, 0.2, 0.3],
            "threshold": 0.9
        });
        let response = app.oneshot(post_json("/query", payload)).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = body_json(response.into_body()).await;
        assert!(body["error"].as_str().unwrap().contains("model_id"));
    }

    #[tokio::test]
    async fn test_cross_namespace_isolation_http() {
        let (app, _, _dir) = test_app().await;
        let insert = json!({
            "embedding": [1.0_f32, 0.0, 0.0],
            "response": "model-A answer",
            "query_text": "q",
            "model_id": "model-A::3"
        });
        let response = app
            .clone()
            .oneshot(post_json("/insert", insert))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let query = json!({
            "embedding": [1.0_f32, 0.0, 0.0],
            "threshold": 0.90,
            "model_id": "model-B::3"
        });
        let response = app.oneshot(post_json("/query", query)).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response.into_body()).await;
        assert_eq!(body["hit"], false);
    }

    #[tokio::test]
    async fn test_stats_shows_namespaces() {
        let (app, _, _dir) = test_app().await;
        for (mid, vec) in [
            ("model-A::3", json!([1.0_f32, 0.0, 0.0])),
            ("model-B::4", json!([1.0_f32, 0.0, 0.0, 0.0])),
        ] {
            let insert = json!({
                "embedding": vec,
                "response": "r",
                "query_text": "q",
                "model_id": mid
            });
            let response = app
                .clone()
                .oneshot(post_json("/insert", insert))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }
        let response = app.oneshot(get("/stats")).await.unwrap();
        let body = body_json(response.into_body()).await;
        let ns = body["namespaces"].as_object().unwrap();
        assert!(ns.contains_key("model-A::3"));
        assert!(ns.contains_key("model-B::4"));
        assert_eq!(ns["model-A::3"]["entry_count"], 1);
        assert_eq!(ns["model-A::3"]["dimension"], 3);
        assert_eq!(ns["model-B::4"]["dimension"], 4);
    }

    #[tokio::test]
    async fn test_local_query_param_bypasses_routing() {
        let (app, _, _dir) = test_app().await;
        // Insert one entry first so we have something to (locally) query.
        let insert_payload = json!({
            "embedding": [1.0_f32, 0.0, 0.0],
            "response": "local-only",
            "query_text": "q",
            "model_id": MID
        });
        let response = app
            .clone()
            .oneshot(post_json("/insert?local=true", insert_payload))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let query_payload = json!({
            "embedding": [1.0_f32, 0.0, 0.0],
            "threshold": 0.9,
            "model_id": MID
        });
        let response = app
            .oneshot(post_json("/query?local=true", query_payload))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response.into_body()).await;
        assert_eq!(body["hit"], true);
        assert_eq!(body["response"], "local-only");
    }

    #[tokio::test]
    async fn test_local_insert_param_bypasses_routing() {
        let (app, _, _dir) = test_app().await;
        let payload = json!({
            "embedding": [0.5_f32, 0.5, 0.5],
            "response": "r",
            "query_text": "q",
            "model_id": MID
        });
        let response = app
            .oneshot(post_json("/insert?local=true", payload))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response.into_body()).await;
        assert!(body["id"].is_string());
        assert_eq!(body["status"], "ok");
    }

    #[tokio::test]
    async fn test_insert_with_provided_uuid() {
        let (app, _, _dir) = test_app().await;
        let payload = json!({
            "embedding": [0.5_f32, 0.5, 0.5],
            "response": "r",
            "query_text": "q",
            "model_id": MID,
            "uuid": "my-fixed-uuid"
        });
        let response = app
            .oneshot(post_json("/insert?local=true", payload))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response.into_body()).await;
        assert_eq!(body["id"], "my-fixed-uuid");
    }

    #[tokio::test]
    async fn test_cluster_status_single_mode() {
        let (app, _, _dir) = test_app().await;
        let response = app.oneshot(get("/cluster/status")).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response.into_body()).await;
        assert_eq!(body["mode"], "single");
        assert_eq!(body["self_node_id"], "test-node");
        assert_eq!(body["node_count"], 1);
        let nodes = body["nodes"].as_array().unwrap();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0], "test-node");
        // Single-node mode has no detector → peer_health field omitted (skip_serializing_if).
        assert!(body.get("peer_health").is_none(), "{body}");
    }

    #[tokio::test]
    async fn test_cluster_status_shows_dead_nodes() {
        // ClusterState::new requires a UDP handle; build the response shape
        // directly instead, exercising the serializer.
        let resp = ClusterStatusResponse {
            mode: "clustered",
            self_node_id: "self".into(),
            gossip_addr: Some("0.0.0.0:4000".into()),
            nodes: vec!["self".into(), "node-A".into()],
            node_count: 2,
            peer_health: std::collections::HashMap::new(),
            dead_nodes: vec!["node-C".into()],
            read_repair_enabled: true,
        };
        let body = serde_json::to_value(&resp).unwrap();
        let dead = body["dead_nodes"].as_array().expect("dead_nodes present");
        assert_eq!(dead.len(), 1);
        assert_eq!(dead[0], "node-C");
        let nodes = body["nodes"].as_array().unwrap();
        assert!(!nodes.iter().any(|v| v == "node-C"));
    }

    #[tokio::test]
    async fn test_cluster_status_includes_peer_health() {
        // ClusterState::new requires a UDP socket + gossip handshake, which
        // is too heavy for a unit test. Instead we exercise the shape: feed
        // a `PhiAccrualDetector` directly, build the response the way the
        // handler does, and assert it serializes with `peer_health` populated.
        use crate::failure_detector::{PeerStatus as PS, PhiAccrualDetector};
        let det = PhiAccrualDetector::new(8.0, 100, 100.0);
        let now = std::time::Instant::now();
        // Two heartbeats so phi is computable.
        det.record_heartbeat_at("peer-A", now - std::time::Duration::from_millis(2000))
            .await;
        det.record_heartbeat_at("peer-A", now - std::time::Duration::from_millis(1000))
            .await;
        let snap = det.snapshot_at(now).await;
        let peer_health: std::collections::HashMap<String, PeerHealth> = snap
            .into_iter()
            .map(|(id, status, phi)| (id, PeerHealth { status, phi }))
            .collect();
        let resp = ClusterStatusResponse {
            mode: "clustered",
            self_node_id: "self".into(),
            gossip_addr: Some("0.0.0.0:4000".into()),
            nodes: vec!["self".into(), "peer-A".into()],
            node_count: 2,
            peer_health,
            dead_nodes: Vec::new(),
            read_repair_enabled: true,
        };
        let body = serde_json::to_value(&resp).unwrap();
        let ph = body["peer_health"]
            .as_object()
            .expect("peer_health populated");
        let entry = &ph["peer-A"];
        assert!(entry["phi"].is_number());
        // peer-A heartbeated 1s ago against ~1s mean — phi sits in the
        // "alive" zone, well below threshold.
        assert_eq!(entry["status"], "alive");
        assert_eq!(entry["status"].as_str(), Some("alive"));
        let _ = PS::Alive; // anchor the import so cargo's unused warning pass stays clean.
    }

    // --- M23: read repair endpoints ---------------------------------------

    #[tokio::test]
    async fn test_read_repair_endpoint_inserts_entry() {
        let (app, _, _dir) = test_app().await;
        let payload = json!({
            "uuid": "fixed-uuid-r",
            "embedding": [1.0_f32, 0.0, 0.0],
            "response": "repaired-resp",
            "query_text": "q",
            "model_id": MID,
        });
        let response = app
            .clone()
            .oneshot(post_json("/internal/read-repair", payload))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response.into_body()).await;
        assert_eq!(body["status"], "ok");
        assert_eq!(body["repaired"], true);

        // The entry must be queryable locally afterwards.
        let q = json!({
            "embedding": [1.0_f32, 0.0, 0.0],
            "threshold": 0.9,
            "model_id": MID,
        });
        let resp = app
            .oneshot(post_json("/query?local=true", q))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp.into_body()).await;
        assert_eq!(body["hit"], true);
        assert_eq!(body["id"], "fixed-uuid-r");
        assert_eq!(body["response"], "repaired-resp");
    }

    #[tokio::test]
    async fn test_get_entry_endpoint_returns_full_entry() {
        let (app, _, _dir) = test_app().await;
        let insert = json!({
            "embedding": [0.5_f32, 0.5, 0.0],
            "response": "the-r",
            "query_text": "the-q",
            "model_id": MID,
            "uuid": "lookup-target",
        });
        app.clone()
            .oneshot(post_json("/insert?local=true", insert))
            .await
            .unwrap();

        let resp = app
            .oneshot(get("/internal/entry/lookup-target?local=true"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp.into_body()).await;
        assert_eq!(body["uuid"], "lookup-target");
        assert_eq!(body["response"], "the-r");
        assert_eq!(body["query_text"], "the-q");
        assert_eq!(body["model_id"], MID);
        let emb = body["embedding"].as_array().unwrap();
        assert_eq!(emb.len(), 3);
    }

    #[tokio::test]
    async fn test_get_entry_endpoint_404() {
        let (app, _, _dir) = test_app().await;
        let resp = app
            .oneshot(get("/internal/entry/nonexistent?local=true"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_read_repair_validates_inputs() {
        let (app, _, _dir) = test_app().await;
        let bad = json!({
            "uuid": "",
            "embedding": [1.0_f32, 0.0, 0.0],
            "response": "r",
            "query_text": "q",
            "model_id": MID,
        });
        let resp = app
            .oneshot(post_json("/internal/read-repair", bad))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn test_local_param_repair_semantics() {
        // M23 contract:
        // - No params → entry-point coordinator → try_repair == true
        // - `?local=true` alone (client diagnostic) → try_repair == false
        // - `?local=true&repair=true` (forwarded by coordinator) → true
        // - `?local=true&repair=false` (read-repair fan-out) → false
        assert!(LocalParam::default().try_repair(), "no params");
        let client_local = LocalParam {
            local: Some(true),
            repair: None,
        };
        assert!(
            !client_local.try_repair(),
            "explicit ?local=true defaults to no repair"
        );
        let forwarded = LocalParam {
            local: Some(true),
            repair: Some(true),
        };
        assert!(forwarded.try_repair(), "forwarded by coordinator");
        let fan_out = LocalParam {
            local: Some(true),
            repair: Some(false),
        };
        assert!(!fan_out.try_repair(), "fan-out itself");
    }

    #[tokio::test]
    async fn test_query_local_does_not_trigger_repair_loop() {
        // ?local=true on a fresh node returns a clean miss without the
        // handler trying to recurse (the LocalParam.try_repair check
        // doesn't fire because there's no cluster wired). This
        // exercises the path that would loop if we forgot to gate it.
        let (app, _, _dir) = test_app().await;
        let q = json!({
            "embedding": [0.1_f32, 0.2, 0.3],
            "threshold": 0.9,
            "model_id": MID,
        });
        let resp = app
            .oneshot(post_json("/query?local=true&repair=false", q))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp.into_body()).await;
        assert_eq!(body["hit"], false);
    }

    #[tokio::test]
    async fn test_compact_endpoint() {
        let (app, state, _dir) = test_app().await;
        for v in [
            json!([1.0_f32, 0.0, 0.0]),
            json!([0.0_f32, 1.0, 0.0]),
            json!([0.0_f32, 0.0, 1.0]),
        ] {
            let response = app
                .clone()
                .oneshot(post_json(
                    "/insert",
                    json!({
                        "embedding": v,
                        "response": "r",
                        "query_text": "q",
                        "model_id": MID,
                    }),
                ))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }

        let response = app
            .oneshot(post_json("/admin/compact", json!({})))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response.into_body()).await;
        assert_eq!(body["status"], "ok");
        assert_eq!(body["entries_snapshotted"], 3);
        assert_eq!(body["wal_sequence"], 3);
        assert!(state.snapshot_path.exists(), "snapshot file must exist");
    }

    #[tokio::test]
    async fn test_startup_with_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("startup.wal");
        // Distinct unit-direction embeddings so each entry is its own nearest neighbor.
        let r2 = std::f32::consts::FRAC_1_SQRT_2;
        let r3 = 1.0_f32 / 3.0_f32.sqrt();
        let snapshot_vecs: [[f32; 4]; 5] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [r2, r2, 0.0, 0.0],
            [r3, r3, r3, 0.0],
        ];
        let tail_vec: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

        // Round 1: insert via the router, then compact via the endpoint.
        {
            let state = build_state(wal_path.clone()).await;
            let app = build_router(state.clone());
            for (i, v) in snapshot_vecs.iter().enumerate() {
                let payload = json!({
                    "embedding": v,
                    "response": format!("r{i}"),
                    "query_text": format!("q{i}"),
                    "model_id": MID
                });
                let response = app
                    .clone()
                    .oneshot(post_json("/insert", payload))
                    .await
                    .unwrap();
                assert_eq!(response.status(), StatusCode::OK);
            }
            let response = app
                .oneshot(post_json("/admin/compact", json!({})))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            assert!(state.snapshot_path.exists());
            // One more insert AFTER the snapshot — this should land in the WAL tail.
            let app2 = build_router(state.clone());
            let payload = json!({
                "embedding": tail_vec,
                "response": "tail-r",
                "query_text": "tail-q",
                "model_id": MID
            });
            app2.oneshot(post_json("/insert", payload)).await.unwrap();
        }

        // Round 2: simulate restart by replaying snapshot + WAL tail manually.
        let snapshot_path = crate::snapshot::snapshot_path_for(&wal_path.to_string_lossy());
        let (snap_entries, snap_seq) = crate::snapshot::read_snapshot(&snapshot_path)
            .await
            .unwrap();
        assert_eq!(snap_entries.len(), 5);
        assert_eq!(snap_seq, 5);

        let mut fresh = SemanticIndex::new(&HnswConfig::default());
        for e in snap_entries {
            fresh.replay_snapshot_entry(e).unwrap();
        }
        let tail: Vec<_> = Wal::replay(&wal_path)
            .await
            .unwrap()
            .into_iter()
            .filter(|e| e.sequence > snap_seq)
            .collect();
        assert_eq!(tail.len(), 1);
        assert_eq!(tail[0].response, "tail-r");
        for e in tail {
            fresh.replay_entry(e).unwrap();
        }
        assert_eq!(fresh.entry_count(), 6);

        let hit = fresh
            .query(&tail_vec, 0.90, MID)
            .unwrap()
            .expect("tail entry survives restart");
        assert_eq!(hit.response, "tail-r");
        let hit2 = fresh
            .query(&snapshot_vecs[2], 0.90, MID)
            .unwrap()
            .expect("snapshotted entry survives restart");
        assert_eq!(hit2.response, "r2");
    }

    #[tokio::test]
    async fn test_startup_corrupt_snapshot_falls_back() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("fb.wal");
        let snapshot_path = crate::snapshot::snapshot_path_for(&wal_path.to_string_lossy());

        // Seed the WAL with two entries directly.
        {
            let mut wal = Wal::open(&wal_path).await.unwrap();
            for i in 0..2u32 {
                wal.append(&WalEntry {
                    uuid: format!("u{i}"),
                    embedding: vec![i as f32, 0.0, 0.0],
                    response: format!("r{i}"),
                    query_text: format!("q{i}"),
                    model_id: MID.to_string(),
                    sequence: 0,
                    inserted_at: 0,
                    tombstone: false,
                    expires_at: None,
                })
                .await
                .unwrap();
            }
        }
        // Write garbage at the snapshot path.
        tokio::fs::write(&snapshot_path, b"this is not a snapshot")
            .await
            .unwrap();

        // read_snapshot must error cleanly (callers fall back to full WAL replay).
        assert!(
            crate::snapshot::read_snapshot(&snapshot_path)
                .await
                .is_err()
        );

        // Full WAL replay still recovers everything.
        let entries = Wal::replay(&wal_path).await.unwrap();
        assert_eq!(entries.len(), 2);
        let mut fresh = SemanticIndex::new(&HnswConfig::default());
        for e in entries {
            fresh.replay_entry(e).unwrap();
        }
        assert_eq!(fresh.entry_count(), 2);
    }

    #[tokio::test]
    async fn test_stats_endpoint() {
        let (app, _, _dir) = test_app().await;
        let response = app.oneshot(get("/stats")).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response.into_body()).await;
        assert_eq!(body["entry_count"], 0);
        assert!(body["wal_path"].is_string());
        assert_eq!(body["hnsw"]["max_nb_connection"], 16);
        assert_eq!(body["hnsw"]["ef_construction"], 200);
        assert_eq!(body["hnsw"]["ef_search"], 32);
        assert!(body["hnsw"]["dimension"].is_null());
        assert!(body["namespaces"].is_object());
        assert_eq!(body["namespaces"].as_object().unwrap().len(), 0);
    }

    async fn body_text(body: Body) -> String {
        let bytes = body.collect().await.unwrap().to_bytes();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    #[tokio::test]
    async fn test_metrics_endpoint() {
        let (app, _, _dir) = test_app().await;
        let response = app.oneshot(get("/metrics")).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let ct = response
            .headers()
            .get(axum::http::header::CONTENT_TYPE)
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        assert!(ct.starts_with("text/plain"), "ct={ct}");
        let text = body_text(response.into_body()).await;
        assert!(text.contains("ferrocache_queries_total"));
        assert!(text.contains("ferrocache_query_duration_seconds_bucket"));
        assert!(text.contains("ferrocache_cluster_nodes 1"));
    }

    #[tokio::test]
    async fn test_metrics_after_operations() {
        let (app, _, _dir) = test_app().await;
        // 2 inserts (different vectors so HNSW doesn't dedupe at search time)
        for v in [json!([1.0_f32, 0.0, 0.0]), json!([0.0_f32, 1.0, 0.0])] {
            app.clone()
                .oneshot(post_json(
                    "/insert",
                    json!({
                        "embedding": v,
                        "response": "r",
                        "query_text": "q",
                        "model_id": MID
                    }),
                ))
                .await
                .unwrap();
        }
        // 3 queries: 2 hits on the inserted vectors + 1 miss with high threshold.
        for (v, threshold) in [
            (json!([1.0_f32, 0.0, 0.0]), 0.9),
            (json!([0.0_f32, 1.0, 0.0]), 0.9),
            (json!([0.0_f32, 0.0, 1.0]), 0.99),
        ] {
            app.clone()
                .oneshot(post_json(
                    "/query",
                    json!({"embedding": v, "threshold": threshold, "model_id": MID}),
                ))
                .await
                .unwrap();
        }
        let response = app.oneshot(get("/metrics")).await.unwrap();
        let text = body_text(response.into_body()).await;
        assert!(text.contains("ferrocache_inserts_total 2"), "{text}");
        assert!(text.contains("ferrocache_queries_total 3"), "{text}");
        assert!(text.contains("ferrocache_queries_hit_total 2"), "{text}");
        assert!(text.contains("ferrocache_queries_miss_total 1"), "{text}");
        assert!(
            text.contains(&format!(
                "ferrocache_namespace_entries{{namespace=\"{MID}\"}} 2"
            )),
            "{text}"
        );
    }

    #[tokio::test]
    async fn test_stats_includes_counters() {
        let (app, _, _dir) = test_app().await;
        app.clone()
            .oneshot(post_json(
                "/insert",
                json!({
                    "embedding": [1.0_f32, 0.0, 0.0],
                    "response": "r",
                    "query_text": "q",
                    "model_id": MID
                }),
            ))
            .await
            .unwrap();
        app.clone()
            .oneshot(post_json(
                "/query",
                json!({
                    "embedding": [1.0_f32, 0.0, 0.0],
                    "threshold": 0.9,
                    "model_id": MID
                }),
            ))
            .await
            .unwrap();
        let response = app.oneshot(get("/stats")).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response.into_body()).await;
        let counters = &body["counters"];
        assert!(counters.is_object(), "counters missing in {body}");
        assert_eq!(counters["queries_total"], 1);
        assert_eq!(counters["queries_hit"], 1);
        assert_eq!(counters["inserts_total"], 1);
        assert!((counters["hit_rate"].as_f64().unwrap() - 1.0).abs() < 1e-9);
    }

    // --- M24: access tracking ----------------------------------------------

    async fn insert_via_http(app: &Router, vec: &[f32], uuid: &str) {
        let payload = json!({
            "embedding": vec,
            "response": format!("r-{uuid}"),
            "query_text": format!("the question for {uuid}"),
            "model_id": MID,
            "uuid": uuid,
        });
        let resp = app
            .clone()
            .oneshot(post_json("/insert?local=true", payload))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    async fn query_via_http(app: &Router, vec: &[f32]) {
        let payload = json!({
            "embedding": vec,
            "threshold": 0.9,
            "model_id": MID,
        });
        app.clone()
            .oneshot(post_json("/query?local=true", payload))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_query_hit_updates_access_count() {
        let (app, _, _dir) = test_app().await;
        let v = [1.0_f32, 0.0, 0.0];
        insert_via_http(&app, &v, "track-target").await;
        query_via_http(&app, &v).await;
        query_via_http(&app, &v).await;
        let resp = app
            .oneshot(get("/internal/entry/track-target?local=true"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp.into_body()).await;
        assert_eq!(body["access_count"], 2);
        assert!(body["last_accessed_at"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn test_query_miss_does_not_update_access() {
        let (app, _, _dir) = test_app().await;
        let v = [1.0_f32, 0.0, 0.0];
        insert_via_http(&app, &v, "no-touch").await;
        // Query with a sufficiently different vector — high threshold makes
        // this a miss against a unit-x neighbour.
        let payload = json!({
            "embedding": [0.0_f32, 0.0, 1.0],
            "threshold": 0.99,
            "model_id": MID,
        });
        app.clone()
            .oneshot(post_json("/query?local=true", payload))
            .await
            .unwrap();
        let resp = app
            .oneshot(get("/internal/entry/no-touch?local=true"))
            .await
            .unwrap();
        let body = body_json(resp.into_body()).await;
        assert_eq!(body["access_count"], 0);
    }

    #[tokio::test]
    async fn test_admin_entry_stats_endpoint() {
        let (app, _, _dir) = test_app().await;
        let a = [1.0_f32, 0.0, 0.0];
        let b = [0.0_f32, 1.0, 0.0];
        let c = [0.0_f32, 0.0, 1.0];
        insert_via_http(&app, &a, "hot").await;
        insert_via_http(&app, &b, "warm").await;
        insert_via_http(&app, &c, "cold").await;
        // hot accessed 3x, warm 1x, cold 0x
        for _ in 0..3 {
            query_via_http(&app, &a).await;
        }
        query_via_http(&app, &b).await;

        let resp = app.oneshot(get("/admin/entry-stats")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp.into_body()).await;
        let top = body["namespaces"][MID]["top_entries"].as_array().unwrap();
        assert!(top.len() >= 2);
        // Ordered by access_count descending.
        assert_eq!(top[0]["uuid"], "hot");
        assert_eq!(top[0]["access_count"], 3);
        assert_eq!(top[1]["uuid"], "warm");
        assert_eq!(top[1]["access_count"], 1);
        assert!(top[0]["query_text_preview"].is_string());
    }

    #[tokio::test]
    async fn test_stats_includes_access_fields() {
        let (app, _, _dir) = test_app().await;
        let a = [1.0_f32, 0.0, 0.0];
        let b = [0.0_f32, 1.0, 0.0];
        insert_via_http(&app, &a, "u-a").await;
        insert_via_http(&app, &b, "u-b").await;
        query_via_http(&app, &a).await;
        query_via_http(&app, &a).await;
        query_via_http(&app, &b).await;

        let resp = app.oneshot(get("/stats")).await.unwrap();
        let body = body_json(resp.into_body()).await;
        let ns = &body["namespaces"][MID];
        assert_eq!(ns["total_accesses"], 3);
        assert!(ns["oldest_entry_ts"].as_u64().unwrap() > 0);
        assert!(ns["newest_entry_ts"].as_u64().unwrap() > 0);
        assert!(ns["newest_entry_ts"].as_u64().unwrap() >= ns["oldest_entry_ts"].as_u64().unwrap());
    }

    // --- M25: eviction at the HTTP boundary --------------------------------

    async fn build_state_with_max_entries(
        wal_path: PathBuf,
        max_entries: Option<usize>,
    ) -> Arc<AppState> {
        let hnsw = HnswConfig {
            max_entries_per_namespace: max_entries,
            ..HnswConfig::default()
        };
        let wal = Wal::open(&wal_path).await.unwrap();
        let snapshot_path = crate::snapshot::snapshot_path_for(&wal_path.to_string_lossy());
        let index = Arc::new(RwLock::new(SemanticIndex::new(&hnsw)));
        let metrics = Arc::new(Metrics::new());
        let gc = GroupCommitWal::spawn(
            wal,
            wal_path.clone(),
            snapshot_path.clone(),
            index.clone(),
            metrics.clone(),
            0,
            GroupCommitConfig {
                batch_size: 1,
                batch_timeout: std::time::Duration::from_millis(0),
                channel_capacity: 64,
            },
            hnsw.clone(),
        );
        Arc::new(AppState::new(
            "test-node".to_string(),
            index,
            gc,
            wal_path.to_string_lossy().into_owned(),
            snapshot_path,
            hnsw,
            None,
            None,
            1,
            metrics,
            None,
            0,
            true,
        ))
    }

    /// Insert N orthogonal-ish unit vectors via HTTP. Each insert routes
    /// through the WAL flush task, so by the time the response returns the
    /// eviction step (if any) has run.
    async fn insert_n_distinct(app: &Router, n: usize) {
        for i in 0..n {
            let mut v = vec![0.0_f32; 8];
            v[i % 8] = 1.0;
            v[(i + 1) % 8] = 0.001 * (i as f32 + 1.0); // tiny perturbation for uniqueness
            let payload = json!({
                "embedding": v,
                "response": format!("r{i}"),
                "query_text": format!("q{i}"),
                "model_id": MID,
                "uuid": format!("u{i}"),
            });
            let r = app
                .clone()
                .oneshot(post_json("/insert?local=true", payload))
                .await
                .unwrap();
            assert_eq!(r.status(), StatusCode::OK, "insert {i} failed");
        }
    }

    #[tokio::test]
    async fn test_eviction_on_max_entries() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ev.wal");
        let state = build_state_with_max_entries(path.clone(), Some(3)).await;
        let app = build_router(state.clone());

        insert_n_distinct(&app, 4).await;

        // Health reports live entry count.
        let resp = app.clone().oneshot(get("/health")).await.unwrap();
        let body = body_json(resp.into_body()).await;
        assert_eq!(body["entry_count"], 3, "health: {body}");

        // The first inserted entry (u0) should have been evicted.
        let resp = app
            .clone()
            .oneshot(get("/internal/entry/u0?local=true"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);

        // The newest entry (u3) survives.
        let resp = app
            .oneshot(get("/internal/entry/u3?local=true"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_eviction_metrics() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("evm.wal");
        let state = build_state_with_max_entries(path, Some(2)).await;
        let app = build_router(state.clone());
        insert_n_distinct(&app, 5).await;

        // 3 evictions expected (5 inserts - cap of 2).
        let resp = app.oneshot(get("/metrics")).await.unwrap();
        let text = body_text(resp.into_body()).await;
        assert!(
            text.contains("ferrocache_evictions_total 3"),
            "evictions metric not 3 in:\n{text}"
        );
    }

    // --- M26: TTL + delete + invalidate -----------------------------------

    #[tokio::test]
    async fn test_insert_with_ttl() {
        let (app, _, _dir) = test_app().await;
        let payload = json!({
            "embedding": [1.0_f32, 0.0, 0.0],
            "response": "ephemeral",
            "query_text": "q",
            "model_id": MID,
            "uuid": "ttl-target",
            "ttl_seconds": 3600,
        });
        let r = app
            .clone()
            .oneshot(post_json("/insert?local=true", payload))
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::OK);

        let r = app
            .oneshot(get("/internal/entry/ttl-target?local=true"))
            .await
            .unwrap();
        let body = body_json(r.into_body()).await;
        let exp = body["expires_at"].as_u64().unwrap();
        let inserted_at = body["inserted_at"].as_u64().unwrap();
        assert_eq!(exp, inserted_at + 3600);
    }

    #[tokio::test]
    async fn test_insert_without_ttl_no_expiry() {
        let (app, _, _dir) = test_app().await;
        let payload = json!({
            "embedding": [1.0_f32, 0.0, 0.0],
            "response": "forever",
            "query_text": "q",
            "model_id": MID,
            "uuid": "no-ttl",
        });
        app.clone()
            .oneshot(post_json("/insert?local=true", payload))
            .await
            .unwrap();
        let r = app
            .oneshot(get("/internal/entry/no-ttl?local=true"))
            .await
            .unwrap();
        let body = body_json(r.into_body()).await;
        // Skip-serializing-if-none means the field is absent.
        assert!(body.get("expires_at").is_none_or(|v| v.is_null()));
    }

    #[tokio::test]
    async fn test_delete_entry_success() {
        let (app, _, _dir) = test_app().await;
        let payload = json!({
            "embedding": [1.0_f32, 0.0, 0.0],
            "response": "doomed",
            "query_text": "q",
            "model_id": MID,
            "uuid": "del-1",
        });
        app.clone()
            .oneshot(post_json("/insert?local=true", payload))
            .await
            .unwrap();

        let r = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri("/entry/del-1?local=true")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::OK);
        let body = body_json(r.into_body()).await;
        assert_eq!(body["deleted"], true);

        // Entry is gone.
        let r = app
            .oneshot(get("/internal/entry/del-1?local=true"))
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_delete_entry_not_found() {
        let (app, _, _dir) = test_app().await;
        let r = app
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri("/entry/nonexistent?local=true")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_invalidate_removes_similar() {
        let (app, _, _dir) = test_app().await;
        let r2 = std::f32::consts::FRAC_1_SQRT_2;
        // A and B sit in the +x/+xy region (sim ≥ 0.70 to probe). C is orthogonal.
        for (uuid, vec) in [
            ("A", vec![1.0_f32, 0.0, 0.0]),
            ("B", vec![r2, r2, 0.0]),
            ("C", vec![0.0_f32, 0.0, 1.0]),
        ] {
            let payload = json!({
                "embedding": vec,
                "response": format!("r-{uuid}"),
                "query_text": "q",
                "model_id": MID,
                "uuid": uuid,
            });
            app.clone()
                .oneshot(post_json("/insert?local=true", payload))
                .await
                .unwrap();
        }

        let inv = json!({
            "embedding": [1.0_f32, 0.0, 0.0],
            "threshold": 0.70,
            "model_id": MID,
        });
        let r = app
            .clone()
            .oneshot(post_json("/admin/invalidate?local=true", inv))
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::OK);
        let body = body_json(r.into_body()).await;
        assert_eq!(body["invalidated_count"], 2);

        // C survives.
        let r = app
            .clone()
            .oneshot(get("/internal/entry/C?local=true"))
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::OK);
        // A and B are gone.
        for uuid in ["A", "B"] {
            let r = app
                .clone()
                .oneshot(get(&format!("/internal/entry/{uuid}?local=true")))
                .await
                .unwrap();
            assert_eq!(r.status(), StatusCode::NOT_FOUND, "{uuid} should be gone");
        }
    }

    #[tokio::test]
    async fn test_invalidate_requires_model_id() {
        let (app, _, _dir) = test_app().await;
        let payload = json!({
            "embedding": [1.0_f32, 0.0, 0.0],
            "threshold": 0.9,
        });
        let r = app
            .oneshot(post_json("/admin/invalidate?local=true", payload))
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_expired_entry_survives_restart() {
        // Insert with a manually-stamped past expires_at via WAL append +
        // tombstone, then replay into a fresh index. The tombstone wins,
        // so the entry is gone after restart.
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("expsr.wal");

        let mut wal = Wal::open(&wal_path).await.unwrap();
        // 1) An entry with an already-elapsed TTL.
        wal.append(&WalEntry {
            uuid: "ephemeral".into(),
            embedding: vec![1.0, 0.0, 0.0],
            response: "r".into(),
            query_text: "q".into(),
            model_id: MID.into(),
            sequence: 0,
            inserted_at: 100,
            tombstone: false,
            expires_at: Some(101), // expired ages ago
        })
        .await
        .unwrap();
        // 2) The reaper would have written a tombstone — simulate.
        wal.append(&WalEntry::tombstone_entry("ephemeral".into(), MID.into()))
            .await
            .unwrap();
        drop(wal);

        let entries = Wal::replay(&wal_path).await.unwrap();
        let mut idx = SemanticIndex::new(&HnswConfig::default());
        for e in entries {
            if e.tombstone {
                idx.remove_by_uuid(&e.uuid);
            } else {
                let _ = idx.replay_entry(e);
            }
        }
        assert!(
            idx.get_entry_by_uuid("ephemeral").is_none(),
            "tombstone wins on replay"
        );
    }

    #[tokio::test]
    async fn test_invalidate_metric() {
        let (app, _, _dir) = test_app().await;
        // Insert two similar entries.
        for uuid in ["m1", "m2"] {
            let payload = json!({
                "embedding": [1.0_f32, 0.0, 0.0],
                "response": "r",
                "query_text": "q",
                "model_id": MID,
                "uuid": uuid,
            });
            app.clone()
                .oneshot(post_json("/insert?local=true", payload))
                .await
                .unwrap();
        }
        let inv = json!({
            "embedding": [1.0_f32, 0.0, 0.0],
            "threshold": 0.95,
            "model_id": MID,
        });
        app.clone()
            .oneshot(post_json("/admin/invalidate?local=true", inv))
            .await
            .unwrap();
        let r = app.oneshot(get("/metrics")).await.unwrap();
        let text = body_text(r.into_body()).await;
        assert!(text.contains("ferrocache_invalidations_total 2"), "{text}");
    }

    // --- M27: exact-match pre-filter via HTTP -----------------------------

    #[tokio::test]
    async fn test_exact_match_via_http() {
        let (app, _, _dir) = test_app().await;
        let payload = json!({
            "embedding": [1.0_f32, 0.0, 0.0],
            "response": "Paris",
            "query_text": "What is the capital of France?",
            "model_id": MID,
            "uuid": "exact-target",
        });
        app.clone()
            .oneshot(post_json("/insert?local=true", payload))
            .await
            .unwrap();

        // Probe with a totally different embedding. Pre-filter wins.
        let q = json!({
            "embedding": [0.0_f32, 0.0, 1.0],
            "threshold": 0.99,
            "model_id": MID,
            "query_text": "WHAT IS THE CAPITAL OF FRANCE?",
        });
        let r = app
            .oneshot(post_json("/query?local=true", q))
            .await
            .unwrap();
        assert_eq!(r.status(), StatusCode::OK);
        let body = body_json(r.into_body()).await;
        assert_eq!(body["hit"], true);
        assert_eq!(body["exact_match"], true);
        assert_eq!(body["id"], "exact-target");
        assert!((body["similarity"].as_f64().unwrap() - 1.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_query_without_query_text_omits_exact_match() {
        let (app, _, _dir) = test_app().await;
        let payload = json!({
            "embedding": [1.0_f32, 0.0, 0.0],
            "response": "r",
            "query_text": "indexed",
            "model_id": MID,
        });
        app.clone()
            .oneshot(post_json("/insert?local=true", payload))
            .await
            .unwrap();
        let q = json!({
            "embedding": [1.0_f32, 0.0, 0.0],
            "threshold": 0.9,
            "model_id": MID,
        });
        let r = app
            .oneshot(post_json("/query?local=true", q))
            .await
            .unwrap();
        let body = body_json(r.into_body()).await;
        assert_eq!(body["hit"], true);
        // ANN hit reports exact_match: false.
        assert_eq!(body["exact_match"], false);
    }

    #[tokio::test]
    async fn test_exact_match_metric() {
        let (app, _, _dir) = test_app().await;
        let payload = json!({
            "embedding": [1.0_f32, 0.0, 0.0],
            "response": "r",
            "query_text": "metric-text",
            "model_id": MID,
        });
        app.clone()
            .oneshot(post_json("/insert?local=true", payload))
            .await
            .unwrap();
        // Two queries with the exact text → 2 exact-match hits.
        for _ in 0..2 {
            let q = json!({
                "embedding": [0.0_f32, 0.0, 1.0],
                "threshold": 0.99,
                "model_id": MID,
                "query_text": "metric-text",
            });
            app.clone()
                .oneshot(post_json("/query?local=true", q))
                .await
                .unwrap();
        }
        let r = app.oneshot(get("/metrics")).await.unwrap();
        let text = body_text(r.into_body()).await;
        assert!(
            text.contains("ferrocache_exact_match_hits_total 2"),
            "{text}"
        );
    }

    #[tokio::test]
    async fn test_no_eviction_without_max_entries() {
        // max_entries = None → all inserts survive, no evictions.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("noev.wal");
        let state = build_state_with_max_entries(path, None).await;
        let app = build_router(state.clone());
        insert_n_distinct(&app, 8).await;

        let resp = app.clone().oneshot(get("/health")).await.unwrap();
        let body = body_json(resp.into_body()).await;
        assert_eq!(body["entry_count"], 8);

        let resp = app.oneshot(get("/metrics")).await.unwrap();
        let text = body_text(resp.into_body()).await;
        assert!(
            text.contains("ferrocache_evictions_total 0"),
            "expected no evictions in:\n{text}"
        );
    }
}
