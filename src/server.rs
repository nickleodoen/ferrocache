use std::sync::Arc;
use std::sync::atomic::Ordering;

use axum::{
    Json, Router,
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::Deserialize;
use tokio::sync::oneshot;

use crate::auth::{AuthToken, auth_middleware};
use crate::failure_detector::PeerStatus;
use crate::metrics::METRICS_CONTENT_TYPE;
use crate::models::{
    ClusterStatusResponse, CompactResponse, CountersResponse, ErrorResponse, HealthResponse,
    InsertRequest, InsertResponse, NamespaceStatsEntry, PeerHealth, QueryRequest, QueryResponse,
    StatsHnsw, StatsResponse,
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
}

impl LocalParam {
    fn is_local(&self) -> bool {
        self.local.unwrap_or(false)
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
        .route("/metrics", get(metrics_handler))
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
                }),
            )
                .into_response();
        }
        return forward_query(&state, &target_id, &target_addr, &req).await;
    }

    process_query_locally(&state, req).await
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

async fn process_query_locally(state: &Arc<AppState>, req: QueryRequest) -> Response {
    let start = std::time::Instant::now();
    let dim = req.embedding.len();
    let model_id = match req.model_id.as_deref() {
        Some(s) if !s.trim().is_empty() => s,
        _ => return bad_request("model_id is required"),
    };
    let result = {
        let index = state.index.read().await;
        index.query(&req.embedding, req.threshold, model_id)
    };
    let elapsed = start.elapsed().as_secs_f64();
    match result {
        Ok(Some(hit)) => {
            state.metrics.record_query_hit(model_id, elapsed);
            tracing::info!(hit = true, similarity = hit.similarity, dim, "query");
            (
                StatusCode::OK,
                Json(QueryResponse {
                    hit: true,
                    id: Some(hit.id),
                    response: Some(hit.response),
                    similarity: Some(hit.similarity),
                }),
            )
                .into_response()
        }
        Ok(None) => {
            state.metrics.record_query_miss(model_id, elapsed);
            tracing::info!(hit = false, dim, "query");
            (
                StatusCode::OK,
                Json(QueryResponse {
                    hit: false,
                    id: None,
                    response: None,
                    similarity: None,
                }),
            )
                .into_response()
        }
        Err(e) => bad_request(e.to_string()),
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
    let entry = WalEntry {
        uuid: uuid.clone(),
        embedding: req.embedding.clone(),
        response: req.response.clone(),
        query_text: req.query_text.clone(),
        model_id: model_id.clone(),
        sequence: 0, // stamped by the flush task
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
            0, // no replication retries in tests
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
}
