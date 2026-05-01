use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::Deserialize;

use crate::models::{
    ClusterStatusResponse, ErrorResponse, HealthResponse, InsertRequest, InsertResponse,
    QueryRequest, QueryResponse, StatsHnsw, StatsResponse,
};
use crate::state::AppState;
use crate::wal::WalEntry;

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
    Router::new()
        .route("/query", post(query_handler))
        .route("/insert", post(insert_handler))
        .route("/health", get(health_handler))
        .route("/stats", get(stats_handler))
        .route("/cluster/status", get(cluster_status_handler))
        .with_state(state)
}

fn bad_request(msg: impl Into<String>) -> Response {
    let msg = msg.into();
    tracing::warn!(error = %msg, "request rejected");
    (StatusCode::BAD_REQUEST, Json(ErrorResponse { error: msg })).into_response()
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
        return bad_gateway(format!("no router configured to reach {target_id}"));
    };
    match router.forward_query(target_addr, req).await {
        Ok(resp) => {
            tracing::info!(target = %target_id, "query forwarded");
            (StatusCode::OK, Json(resp)).into_response()
        }
        Err(e) => bad_gateway(format!("failed to reach node {target_id}: {e}")),
    }
}

async fn process_query_locally(state: &Arc<AppState>, req: QueryRequest) -> Response {
    let dim = req.embedding.len();
    let index = state.index.read().await;
    match index.query(&req.embedding, req.threshold) {
        Ok(Some(hit)) => {
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
        return bad_gateway("no router configured for replication");
    };

    for (peer_id, peer_addr) in replicas.iter().filter(|(id, _)| id != self_id) {
        if let Err(e) = router.forward_insert(peer_addr, &req_with_uuid).await {
            return bad_gateway(format!("replica {peer_id} failed: {e}"));
        }
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
async fn local_insert_inner(
    state: &Arc<AppState>,
    req: &InsertRequest,
) -> Result<String, Response> {
    let dim = req.embedding.len();
    let mut wal = state.wal.lock().await;

    if let Some(existing) = state.index.read().await.dimension()
        && existing != dim
    {
        return Err(bad_request(format!(
            "dimension mismatch: expected {existing}, got {dim}"
        )));
    }

    let uuid = req
        .uuid
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    let entry = WalEntry {
        uuid: uuid.clone(),
        embedding: req.embedding.clone(),
        response: req.response.clone(),
        query_text: req.query_text.clone(),
    };

    if let Err(e) = wal.append(&entry).await {
        tracing::error!(error = %e, "WAL append failed");
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("WAL append failed: {e}"),
            }),
        )
            .into_response());
    }

    if let Err(e) = state.index.write().await.replay_entry(entry) {
        tracing::error!(error = %e, "index insert failed after WAL append");
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
            .into_response());
    }

    tracing::info!(id = %uuid, dim, "inserted");
    Ok(uuid)
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
            Json(ClusterStatusResponse {
                mode: "clustered",
                self_node_id: cluster.self_node_id().to_string(),
                gossip_addr: Some(cluster.gossip_addr().to_string()),
                nodes,
                node_count,
            })
        }
        None => Json(ClusterStatusResponse {
            mode: "single",
            self_node_id: state.node_id.clone(),
            gossip_addr: None,
            nodes: vec![state.node_id.clone()],
            node_count: 1,
        }),
    }
}

async fn stats_handler(State(state): State<Arc<AppState>>) -> Json<StatsResponse> {
    let index = state.index.read().await;
    Json(StatsResponse {
        entry_count: index.entry_count() as u64,
        wal_path: state.wal_path.clone(),
        hnsw: StatsHnsw {
            max_nb_connection: state.hnsw_config.max_nb_connection,
            ef_construction: state.hnsw_config.ef_construction,
            ef_search: state.hnsw_config.ef_search,
            dimension: index.dimension(),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::HnswConfig;
    use crate::index::SemanticIndex;
    use crate::wal::Wal;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use serde_json::{Value, json};
    use std::path::PathBuf;
    use tower::ServiceExt;

    async fn build_state(wal_path: PathBuf) -> Arc<AppState> {
        let hnsw = HnswConfig::default();
        let wal = Wal::open(&wal_path).await.unwrap();
        Arc::new(AppState::new(
            "test-node".to_string(),
            SemanticIndex::new(&hnsw),
            wal,
            wal_path.to_string_lossy().into_owned(),
            hnsw,
            None,
            None,
            1,
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

    #[tokio::test]
    async fn test_insert_returns_id() {
        let (app, _, _dir) = test_app().await;
        let payload = json!({
            "embedding": [0.1f32, 0.2, 0.3],
            "response": "hello",
            "query_text": "hi"
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
            "threshold": 0.9
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
            "query_text": "hi"
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
            "query_text": "the question"
        });
        let response = app
            .clone()
            .oneshot(post_json("/insert", insert_payload))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let query_payload = json!({
            "embedding": [1.0_f32, 0.0, 0.0],
            "threshold": 0.90
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
            "query_text": "q"
        });
        let response = app
            .clone()
            .oneshot(post_json("/insert", insert_payload))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let query_payload = json!({
            "embedding": [0.0_f32, 0.0, 1.0],
            "threshold": 0.99
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
    async fn test_insert_dimension_mismatch() {
        let (app, _, _dir) = test_app().await;
        let first = json!({
            "embedding": [1.0_f32, 0.0],
            "response": "r",
            "query_text": "q"
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
            "query_text": "q"
        });
        let response = app.oneshot(post_json("/insert", second)).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
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
                "query_text": "persisted question"
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
            .query(&[1.0_f32, 0.0, 0.0], 0.90)
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
            "query_text": "q"
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
            "query_text": "q"
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
            "query_text": "q"
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
            "threshold": 1.5
        });
        let response = app.oneshot(post_json("/query", payload)).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = body_json(response.into_body()).await;
        assert!(body["error"].as_str().unwrap().contains("threshold"));
    }

    #[tokio::test]
    async fn test_local_query_param_bypasses_routing() {
        let (app, _, _dir) = test_app().await;
        // Insert one entry first so we have something to (locally) query.
        let insert_payload = json!({
            "embedding": [1.0_f32, 0.0, 0.0],
            "response": "local-only",
            "query_text": "q"
        });
        let response = app
            .clone()
            .oneshot(post_json("/insert?local=true", insert_payload))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let query_payload = json!({
            "embedding": [1.0_f32, 0.0, 0.0],
            "threshold": 0.9
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
            "query_text": "q"
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
    }
}
