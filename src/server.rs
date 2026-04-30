use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};

use crate::models::{
    ErrorResponse, HealthResponse, InsertRequest, InsertResponse, QueryRequest, QueryResponse,
};
use crate::state::AppState;
use crate::wal::WalEntry;

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/query", post(query_handler))
        .route("/insert", post(insert_handler))
        .route("/health", get(health_handler))
        .with_state(state)
}

async fn query_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<QueryRequest>,
) -> Response {
    let dim = req.embedding.len();
    let index = state.index.read().await;
    match index.query(&req.embedding, req.threshold) {
        Ok(Some(hit)) => {
            tracing::info!(hit = true, similarity = hit.similarity, dim, "query");
            (
                StatusCode::OK,
                Json(QueryResponse {
                    hit: true,
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
                    response: None,
                    similarity: None,
                }),
            )
                .into_response()
        }
        Err(e) => {
            tracing::warn!(error = %e, "query failed");
            (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
                .into_response()
        }
    }
}

async fn insert_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<InsertRequest>,
) -> Response {
    let dim = req.embedding.len();

    // Hold the WAL mutex for the whole critical section: this serializes
    // inserters so dimension cannot change between our peek and our index write.
    let mut wal = state.wal.lock().await;

    // Validate dimension up-front to avoid persisting WAL lines we'd reject anyway.
    if let Some(existing) = state.index.read().await.dimension()
        && existing != dim
    {
        let msg = format!("dimension mismatch: expected {existing}, got {dim}");
        tracing::warn!(error = %msg, "insert rejected");
        return (StatusCode::BAD_REQUEST, Json(ErrorResponse { error: msg })).into_response();
    }

    let uuid = uuid::Uuid::new_v4().to_string();
    let entry = WalEntry {
        uuid: uuid.clone(),
        embedding: req.embedding,
        response: req.response,
        query_text: req.query_text,
    };

    if let Err(e) = wal.append(&entry).await {
        tracing::error!(error = %e, "WAL append failed");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: format!("WAL append failed: {e}"),
            }),
        )
            .into_response();
    }

    if let Err(e) = state.index.write().await.replay_entry(entry) {
        // Should not happen — dimension already validated under the WAL lock.
        tracing::error!(error = %e, "index insert failed after WAL append");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
            .into_response();
    }

    tracing::info!(id = %uuid, dim, "inserted");
    (
        StatusCode::OK,
        Json(InsertResponse {
            id: uuid,
            status: "ok".to_string(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::SemanticIndex;
    use crate::wal::Wal;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use serde_json::{Value, json};
    use std::path::PathBuf;
    use tower::ServiceExt;

    async fn test_app() -> (Router, Arc<AppState>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.wal");
        let wal = Wal::open(&path).await.unwrap();
        let state = Arc::new(AppState::new(SemanticIndex::new(), wal));
        (build_router(state.clone()), state, dir)
    }

    async fn test_app_with_wal_path(path: PathBuf) -> (Router, Arc<AppState>) {
        let wal = Wal::open(&path).await.unwrap();
        let state = Arc::new(AppState::new(SemanticIndex::new(), wal));
        (build_router(state.clone()), state)
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
        let response = app.oneshot(post_json("/query", query_payload)).await.unwrap();
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
        let response = app.oneshot(post_json("/query", query_payload)).await.unwrap();
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
        assert!(body["error"].as_str().unwrap().contains("dimension mismatch"));
    }

    #[tokio::test]
    async fn test_insert_persists_via_wal() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("persist.wal");

        // Run an insert through the HTTP layer.
        {
            let (app, state) = test_app_with_wal_path(wal_path.clone()).await;
            let payload = json!({
                "embedding": [1.0_f32, 0.0, 0.0],
                "response": "persisted answer",
                "query_text": "persisted question"
            });
            let response = app.oneshot(post_json("/insert", payload)).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            // Drop the AppState (and its WAL handle) before replaying.
            drop(state);
        }

        // Simulate restart: replay WAL into a fresh index.
        let entries = Wal::replay(&wal_path).await.unwrap();
        assert_eq!(entries.len(), 1);
        let mut fresh = SemanticIndex::new();
        for entry in entries {
            fresh.replay_entry(entry).unwrap();
        }

        let hit = fresh
            .query(&[1.0_f32, 0.0, 0.0], 0.90)
            .unwrap()
            .expect("should hit after replay");
        assert_eq!(hit.response, "persisted answer");
    }
}
