use std::sync::Arc;
use std::sync::atomic::Ordering;

use axum::{
    Json, Router,
    extract::State,
    routing::{get, post},
};

use crate::models::{
    HealthResponse, InsertRequest, InsertResponse, QueryRequest, QueryResponse,
};
use crate::state::AppState;

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/query", post(query_handler))
        .route("/insert", post(insert_handler))
        .route("/health", get(health_handler))
        .with_state(state)
}

async fn query_handler(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<QueryRequest>,
) -> Json<QueryResponse> {
    tracing::info!(
        embedding_dim = req.embedding.len(),
        threshold = req.threshold,
        "query received"
    );
    Json(QueryResponse { hit: false })
}

async fn insert_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<InsertRequest>,
) -> Json<InsertResponse> {
    let id = uuid::Uuid::new_v4().to_string();
    state.entry_count.fetch_add(1, Ordering::SeqCst);
    tracing::info!(
        id = %id,
        embedding_dim = req.embedding.len(),
        "insert received"
    );
    Json(InsertResponse {
        id,
        status: "ok".to_string(),
    })
}

async fn health_handler(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        node_id: state.node_id.clone(),
        entry_count: state.entry_count.load(Ordering::SeqCst),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use serde_json::{Value, json};
    use tower::ServiceExt;

    fn test_app() -> (Router, Arc<AppState>) {
        let state = Arc::new(AppState::new());
        (build_router(state.clone()), state)
    }

    async fn body_json(body: Body) -> Value {
        let bytes = body.collect().await.unwrap().to_bytes();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn test_health_returns_ok() {
        let (app, _) = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response.into_body()).await;
        assert_eq!(body["status"], "ok");
    }

    #[tokio::test]
    async fn test_insert_returns_id() {
        let (app, _) = test_app();
        let payload = json!({
            "embedding": [0.1f32, 0.2, 0.3],
            "response": "hello",
            "query_text": "hi"
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/insert")
                    .header("content-type", "application/json")
                    .body(Body::from(payload.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response.into_body()).await;
        assert!(body["id"].is_string());
        assert!(!body["id"].as_str().unwrap().is_empty());
        assert_eq!(body["status"], "ok");
    }

    #[tokio::test]
    async fn test_query_returns_miss() {
        let (app, _) = test_app();
        let payload = json!({
            "embedding": [0.1f32, 0.2, 0.3],
            "threshold": 0.9
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/query")
                    .header("content-type", "application/json")
                    .body(Body::from(payload.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response.into_body()).await;
        assert_eq!(body["hit"], false);
    }

    #[tokio::test]
    async fn test_insert_increments_count() {
        let (app, _) = test_app();
        let payload = json!({
            "embedding": [0.1f32, 0.2, 0.3],
            "response": "hello",
            "query_text": "hi"
        });

        for _ in 0..2 {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/insert")
                        .header("content-type", "application/json")
                        .body(Body::from(payload.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }

        let response = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response.into_body()).await;
        assert_eq!(body["entry_count"], 2);
    }
}
