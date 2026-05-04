//! Bearer-token auth middleware for the public HTTP API.
//!
//! Opt-in: when `FERROCACHE_AUTH_TOKEN` is unset, the middleware is never
//! installed and behavior is identical to pre-M17. When set, every request to
//! a data route must carry `Authorization: Bearer <token>`. `/health` and
//! `/metrics` are always allowed through (load balancers + Prometheus).

use std::sync::Arc;

use axum::{
    Json,
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use subtle::ConstantTimeEq;

use crate::models::ErrorResponse;

pub struct AuthToken {
    pub value: String,
}

/// Paths that bypass auth even when it's enabled.
fn is_exempt(path: &str) -> bool {
    path == "/health" || path == "/metrics"
}

pub async fn auth_middleware(
    State(token): State<Arc<AuthToken>>,
    request: Request,
    next: Next,
) -> Response {
    if is_exempt(request.uri().path()) {
        return next.run(request).await;
    }

    let auth_header = request.headers().get("authorization");
    let value = match auth_header.and_then(|v| v.to_str().ok()) {
        Some(v) => v,
        None => return unauthorized(),
    };

    let provided = match value.strip_prefix("Bearer ") {
        Some(p) => p,
        None => return unauthorized(),
    };

    let expected = token.value.as_bytes();
    let provided_bytes = provided.as_bytes();
    if expected.len() != provided_bytes.len() || expected.ct_eq(provided_bytes).unwrap_u8() != 1 {
        return unauthorized();
    }

    next.run(request).await
}

fn unauthorized() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        Json(ErrorResponse {
            error: "unauthorized".to_string(),
        }),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::HnswConfig;
    use crate::index::SemanticIndex;
    use crate::metrics::Metrics;
    use crate::server::build_router;
    use crate::state::AppState;
    use crate::wal::{GroupCommitConfig, GroupCommitWal, Wal};
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use serde_json::{Value, json};
    use std::path::PathBuf;
    use tokio::sync::RwLock;
    use tower::ServiceExt;

    const MID: &str = "test-model::3";

    async fn build_state(wal_path: PathBuf, auth_token: Option<String>) -> Arc<AppState> {
        let hnsw = HnswConfig::default();
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
            auth_token,
            0,
            true,
            None, // conversation_ttl_seconds (M29) — auth tests don't use it
        ))
    }

    async fn test_app(
        auth_token: Option<String>,
    ) -> (axum::Router, Arc<AppState>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auth.wal");
        let state = build_state(path, auth_token).await;
        (build_router(state.clone()), state, dir)
    }

    fn post_json_with_auth(uri: &str, payload: Value, auth: Option<&str>) -> Request<Body> {
        let mut b = Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json");
        if let Some(a) = auth {
            b = b.header("authorization", a);
        }
        b.body(Body::from(payload.to_string())).unwrap()
    }

    fn get_with_auth(uri: &str, auth: Option<&str>) -> Request<Body> {
        let mut b = Request::builder().method("GET").uri(uri);
        if let Some(a) = auth {
            b = b.header("authorization", a);
        }
        b.body(Body::empty()).unwrap()
    }

    fn valid_query() -> Value {
        json!({
            "embedding": [0.1f32, 0.2, 0.3],
            "threshold": 0.9,
            "model_id": MID,
        })
    }

    fn valid_insert() -> Value {
        json!({
            "embedding": [0.1f32, 0.2, 0.3],
            "response": "r",
            "query_text": "q",
            "model_id": MID,
        })
    }

    async fn body_text(body: Body) -> String {
        let bytes = body.collect().await.unwrap().to_bytes();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    #[tokio::test]
    async fn test_auth_rejects_no_header() {
        let (app, _, _dir) = test_app(Some("secret".to_string())).await;
        let resp = app
            .oneshot(post_json_with_auth("/query", valid_query(), None))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        let body = body_text(resp.into_body()).await;
        assert!(body.contains("unauthorized"), "body={body}");
    }

    #[tokio::test]
    async fn test_auth_rejects_wrong_token() {
        let (app, _, _dir) = test_app(Some("secret".to_string())).await;
        let resp = app
            .oneshot(post_json_with_auth(
                "/query",
                valid_query(),
                Some("Bearer wrong-token"),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_auth_accepts_correct_token() {
        let (app, _, _dir) = test_app(Some("secret".to_string())).await;
        let resp = app
            .oneshot(post_json_with_auth(
                "/query",
                valid_query(),
                Some("Bearer secret"),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_auth_allows_health_unauthenticated() {
        let (app, _, _dir) = test_app(Some("secret".to_string())).await;
        let resp = app.oneshot(get_with_auth("/health", None)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_auth_allows_metrics_unauthenticated() {
        let (app, _, _dir) = test_app(Some("secret".to_string())).await;
        let resp = app.oneshot(get_with_auth("/metrics", None)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_auth_disabled_allows_all() {
        let (app, _, _dir) = test_app(None).await;
        let resp = app
            .oneshot(post_json_with_auth("/query", valid_query(), None))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_auth_rejects_prefix_token() {
        let (app, _, _dir) = test_app(Some("my-secret-token".to_string())).await;
        let resp = app
            .oneshot(post_json_with_auth(
                "/query",
                valid_query(),
                Some("Bearer my-secret"),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_auth_rejects_empty_bearer() {
        let (app, _, _dir) = test_app(Some("secret".to_string())).await;
        let resp = app
            .oneshot(post_json_with_auth(
                "/query",
                valid_query(),
                Some("Bearer "),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_auth_rejects_non_bearer_scheme() {
        let (app, _, _dir) = test_app(Some("secret".to_string())).await;
        let resp = app
            .oneshot(post_json_with_auth(
                "/query",
                valid_query(),
                Some("Basic abc123"),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_insert_with_auth() {
        let (app, _, _dir) = test_app(Some("secret".to_string())).await;
        let resp = app
            .oneshot(post_json_with_auth(
                "/insert",
                valid_insert(),
                Some("Bearer secret"),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_stats_requires_auth() {
        let (app, _, _dir) = test_app(Some("secret".to_string())).await;
        let resp = app.oneshot(get_with_auth("/stats", None)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_admin_compact_requires_auth() {
        let (app, _, _dir) = test_app(Some("secret".to_string())).await;
        let resp = app
            .oneshot(post_json_with_auth("/admin/compact", json!({}), None))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }
}
