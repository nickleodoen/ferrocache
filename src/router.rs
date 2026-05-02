use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use reqwest::Client;

use crate::models::{InsertRequest, InsertResponse, QueryRequest, QueryResponse};

const FORWARD_TIMEOUT: Duration = Duration::from_secs(5);

pub struct ClusterRouter {
    client: Client,
    auth_token: Option<String>,
}

impl Default for ClusterRouter {
    fn default() -> Self {
        Self::new(None)
    }
}

impl ClusterRouter {
    pub fn new(auth_token: Option<String>) -> Self {
        let client = Client::builder()
            .timeout(FORWARD_TIMEOUT)
            .build()
            .expect("reqwest::Client build");
        Self { client, auth_token }
    }

    fn with_auth(&self, b: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.auth_token {
            Some(t) => b.header("Authorization", format!("Bearer {t}")),
            None => b,
        }
    }

    pub async fn forward_query(
        &self,
        target_addr: &str,
        req: &QueryRequest,
    ) -> Result<QueryResponse> {
        let url = format!("http://{target_addr}/query?local=true");
        let resp = self
            .with_auth(self.client.post(&url).json(req))
            .send()
            .await
            .with_context(|| format!("forward /query to {target_addr}"))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("peer returned {status}: {body}"));
        }
        resp.json::<QueryResponse>()
            .await
            .context("decode QueryResponse from peer")
    }

    pub async fn forward_insert(
        &self,
        target_addr: &str,
        req: &InsertRequest,
    ) -> Result<InsertResponse> {
        let url = format!("http://{target_addr}/insert?local=true");
        let resp = self
            .with_auth(self.client.post(&url).json(req))
            .send()
            .await
            .with_context(|| format!("forward /insert to {target_addr}"))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("peer returned {status}: {body}"));
        }
        resp.json::<InsertResponse>()
            .await
            .context("decode InsertResponse from peer")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Json, Router, extract::Query, routing::post};
    use std::collections::HashMap;
    use tokio::net::TcpListener;

    async fn spawn_mock(app: Router) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        addr.to_string()
    }

    #[tokio::test]
    async fn test_forward_query_format() {
        let app = Router::new().route(
            "/query",
            post(
                |Query(q): Query<HashMap<String, String>>, Json(_): Json<QueryRequest>| async move {
                    assert_eq!(q.get("local").map(String::as_str), Some("true"));
                    Json(QueryResponse {
                        hit: true,
                        id: Some("u-42".to_string()),
                        response: Some("from peer".to_string()),
                        similarity: Some(0.97),
                    })
                },
            ),
        );
        let addr = spawn_mock(app).await;
        let router = ClusterRouter::new(None);
        let resp = router
            .forward_query(
                &addr,
                &QueryRequest {
                    embedding: vec![1.0, 0.0],
                    threshold: 0.9,
                    model_id: Some("test::2".into()),
                },
            )
            .await
            .unwrap();
        assert!(resp.hit);
        assert_eq!(resp.response.as_deref(), Some("from peer"));
        assert_eq!(resp.id.as_deref(), Some("u-42"));
    }

    #[tokio::test]
    async fn test_forward_includes_auth_header_when_configured() {
        use axum::http::HeaderMap;
        let app = Router::new().route(
            "/insert",
            post(
                |headers: HeaderMap, Json(req): Json<InsertRequest>| async move {
                    let auth = headers
                        .get("authorization")
                        .map(|v| v.to_str().unwrap().to_string());
                    assert_eq!(auth.as_deref(), Some("Bearer s3cr3t"));
                    Json(InsertResponse {
                        id: req.uuid.unwrap_or_else(|| "g".into()),
                        status: "ok".into(),
                    })
                },
            ),
        );
        let addr = spawn_mock(app).await;
        let router = ClusterRouter::new(Some("s3cr3t".to_string()));
        let resp = router
            .forward_insert(
                &addr,
                &InsertRequest {
                    embedding: vec![1.0, 0.0],
                    response: "r".into(),
                    query_text: "q".into(),
                    model_id: Some("test::2".into()),
                    uuid: Some("u".into()),
                },
            )
            .await
            .unwrap();
        assert_eq!(resp.id, "u");
    }

    #[tokio::test]
    async fn test_forward_insert_format() {
        let app = Router::new().route(
            "/insert",
            post(
                |Query(q): Query<HashMap<String, String>>, Json(req): Json<InsertRequest>| async move {
                    assert_eq!(q.get("local").map(String::as_str), Some("true"));
                    let id = req.uuid.unwrap_or_else(|| "generated".to_string());
                    Json(InsertResponse {
                        id,
                        status: "ok".to_string(),
                    })
                },
            ),
        );
        let addr = spawn_mock(app).await;
        let router = ClusterRouter::new(None);
        let resp = router
            .forward_insert(
                &addr,
                &InsertRequest {
                    embedding: vec![1.0, 0.0],
                    response: "r".into(),
                    query_text: "q".into(),
                    model_id: Some("test::2".into()),
                    uuid: Some("fixed-uuid".into()),
                },
            )
            .await
            .unwrap();
        assert_eq!(resp.id, "fixed-uuid");
        assert_eq!(resp.status, "ok");
    }
}
