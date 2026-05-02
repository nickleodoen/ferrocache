use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use reqwest::Client;

use crate::models::{InsertRequest, InsertResponse, QueryRequest, QueryResponse};
use crate::tls::TlsBundle;

const FORWARD_TIMEOUT: Duration = Duration::from_secs(5);

pub struct ClusterRouter {
    client: Client,
    auth_token: Option<String>,
    tls_enabled: bool,
}

impl Default for ClusterRouter {
    fn default() -> Self {
        Self::new(None, None).expect("plain reqwest client builds")
    }
}

impl ClusterRouter {
    /// Build the inter-node forwarding client.
    ///
    /// When `tls_bundle` is `Some`, the client only trusts the cluster CA
    /// (system roots disabled) and presents this node's leaf cert as a
    /// client identity — i.e. mTLS, not bearer-token-over-TLS.
    pub fn new(auth_token: Option<String>, tls_bundle: Option<&TlsBundle>) -> Result<Self> {
        let mut builder = Client::builder().timeout(FORWARD_TIMEOUT);
        let tls_enabled = tls_bundle.is_some();
        if let Some(bundle) = tls_bundle {
            let ca = reqwest::Certificate::from_pem(bundle.ca_cert_pem.as_bytes())
                .context("parse cluster CA PEM for reqwest")?;
            // Concatenated cert+key PEM is what reqwest's rustls-tls
            // Identity::from_pem expects.
            let identity_pem = format!("{}\n{}", bundle.node_cert_pem, bundle.node_key_pem);
            let identity = reqwest::Identity::from_pem(identity_pem.as_bytes())
                .context("build reqwest identity from node cert/key")?;
            builder = builder
                .tls_built_in_root_certs(false)
                .add_root_certificate(ca)
                .identity(identity);
        }
        let client = builder.build().context("build reqwest client")?;
        Ok(Self {
            client,
            auth_token,
            tls_enabled,
        })
    }

    fn scheme(&self) -> &'static str {
        if self.tls_enabled { "https" } else { "http" }
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
        let scheme = self.scheme();
        let url = format!("{scheme}://{target_addr}/query?local=true");
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
        let scheme = self.scheme();
        let url = format!("{scheme}://{target_addr}/insert?local=true");
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
        let router = ClusterRouter::new(None, None).unwrap();
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
        let router = ClusterRouter::new(Some("s3cr3t".to_string()), None).unwrap();
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
        let router = ClusterRouter::new(None, None).unwrap();
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
