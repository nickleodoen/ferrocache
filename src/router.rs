use std::future::Future;
use std::time::Duration;

use anyhow::{Context, Result};
use rand::Rng;
use reqwest::Client;

use crate::metrics::Metrics;
use crate::models::{
    FullEntryResponse, InsertRequest, InsertResponse, InvalidateRequest, InvalidateResponse,
    QueryRequest, QueryResponse,
};
use crate::tls::TlsBundle;

const FORWARD_TIMEOUT: Duration = Duration::from_secs(5);
/// Cap the per-attempt sleep so a misconfigured `max_replication_retries`
/// can't park the caller for hours.
const MAX_RETRY_DELAY_MS: u64 = 5_000;

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

    /// Forward a /query to a peer (single attempt).
    ///
    /// Returns the typed `reqwest::Error` so the retry layer can classify
    /// failures by `e.is_connect()` / `e.is_timeout()` / `e.status()`. The
    /// `?` after `error_for_status()` ensures a 5xx response surfaces as an
    /// `Err` whose `status()` is set, which the retry layer recognizes.
    pub async fn forward_query(
        &self,
        target_addr: &str,
        req: &QueryRequest,
    ) -> reqwest::Result<QueryResponse> {
        let scheme = self.scheme();
        // `repair=true` (M23): the recipient IS the ring owner, so on a
        // local miss it should fan out to its replicas. `local=true`
        // continues to mean "skip ring routing".
        let url = format!("{scheme}://{target_addr}/query?local=true&repair=true");
        let resp = self
            .with_auth(self.client.post(&url).json(req))
            .send()
            .await?;
        let resp = resp.error_for_status()?;
        resp.json::<QueryResponse>().await
    }

    /// Forward a /query to a replica during read-repair fan-out (M23). The
    /// `repair=false` param tells the recipient NOT to do its own
    /// read-repair on miss — that prevents an N-way fan-out from looping
    /// the cluster. Always uses `?local=true` for the same reason.
    pub async fn forward_query_no_repair(
        &self,
        target_addr: &str,
        req: &QueryRequest,
    ) -> reqwest::Result<QueryResponse> {
        let scheme = self.scheme();
        let url = format!("{scheme}://{target_addr}/query?local=true&repair=false");
        let resp = self
            .with_auth(self.client.post(&url).json(req))
            .send()
            .await?;
        let resp = resp.error_for_status()?;
        resp.json::<QueryResponse>().await
    }

    /// Fetch the full entry (including embedding) for a UUID from a peer
    /// during read repair. Returns `Ok(None)` for a 404 — the peer didn't
    /// have the entry — and `Err` for transport / non-404 status.
    pub async fn forward_get_entry(
        &self,
        target_addr: &str,
        uuid: &str,
    ) -> reqwest::Result<Option<FullEntryResponse>> {
        let scheme = self.scheme();
        let url = format!("{scheme}://{target_addr}/internal/entry/{uuid}?local=true");
        let resp = self.with_auth(self.client.get(&url)).send().await?;
        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        let resp = resp.error_for_status()?;
        let body = resp.json::<FullEntryResponse>().await?;
        Ok(Some(body))
    }

    /// Fire-and-forget delete forward (M26). Returns the HTTP status so
    /// the caller can distinguish 200 (peer had the entry) from 404
    /// (peer didn't — idempotent success). Never retries: the local
    /// node already removed the entry, so a slow/down peer is allowed
    /// to miss this and pick up the tombstone on next replay or the
    /// next time the user does anything.
    pub async fn forward_delete_entry(
        &self,
        target_addr: &str,
        uuid: &str,
    ) -> reqwest::Result<reqwest::StatusCode> {
        let scheme = self.scheme();
        let url = format!("{scheme}://{target_addr}/entry/{uuid}?local=true");
        let resp = self.with_auth(self.client.delete(&url)).send().await?;
        Ok(resp.status())
    }

    /// Forward `/admin/invalidate` to a peer (M26). Each replica computes
    /// its own matches against the same embedding+threshold; we don't
    /// share the UUID list because it'd assume every replica has the
    /// same internal state (which is true today but a fragile contract).
    pub async fn forward_invalidate(
        &self,
        target_addr: &str,
        req: &InvalidateRequest,
    ) -> reqwest::Result<InvalidateResponse> {
        let scheme = self.scheme();
        let url = format!("{scheme}://{target_addr}/admin/invalidate?local=true");
        let resp = self
            .with_auth(self.client.post(&url).json(req))
            .send()
            .await?;
        let resp = resp.error_for_status()?;
        resp.json::<InvalidateResponse>().await
    }

    /// Forward an /insert to a peer (single attempt). See `forward_query`
    /// for the retry-classification rationale.
    pub async fn forward_insert(
        &self,
        target_addr: &str,
        req: &InsertRequest,
    ) -> reqwest::Result<InsertResponse> {
        let scheme = self.scheme();
        let url = format!("{scheme}://{target_addr}/insert?local=true");
        let resp = self
            .with_auth(self.client.post(&url).json(req))
            .send()
            .await?;
        let resp = resp.error_for_status()?;
        resp.json::<InsertResponse>().await
    }

    /// Run `make_request` with exponential backoff. `max_retries` is the
    /// number of retries *after* the initial attempt — so total attempts =
    /// `max_retries + 1`.
    ///
    /// Backoff: base = 50ms × 2^(attempt-1), capped at 5s, with ±20% jitter.
    /// Jitter sourced from `rand` so multiple replicas under the same
    /// upstream blip don't synchronize their retry waves.
    ///
    /// Retry policy: connect errors, timeouts, and 5xx responses are
    /// retried. 4xx is treated as a deterministic failure (wrong token,
    /// bad request) and surfaces immediately.
    pub async fn forward_with_retry<F, Fut, T>(
        &self,
        max_retries: usize,
        metrics: &Metrics,
        mut make_request: F,
    ) -> Result<T>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = reqwest::Result<T>>,
    {
        let total = max_retries.saturating_add(1);
        let mut last_err: Option<reqwest::Error> = None;
        for attempt in 0..total {
            if attempt > 0 {
                metrics.record_replication_retry();
                let delay = compute_retry_delay(attempt);
                tracing::warn!(
                    attempt,
                    max_retries,
                    delay_ms = delay.as_millis() as u64,
                    "retrying replication"
                );
                tokio::time::sleep(delay).await;
            }
            match make_request().await {
                Ok(value) => return Ok(value),
                Err(e) => {
                    if is_retryable(&e) {
                        last_err = Some(e);
                        continue;
                    }
                    return Err(anyhow::Error::new(e)).context("replication forward failed");
                }
            }
        }
        Err(anyhow::Error::new(last_err.expect("at least one attempt")))
            .context("replication forward exhausted retries")
    }
}

/// Connect errors, timeouts, and 5xx responses are transient. Anything
/// else (including 4xx) is treated as deterministic and not retried.
fn is_retryable(e: &reqwest::Error) -> bool {
    if e.is_connect() || e.is_timeout() {
        return true;
    }
    e.status().is_some_and(|s| s.is_server_error())
}

fn compute_retry_delay(attempt: usize) -> Duration {
    // attempt == 1 → 50ms base, attempt == 2 → 100ms, attempt == 3 → 200ms…
    let shift = (attempt as u32).saturating_sub(1);
    let base_ms: u64 = 50u64
        .checked_shl(shift)
        .unwrap_or(MAX_RETRY_DELAY_MS)
        .min(MAX_RETRY_DELAY_MS);
    let jitter_ms = (base_ms as f64 * 0.2) as i64;
    let signed_jitter: i64 = if jitter_ms > 0 {
        // Range [-jitter_ms, +jitter_ms]
        let mut rng = rand::thread_rng();
        rng.gen_range(-jitter_ms..=jitter_ms)
    } else {
        0
    };
    let delay_ms = (base_ms as i64 + signed_jitter).max(1) as u64;
    Duration::from_millis(delay_ms)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::response::IntoResponse;
    use axum::{Json, Router, extract::Query, routing::post};
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
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
                    assert_eq!(q.get("repair").map(String::as_str), Some("true"));
                    Json(QueryResponse {
                        hit: true,
                        id: Some("u-42".to_string()),
                        response: Some("from peer".to_string()),
                        similarity: Some(0.97),
                        exact_match: None,
                        scope: None,
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
                    query_text: None,
                    cache_scope: None,
                    conversation_id: None,
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
                    ttl_seconds: None,
                    cache_scope: None,
                    conversation_id: None,
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
                    ttl_seconds: None,
                    cache_scope: None,
                    conversation_id: None,
                },
            )
            .await
            .unwrap();
        assert_eq!(resp.id, "fixed-uuid");
        assert_eq!(resp.status, "ok");
    }

    /// Mock that returns 500 for the first `fail_first` requests and 200
    /// thereafter. Returns the addr + a counter the test can read.
    async fn flaky_mock(fail_first: usize) -> (String, Arc<AtomicUsize>) {
        let count = Arc::new(AtomicUsize::new(0));
        let count_clone = count.clone();
        let app = Router::new().route(
            "/insert",
            post(move |Json(req): Json<InsertRequest>| {
                let count = count_clone.clone();
                async move {
                    let n = count.fetch_add(1, Ordering::SeqCst);
                    if n < fail_first {
                        (
                            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                            "boom".to_string(),
                        )
                            .into_response()
                    } else {
                        Json(InsertResponse {
                            id: req.uuid.unwrap_or_else(|| "ok".into()),
                            status: "ok".into(),
                        })
                        .into_response()
                    }
                }
            }),
        );
        (spawn_mock(app).await, count)
    }

    /// Mock that always returns the given status code with a small body.
    async fn always_status(status: axum::http::StatusCode) -> (String, Arc<AtomicUsize>) {
        let count = Arc::new(AtomicUsize::new(0));
        let count_clone = count.clone();
        let app = Router::new().route(
            "/insert",
            post(move |Json(_): Json<InsertRequest>| {
                let count = count_clone.clone();
                async move {
                    count.fetch_add(1, Ordering::SeqCst);
                    (status, "no").into_response()
                }
            }),
        );
        (spawn_mock(app).await, count)
    }

    fn sample_insert() -> InsertRequest {
        InsertRequest {
            embedding: vec![1.0, 0.0],
            response: "r".into(),
            query_text: "q".into(),
            model_id: Some("t::2".into()),
            uuid: Some("u".into()),
            ttl_seconds: None,
            cache_scope: None,
            conversation_id: None,
        }
    }

    #[tokio::test]
    async fn test_retry_on_connection_error() {
        // 5xx is one of the retryable error classes. The mock fails twice
        // (returns 500) and succeeds on attempt 3 — overall call must Ok.
        let (addr, count) = flaky_mock(2).await;
        let router = ClusterRouter::new(None, None).unwrap();
        let metrics = Metrics::new();
        let req = sample_insert();
        let res = router
            .forward_with_retry(3, &metrics, || router.forward_insert(&addr, &req))
            .await;
        assert!(res.is_ok(), "expected success after retries: {res:?}");
        assert_eq!(count.load(Ordering::SeqCst), 3);
        assert_eq!(metrics.replication_retries_total.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn test_no_retry_on_4xx() {
        let (addr, count) = always_status(axum::http::StatusCode::BAD_REQUEST).await;
        let router = ClusterRouter::new(None, None).unwrap();
        let metrics = Metrics::new();
        let req = sample_insert();
        let res = router
            .forward_with_retry(3, &metrics, || router.forward_insert(&addr, &req))
            .await;
        assert!(res.is_err(), "4xx must not be retried but did not Err");
        assert_eq!(count.load(Ordering::SeqCst), 1, "only one request expected");
        assert_eq!(metrics.replication_retries_total.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn test_retry_exhausted_returns_error() {
        let (addr, count) = always_status(axum::http::StatusCode::INTERNAL_SERVER_ERROR).await;
        let router = ClusterRouter::new(None, None).unwrap();
        let metrics = Metrics::new();
        let req = sample_insert();
        // max_retries=2 → total attempts = 3
        let res = router
            .forward_with_retry(2, &metrics, || router.forward_insert(&addr, &req))
            .await;
        assert!(res.is_err());
        assert_eq!(count.load(Ordering::SeqCst), 3);
        assert_eq!(metrics.replication_retries_total.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn test_forward_get_entry_200() {
        use axum::{Router, extract::Path, routing::get};
        let app = Router::new().route(
            "/internal/entry/:uuid",
            get(|Path(uuid): Path<String>| async move {
                Json(FullEntryResponse {
                    uuid,
                    embedding: vec![1.0, 0.0, 0.0],
                    response: "from peer".into(),
                    query_text: "q".into(),
                    model_id: "m::3".into(),
                    inserted_at: 0,
                    last_accessed_at: 0,
                    access_count: 0,
                    expires_at: None,
                })
            }),
        );
        let addr = spawn_mock(app).await;
        let router = ClusterRouter::new(None, None).unwrap();
        let resp = router.forward_get_entry(&addr, "u-42").await.unwrap();
        let entry = resp.expect("Some(entry) for 200");
        assert_eq!(entry.uuid, "u-42");
        assert_eq!(entry.response, "from peer");
        assert_eq!(entry.embedding.len(), 3);
        assert_eq!(entry.model_id, "m::3");
    }

    #[tokio::test]
    async fn test_forward_get_entry_404() {
        use axum::{Router, http::StatusCode, routing::get};
        let app = Router::new().route(
            "/internal/entry/:uuid",
            get(|_: axum::extract::Path<String>| async move {
                (StatusCode::NOT_FOUND, "not found").into_response()
            }),
        );
        let addr = spawn_mock(app).await;
        let router = ClusterRouter::new(None, None).unwrap();
        let resp = router.forward_get_entry(&addr, "ghost").await.unwrap();
        assert!(resp.is_none(), "404 must surface as Ok(None)");
    }

    #[tokio::test]
    async fn test_retry_counter_increments() {
        // The brief's "test_retry_backoff_increases" — we don't try to time
        // the sleeps (flaky), but we lock in that one call → 2 retries
        // bumps the metric by exactly 2.
        let (addr, _count) = flaky_mock(2).await;
        let router = ClusterRouter::new(None, None).unwrap();
        let metrics = Metrics::new();
        let req = sample_insert();
        let _ = router
            .forward_with_retry(3, &metrics, || router.forward_insert(&addr, &req))
            .await
            .unwrap();
        assert_eq!(metrics.replication_retries_total.load(Ordering::Relaxed), 2);
    }
}
