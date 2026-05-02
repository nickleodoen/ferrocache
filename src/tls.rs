//! mTLS for inter-node cluster traffic (M18).
//!
//! Provides:
//! - `generate_ca` / `generate_node_cert` — rcgen 0.13 wrappers that produce a
//!   CA and per-node leaf certs signed by it (dev mode).
//! - `load_or_generate` — picks load-from-disk vs. dev generation based on
//!   the `ClusterTlsConfig`.
//! - `build_server_config` — a `rustls::ServerConfig` that demands a client
//!   cert chained to the cluster CA (mTLS, not just TLS).
//! - `install_default_crypto_provider` — call once at startup so subsequent
//!   `WebPkiClientVerifier::builder` etc. don't panic.
//!
//! Scope: this module is concerned only with cluster traffic. The public API
//! port stays plain HTTP and is terminated by a reverse proxy in production.

use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use rcgen::{
    BasicConstraints, Certificate, CertificateParams, DnType, ExtendedKeyUsagePurpose, IsCa,
    KeyPair, KeyUsagePurpose,
};
use rustls::RootCertStore;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use rustls::server::WebPkiClientVerifier;

use crate::config::ClusterTlsConfig;

/// PEM bundle a node needs at runtime to participate in mTLS:
/// the cluster CA (used to verify peers + as a client-cert verifier root)
/// plus this node's own leaf cert + private key.
#[derive(Clone, Debug)]
pub struct TlsBundle {
    pub ca_cert_pem: String,
    pub node_cert_pem: String,
    pub node_key_pem: String,
}

/// Install the aws-lc-rs CryptoProvider as the rustls process default.
///
/// Idempotent — safe to call multiple times. rustls 0.23 requires a default
/// provider before `WebPkiClientVerifier::builder` etc. work; we install it
/// explicitly so initialization order is deterministic.
pub fn install_default_crypto_provider() {
    // `install_default` returns `Err(_)` if a provider was already installed,
    // which is fine — we just want one to be there.
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
}

/// Build a self-signed CA suitable for signing a small homogeneous cluster.
///
/// Sets `is_ca = Ca(Unconstrained)` and `KeyCertSign` usage so the resulting
/// cert is recognized as a CA by rustls' webpki verifier.
pub fn generate_ca() -> Result<(Certificate, KeyPair)> {
    let key_pair = KeyPair::generate().context("CA keypair")?;
    let mut params = CertificateParams::new(Vec::<String>::new()).context("CA params")?;
    params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    params.key_usages = vec![KeyUsagePurpose::KeyCertSign, KeyUsagePurpose::CrlSign];
    params
        .distinguished_name
        .push(DnType::CommonName, "ferrocache-dev-ca");
    let cert = params.self_signed(&key_pair).context("self-sign CA")?;
    Ok((cert, key_pair))
}

/// Build a leaf cert for `node_id`, signed by `(ca_cert, ca_key)`.
///
/// Each entry in `san_addresses` is parsed as an IP address if it round-trips
/// through `IpAddr::from_str`; otherwise it's treated as a DNS name.
/// Both `ServerAuth` and `ClientAuth` EKUs are set so the same cert can play
/// either role in a mTLS handshake — node A is a "client" when it forwards a
/// replication request, but a "server" when it receives one.
pub fn generate_node_cert(
    ca_cert: &Certificate,
    ca_key: &KeyPair,
    node_id: &str,
    san_addresses: &[String],
) -> Result<(Certificate, KeyPair)> {
    let key_pair = KeyPair::generate().context("node keypair")?;
    let mut params = CertificateParams::new(san_addresses.to_vec()).context("node cert params")?;
    params.distinguished_name.push(DnType::CommonName, node_id);
    params.key_usages = vec![
        KeyUsagePurpose::DigitalSignature,
        KeyUsagePurpose::KeyEncipherment,
    ];
    params.extended_key_usages = vec![
        ExtendedKeyUsagePurpose::ServerAuth,
        ExtendedKeyUsagePurpose::ClientAuth,
    ];
    let cert = params
        .signed_by(&key_pair, ca_cert, ca_key)
        .context("sign node cert")?;
    Ok((cert, key_pair))
}

/// Read a PEM file from disk and return its UTF-8 contents.
fn read_pem(path: &str) -> Result<String> {
    std::fs::read_to_string(path).with_context(|| format!("read TLS PEM at {path}"))
}

/// Resolve the runtime TLS bundle for this node.
///
/// - All three paths set → load from disk (production mode).
/// - Any path missing → generate a fresh CA + leaf cert in memory and emit a
///   warning. Useful for local-cluster smoke tests; never appropriate for
///   production because every node generates its own CA and they don't trust
///   each other.
pub fn load_or_generate(
    config: &ClusterTlsConfig,
    node_id: &str,
    api_addr: &str,
) -> Result<TlsBundle> {
    match (
        config.ca_cert_path.as_deref(),
        config.node_cert_path.as_deref(),
        config.node_key_path.as_deref(),
    ) {
        (Some(ca), Some(cert), Some(key)) => {
            tracing::info!(
                ca = ca,
                cert = cert,
                key = key,
                "loading TLS certs from disk"
            );
            Ok(TlsBundle {
                ca_cert_pem: read_pem(ca)?,
                node_cert_pem: read_pem(cert)?,
                node_key_pem: read_pem(key)?,
            })
        }
        _ => {
            tracing::warn!(
                "generating self-signed TLS certs in memory (DEV MODE — \
                 every node will have its own CA and won't trust peers; use \
                 cluster.tls.{{ca_cert_path,node_cert_path,node_key_path}} \
                 in production)"
            );
            let (ca_cert, ca_key) = generate_ca()?;
            let host = api_addr
                .split(':')
                .next()
                .unwrap_or("localhost")
                .to_string();
            let mut sans = vec![host];
            if !sans.iter().any(|s| s == "localhost") {
                sans.push("localhost".to_string());
            }
            sans.push("127.0.0.1".to_string());
            let (node_cert, node_key) = generate_node_cert(&ca_cert, &ca_key, node_id, &sans)?;
            Ok(TlsBundle {
                ca_cert_pem: ca_cert.pem(),
                node_cert_pem: node_cert.pem(),
                node_key_pem: node_key.serialize_pem(),
            })
        }
    }
}

/// Parse a PEM bundle into `(cert_chain, private_key)` ready for rustls.
fn pem_to_der(
    cert_pem: &str,
    key_pem: &str,
) -> Result<(Vec<CertificateDer<'static>>, PrivateKeyDer<'static>)> {
    let mut cert_reader = std::io::Cursor::new(cert_pem.as_bytes());
    let cert_chain: Vec<CertificateDer<'static>> = rustls_pemfile::certs(&mut cert_reader)
        .collect::<std::io::Result<_>>()
        .context("parse cert chain PEM")?;
    if cert_chain.is_empty() {
        return Err(anyhow!("no certificates found in cert PEM"));
    }
    let mut key_reader = std::io::Cursor::new(key_pem.as_bytes());
    let pkcs8 = rustls_pemfile::pkcs8_private_keys(&mut key_reader)
        .next()
        .ok_or_else(|| anyhow!("no PKCS8 private key found in key PEM"))?
        .context("parse PKCS8 private key")?;
    let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(pkcs8.secret_pkcs8_der().to_vec()));
    Ok((cert_chain, key))
}

/// Build a `rustls::ServerConfig` that:
/// - presents `bundle.node_cert_pem` + `bundle.node_key_pem` as our identity,
/// - requires a client cert and verifies it against `bundle.ca_cert_pem`.
///
/// I.e. mTLS, not just TLS — anonymous clients are rejected at handshake.
pub fn build_server_config(bundle: &TlsBundle) -> Result<rustls::ServerConfig> {
    let mut roots = RootCertStore::empty();
    let mut ca_reader = std::io::Cursor::new(bundle.ca_cert_pem.as_bytes());
    for ca in rustls_pemfile::certs(&mut ca_reader) {
        let ca = ca.context("parse cluster CA PEM")?;
        roots.add(ca).context("add CA to root store")?;
    }
    if roots.is_empty() {
        return Err(anyhow!("cluster CA PEM contained no certificates"));
    }
    let client_verifier = WebPkiClientVerifier::builder(Arc::new(roots))
        .build()
        .context("build client cert verifier")?;
    let (cert_chain, key) = pem_to_der(&bundle.node_cert_pem, &bundle.node_key_pem)?;
    let server_config = rustls::ServerConfig::builder()
        .with_client_cert_verifier(client_verifier)
        .with_single_cert(cert_chain, key)
        .context("build rustls ServerConfig")?;
    Ok(server_config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn dev_config() -> ClusterTlsConfig {
        ClusterTlsConfig::default()
    }

    #[test]
    fn test_generate_ca_returns_pem() {
        let (cert, key) = generate_ca().unwrap();
        let cert_pem = cert.pem();
        let key_pem = key.serialize_pem();
        assert!(cert_pem.starts_with("-----BEGIN CERTIFICATE-----"));
        assert!(key_pem.starts_with("-----BEGIN PRIVATE KEY-----"));
    }

    #[test]
    fn test_generate_node_cert_distinct_from_ca() {
        let (ca_cert, ca_key) = generate_ca().unwrap();
        let (node_cert, _node_key) = generate_node_cert(
            &ca_cert,
            &ca_key,
            "node1",
            &["node1".to_string(), "127.0.0.1".to_string()],
        )
        .unwrap();
        let ca_pem = ca_cert.pem();
        let node_pem = node_cert.pem();
        assert!(node_pem.starts_with("-----BEGIN CERTIFICATE-----"));
        assert_ne!(ca_pem, node_pem, "node cert must not equal CA cert");
    }

    #[test]
    fn test_load_or_generate_dev_mode() {
        install_default_crypto_provider();
        let bundle = load_or_generate(&dev_config(), "node1", "node1:3000").unwrap();
        assert!(!bundle.ca_cert_pem.is_empty());
        assert!(!bundle.node_cert_pem.is_empty());
        assert!(!bundle.node_key_pem.is_empty());
        // Sanity: the node key really is PKCS#8 PEM (what reqwest's
        // rustls-tls Identity::from_pem expects).
        assert!(bundle.node_key_pem.contains("BEGIN PRIVATE KEY"));
    }

    #[test]
    fn test_load_or_generate_production_mode() {
        let dir = tempfile::tempdir().unwrap();
        let (ca_cert, ca_key) = generate_ca().unwrap();
        let (node_cert, node_key) =
            generate_node_cert(&ca_cert, &ca_key, "node1", &["node1".to_string()]).unwrap();
        let ca_path = dir.path().join("ca.pem");
        let cert_path = dir.path().join("cert.pem");
        let key_path = dir.path().join("key.pem");
        std::fs::write(&ca_path, ca_cert.pem()).unwrap();
        std::fs::write(&cert_path, node_cert.pem()).unwrap();
        std::fs::write(&key_path, node_key.serialize_pem()).unwrap();

        let cfg = ClusterTlsConfig {
            enabled: true,
            ca_cert_path: Some(ca_path.to_string_lossy().into_owned()),
            node_cert_path: Some(cert_path.to_string_lossy().into_owned()),
            node_key_path: Some(key_path.to_string_lossy().into_owned()),
            internal_port: None,
        };
        let bundle = load_or_generate(&cfg, "node1", "node1:3000").unwrap();
        assert_eq!(bundle.ca_cert_pem, ca_cert.pem());
        assert_eq!(bundle.node_cert_pem, node_cert.pem());
        assert_eq!(bundle.node_key_pem, node_key.serialize_pem());
        let _ = PathBuf::from(dir.path());
    }

    #[test]
    fn test_build_server_config_smoke() {
        install_default_crypto_provider();
        let bundle = load_or_generate(&dev_config(), "node1", "node1:3000").unwrap();
        // The function takes ~3 fallible parsing steps; smoke-testing that it
        // returns Ok proves all of them line up with what rcgen emits.
        let _server_cfg = build_server_config(&bundle).expect("server config builds");
    }

    #[tokio::test]
    async fn test_mtls_replication_roundtrip() {
        // End-to-end: spin up a TLS axum listener with mTLS required, hit it
        // with a reqwest client carrying a peer cert from the same CA. Both
        // sides verify each other's chain.
        install_default_crypto_provider();
        let (ca_cert, ca_key) = generate_ca().unwrap();
        let (server_cert, server_key) =
            generate_node_cert(&ca_cert, &ca_key, "server", &["127.0.0.1".to_string()]).unwrap();
        let (client_cert, client_key) =
            generate_node_cert(&ca_cert, &ca_key, "client", &["client".to_string()]).unwrap();

        let server_bundle = TlsBundle {
            ca_cert_pem: ca_cert.pem(),
            node_cert_pem: server_cert.pem(),
            node_key_pem: server_key.serialize_pem(),
        };
        let server_config = build_server_config(&server_bundle).unwrap();
        let rustls_config =
            axum_server::tls_rustls::RustlsConfig::from_config(Arc::new(server_config));

        let app = axum::Router::new().route(
            "/health",
            axum::routing::get(|| async { axum::Json(serde_json::json!({"status":"ok"})) }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let std_listener = listener.into_std().unwrap();
        std_listener.set_nonblocking(true).unwrap();
        tokio::spawn(async move {
            axum_server::from_tcp_rustls(std_listener, rustls_config)
                .serve(app.into_make_service())
                .await
                .unwrap();
        });

        // Build a reqwest client with the same CA + the *client* node's cert
        // as identity. Disable the system root store so any leak from the
        // host's truststore can't accidentally make the test pass.
        let identity_pem = format!("{}\n{}", client_cert.pem(), client_key.serialize_pem());
        let identity = reqwest::Identity::from_pem(identity_pem.as_bytes()).unwrap();
        let ca = reqwest::Certificate::from_pem(ca_cert.pem().as_bytes()).unwrap();
        let client = reqwest::Client::builder()
            .tls_built_in_root_certs(false)
            .add_root_certificate(ca)
            .identity(identity)
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap();
        let url = format!("https://{addr}/health");
        // Brief warmup — axum_server::from_tcp_rustls spawns its accept loop
        // asynchronously; without a tiny yield the first connect can race
        // the bind on slow CI.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let resp = client.get(&url).send().await.expect("mTLS request");
        assert_eq!(resp.status(), 200);
    }

    #[tokio::test]
    async fn test_mtls_rejects_unknown_ca() {
        install_default_crypto_provider();
        let (server_ca, server_ca_key) = generate_ca().unwrap();
        let (server_cert, server_key) = generate_node_cert(
            &server_ca,
            &server_ca_key,
            "server",
            &["127.0.0.1".to_string()],
        )
        .unwrap();
        // Client cert from a DIFFERENT CA → server must reject.
        let (foreign_ca, foreign_ca_key) = generate_ca().unwrap();
        let (foreign_cert, foreign_key) = generate_node_cert(
            &foreign_ca,
            &foreign_ca_key,
            "intruder",
            &["intruder".to_string()],
        )
        .unwrap();

        let server_bundle = TlsBundle {
            ca_cert_pem: server_ca.pem(),
            node_cert_pem: server_cert.pem(),
            node_key_pem: server_key.serialize_pem(),
        };
        let server_config = build_server_config(&server_bundle).unwrap();
        let rustls_config =
            axum_server::tls_rustls::RustlsConfig::from_config(Arc::new(server_config));

        let app = axum::Router::new().route("/health", axum::routing::get(|| async { "ok" }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let std_listener = listener.into_std().unwrap();
        std_listener.set_nonblocking(true).unwrap();
        tokio::spawn(async move {
            // Errors from a rejected handshake are fine; the test cares about
            // the *client* observing the rejection.
            let _ = axum_server::from_tcp_rustls(std_listener, rustls_config)
                .serve(app.into_make_service())
                .await;
        });

        // Client trusts the *server's* CA but presents a cert from the
        // *foreign* CA — the server should refuse the handshake.
        let identity_pem = format!("{}\n{}", foreign_cert.pem(), foreign_key.serialize_pem());
        let identity = reqwest::Identity::from_pem(identity_pem.as_bytes()).unwrap();
        let server_ca_for_client =
            reqwest::Certificate::from_pem(server_ca.pem().as_bytes()).unwrap();
        let client = reqwest::Client::builder()
            .tls_built_in_root_certs(false)
            .add_root_certificate(server_ca_for_client)
            .identity(identity)
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap();
        let url = format!("https://{addr}/health");
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let result = client.get(&url).send().await;
        assert!(
            result.is_err(),
            "request unexpectedly succeeded with an untrusted client cert: {result:?}"
        );
    }
}
