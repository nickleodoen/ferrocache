use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use tokio::sync::RwLock;
use tracing_subscriber::EnvFilter;

use ferrocache::cluster::ClusterState;
use ferrocache::config::FerrocacheConfig;
use ferrocache::index::SemanticIndex;
use ferrocache::metrics::Metrics;
use ferrocache::router::ClusterRouter;
use ferrocache::server;
use ferrocache::snapshot;
use ferrocache::state::AppState;
use ferrocache::tls;
use ferrocache::wal::{DEFAULT_CHANNEL_CAPACITY, GroupCommitConfig, GroupCommitWal, Wal};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let config = FerrocacheConfig::load().context("config load failed")?;
    // Avoid `?config` here so the auth_token never lands in logs via Debug.
    tracing::info!(
        port = config.port,
        wal_path = %config.wal_path,
        cluster_enabled = config.cluster.enabled,
        compact_interval_inserts = config.compact_interval_inserts,
        "loaded config"
    );
    let auth_enabled = config.auth_token.as_ref().is_some_and(|t| !t.is_empty());
    if auth_enabled {
        tracing::info!("bearer token auth enabled");
    } else {
        tracing::info!("bearer token auth disabled (FERROCACHE_AUTH_TOKEN not set)");
    }

    let node_id = config
        .node_id
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    let snapshot_path = snapshot::snapshot_path_for(&config.wal_path);
    tracing::info!(
        wal_path = %config.wal_path,
        snapshot_path = %snapshot_path.display(),
        "WAL paths"
    );

    let mut index = SemanticIndex::new(&config.hnsw);

    // Try snapshot first; fall back to full WAL replay if missing or corrupt.
    let snapshot_sequence: Option<u64> = if snapshot_path.exists() {
        match snapshot::read_snapshot(&snapshot_path).await {
            Ok((entries, seq)) => {
                let count = entries.len();
                let mut loaded = 0usize;
                for e in entries {
                    match index.replay_snapshot_entry(e) {
                        Ok(()) => loaded += 1,
                        Err(err) => tracing::warn!(error = %err, "snapshot entry rejected"),
                    }
                }
                tracing::info!(loaded, total = count, wal_sequence = seq, "snapshot loaded");
                Some(seq)
            }
            Err(e) => {
                tracing::warn!(error = %e, "snapshot read failed; falling back to full WAL replay");
                None
            }
        }
    } else {
        None
    };

    let entries = Wal::replay(&config.wal_path)
        .await
        .context("WAL replay failed")?;
    let mut replayed = 0usize;
    let mut max_seq = snapshot_sequence.unwrap_or(0);
    let watermark = snapshot_sequence.unwrap_or(0);
    let only_tail = snapshot_sequence.is_some();
    for entry in entries {
        if only_tail && entry.sequence <= watermark {
            continue;
        }
        if entry.sequence > max_seq {
            max_seq = entry.sequence;
        }
        match index.replay_entry(entry) {
            Ok(()) => replayed += 1,
            Err(e) => tracing::warn!(error = %e, "WAL replay entry rejected"),
        }
    }
    tracing::info!(
        wal_tail_entries = replayed,
        snapshot_watermark = watermark,
        "startup replay complete"
    );

    let wal = Wal::open_with_sequence(&config.wal_path, max_seq)
        .await
        .context("WAL open failed")?;

    // mTLS materialization is bound to cluster mode — there's no peer
    // traffic in single-node mode, so the bundle stays None.
    let tls_bundle = if config.cluster.enabled && config.cluster.tls.enabled {
        tls::install_default_crypto_provider();
        let bundle = tls::load_or_generate(&config.cluster.tls, &node_id, &config.cluster.api_addr)
            .context("TLS bundle init")?;
        tracing::info!("cluster mTLS enabled");
        Some(bundle)
    } else {
        if config.cluster.tls.enabled {
            tracing::warn!(
                "cluster.tls.enabled=true ignored because cluster.enabled=false (single-node mode)"
            );
        }
        None
    };

    // The internal port: explicit override > public port + 1000. We log a
    // warning if the derived port is the same number as the gossip UDP port —
    // legal (different proto) but a footgun for operators reading firewall
    // rules.
    let internal_port = config
        .cluster
        .tls
        .internal_port
        .unwrap_or_else(|| config.port.saturating_add(1000));
    if config.cluster.enabled
        && config.cluster.tls.enabled
        && config
            .cluster
            .gossip_addr
            .rsplit_once(':')
            .and_then(|(_, p)| p.parse::<u16>().ok())
            == Some(internal_port)
    {
        tracing::warn!(
            internal_port,
            gossip_port = internal_port,
            "TLS internal_port equals gossip UDP port — set cluster.tls.internal_port explicitly to avoid confusion"
        );
    }
    // Derive the host portion from api_addr (e.g. "node1:3000" → "node1").
    // This is what peers will dial when forwarding replication via TLS.
    let api_host = config
        .cluster
        .api_addr
        .rsplit_once(':')
        .map(|(host, _)| host)
        .unwrap_or(&config.cluster.api_addr)
        .to_string();
    let internal_addr = format!("{api_host}:{internal_port}");
    let forward_addr = if tls_bundle.is_some() {
        internal_addr.clone()
    } else {
        config.cluster.api_addr.clone()
    };

    // Build shared state holders (metrics + index) up front so the cluster
    // reconciler and the WAL flush task can both share them.
    let index_arc = Arc::new(RwLock::new(index));
    let metrics = Arc::new(Metrics::new());

    let (cluster, router) = if config.cluster.enabled {
        let cs = ClusterState::new(&node_id, &config.cluster, &forward_addr, metrics.clone())
            .await
            .context("cluster init failed")?;
        tracing::info!(
            gossip_addr = %cs.gossip_addr(),
            forward_addr = %forward_addr,
            seeds = ?config.cluster.seed_nodes,
            replication_factor = config.cluster.replication_factor,
            tls = tls_bundle.is_some(),
            dead_node_removal_enabled = config.cluster.dead_node_removal_enabled,
            "cluster enabled"
        );
        let router = ClusterRouter::new(config.auth_token.clone(), tls_bundle.as_ref())
            .context("cluster router build")?;
        (Some(Arc::new(cs)), Some(Arc::new(router)))
    } else {
        tracing::info!("cluster disabled — running in single-node mode");
        (None, None)
    };

    let addr = format!("0.0.0.0:{}", config.port);

    // Spawn the group-commit flush task. The task takes ownership of the
    // sole `Wal` and a clone of the index/metrics; from here on out, no
    // other code path writes to the WAL directly.
    let group_commit_config = GroupCommitConfig {
        batch_size: config.wal_batch_size.max(1),
        batch_timeout: Duration::from_millis(config.wal_batch_timeout_ms),
        channel_capacity: DEFAULT_CHANNEL_CAPACITY,
    };
    tracing::info!(
        batch_size = group_commit_config.batch_size,
        batch_timeout_ms = config.wal_batch_timeout_ms,
        "WAL group-commit configured"
    );
    let wal_handle = GroupCommitWal::spawn(
        wal,
        std::path::PathBuf::from(&config.wal_path),
        snapshot_path.clone(),
        index_arc.clone(),
        metrics.clone(),
        config.compact_interval_inserts,
        group_commit_config,
    );

    let state = Arc::new(AppState::new(
        node_id,
        index_arc,
        wal_handle,
        config.wal_path.clone(),
        snapshot_path,
        config.hnsw.clone(),
        cluster,
        router,
        config.cluster.replication_factor.max(1),
        metrics,
        config.auth_token.clone(),
        config.cluster.max_replication_retries,
        config.cluster.read_repair_enabled,
    ));
    let app = server::build_router(state.clone());

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("failed to bind {addr}"))?;

    tracing::info!(node_id = %state.node_id, %addr, "ferrocache listening");

    // When TLS is enabled, run a second listener on the internal port that
    // serves the same router under mTLS. The two listeners share the
    // AppState by Arc, so /metrics + /stats reflect both surfaces.
    if let Some(bundle) = tls_bundle {
        let server_config = tls::build_server_config(&bundle).context("TLS server config")?;
        let rustls_config =
            axum_server::tls_rustls::RustlsConfig::from_config(Arc::new(server_config));
        let internal_bind = format!("0.0.0.0:{internal_port}");
        let internal_socket: std::net::SocketAddr = internal_bind
            .parse()
            .with_context(|| format!("invalid internal bind addr {internal_bind}"))?;
        tracing::info!(addr = %internal_bind, "ferrocache TLS listening (cluster)");
        let tls_app = app.clone();
        let tls_handle = tokio::spawn(async move {
            axum_server::bind_rustls(internal_socket, rustls_config)
                .serve(tls_app.into_make_service())
                .await
        });
        let plain = async {
            axum::serve(listener, app)
                .await
                .context("axum server failed")
        };
        tokio::select! {
            r = plain => { r?; }
            r = tls_handle => {
                r.context("TLS server task join")?
                    .context("TLS server failed")?;
            }
        }
    } else {
        axum::serve(listener, app)
            .await
            .context("axum server failed")?;
    }

    Ok(())
}
