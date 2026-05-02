use std::sync::Arc;

use anyhow::Context;
use tracing_subscriber::EnvFilter;

use ferrocache::cluster::ClusterState;
use ferrocache::config::FerrocacheConfig;
use ferrocache::index::SemanticIndex;
use ferrocache::router::ClusterRouter;
use ferrocache::server;
use ferrocache::snapshot;
use ferrocache::state::AppState;
use ferrocache::wal::Wal;

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

    let (cluster, router) = if config.cluster.enabled {
        let cs = ClusterState::new(&node_id, &config.cluster)
            .await
            .context("cluster init failed")?;
        tracing::info!(
            gossip_addr = %cs.gossip_addr(),
            api_addr = %config.cluster.api_addr,
            seeds = ?config.cluster.seed_nodes,
            replication_factor = config.cluster.replication_factor,
            "cluster enabled"
        );
        (
            Some(Arc::new(cs)),
            Some(Arc::new(ClusterRouter::new(config.auth_token.clone()))),
        )
    } else {
        tracing::info!("cluster disabled — running in single-node mode");
        (None, None)
    };

    let addr = format!("0.0.0.0:{}", config.port);
    let state = Arc::new(AppState::new(
        node_id,
        index,
        wal,
        config.wal_path.clone(),
        snapshot_path,
        config.hnsw.clone(),
        cluster,
        router,
        config.cluster.replication_factor.max(1),
        config.compact_interval_inserts,
        config.auth_token.clone(),
    ));
    let app = server::build_router(state.clone());

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("failed to bind {addr}"))?;

    tracing::info!(node_id = %state.node_id, %addr, "ferrocache listening");

    axum::serve(listener, app)
        .await
        .context("axum server failed")?;

    Ok(())
}
