use std::sync::Arc;

use anyhow::Context;
use tracing_subscriber::EnvFilter;

mod cluster;
mod config;
mod index;
mod models;
mod ring;
mod router;
mod server;
mod state;
mod wal;

use crate::cluster::ClusterState;
use crate::config::FerrocacheConfig;
use crate::index::SemanticIndex;
use crate::router::ClusterRouter;
use crate::state::AppState;
use crate::wal::Wal;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let config = FerrocacheConfig::load().context("config load failed")?;
    tracing::info!(?config, "loaded config");

    let node_id = config
        .node_id
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    tracing::info!(path = %config.wal_path, "WAL path");
    let entries = Wal::replay(&config.wal_path)
        .await
        .context("WAL replay failed")?;
    let mut index = SemanticIndex::new(&config.hnsw);
    let mut replayed = 0usize;
    for entry in entries {
        match index.replay_entry(entry) {
            Ok(()) => replayed += 1,
            Err(e) => tracing::warn!(error = %e, "WAL replay entry rejected"),
        }
    }
    tracing::info!(count = replayed, "WAL replay complete");

    let wal = Wal::open(&config.wal_path)
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
        (Some(Arc::new(cs)), Some(Arc::new(ClusterRouter::new())))
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
        config.hnsw.clone(),
        cluster,
        router,
        config.cluster.replication_factor.max(1),
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
