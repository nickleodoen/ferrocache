use std::sync::Arc;

use anyhow::Context;
use tracing_subscriber::EnvFilter;

mod config;
mod index;
mod models;
mod server;
mod state;
mod wal;

use crate::config::FerrocacheConfig;
use crate::index::SemanticIndex;
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

    let addr = format!("0.0.0.0:{}", config.port);
    let state = Arc::new(AppState::new(
        node_id,
        index,
        wal,
        config.wal_path.clone(),
        config.hnsw.clone(),
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
