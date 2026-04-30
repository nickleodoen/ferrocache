use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use tracing_subscriber::EnvFilter;

mod index;
mod models;
mod server;
mod state;
mod wal;

use crate::index::SemanticIndex;
use crate::state::AppState;
use crate::wal::Wal;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .init();

    let port: u16 = std::env::var("FERROCACHE_PORT")
        .ok()
        .map(|s| s.parse())
        .transpose()
        .context("FERROCACHE_PORT must be a valid u16")?
        .unwrap_or(3000);

    let wal_path: PathBuf = std::env::var("FERROCACHE_WAL_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("./ferrocache.wal"));
    tracing::info!(path = %wal_path.display(), "WAL path");

    let entries = Wal::replay(&wal_path).await.context("WAL replay failed")?;
    let mut index = SemanticIndex::new();
    let mut replayed = 0usize;
    for entry in entries {
        match index.replay_entry(entry) {
            Ok(()) => replayed += 1,
            Err(e) => tracing::warn!(error = %e, "WAL replay entry rejected"),
        }
    }
    tracing::info!(count = replayed, "WAL replay complete");

    let wal = Wal::open(&wal_path).await.context("WAL open failed")?;

    let addr = format!("0.0.0.0:{port}");
    let state = Arc::new(AppState::new(index, wal));
    let app = server::build_router(state.clone());

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("failed to bind {addr}"))?;

    tracing::info!(node_id = %state.node_id, %addr, "ferrocache listening");

    axum::serve(listener, app).await.context("axum server failed")?;

    Ok(())
}
