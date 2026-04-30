use std::sync::Arc;

use anyhow::Context;
use tracing_subscriber::EnvFilter;

mod models;
mod server;
mod state;

use crate::state::AppState;

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

    let addr = format!("0.0.0.0:{port}");
    let state = Arc::new(AppState::new());
    let app = server::build_router(state.clone());

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("failed to bind {addr}"))?;

    tracing::info!(node_id = %state.node_id, %addr, "ferrocache listening");

    axum::serve(listener, app).await.context("axum server failed")?;

    Ok(())
}
