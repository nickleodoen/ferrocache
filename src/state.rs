use std::sync::Arc;

use tokio::sync::{Mutex, RwLock};

use crate::config::HnswConfig;
use crate::index::SemanticIndex;
use crate::wal::Wal;

pub struct AppState {
    pub node_id: String,
    pub index: Arc<RwLock<SemanticIndex>>,
    pub wal: Arc<Mutex<Wal>>,
    pub wal_path: String,
    pub hnsw_config: HnswConfig,
}

impl AppState {
    pub fn new(
        node_id: String,
        index: SemanticIndex,
        wal: Wal,
        wal_path: String,
        hnsw_config: HnswConfig,
    ) -> Self {
        Self {
            node_id,
            index: Arc::new(RwLock::new(index)),
            wal: Arc::new(Mutex::new(wal)),
            wal_path,
            hnsw_config,
        }
    }
}
