use std::sync::Arc;

use tokio::sync::{Mutex, RwLock};

use crate::cluster::ClusterState;
use crate::config::HnswConfig;
use crate::index::SemanticIndex;
use crate::router::ClusterRouter;
use crate::wal::Wal;

pub struct AppState {
    pub node_id: String,
    pub index: Arc<RwLock<SemanticIndex>>,
    pub wal: Arc<Mutex<Wal>>,
    pub wal_path: String,
    pub hnsw_config: HnswConfig,
    pub cluster: Option<Arc<ClusterState>>,
    pub router: Option<Arc<ClusterRouter>>,
    pub replication_factor: usize,
}

impl AppState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        node_id: String,
        index: SemanticIndex,
        wal: Wal,
        wal_path: String,
        hnsw_config: HnswConfig,
        cluster: Option<Arc<ClusterState>>,
        router: Option<Arc<ClusterRouter>>,
        replication_factor: usize,
    ) -> Self {
        Self {
            node_id,
            index: Arc::new(RwLock::new(index)),
            wal: Arc::new(Mutex::new(wal)),
            wal_path,
            hnsw_config,
            cluster,
            router,
            replication_factor,
        }
    }
}
