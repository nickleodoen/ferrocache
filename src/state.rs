use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use tokio::sync::{Mutex, RwLock};

use crate::cluster::ClusterState;
use crate::config::HnswConfig;
use crate::index::SemanticIndex;
use crate::metrics::Metrics;
use crate::router::ClusterRouter;
use crate::wal::Wal;

pub struct AppState {
    pub node_id: String,
    pub index: Arc<RwLock<SemanticIndex>>,
    pub wal: Arc<Mutex<Wal>>,
    pub wal_path: String,
    pub snapshot_path: PathBuf,
    pub hnsw_config: HnswConfig,
    pub cluster: Option<Arc<ClusterState>>,
    pub router: Option<Arc<ClusterRouter>>,
    pub replication_factor: usize,
    pub compact_interval_inserts: u64,
    pub inserts_since_compact: AtomicU64,
    pub metrics: Arc<Metrics>,
}

impl AppState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        node_id: String,
        index: SemanticIndex,
        wal: Wal,
        wal_path: String,
        snapshot_path: PathBuf,
        hnsw_config: HnswConfig,
        cluster: Option<Arc<ClusterState>>,
        router: Option<Arc<ClusterRouter>>,
        replication_factor: usize,
        compact_interval_inserts: u64,
    ) -> Self {
        Self {
            node_id,
            index: Arc::new(RwLock::new(index)),
            wal: Arc::new(Mutex::new(wal)),
            wal_path,
            snapshot_path,
            hnsw_config,
            cluster,
            router,
            replication_factor,
            compact_interval_inserts,
            inserts_since_compact: AtomicU64::new(0),
            metrics: Arc::new(Metrics::new()),
        }
    }
}
