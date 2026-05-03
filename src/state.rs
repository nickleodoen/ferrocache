use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::cluster::ClusterState;
use crate::config::HnswConfig;
use crate::index::SemanticIndex;
use crate::metrics::Metrics;
use crate::router::ClusterRouter;
use crate::wal::GroupCommitWal;

pub struct AppState {
    pub node_id: String,
    pub index: Arc<RwLock<SemanticIndex>>,
    /// Producer handle to the group-commit flush task. The flush task owns
    /// the only `Wal` and is the sole writer; handlers send commands here
    /// and await a oneshot reply.
    pub wal: GroupCommitWal,
    pub wal_path: String,
    pub snapshot_path: PathBuf,
    pub hnsw_config: HnswConfig,
    pub cluster: Option<Arc<ClusterState>>,
    pub router: Option<Arc<ClusterRouter>>,
    pub replication_factor: usize,
    pub metrics: Arc<Metrics>,
    /// Bearer token for the `/query`, `/insert`, `/stats`, `/cluster/status`,
    /// and `/admin/compact` routes. `None` disables auth entirely.
    pub auth_token: Option<String>,
    /// Max number of retries (after the initial attempt) for replication
    /// forwards. Total attempts = `max_replication_retries + 1`.
    pub max_replication_retries: usize,
    /// Read repair (M23): on a query miss, fan out to non-dead replicas
    /// and return the first hit while spawning an async repair task.
    /// `false` makes a coordinator miss return immediately without
    /// touching replicas.
    pub read_repair_enabled: bool,
}

impl AppState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        node_id: String,
        index: Arc<RwLock<SemanticIndex>>,
        wal: GroupCommitWal,
        wal_path: String,
        snapshot_path: PathBuf,
        hnsw_config: HnswConfig,
        cluster: Option<Arc<ClusterState>>,
        router: Option<Arc<ClusterRouter>>,
        replication_factor: usize,
        metrics: Arc<Metrics>,
        auth_token: Option<String>,
        max_replication_retries: usize,
        read_repair_enabled: bool,
    ) -> Self {
        Self {
            node_id,
            index,
            wal,
            wal_path,
            snapshot_path,
            hnsw_config,
            cluster,
            router,
            replication_factor,
            metrics,
            auth_token,
            max_replication_retries,
            read_repair_enabled,
        }
    }
}
