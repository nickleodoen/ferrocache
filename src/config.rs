use anyhow::{Context, Result};
use config::{Config, Environment, File};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FerrocacheConfig {
    pub port: u16,
    #[serde(default)]
    pub node_id: Option<String>,
    pub wal_path: String,
    pub hnsw: HnswConfig,
    #[serde(default)]
    pub cluster: ClusterConfig,
    #[serde(default = "default_compact_interval_inserts")]
    pub compact_interval_inserts: u64,
    /// When set, all data routes require `Authorization: Bearer <token>`.
    /// Unset/empty disables auth entirely (default). Configure via env
    /// `FERROCACHE_AUTH_TOKEN`. NEVER log this value.
    #[serde(default)]
    pub auth_token: Option<String>,
    /// Group-commit batch size. Inserts are coalesced up to this many per
    /// fsync. Set to `1` for per-insert fsync (the pre-M20 behavior).
    #[serde(default = "default_wal_batch_size")]
    pub wal_batch_size: usize,
    /// Max time the flush task waits for additional inserts to join the
    /// current batch after the first one arrives.
    #[serde(default = "default_wal_batch_timeout_ms")]
    pub wal_batch_timeout_ms: u64,
    /// Seconds between TTL reaper scans (M26). The reaper sweeps every
    /// namespace under the index write lock, evicts expired entries, and
    /// writes tombstones. Default 60s; set to 0 to disable.
    #[serde(default = "default_expire_scan_interval_secs")]
    pub expire_scan_interval_secs: u64,
}

fn default_compact_interval_inserts() -> u64 {
    10_000
}

fn default_wal_batch_size() -> usize {
    256
}

fn default_wal_batch_timeout_ms() -> u64 {
    1
}

fn default_expire_scan_interval_secs() -> u64 {
    60
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ClusterConfig {
    pub enabled: bool,
    pub gossip_addr: String,
    pub api_addr: String,
    pub seed_nodes: Vec<String>,
    pub virtual_nodes: usize,
    pub replication_factor: usize,
    #[serde(default)]
    pub tls: ClusterTlsConfig,
    #[serde(default = "default_max_replication_retries")]
    pub max_replication_retries: usize,
    /// Phi accrual failure detector threshold. Above this, a peer flips to
    /// `Suspected`; above 2× this, to `Dead`. Cassandra/Akka default is 8.0.
    #[serde(default = "default_phi_threshold")]
    pub phi_threshold: f64,
    /// Sliding-window length for heartbeat inter-arrival samples.
    #[serde(default = "default_phi_window_size")]
    pub phi_window_size: usize,
    /// Floor on the std-dev of the inter-arrival window to keep the CDF
    /// well-defined (and absorb tiny network jitter as "still normal").
    #[serde(default = "default_phi_min_std_dev_ms")]
    pub phi_min_std_dev_ms: f64,
    /// When `true` (default), peers the failure detector marks `Dead` are
    /// automatically removed from the hash ring (M22). Set `false` for a
    /// monitoring-only mode where phi values are reported via metrics and
    /// `/cluster/status` but ring membership stays static.
    #[serde(default = "default_dead_node_removal_enabled")]
    pub dead_node_removal_enabled: bool,
    /// Read repair (M23): on a query miss, the coordinator queries
    /// non-dead replicas in parallel. If any reports a hit, the result is
    /// returned to the client immediately and the entry is fetched +
    /// re-inserted locally in a fire-and-forget background task. Set
    /// `false` to disable replica fan-out on miss entirely.
    #[serde(default = "default_read_repair_enabled")]
    pub read_repair_enabled: bool,
}

fn default_max_replication_retries() -> usize {
    3
}

fn default_phi_threshold() -> f64 {
    crate::failure_detector::DEFAULT_PHI_THRESHOLD
}

fn default_phi_window_size() -> usize {
    crate::failure_detector::DEFAULT_WINDOW_SIZE
}

fn default_phi_min_std_dev_ms() -> f64 {
    crate::failure_detector::DEFAULT_MIN_STD_DEV_MS
}

fn default_dead_node_removal_enabled() -> bool {
    true
}

fn default_read_repair_enabled() -> bool {
    true
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            gossip_addr: "0.0.0.0:4000".to_string(),
            api_addr: "0.0.0.0:3000".to_string(),
            seed_nodes: Vec::new(),
            virtual_nodes: 64,
            replication_factor: 2,
            tls: ClusterTlsConfig::default(),
            max_replication_retries: default_max_replication_retries(),
            phi_threshold: default_phi_threshold(),
            phi_window_size: default_phi_window_size(),
            phi_min_std_dev_ms: default_phi_min_std_dev_ms(),
            dead_node_removal_enabled: default_dead_node_removal_enabled(),
            read_repair_enabled: default_read_repair_enabled(),
        }
    }
}

/// mTLS for inter-node cluster traffic. Off by default; behavior is
/// identical to pre-M18 when `enabled = false`.
///
/// When `enabled = true` and all three `*_path` fields are set, certs are
/// loaded from disk (production mode). When any path is missing, a fresh CA
/// + leaf cert pair is generated in memory and a warning is logged — useful
///   for local smoke tests, never appropriate for production (every node
///   would synthesize its own CA and they wouldn't trust each other).
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ClusterTlsConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub ca_cert_path: Option<String>,
    #[serde(default)]
    pub node_cert_path: Option<String>,
    #[serde(default)]
    pub node_key_path: Option<String>,
    /// TLS listener port. When `None`, derived as `port + 1000` at startup.
    #[serde(default)]
    pub internal_port: Option<u16>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HnswConfig {
    pub max_nb_connection: usize,
    pub max_elements: usize,
    pub max_layer: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub default_threshold: f32,
    /// LRU eviction cap per namespace (M25). When `None`, the namespace is
    /// unlimited (pre-M25 behavior). When `Some(N)`, the flush task evicts
    /// the least-recently-accessed entries after each insert batch until
    /// the namespace holds ≤ N live entries.
    #[serde(default)]
    pub max_entries_per_namespace: Option<usize>,
}

impl Default for FerrocacheConfig {
    fn default() -> Self {
        Self {
            port: 3000,
            node_id: None,
            wal_path: "./ferrocache.wal".to_string(),
            hnsw: HnswConfig::default(),
            cluster: ClusterConfig::default(),
            compact_interval_inserts: default_compact_interval_inserts(),
            auth_token: None,
            wal_batch_size: default_wal_batch_size(),
            wal_batch_timeout_ms: default_wal_batch_timeout_ms(),
            expire_scan_interval_secs: default_expire_scan_interval_secs(),
        }
    }
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            max_nb_connection: 16,
            max_elements: 100_000,
            max_layer: 16,
            ef_construction: 200,
            ef_search: 32,
            default_threshold: 0.92,
            max_entries_per_namespace: None,
        }
    }
}

impl FerrocacheConfig {
    pub fn load() -> Result<Self> {
        let defaults = FerrocacheConfig::default();
        let cfg = Config::builder()
            .set_default("port", defaults.port as i64)?
            .set_default("wal_path", defaults.wal_path.clone())?
            .set_default(
                "hnsw.max_nb_connection",
                defaults.hnsw.max_nb_connection as i64,
            )?
            .set_default("hnsw.max_elements", defaults.hnsw.max_elements as i64)?
            .set_default("hnsw.max_layer", defaults.hnsw.max_layer as i64)?
            .set_default("hnsw.ef_construction", defaults.hnsw.ef_construction as i64)?
            .set_default("hnsw.ef_search", defaults.hnsw.ef_search as i64)?
            .set_default(
                "hnsw.default_threshold",
                defaults.hnsw.default_threshold as f64,
            )?
            .set_default("cluster.enabled", defaults.cluster.enabled)?
            .set_default("cluster.gossip_addr", defaults.cluster.gossip_addr.clone())?
            .set_default("cluster.api_addr", defaults.cluster.api_addr.clone())?
            .set_default("cluster.seed_nodes", Vec::<String>::new())?
            .set_default(
                "cluster.virtual_nodes",
                defaults.cluster.virtual_nodes as i64,
            )?
            .set_default(
                "cluster.replication_factor",
                defaults.cluster.replication_factor as i64,
            )?
            .set_default(
                "cluster.max_replication_retries",
                defaults.cluster.max_replication_retries as i64,
            )?
            .set_default("cluster.phi_threshold", defaults.cluster.phi_threshold)?
            .set_default(
                "cluster.phi_window_size",
                defaults.cluster.phi_window_size as i64,
            )?
            .set_default(
                "cluster.phi_min_std_dev_ms",
                defaults.cluster.phi_min_std_dev_ms,
            )?
            .set_default(
                "cluster.dead_node_removal_enabled",
                defaults.cluster.dead_node_removal_enabled,
            )?
            .set_default(
                "cluster.read_repair_enabled",
                defaults.cluster.read_repair_enabled,
            )?
            .set_default(
                "compact_interval_inserts",
                defaults.compact_interval_inserts as i64,
            )?
            .set_default("wal_batch_size", defaults.wal_batch_size as i64)?
            .set_default("wal_batch_timeout_ms", defaults.wal_batch_timeout_ms as i64)?
            .set_default(
                "expire_scan_interval_secs",
                defaults.expire_scan_interval_secs as i64,
            )?
            .add_source(File::with_name("ferrocache").required(false))
            .add_source(
                // FERROCACHE_PORT → port; FERROCACHE_CLUSTER__ENABLED → cluster.enabled.
                // try_parsing must be true for bool/int/list coercion from env strings.
                Environment::with_prefix("FERROCACHE")
                    .prefix_separator("_")
                    .separator("__")
                    .try_parsing(true)
                    .list_separator(",")
                    .with_list_parse_key("cluster.seed_nodes"),
            )
            .build()
            .context("failed to build config")?;
        cfg.try_deserialize()
            .context("failed to deserialize config")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let c = FerrocacheConfig::default();
        assert_eq!(c.port, 3000);
        assert_eq!(c.wal_path, "./ferrocache.wal");
        assert!(c.node_id.is_none());
        assert_eq!(c.hnsw.max_nb_connection, 16);
        assert_eq!(c.hnsw.max_elements, 100_000);
        assert_eq!(c.hnsw.max_layer, 16);
        assert_eq!(c.hnsw.ef_construction, 200);
        assert_eq!(c.hnsw.ef_search, 32);
        assert!((c.hnsw.default_threshold - 0.92).abs() < 1e-6);
        assert!(c.hnsw.max_entries_per_namespace.is_none());
        assert!(!c.cluster.enabled);
        assert_eq!(c.cluster.gossip_addr, "0.0.0.0:4000");
        assert_eq!(c.cluster.api_addr, "0.0.0.0:3000");
        assert!(c.cluster.seed_nodes.is_empty());
        assert_eq!(c.cluster.virtual_nodes, 64);
        assert_eq!(c.cluster.replication_factor, 2);
        assert_eq!(c.compact_interval_inserts, 10_000);
        assert!(c.auth_token.is_none());
        assert!(!c.cluster.tls.enabled);
        assert!(c.cluster.tls.ca_cert_path.is_none());
        assert!(c.cluster.tls.node_cert_path.is_none());
        assert!(c.cluster.tls.node_key_path.is_none());
        assert!(c.cluster.tls.internal_port.is_none());
        assert_eq!(c.cluster.max_replication_retries, 3);
        assert!((c.cluster.phi_threshold - 8.0).abs() < 1e-9);
        assert_eq!(c.cluster.phi_window_size, 100);
        assert!((c.cluster.phi_min_std_dev_ms - 100.0).abs() < 1e-9);
        assert!(c.cluster.dead_node_removal_enabled);
        assert!(c.cluster.read_repair_enabled);
        assert_eq!(c.wal_batch_size, 256);
        assert_eq!(c.wal_batch_timeout_ms, 1);
        assert_eq!(c.expire_scan_interval_secs, 60);
    }
}
