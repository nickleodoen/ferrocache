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
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ClusterConfig {
    pub enabled: bool,
    pub gossip_addr: String,
    pub api_addr: String,
    pub seed_nodes: Vec<String>,
    pub virtual_nodes: usize,
    pub replication_factor: usize,
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
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HnswConfig {
    pub max_nb_connection: usize,
    pub max_elements: usize,
    pub max_layer: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub default_threshold: f32,
}

impl Default for FerrocacheConfig {
    fn default() -> Self {
        Self {
            port: 3000,
            node_id: None,
            wal_path: "./ferrocache.wal".to_string(),
            hnsw: HnswConfig::default(),
            cluster: ClusterConfig::default(),
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
            .add_source(File::with_name("ferrocache").required(false))
            .add_source(Environment::with_prefix("FERROCACHE").separator("__"))
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
        assert!(!c.cluster.enabled);
        assert_eq!(c.cluster.gossip_addr, "0.0.0.0:4000");
        assert_eq!(c.cluster.api_addr, "0.0.0.0:3000");
        assert!(c.cluster.seed_nodes.is_empty());
        assert_eq!(c.cluster.virtual_nodes, 64);
        assert_eq!(c.cluster.replication_factor, 2);
    }
}
