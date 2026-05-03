use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRequest {
    pub embedding: Vec<f32>,
    pub threshold: f32,
    /// Required since M14 — queries only search the namespace matching this
    /// `model_id`. Modeled as `Option` so missing values produce a 400 error
    /// rather than a serde rejection.
    #[serde(default)]
    pub model_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResponse {
    pub hit: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub similarity: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsertRequest {
    pub embedding: Vec<f32>,
    pub response: String,
    pub query_text: String,
    /// Required since M14 — namespace partition for the entry.
    #[serde(default)]
    pub model_id: Option<String>,
    /// When set on a `local=true` request, the receiving node uses this UUID
    /// instead of generating a new one. Used by the coordinator so all
    /// replicas store the same id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uuid: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsertResponse {
    pub id: String,
    pub status: String,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub node_id: String,
    pub entry_count: u64,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

#[derive(Debug, Serialize)]
pub struct StatsResponse {
    pub entry_count: u64,
    pub wal_path: String,
    pub hnsw: StatsHnsw,
    pub namespaces: HashMap<String, NamespaceStatsEntry>,
    pub counters: CountersResponse,
}

#[derive(Debug, Serialize)]
pub struct CountersResponse {
    pub queries_total: u64,
    pub queries_hit: u64,
    pub queries_miss: u64,
    pub hit_rate: f64,
    pub inserts_total: u64,
    pub replication_forwards: u64,
    pub replication_failures: u64,
    pub replication_retries: u64,
    pub compactions: u64,
}

#[derive(Debug, Serialize)]
pub struct StatsHnsw {
    pub max_nb_connection: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub dimension: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct NamespaceStatsEntry {
    pub entry_count: usize,
    pub dimension: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct CompactResponse {
    pub status: String,
    pub entries_snapshotted: usize,
    pub wal_sequence: u64,
}

#[derive(Debug, Serialize)]
pub struct ClusterStatusResponse {
    pub mode: &'static str,
    pub self_node_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gossip_addr: Option<String>,
    pub nodes: Vec<String>,
    pub node_count: usize,
    /// Phi accrual failure detector state (M21). Empty when cluster is
    /// disabled or no peers have been observed yet.
    #[serde(skip_serializing_if = "HashMap::is_empty", default)]
    pub peer_health: HashMap<String, PeerHealth>,
    /// Nodes the failure detector confirmed dead and the reconciler removed
    /// from the ring (M22). They re-enter `nodes` automatically on re-join.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub dead_nodes: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct PeerHealth {
    pub status: crate::failure_detector::PeerStatus,
    pub phi: f64,
}
