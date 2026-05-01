use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRequest {
    pub embedding: Vec<f32>,
    pub threshold: f32,
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
}

#[derive(Debug, Serialize)]
pub struct StatsHnsw {
    pub max_nb_connection: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub dimension: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct ClusterStatusResponse {
    pub mode: &'static str,
    pub self_node_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gossip_addr: Option<String>,
    pub nodes: Vec<String>,
    pub node_count: usize,
}
