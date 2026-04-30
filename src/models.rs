use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct QueryRequest {
    pub embedding: Vec<f32>,
    pub threshold: f32,
}

#[derive(Debug, Serialize)]
pub struct QueryResponse {
    pub hit: bool,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct InsertRequest {
    pub embedding: Vec<f32>,
    pub response: String,
    pub query_text: String,
}

#[derive(Debug, Serialize)]
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
