use std::sync::Arc;

use tokio::sync::RwLock;

use crate::index::SemanticIndex;

pub struct AppState {
    pub node_id: String,
    pub index: Arc<RwLock<SemanticIndex>>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            node_id: uuid::Uuid::new_v4().to_string(),
            index: Arc::new(RwLock::new(SemanticIndex::new())),
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}
