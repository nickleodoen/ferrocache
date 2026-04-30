use std::sync::Arc;
use std::sync::atomic::AtomicU64;

pub struct AppState {
    pub node_id: String,
    pub entry_count: Arc<AtomicU64>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            node_id: uuid::Uuid::new_v4().to_string(),
            entry_count: Arc::new(AtomicU64::new(0)),
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}
