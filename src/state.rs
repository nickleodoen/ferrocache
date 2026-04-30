use std::sync::Arc;

use tokio::sync::{Mutex, RwLock};

use crate::index::SemanticIndex;
use crate::wal::Wal;

pub struct AppState {
    pub node_id: String,
    pub index: Arc<RwLock<SemanticIndex>>,
    pub wal: Arc<Mutex<Wal>>,
}

impl AppState {
    pub fn new(index: SemanticIndex, wal: Wal) -> Self {
        Self {
            node_id: uuid::Uuid::new_v4().to_string(),
            index: Arc::new(RwLock::new(index)),
            wal: Arc::new(Mutex::new(wal)),
        }
    }
}
