use std::collections::HashMap;

use anyhow::{Result, anyhow};
use hnsw_rs::prelude::*;

use crate::config::HnswConfig;
use crate::wal::WalEntry;

pub struct CacheEntry {
    pub uuid: String,
    pub response: String,
}

#[derive(Debug)]
pub struct QueryHit {
    pub id: String,
    pub response: String,
    pub similarity: f32,
}

pub struct SemanticIndex {
    hnsw: Hnsw<'static, f32, DistCosine>,
    entries: HashMap<usize, CacheEntry>,
    next_id: usize,
    dimension: Option<usize>,
    ef_search: usize,
}

impl SemanticIndex {
    pub fn new(cfg: &HnswConfig) -> Self {
        let hnsw = Hnsw::<f32, DistCosine>::new(
            cfg.max_nb_connection,
            cfg.max_elements,
            cfg.max_layer,
            cfg.ef_construction,
            DistCosine,
        );
        Self {
            hnsw,
            entries: HashMap::new(),
            next_id: 0,
            dimension: None,
            ef_search: cfg.ef_search,
        }
    }

    #[cfg(test)]
    pub fn insert(
        &mut self,
        embedding: Vec<f32>,
        response: String,
        query_text: String,
    ) -> Result<String> {
        let uuid = uuid::Uuid::new_v4().to_string();
        self.insert_with_uuid(uuid.clone(), embedding, response, query_text)?;
        Ok(uuid)
    }

    pub fn replay_entry(&mut self, entry: WalEntry) -> Result<()> {
        self.insert_with_uuid(
            entry.uuid,
            entry.embedding,
            entry.response,
            entry.query_text,
        )
    }

    fn insert_with_uuid(
        &mut self,
        uuid: String,
        embedding: Vec<f32>,
        response: String,
        _query_text: String,
    ) -> Result<()> {
        match self.dimension {
            None => self.dimension = Some(embedding.len()),
            Some(d) if d != embedding.len() => {
                return Err(anyhow!(
                    "dimension mismatch: expected {}, got {}",
                    d,
                    embedding.len()
                ));
            }
            _ => {}
        }

        let id = self.next_id;
        self.next_id += 1;

        self.hnsw.insert((&embedding, id));

        self.entries.insert(id, CacheEntry { uuid, response });

        Ok(())
    }

    pub fn query(&self, embedding: &[f32], threshold: f32) -> Result<Option<QueryHit>> {
        if let Some(d) = self.dimension {
            if d != embedding.len() {
                return Err(anyhow!(
                    "dimension mismatch: expected {}, got {}",
                    d,
                    embedding.len()
                ));
            }
        } else {
            return Ok(None);
        }

        let neighbours = self.hnsw.search(embedding, 1, self.ef_search);
        let Some(n) = neighbours.first() else {
            return Ok(None);
        };

        let similarity = 1.0 - n.get_distance();
        if similarity < threshold {
            return Ok(None);
        }

        let entry = self
            .entries
            .get(&n.get_origin_id())
            .ok_or_else(|| anyhow!("neighbour id {} not in side-table", n.get_origin_id()))?;

        Ok(Some(QueryHit {
            id: entry.uuid.clone(),
            response: entry.response.clone(),
            similarity,
        }))
    }

    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    pub fn dimension(&self) -> Option<usize> {
        self.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_index() -> SemanticIndex {
        SemanticIndex::new(&HnswConfig::default())
    }

    #[test]
    fn test_insert_and_query_hit() {
        let mut idx = test_index();
        let v = vec![1.0_f32, 0.0, 0.0];
        let uuid = idx
            .insert(v.clone(), "cached-response".to_string(), "q".to_string())
            .unwrap();
        assert!(!uuid.is_empty());

        let hit = idx.query(&v, 0.90).unwrap().expect("should hit");
        assert_eq!(hit.response, "cached-response");
        assert!(hit.similarity > 0.999, "similarity={}", hit.similarity);
        assert_eq!(hit.id, uuid);
    }

    #[test]
    fn test_query_miss_below_threshold() {
        let mut idx = test_index();
        idx.insert(vec![1.0_f32, 0.0, 0.0], "r".to_string(), "q".to_string())
            .unwrap();
        let result = idx.query(&[0.0_f32, 0.0, 1.0], 0.99).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_dimension_mismatch_insert() {
        let mut idx = test_index();
        idx.insert(vec![1.0, 2.0, 3.0], "r".into(), "q".into())
            .unwrap();
        let err = idx
            .insert(vec![1.0, 2.0, 3.0, 4.0, 5.0], "r".into(), "q".into())
            .unwrap_err();
        assert!(err.to_string().contains("dimension mismatch"));
    }

    #[test]
    fn test_dimension_mismatch_query() {
        let mut idx = test_index();
        idx.insert(vec![1.0, 2.0, 3.0], "r".into(), "q".into())
            .unwrap();
        let err = idx.query(&[1.0, 2.0, 3.0, 4.0, 5.0], 0.5).unwrap_err();
        assert!(err.to_string().contains("dimension mismatch"));
    }

    #[test]
    fn test_entry_count() {
        let mut idx = test_index();
        idx.insert(vec![1.0, 0.0, 0.0], "a".into(), "qa".into())
            .unwrap();
        idx.insert(vec![0.0, 1.0, 0.0], "b".into(), "qb".into())
            .unwrap();
        idx.insert(vec![0.0, 0.0, 1.0], "c".into(), "qc".into())
            .unwrap();
        assert_eq!(idx.entry_count(), 3);
    }
}
