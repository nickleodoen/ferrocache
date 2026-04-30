use std::collections::HashMap;

use anyhow::{Result, anyhow};
use hnsw_rs::prelude::*;

#[allow(dead_code)]
pub struct CacheEntry {
    pub uuid: String,
    pub embedding: Vec<f32>,
    pub response: String,
    pub query_text: String,
}

#[derive(Debug)]
pub struct QueryHit {
    #[allow(dead_code)]
    pub id: String,
    pub response: String,
    pub similarity: f32,
}

pub struct SemanticIndex {
    hnsw: Hnsw<'static, f32, DistCosine>,
    entries: HashMap<usize, CacheEntry>,
    next_id: usize,
    dimension: Option<usize>,
}

const HNSW_MAX_NB_CONNECTION: usize = 16;
const HNSW_MAX_ELEMENTS: usize = 100_000;
const HNSW_MAX_LAYER: usize = 16;
const HNSW_EF_CONSTRUCTION: usize = 200;
const HNSW_EF_SEARCH: usize = 32;

impl SemanticIndex {
    pub fn new() -> Self {
        let hnsw = Hnsw::<f32, DistCosine>::new(
            HNSW_MAX_NB_CONNECTION,
            HNSW_MAX_ELEMENTS,
            HNSW_MAX_LAYER,
            HNSW_EF_CONSTRUCTION,
            DistCosine,
        );
        Self {
            hnsw,
            entries: HashMap::new(),
            next_id: 0,
            dimension: None,
        }
    }

    pub fn insert(
        &mut self,
        embedding: Vec<f32>,
        response: String,
        query_text: String,
    ) -> Result<String> {
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
        let uuid = uuid::Uuid::new_v4().to_string();

        self.hnsw.insert((&embedding, id));

        self.entries.insert(
            id,
            CacheEntry {
                uuid: uuid.clone(),
                embedding,
                response,
                query_text,
            },
        );

        Ok(uuid)
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

        let neighbours = self.hnsw.search(embedding, 1, HNSW_EF_SEARCH);
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
}

impl Default for SemanticIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_query_hit() {
        let mut idx = SemanticIndex::new();
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
        let mut idx = SemanticIndex::new();
        idx.insert(
            vec![1.0_f32, 0.0, 0.0],
            "r".to_string(),
            "q".to_string(),
        )
        .unwrap();
        let result = idx.query(&[0.0_f32, 0.0, 1.0], 0.99).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_dimension_mismatch_insert() {
        let mut idx = SemanticIndex::new();
        idx.insert(vec![1.0, 2.0, 3.0], "r".into(), "q".into())
            .unwrap();
        let err = idx
            .insert(vec![1.0, 2.0, 3.0, 4.0, 5.0], "r".into(), "q".into())
            .unwrap_err();
        assert!(err.to_string().contains("dimension mismatch"));
    }

    #[test]
    fn test_dimension_mismatch_query() {
        let mut idx = SemanticIndex::new();
        idx.insert(vec![1.0, 2.0, 3.0], "r".into(), "q".into())
            .unwrap();
        let err = idx.query(&[1.0, 2.0, 3.0, 4.0, 5.0], 0.5).unwrap_err();
        assert!(err.to_string().contains("dimension mismatch"));
    }

    #[test]
    fn test_entry_count() {
        let mut idx = SemanticIndex::new();
        idx.insert(vec![1.0, 0.0, 0.0], "a".into(), "qa".into())
            .unwrap();
        idx.insert(vec![0.0, 1.0, 0.0], "b".into(), "qb".into())
            .unwrap();
        idx.insert(vec![0.0, 0.0, 1.0], "c".into(), "qc".into())
            .unwrap();
        assert_eq!(idx.entry_count(), 3);
    }
}
