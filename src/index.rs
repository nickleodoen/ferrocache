use std::collections::HashMap;

use anyhow::{Result, anyhow};
use hnsw_rs::prelude::*;

use crate::config::HnswConfig;
use crate::snapshot::SnapshotEntry;
use crate::wal::WalEntry;

pub struct CacheEntry {
    pub uuid: String,
    pub embedding: Vec<f32>,
    pub response: String,
    pub query_text: String,
}

#[derive(Debug)]
pub struct QueryHit {
    pub id: String,
    pub response: String,
    pub similarity: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct NamespaceStats {
    pub entry_count: usize,
    pub dimension: Option<usize>,
}

/// One HNSW index + side-table, scoped to a single `model_id`.
/// Vectors from different namespaces are never compared.
pub struct NamespacedIndex {
    hnsw: Hnsw<'static, f32, DistCosine>,
    entries: HashMap<usize, CacheEntry>,
    next_id: usize,
    dimension: Option<usize>,
    ef_search: usize,
}

impl NamespacedIndex {
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

    pub fn insert_with_uuid(
        &mut self,
        uuid: String,
        embedding: Vec<f32>,
        response: String,
        query_text: String,
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
        self.entries.insert(
            id,
            CacheEntry {
                uuid,
                embedding,
                response,
                query_text,
            },
        );
        Ok(())
    }

    pub fn entries(&self) -> impl Iterator<Item = &CacheEntry> {
        self.entries.values()
    }

    pub fn query(&self, embedding: &[f32], threshold: f32) -> Result<Option<QueryHit>> {
        let Some(d) = self.dimension else {
            return Ok(None);
        };
        if d != embedding.len() {
            return Err(anyhow!(
                "dimension mismatch: expected {}, got {}",
                d,
                embedding.len()
            ));
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

/// Top-level index: a map of `model_id` → `NamespacedIndex`. Each namespace
/// owns its own HNSW; cross-namespace queries are impossible by construction.
pub struct SemanticIndex {
    namespaces: HashMap<String, NamespacedIndex>,
    hnsw_config: HnswConfig,
}

impl SemanticIndex {
    pub fn new(cfg: &HnswConfig) -> Self {
        Self {
            namespaces: HashMap::new(),
            hnsw_config: cfg.clone(),
        }
    }

    fn namespace_mut(&mut self, model_id: &str) -> &mut NamespacedIndex {
        self.namespaces
            .entry(model_id.to_string())
            .or_insert_with(|| NamespacedIndex::new(&self.hnsw_config))
    }

    #[cfg(test)]
    pub fn insert(
        &mut self,
        embedding: Vec<f32>,
        response: String,
        query_text: String,
        model_id: &str,
    ) -> Result<String> {
        let uuid = uuid::Uuid::new_v4().to_string();
        self.namespace_mut(model_id).insert_with_uuid(
            uuid.clone(),
            embedding,
            response,
            query_text,
        )?;
        Ok(uuid)
    }

    pub fn replay_entry(&mut self, entry: WalEntry) -> Result<()> {
        let WalEntry {
            uuid,
            embedding,
            response,
            query_text,
            model_id,
            ..
        } = entry;
        self.namespace_mut(&model_id)
            .insert_with_uuid(uuid, embedding, response, query_text)
    }

    pub fn replay_snapshot_entry(&mut self, entry: SnapshotEntry) -> Result<()> {
        let SnapshotEntry {
            uuid,
            embedding,
            response,
            query_text,
            model_id,
        } = entry;
        self.namespace_mut(&model_id)
            .insert_with_uuid(uuid, embedding, response, query_text)
    }

    /// Flatten every namespace into a `Vec<SnapshotEntry>`. Used by
    /// compaction; the result is a complete picture of in-memory state.
    pub fn snapshot_entries(&self) -> Vec<SnapshotEntry> {
        let mut out = Vec::with_capacity(self.entry_count());
        for (model_id, ns) in &self.namespaces {
            for entry in ns.entries() {
                out.push(SnapshotEntry {
                    uuid: entry.uuid.clone(),
                    embedding: entry.embedding.clone(),
                    response: entry.response.clone(),
                    query_text: entry.query_text.clone(),
                    model_id: model_id.clone(),
                });
            }
        }
        out
    }

    pub fn query(
        &self,
        embedding: &[f32],
        threshold: f32,
        model_id: &str,
    ) -> Result<Option<QueryHit>> {
        match self.namespaces.get(model_id) {
            Some(ns) => ns.query(embedding, threshold),
            None => Ok(None),
        }
    }

    pub fn entry_count(&self) -> usize {
        self.namespaces.values().map(|n| n.entry_count()).sum()
    }

    pub fn namespace_stats(&self) -> HashMap<String, NamespaceStats> {
        self.namespaces
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    NamespaceStats {
                        entry_count: v.entry_count(),
                        dimension: v.dimension(),
                    },
                )
            })
            .collect()
    }

    /// Returns the dimension of the first namespace encountered, if any.
    /// Kept for backward-compat with the pre-M14 single-index API.
    pub fn dimension(&self) -> Option<usize> {
        self.namespaces.values().find_map(|n| n.dimension())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: &str = "test-model::3";

    fn test_index() -> SemanticIndex {
        SemanticIndex::new(&HnswConfig::default())
    }

    #[test]
    fn test_insert_and_query_hit() {
        let mut idx = test_index();
        let v = vec![1.0_f32, 0.0, 0.0];
        let uuid = idx
            .insert(v.clone(), "cached-response".to_string(), "q".to_string(), M)
            .unwrap();
        assert!(!uuid.is_empty());

        let hit = idx.query(&v, 0.90, M).unwrap().expect("should hit");
        assert_eq!(hit.response, "cached-response");
        assert!(hit.similarity > 0.999, "similarity={}", hit.similarity);
        assert_eq!(hit.id, uuid);
    }

    #[test]
    fn test_query_miss_below_threshold() {
        let mut idx = test_index();
        idx.insert(vec![1.0_f32, 0.0, 0.0], "r".to_string(), "q".to_string(), M)
            .unwrap();
        let result = idx.query(&[0.0_f32, 0.0, 1.0], 0.99, M).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_dimension_mismatch_insert() {
        let mut idx = test_index();
        idx.insert(vec![1.0, 2.0, 3.0], "r".into(), "q".into(), M)
            .unwrap();
        let err = idx
            .insert(vec![1.0, 2.0, 3.0, 4.0, 5.0], "r".into(), "q".into(), M)
            .unwrap_err();
        assert!(err.to_string().contains("dimension mismatch"));
    }

    #[test]
    fn test_dimension_mismatch_query() {
        let mut idx = test_index();
        idx.insert(vec![1.0, 2.0, 3.0], "r".into(), "q".into(), M)
            .unwrap();
        let err = idx.query(&[1.0, 2.0, 3.0, 4.0, 5.0], 0.5, M).unwrap_err();
        assert!(err.to_string().contains("dimension mismatch"));
    }

    #[test]
    fn test_entry_count() {
        let mut idx = test_index();
        idx.insert(vec![1.0, 0.0, 0.0], "a".into(), "qa".into(), M)
            .unwrap();
        idx.insert(vec![0.0, 1.0, 0.0], "b".into(), "qb".into(), M)
            .unwrap();
        idx.insert(vec![0.0, 0.0, 1.0], "c".into(), "qc".into(), M)
            .unwrap();
        assert_eq!(idx.entry_count(), 3);
    }

    #[test]
    fn test_namespace_isolation() {
        let mut idx = test_index();
        idx.insert(
            vec![1.0_f32, 0.0, 0.0],
            "in model-a".into(),
            "q".into(),
            "model-a::3",
        )
        .unwrap();
        let result = idx.query(&[1.0_f32, 0.0, 0.0], 0.90, "model-b::3").unwrap();
        assert!(result.is_none(), "cross-namespace lookup must miss");
    }

    #[test]
    fn test_same_dim_different_namespace() {
        let mut idx = test_index();
        idx.insert(
            vec![1.0_f32, 0.0, 0.0],
            "answer-a".into(),
            "q".into(),
            "model-a::3",
        )
        .unwrap();
        idx.insert(
            vec![0.0_f32, 1.0, 0.0],
            "answer-b".into(),
            "q".into(),
            "model-b::3",
        )
        .unwrap();
        let hit = idx
            .query(&[1.0_f32, 0.0, 0.0], 0.90, "model-a::3")
            .unwrap()
            .expect("should hit");
        assert_eq!(hit.response, "answer-a");
    }

    #[test]
    fn test_namespace_stats() {
        let mut idx = test_index();
        idx.insert(vec![1.0, 0.0, 0.0], "1".into(), "q".into(), "model-a::3")
            .unwrap();
        idx.insert(vec![0.0, 1.0, 0.0], "2".into(), "q".into(), "model-a::3")
            .unwrap();
        idx.insert(
            vec![1.0, 0.0, 0.0, 0.0],
            "3".into(),
            "q".into(),
            "model-b::4",
        )
        .unwrap();
        idx.insert(
            vec![0.0, 1.0, 0.0, 0.0],
            "4".into(),
            "q".into(),
            "model-b::4",
        )
        .unwrap();
        idx.insert(
            vec![0.0, 0.0, 1.0, 0.0],
            "5".into(),
            "q".into(),
            "model-b::4",
        )
        .unwrap();
        let stats = idx.namespace_stats();
        assert_eq!(stats.get("model-a::3").unwrap().entry_count, 2);
        assert_eq!(stats.get("model-a::3").unwrap().dimension, Some(3));
        assert_eq!(stats.get("model-b::4").unwrap().entry_count, 3);
        assert_eq!(stats.get("model-b::4").unwrap().dimension, Some(4));
    }

    #[test]
    fn test_entry_count_across_namespaces() {
        let mut idx = test_index();
        idx.insert(vec![1.0, 0.0, 0.0], "a".into(), "q".into(), "model-a::3")
            .unwrap();
        idx.insert(vec![0.0, 1.0, 0.0], "b".into(), "q".into(), "model-a::3")
            .unwrap();
        idx.insert(vec![1.0, 0.0, 0.0], "c".into(), "q".into(), "model-b::3")
            .unwrap();
        assert_eq!(idx.entry_count(), 3);
    }

    #[test]
    fn test_query_nonexistent_namespace() {
        let idx = test_index();
        let result = idx.query(&[0.1, 0.2, 0.3], 0.5, "never-seen::3").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_snapshot_entries_roundtrip() {
        let mut src = test_index();
        src.insert(
            vec![1.0, 0.0, 0.0],
            "model-a-resp".into(),
            "qa".into(),
            "model-a::3",
        )
        .unwrap();
        src.insert(
            vec![0.0, 1.0, 0.0],
            "model-a-resp2".into(),
            "qa2".into(),
            "model-a::3",
        )
        .unwrap();
        src.insert(
            vec![1.0, 0.0, 0.0, 0.0],
            "model-b-resp".into(),
            "qb".into(),
            "model-b::4",
        )
        .unwrap();

        let snap = src.snapshot_entries();
        assert_eq!(snap.len(), 3);

        let mut dst = test_index();
        for e in snap {
            dst.replay_snapshot_entry(e).unwrap();
        }
        assert_eq!(dst.entry_count(), 3);
        let hit = dst
            .query(&[1.0_f32, 0.0, 0.0], 0.90, "model-a::3")
            .unwrap()
            .expect("should hit");
        assert_eq!(hit.response, "model-a-resp");
        let hit_b = dst
            .query(&[1.0_f32, 0.0, 0.0, 0.0], 0.90, "model-b::4")
            .unwrap()
            .expect("should hit");
        assert_eq!(hit_b.response, "model-b-resp");
    }
}
