use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Result, anyhow};
use hnsw_rs::prelude::*;

use crate::config::HnswConfig;
use crate::snapshot::SnapshotEntry;
use crate::wal::WalEntry;

/// Current Unix timestamp in seconds. Used for `inserted_at` /
/// `last_accessed_at` stamps. Returns 0 if the system clock predates UNIX_EPOCH
/// (impossible on a sane host, but we don't want to panic on it).
pub fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub struct CacheEntry {
    pub uuid: String,
    pub embedding: Vec<f32>,
    pub response: String,
    pub query_text: String,
    /// Unix timestamp seconds; set once on insert, never updated. Persisted in
    /// the WAL so it survives restart.
    pub inserted_at: u64,
    /// Unix timestamp seconds; updated on every query hit. NOT in the WAL —
    /// this is in-memory soft state, persisted only via snapshots.
    pub last_accessed_at: u64,
    /// Monotonically incremented on every query hit. NOT in the WAL — same
    /// reasoning as `last_accessed_at`.
    pub access_count: u64,
}

#[derive(Debug)]
pub struct QueryHit {
    pub id: String,
    pub response: String,
    pub similarity: f32,
    /// Side-table internal id, exposed so the handler can call
    /// `record_access` without a second lookup.
    pub internal_id: usize,
}

#[derive(Debug, Clone)]
pub struct NamespaceStats {
    pub entry_count: usize,
    pub dimension: Option<usize>,
    /// Min `inserted_at` across entries (0 if empty).
    pub oldest_entry_ts: u64,
    /// Max `inserted_at` across entries (0 if empty).
    pub newest_entry_ts: u64,
    /// Sum of `access_count` across all entries.
    pub total_accesses: u64,
}

/// One Top-N entry surfaced by `/admin/entry-stats`.
#[derive(Debug, Clone)]
pub struct TopEntrySummary {
    pub uuid: String,
    pub access_count: u64,
    pub last_accessed_at: u64,
    pub query_text_preview: String,
}

/// One HNSW index + side-table, scoped to a single `model_id`.
/// Vectors from different namespaces are never compared.
pub struct NamespacedIndex {
    hnsw: Hnsw<'static, f32, DistCosine>,
    entries: HashMap<usize, CacheEntry>,
    /// Reverse map (M23): UUID → side-table internal id, so read-repair
    /// fetch-by-UUID is O(1) instead of a linear scan over `entries`.
    uuid_to_internal: HashMap<String, usize>,
    next_id: usize,
    dimension: Option<usize>,
    ef_search: usize,
}

/// Full cache entry including its namespace — what `/internal/entry/{uuid}`
/// returns and what read-repair re-inserts on the local node.
#[derive(Debug, Clone, PartialEq)]
pub struct FullEntry {
    pub uuid: String,
    pub embedding: Vec<f32>,
    pub response: String,
    pub query_text: String,
    pub model_id: String,
    pub inserted_at: u64,
    pub last_accessed_at: u64,
    pub access_count: u64,
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
            uuid_to_internal: HashMap::new(),
            next_id: 0,
            dimension: None,
            ef_search: cfg.ef_search,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn insert_with_uuid(
        &mut self,
        uuid: String,
        embedding: Vec<f32>,
        response: String,
        query_text: String,
        inserted_at: u64,
        last_accessed_at: u64,
        access_count: u64,
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

        // De-duplicate on UUID so read-repair re-inserting an entry doesn't
        // create a second copy in the side-table. This is also the right
        // behaviour for WAL replay if the same UUID appears twice.
        if self.uuid_to_internal.contains_key(&uuid) {
            return Ok(());
        }

        let id = self.next_id;
        self.next_id += 1;
        self.hnsw.insert((&embedding, id));
        self.uuid_to_internal.insert(uuid.clone(), id);
        self.entries.insert(
            id,
            CacheEntry {
                uuid,
                embedding,
                response,
                query_text,
                inserted_at,
                last_accessed_at,
                access_count,
            },
        );
        Ok(())
    }

    pub fn get_by_uuid(&self, uuid: &str) -> Option<&CacheEntry> {
        let id = self.uuid_to_internal.get(uuid)?;
        self.entries.get(id)
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

        let internal_id = n.get_origin_id();
        let entry = self
            .entries
            .get(&internal_id)
            .ok_or_else(|| anyhow!("neighbour id {} not in side-table", internal_id))?;

        Ok(Some(QueryHit {
            id: entry.uuid.clone(),
            response: entry.response.clone(),
            similarity,
            internal_id,
        }))
    }

    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    pub fn dimension(&self) -> Option<usize> {
        self.dimension
    }

    /// Update access metadata for the given internal id. Called on every
    /// query hit — tiny critical section under the index write lock.
    fn record_access(&mut self, internal_id: usize) {
        if let Some(entry) = self.entries.get_mut(&internal_id) {
            entry.last_accessed_at = now_unix_secs();
            entry.access_count = entry.access_count.saturating_add(1);
        }
    }
}

/// Top-level index: a map of `model_id` → `NamespacedIndex`. Each namespace
/// owns its own HNSW; cross-namespace queries are impossible by construction.
pub struct SemanticIndex {
    namespaces: HashMap<String, NamespacedIndex>,
    /// UUID → owning `model_id`. Populated on every successful insert so
    /// read-repair (M23) can find an entry by its UUID without scanning
    /// every namespace.
    uuid_to_namespace: HashMap<String, String>,
    hnsw_config: HnswConfig,
}

impl SemanticIndex {
    pub fn new(cfg: &HnswConfig) -> Self {
        Self {
            namespaces: HashMap::new(),
            uuid_to_namespace: HashMap::new(),
            hnsw_config: cfg.clone(),
        }
    }

    fn namespace_mut(&mut self, model_id: &str) -> &mut NamespacedIndex {
        self.namespaces
            .entry(model_id.to_string())
            .or_insert_with(|| NamespacedIndex::new(&self.hnsw_config))
    }

    fn record_uuid(&mut self, uuid: &str, model_id: &str) {
        self.uuid_to_namespace
            .insert(uuid.to_string(), model_id.to_string());
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
        let now = now_unix_secs();
        self.namespace_mut(model_id).insert_with_uuid(
            uuid.clone(),
            embedding,
            response,
            query_text,
            now,
            now,
            0,
        )?;
        self.record_uuid(&uuid, model_id);
        Ok(uuid)
    }

    pub fn replay_entry(&mut self, entry: WalEntry) -> Result<()> {
        let WalEntry {
            uuid,
            embedding,
            response,
            query_text,
            model_id,
            inserted_at,
            ..
        } = entry;
        // Pre-M24 WAL entries lack inserted_at and deserialize to 0; for the
        // initial implementation we keep that 0 — it tags the entry as
        // "older than anything stamped post-M24" which is the correct LRU
        // ordering.
        self.namespace_mut(&model_id).insert_with_uuid(
            uuid.clone(),
            embedding,
            response,
            query_text,
            inserted_at,
            inserted_at,
            0,
        )?;
        self.record_uuid(&uuid, &model_id);
        Ok(())
    }

    pub fn replay_snapshot_entry(&mut self, entry: SnapshotEntry) -> Result<()> {
        let SnapshotEntry {
            uuid,
            embedding,
            response,
            query_text,
            model_id,
            inserted_at,
            last_accessed_at,
            access_count,
        } = entry;
        self.namespace_mut(&model_id).insert_with_uuid(
            uuid.clone(),
            embedding,
            response,
            query_text,
            inserted_at,
            last_accessed_at,
            access_count,
        )?;
        self.record_uuid(&uuid, &model_id);
        Ok(())
    }

    /// Look up a full entry (including its namespace) by UUID. Used by
    /// the `/internal/entry/{uuid}` endpoint so read-repair can re-insert
    /// the entry on a stale primary.
    pub fn get_entry_by_uuid(&self, uuid: &str) -> Option<FullEntry> {
        let model_id = self.uuid_to_namespace.get(uuid)?.clone();
        let ns = self.namespaces.get(&model_id)?;
        let entry = ns.get_by_uuid(uuid)?;
        Some(FullEntry {
            uuid: entry.uuid.clone(),
            embedding: entry.embedding.clone(),
            response: entry.response.clone(),
            query_text: entry.query_text.clone(),
            model_id,
            inserted_at: entry.inserted_at,
            last_accessed_at: entry.last_accessed_at,
            access_count: entry.access_count,
        })
    }

    /// Update access metadata for a query hit. Called by the query handler
    /// under a brief write lock after the HNSW search resolved to a hit.
    pub fn record_access(&mut self, model_id: &str, internal_id: usize) {
        if let Some(ns) = self.namespaces.get_mut(model_id) {
            ns.record_access(internal_id);
        }
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
                    inserted_at: entry.inserted_at,
                    last_accessed_at: entry.last_accessed_at,
                    access_count: entry.access_count,
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
                let mut oldest: u64 = u64::MAX;
                let mut newest: u64 = 0;
                let mut total: u64 = 0;
                let mut have_any = false;
                for e in v.entries() {
                    have_any = true;
                    if e.inserted_at < oldest {
                        oldest = e.inserted_at;
                    }
                    if e.inserted_at > newest {
                        newest = e.inserted_at;
                    }
                    total = total.saturating_add(e.access_count);
                }
                let oldest_entry_ts = if have_any { oldest } else { 0 };
                (
                    k.clone(),
                    NamespaceStats {
                        entry_count: v.entry_count(),
                        dimension: v.dimension(),
                        oldest_entry_ts,
                        newest_entry_ts: newest,
                        total_accesses: total,
                    },
                )
            })
            .collect()
    }

    /// Return the top-N most-accessed entries per namespace, sorted by
    /// access_count descending. Used by `/admin/entry-stats`.
    pub fn top_entries_per_namespace(&self, limit: usize) -> HashMap<String, Vec<TopEntrySummary>> {
        let mut out = HashMap::with_capacity(self.namespaces.len());
        for (model_id, ns) in &self.namespaces {
            let mut entries: Vec<&CacheEntry> = ns.entries().collect();
            entries.sort_by(|a, b| b.access_count.cmp(&a.access_count));
            entries.truncate(limit);
            let summaries = entries
                .into_iter()
                .map(|e| TopEntrySummary {
                    uuid: e.uuid.clone(),
                    access_count: e.access_count,
                    last_accessed_at: e.last_accessed_at,
                    query_text_preview: preview_query_text(&e.query_text),
                })
                .collect();
            out.insert(model_id.clone(), summaries);
        }
        out
    }

    /// Returns the dimension of the first namespace encountered, if any.
    /// Kept for backward-compat with the pre-M14 single-index API.
    pub fn dimension(&self) -> Option<usize> {
        self.namespaces.values().find_map(|n| n.dimension())
    }
}

fn preview_query_text(s: &str) -> String {
    const PREVIEW_LEN: usize = 50;
    if s.chars().count() <= PREVIEW_LEN {
        return s.to_string();
    }
    let truncated: String = s.chars().take(PREVIEW_LEN).collect();
    format!("{truncated}...")
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: &str = "test-model::3";

    fn test_index() -> SemanticIndex {
        SemanticIndex::new(&HnswConfig::default())
    }

    fn wal_entry(uuid: &str, vec: Vec<f32>, model_id: &str) -> WalEntry {
        WalEntry {
            uuid: uuid.into(),
            embedding: vec,
            response: format!("r-{uuid}"),
            query_text: format!("q-{uuid}"),
            model_id: model_id.into(),
            inserted_at: 0,
            sequence: 0,
        }
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
    fn test_get_entry_by_uuid_found() {
        let mut idx = test_index();
        let entry = WalEntry {
            uuid: "fixed-uuid".into(),
            embedding: vec![1.0, 0.0, 0.0],
            response: "the answer".into(),
            query_text: "the question".into(),
            model_id: M.into(),
            inserted_at: 1234,
            sequence: 0,
        };
        idx.replay_entry(entry).unwrap();
        let got = idx.get_entry_by_uuid("fixed-uuid").unwrap();
        assert_eq!(got.uuid, "fixed-uuid");
        assert_eq!(got.embedding, vec![1.0, 0.0, 0.0]);
        assert_eq!(got.response, "the answer");
        assert_eq!(got.query_text, "the question");
        assert_eq!(got.model_id, M);
        assert_eq!(got.inserted_at, 1234);
        assert_eq!(got.last_accessed_at, 1234);
        assert_eq!(got.access_count, 0);
    }

    #[test]
    fn test_get_entry_by_uuid_not_found() {
        let idx = test_index();
        assert!(idx.get_entry_by_uuid("nonexistent").is_none());
    }

    #[test]
    fn test_get_entry_by_uuid_cross_namespace() {
        let mut idx = test_index();
        idx.replay_entry(wal_entry("u-a", vec![1.0, 0.0, 0.0], "model-a::3"))
            .unwrap();
        idx.replay_entry(wal_entry("u-b", vec![0.0, 1.0, 0.0], "model-b::3"))
            .unwrap();
        let hit = idx.get_entry_by_uuid("u-b").unwrap();
        assert_eq!(hit.model_id, "model-b::3");
        assert_eq!(hit.response, "r-u-b");
    }

    #[test]
    fn test_uuid_to_namespace_populated_on_replay() {
        let mut idx = test_index();
        idx.replay_entry(wal_entry("u1", vec![1.0, 0.0, 0.0], "m::3"))
            .unwrap();
        idx.replay_snapshot_entry(SnapshotEntry {
            uuid: "u2".into(),
            embedding: vec![0.0, 1.0, 0.0],
            response: "r".into(),
            query_text: "q".into(),
            model_id: "m::3".into(),
            inserted_at: 0,
            last_accessed_at: 0,
            access_count: 0,
        })
        .unwrap();
        assert_eq!(
            idx.uuid_to_namespace.get("u1").map(String::as_str),
            Some("m::3")
        );
        assert_eq!(
            idx.uuid_to_namespace.get("u2").map(String::as_str),
            Some("m::3")
        );
    }

    #[test]
    fn test_replay_dedupes_on_uuid() {
        // Replaying the same UUID twice must not create a duplicate entry.
        // (Important for read-repair — the coordinator may replay an entry
        // that was already inserted by another concurrent path.)
        let mut idx = test_index();
        let mk = || wal_entry("dup", vec![1.0, 0.0, 0.0], M);
        idx.replay_entry(mk()).unwrap();
        idx.replay_entry(mk()).unwrap();
        assert_eq!(idx.entry_count(), 1);
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

    // --- M24: access tracking ---------------------------------------------

    #[test]
    fn test_insert_sets_inserted_at() {
        let before = now_unix_secs();
        let mut idx = test_index();
        let uuid = idx
            .insert(vec![1.0, 0.0, 0.0], "r".into(), "q".into(), M)
            .unwrap();
        let after = now_unix_secs();
        let entry = idx.get_entry_by_uuid(&uuid).unwrap();
        assert!(
            entry.inserted_at >= before && entry.inserted_at <= after + 1,
            "inserted_at={} not in [{before}, {after}+1]",
            entry.inserted_at
        );
    }

    #[test]
    fn test_insert_sets_initial_access_fields() {
        let mut idx = test_index();
        let uuid = idx
            .insert(vec![1.0, 0.0, 0.0], "r".into(), "q".into(), M)
            .unwrap();
        let entry = idx.get_entry_by_uuid(&uuid).unwrap();
        assert_eq!(entry.access_count, 0);
        assert_eq!(entry.last_accessed_at, entry.inserted_at);
    }

    #[test]
    fn test_record_access_increments_count() {
        let mut idx = test_index();
        let v = vec![1.0_f32, 0.0, 0.0];
        let uuid = idx.insert(v.clone(), "r".into(), "q".into(), M).unwrap();
        let hit = idx.query(&v, 0.9, M).unwrap().unwrap();
        idx.record_access(M, hit.internal_id);
        assert_eq!(idx.get_entry_by_uuid(&uuid).unwrap().access_count, 1);
        idx.record_access(M, hit.internal_id);
        assert_eq!(idx.get_entry_by_uuid(&uuid).unwrap().access_count, 2);
    }

    #[test]
    fn test_record_access_updates_last_accessed() {
        let mut idx = test_index();
        let v = vec![1.0_f32, 0.0, 0.0];
        let uuid = idx.insert(v.clone(), "r".into(), "q".into(), M).unwrap();
        let initial = idx.get_entry_by_uuid(&uuid).unwrap().last_accessed_at;
        let hit = idx.query(&v, 0.9, M).unwrap().unwrap();
        idx.record_access(M, hit.internal_id);
        let after = idx.get_entry_by_uuid(&uuid).unwrap().last_accessed_at;
        assert!(
            after >= initial,
            "last_accessed_at went backwards: {after} < {initial}"
        );
    }

    #[test]
    fn test_inserted_at_does_not_change_on_access() {
        let mut idx = test_index();
        let v = vec![1.0_f32, 0.0, 0.0];
        let uuid = idx.insert(v.clone(), "r".into(), "q".into(), M).unwrap();
        let initial = idx.get_entry_by_uuid(&uuid).unwrap().inserted_at;
        let hit = idx.query(&v, 0.9, M).unwrap().unwrap();
        idx.record_access(M, hit.internal_id);
        idx.record_access(M, hit.internal_id);
        assert_eq!(idx.get_entry_by_uuid(&uuid).unwrap().inserted_at, initial);
    }

    #[test]
    fn test_namespace_stats_access_fields() {
        let mut idx = test_index();
        let v1 = vec![1.0_f32, 0.0, 0.0];
        let v2 = vec![0.0_f32, 1.0, 0.0];
        let v3 = vec![0.0_f32, 0.0, 1.0];
        let _u1 = idx.insert(v1.clone(), "r1".into(), "q1".into(), M).unwrap();
        let _u2 = idx.insert(v2.clone(), "r2".into(), "q2".into(), M).unwrap();
        let _u3 = idx.insert(v3.clone(), "r3".into(), "q3".into(), M).unwrap();

        // Access entry 1 twice, entry 2 once. Entry 3 untouched.
        let h1 = idx.query(&v1, 0.9, M).unwrap().unwrap();
        idx.record_access(M, h1.internal_id);
        idx.record_access(M, h1.internal_id);
        let h2 = idx.query(&v2, 0.9, M).unwrap().unwrap();
        idx.record_access(M, h2.internal_id);

        let stats = idx.namespace_stats();
        let ns = stats.get(M).unwrap();
        assert_eq!(ns.total_accesses, 3);
        assert!(ns.oldest_entry_ts > 0);
        assert!(ns.newest_entry_ts >= ns.oldest_entry_ts);
    }

    #[test]
    fn test_snapshot_preserves_access_fields() {
        let mut idx = test_index();
        let v = vec![1.0_f32, 0.0, 0.0];
        idx.insert(v.clone(), "r".into(), "q".into(), M).unwrap();
        let h = idx.query(&v, 0.9, M).unwrap().unwrap();
        idx.record_access(M, h.internal_id);
        idx.record_access(M, h.internal_id);
        idx.record_access(M, h.internal_id);

        let snap = idx.snapshot_entries();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].access_count, 3);
        assert!(snap[0].last_accessed_at > 0);
        assert!(snap[0].inserted_at > 0);
    }

    #[test]
    fn test_replay_snapshot_preserves_access_fields() {
        let mut idx = test_index();
        idx.replay_snapshot_entry(SnapshotEntry {
            uuid: "u1".into(),
            embedding: vec![1.0, 0.0, 0.0],
            response: "r".into(),
            query_text: "q".into(),
            model_id: M.into(),
            inserted_at: 5000,
            last_accessed_at: 6000,
            access_count: 42,
        })
        .unwrap();
        let entry = idx.get_entry_by_uuid("u1").unwrap();
        assert_eq!(entry.inserted_at, 5000);
        assert_eq!(entry.last_accessed_at, 6000);
        assert_eq!(entry.access_count, 42);
    }
}
