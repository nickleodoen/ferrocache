use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Result, anyhow};
use hnsw_rs::prelude::*;

use crate::config::HnswConfig;
use crate::snapshot::SnapshotEntry;
use crate::wal::WalEntry;

/// Default rebuild trigger: when the fraction of evicted-but-still-in-graph
/// nodes exceeds this, rebuild the HNSW from scratch using only live entries
/// to reclaim graph connectivity wasted by ghosts.
pub const DEFAULT_EVICTED_FRACTION_REBUILD: f64 = 0.20;

/// Cap on the extra HNSW neighbors we ask for to compensate for evicted
/// ghosts in the search results. Beyond this, a rebuild is overdue and we
/// don't want to scan a long candidate list on every query.
const MAX_EVICTION_OVERSCAN: usize = 8;

/// Normalize a `query_text` for exact-match indexing (M27). Lowercases,
/// trims, and collapses internal whitespace runs to a single space.
/// Deterministic and locale-independent — no Unicode normalization on
/// purpose, so behaviour stays predictable for cache operators.
pub fn normalize_query_text(text: &str) -> String {
    text.split_whitespace()
        .map(str::to_lowercase)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Compute the effective namespace key for an insert/query/invalidate (M28).
/// When `cache_scope` is `Some(non-empty-after-trim)`, returns
/// `"{model_id}::{cache_scope}"`. Otherwise returns `model_id` unchanged.
/// Empty/whitespace-only scope is treated as absent so callers passing
/// `cache_scope: ""` don't end up in a separate `"m::"` namespace.
pub fn effective_namespace(model_id: &str, cache_scope: Option<&str>) -> String {
    match cache_scope {
        Some(scope) if !scope.trim().is_empty() => format!("{model_id}::{scope}"),
        _ => model_id.to_string(),
    }
}

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
    /// TTL deadline (M26). `None` = no expiry. Persisted in WAL + snapshot.
    pub expires_at: Option<u64>,
}

#[derive(Debug)]
pub struct QueryHit {
    pub id: String,
    pub response: String,
    pub similarity: f32,
    /// Side-table internal id, exposed so the handler can call
    /// `record_access` without a second lookup.
    pub internal_id: usize,
    /// `true` when the hit came from the M27 exact-match pre-filter
    /// (i.e. `query_text` matched a stored entry's normalized form).
    /// `false` for HNSW ANN hits.
    pub exact_match: bool,
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
    /// HNSW internal IDs evicted but still inert in the graph (M25). A
    /// rebuild reclaims this space. Useful for operators to spot a stuck
    /// rebuild or unusually high churn.
    pub evicted_ghost_count: usize,
}

/// One entry that was just evicted by `evict_lru`. The flush task uses
/// `(uuid, model_id)` to write a tombstone WAL entry so the eviction
/// survives a restart.
#[derive(Debug, Clone)]
pub struct EvictedEntry {
    pub uuid: String,
    pub model_id: String,
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
    /// HNSW internal IDs that have been evicted (M25). The HNSW graph
    /// has no removal API, so the node stays in the graph as an inert
    /// ghost; queries filter against this set before threshold checks.
    /// Cleared on rebuild.
    evicted_ids: HashSet<usize>,
    /// Exact-match pre-filter (M27): normalized `query_text` → owning
    /// UUID. Checked before HNSW search; O(1) hit when the user's text
    /// matches a stored entry. Cleaned up on every removal path.
    exact_match_index: HashMap<String, String>,
    next_id: usize,
    dimension: Option<usize>,
    ef_search: usize,
    /// Rebuild trigger threshold: when `evicted_ids.len() / total > this`,
    /// `needs_rebuild()` returns true. Default 0.20.
    evicted_fraction_rebuild: f64,
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
    pub expires_at: Option<u64>,
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
            evicted_ids: HashSet::new(),
            exact_match_index: HashMap::new(),
            next_id: 0,
            dimension: None,
            ef_search: cfg.ef_search,
            evicted_fraction_rebuild: DEFAULT_EVICTED_FRACTION_REBUILD,
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
        expires_at: Option<u64>,
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
        // M27: register normalized query_text → uuid. Empty after
        // normalization (whitespace-only) is skipped — would otherwise
        // make every "" query match.
        let normalized = normalize_query_text(&query_text);
        if !normalized.is_empty() {
            self.exact_match_index.insert(normalized, uuid.clone());
        }
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
                expires_at,
            },
        );
        Ok(())
    }

    /// Internal: remove the entry with the given internal id and clean
    /// up every side map (uuid_to_internal, evicted_ids, exact_match_index).
    /// This is the single removal point — every public removal path
    /// (`evict_lru`, `remove_by_uuid`, `remove_internal`,
    /// `collect_expired_within`) funnels through it.
    ///
    /// The exact-match cleanup includes a UUID ownership check: if a
    /// later insert with the same normalized text overwrote this entry's
    /// mapping, removing it would clobber the live mapping. We only
    /// remove the key when its current value still references the
    /// dropped UUID.
    fn drop_entry(&mut self, internal_id: usize) -> Option<CacheEntry> {
        let entry = self.entries.remove(&internal_id)?;
        self.uuid_to_internal.remove(&entry.uuid);
        self.evicted_ids.insert(internal_id);
        let normalized = normalize_query_text(&entry.query_text);
        if !normalized.is_empty()
            && self
                .exact_match_index
                .get(&normalized)
                .is_some_and(|u| u == &entry.uuid)
        {
            self.exact_match_index.remove(&normalized);
        }
        Some(entry)
    }

    pub fn get_by_uuid(&self, uuid: &str) -> Option<&CacheEntry> {
        let id = self.uuid_to_internal.get(uuid)?;
        self.entries.get(id)
    }

    pub fn entries(&self) -> impl Iterator<Item = &CacheEntry> {
        self.entries.values()
    }

    /// Backward-compat wrapper: HNSW-only path with no exact-match
    /// pre-filter. Used by tests + any caller that doesn't have a
    /// `query_text` to look up by.
    pub fn query(&self, embedding: &[f32], threshold: f32) -> Result<Option<QueryHit>> {
        self.query_with_text(embedding, threshold, None, now_unix_secs())
    }

    /// Full M27 query path: try the exact-match pre-filter first when
    /// `query_text` is `Some`, then fall through to HNSW ANN search.
    /// `now` is passed in (not read from the wall clock) so the same
    /// instant evaluates every candidate consistently in one call.
    pub fn query_with_text(
        &self,
        embedding: &[f32],
        threshold: f32,
        query_text: Option<&str>,
        now: u64,
    ) -> Result<Option<QueryHit>> {
        // M27 exact-match pre-filter — runs BEFORE the HNSW search. A
        // miss falls through silently. Conditions for a hit: normalized
        // text key exists → UUID resolves to an internal id → not in
        // `evicted_ids` → entry still in side-table → not expired.
        if let Some(text) = query_text {
            let normalized = normalize_query_text(text);
            if !normalized.is_empty()
                && let Some(uuid) = self.exact_match_index.get(&normalized)
                && let Some(&internal_id) = self.uuid_to_internal.get(uuid)
                && !self.evicted_ids.contains(&internal_id)
                && let Some(entry) = self.entries.get(&internal_id)
                && entry.expires_at.is_none_or(|exp| now <= exp)
            {
                return Ok(Some(QueryHit {
                    id: entry.uuid.clone(),
                    response: entry.response.clone(),
                    similarity: 1.0,
                    internal_id,
                    exact_match: true,
                }));
            }
        }

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

        // Ask for extra candidates so eviction-filtering doesn't turn a hit
        // into a miss. Capped because beyond ~8 ghosts a rebuild is overdue.
        let want = 1 + self.evicted_ids.len().min(MAX_EVICTION_OVERSCAN);
        let neighbours = self.hnsw.search(embedding, want, self.ef_search);

        for n in neighbours {
            let internal_id = n.get_origin_id();
            // Skip ghost nodes (evicted-but-still-in-graph).
            if self.evicted_ids.contains(&internal_id) {
                continue;
            }
            let similarity = 1.0 - n.get_distance();
            if similarity < threshold {
                // Best non-evicted candidate falls below threshold — miss.
                return Ok(None);
            }
            let entry = self
                .entries
                .get(&internal_id)
                .ok_or_else(|| anyhow!("neighbour id {} not in side-table", internal_id))?;
            // M26: TTL check. Expired entries return miss without mutating
            // state (the reaper handles cleanup); we move on to the next
            // candidate so a still-live neighbour can match.
            if let Some(exp) = entry.expires_at
                && now > exp
            {
                continue;
            }
            return Ok(Some(QueryHit {
                id: entry.uuid.clone(),
                response: entry.response.clone(),
                similarity,
                internal_id,
                exact_match: false,
            }));
        }
        Ok(None)
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

    pub fn evicted_ghost_count(&self) -> usize {
        self.evicted_ids.len()
    }

    /// Return all live (non-evicted, non-expired) entries whose cosine
    /// similarity to `embedding` is at or above `threshold`. O(n) scan
    /// over the side-table — used by `/admin/invalidate`, which is an
    /// infrequent admin path. Returns `(internal_id, uuid)` pairs.
    pub fn find_similar(
        &self,
        embedding: &[f32],
        threshold: f32,
        now: u64,
    ) -> Vec<(usize, String)> {
        let mut matches: Vec<(usize, String)> = Vec::new();
        for (&id, entry) in &self.entries {
            if self.evicted_ids.contains(&id) {
                continue;
            }
            if let Some(exp) = entry.expires_at
                && now > exp
            {
                continue;
            }
            let sim = cosine_similarity(embedding, &entry.embedding);
            if sim >= threshold {
                matches.push((id, entry.uuid.clone()));
            }
        }
        matches
    }

    /// Drop all entries with `expires_at < now`. Returns the evicted
    /// `(uuid, internal_id)` pairs so the caller can write tombstones.
    pub fn collect_expired_within(&mut self, now: u64) -> Vec<(String, usize)> {
        let expired_ids: Vec<usize> = self
            .entries
            .iter()
            .filter(|(_, e)| e.expires_at.is_some_and(|exp| now > exp))
            .map(|(&id, _)| id)
            .collect();
        let mut out: Vec<(String, usize)> = Vec::with_capacity(expired_ids.len());
        for id in expired_ids {
            if let Some(entry) = self.drop_entry(id) {
                out.push((entry.uuid, id));
            }
        }
        out
    }

    /// Evict the entry with the smallest `last_accessed_at`. O(n) scan; fine
    /// for a cache that runs evictions per insert batch (cap-bounded), not
    /// per request. Returns the evicted (uuid, model_id) so the caller can
    /// write a tombstone to the WAL.
    ///
    /// Tie-break: when `last_accessed_at` (and `inserted_at`) match — typical
    /// when several entries land within the same wall-clock second — fall
    /// through to the side-table internal id, which is monotonic with insert
    /// order. That gives FIFO eviction on ties, matching the intuitive
    /// "oldest entry first" semantics.
    pub fn evict_lru(&mut self, model_id: &str) -> Option<EvictedEntry> {
        let (&evict_id, _) = self
            .entries
            .iter()
            .min_by_key(|(id, e)| (e.last_accessed_at, e.inserted_at, **id))?;
        let entry = self.drop_entry(evict_id)?;
        Some(EvictedEntry {
            uuid: entry.uuid,
            model_id: model_id.to_string(),
        })
    }

    /// Remove the entry with the given UUID. Used for explicit deletion
    /// (M25 tombstone replay). Returns `true` if the entry existed.
    pub fn remove_by_uuid(&mut self, uuid: &str) -> bool {
        let Some(&internal_id) = self.uuid_to_internal.get(uuid) else {
            return false;
        };
        self.drop_entry(internal_id).is_some()
    }

    /// Remove a known internal id (M26). Used by `/admin/invalidate`,
    /// which already has the id from `find_similar`.
    pub fn remove_internal(&mut self, internal_id: usize) -> Option<String> {
        let entry = self.drop_entry(internal_id)?;
        Some(entry.uuid)
    }

    /// True when a HNSW rebuild would meaningfully reclaim space. Threshold
    /// is `evicted_fraction_rebuild` of `(live + evicted)`.
    pub fn needs_rebuild(&self) -> bool {
        let total = self.entries.len() + self.evicted_ids.len();
        if total == 0 {
            return false;
        }
        (self.evicted_ids.len() as f64 / total as f64) > self.evicted_fraction_rebuild
    }

    /// Rebuild the HNSW from scratch using only live entries. Reclaims the
    /// memory + graph connectivity wasted by ghosts. Internal IDs are
    /// reassigned; `uuid_to_internal` is updated; `evicted_ids` is cleared.
    pub fn rebuild(&mut self, hnsw_config: &HnswConfig) {
        let new_hnsw = Hnsw::<f32, DistCosine>::new(
            hnsw_config.max_nb_connection,
            hnsw_config.max_elements,
            hnsw_config.max_layer,
            hnsw_config.ef_construction,
            DistCosine,
        );

        let mut new_entries: HashMap<usize, CacheEntry> = HashMap::new();
        let mut new_uuid_to_internal: HashMap<String, usize> = HashMap::new();
        let mut new_next_id: usize = 0;
        for (_, entry) in self.entries.drain() {
            let id = new_next_id;
            new_next_id += 1;
            new_hnsw.insert((&entry.embedding, id));
            new_uuid_to_internal.insert(entry.uuid.clone(), id);
            new_entries.insert(id, entry);
        }

        self.hnsw = new_hnsw;
        self.entries = new_entries;
        self.uuid_to_internal = new_uuid_to_internal;
        self.next_id = new_next_id;
        self.evicted_ids.clear();
        tracing::info!(live_entries = new_next_id, "HNSW rebuild complete");
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
            None,
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
            expires_at,
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
            expires_at,
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
            tombstone: _, // snapshots only carry live entries by construction
            expires_at,
        } = entry;
        self.namespace_mut(&model_id).insert_with_uuid(
            uuid.clone(),
            embedding,
            response,
            query_text,
            inserted_at,
            last_accessed_at,
            access_count,
            expires_at,
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
            expires_at: entry.expires_at,
        })
    }

    /// Update access metadata for a query hit. Called by the query handler
    /// under a brief write lock after the HNSW search resolved to a hit.
    pub fn record_access(&mut self, model_id: &str, internal_id: usize) {
        if let Some(ns) = self.namespaces.get_mut(model_id) {
            ns.record_access(internal_id);
        }
    }

    /// Remove the entry with the given UUID from any namespace. Called by
    /// WAL tombstone replay (M25). Returns `true` if the entry existed.
    pub fn remove_by_uuid(&mut self, uuid: &str) -> bool {
        let Some(model_id) = self.uuid_to_namespace.remove(uuid) else {
            return false;
        };
        let Some(ns) = self.namespaces.get_mut(&model_id) else {
            return false;
        };
        ns.remove_by_uuid(uuid)
    }

    /// For each namespace exceeding `max_entries`, evict the least-recently-
    /// used entries until at cap. Returns the evicted entries so the caller
    /// (the flush task) can write tombstone WAL entries.
    pub fn evict_to_cap(&mut self, max_entries: usize) -> Vec<EvictedEntry> {
        let mut evicted: Vec<EvictedEntry> = Vec::new();
        for (model_id, ns) in self.namespaces.iter_mut() {
            while ns.entry_count() > max_entries {
                let Some(e) = ns.evict_lru(model_id) else {
                    break;
                };
                evicted.push(e);
            }
        }
        // Mirror the eviction at the top-level uuid → namespace map so
        // `get_entry_by_uuid` doesn't keep returning a ghost.
        for e in &evicted {
            self.uuid_to_namespace.remove(&e.uuid);
        }
        evicted
    }

    /// Rebuild every namespace whose ghost ratio is above the trigger
    /// threshold. Returns the model_ids that were rebuilt so the caller
    /// can bump per-namespace metrics.
    pub fn rebuild_dirty_namespaces(&mut self, hnsw_config: &HnswConfig) -> Vec<String> {
        let mut rebuilt: Vec<String> = Vec::new();
        for (model_id, ns) in self.namespaces.iter_mut() {
            if ns.needs_rebuild() {
                ns.rebuild(hnsw_config);
                rebuilt.push(model_id.clone());
            }
        }
        rebuilt
    }

    /// Sweep every namespace for entries with `expires_at < now` (M26).
    /// Removes them from the side-table + reverse maps and returns
    /// `(uuid, model_id)` pairs so the caller can write tombstones.
    pub fn collect_expired(&mut self, now: u64) -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = Vec::new();
        for (model_id, ns) in self.namespaces.iter_mut() {
            for (uuid, _id) in ns.collect_expired_within(now) {
                out.push((uuid, model_id.clone()));
            }
        }
        for (uuid, _) in &out {
            self.uuid_to_namespace.remove(uuid);
        }
        out
    }

    /// Find every live entry in the namespace whose cosine similarity to
    /// `embedding` is at or above `threshold`. Used by `/admin/invalidate`.
    /// Returns `(internal_id, uuid)` pairs.
    pub fn find_similar(
        &self,
        embedding: &[f32],
        threshold: f32,
        model_id: &str,
        now: u64,
    ) -> Vec<(usize, String)> {
        match self.namespaces.get(model_id) {
            Some(ns) => ns.find_similar(embedding, threshold, now),
            None => Vec::new(),
        }
    }

    /// Look up a UUID's owning namespace without fetching the full entry.
    /// Cheap O(1); used by the DELETE handler to find the model_id before
    /// taking the index write lock.
    pub fn namespace_for(&self, uuid: &str) -> Option<&str> {
        self.uuid_to_namespace.get(uuid).map(String::as_str)
    }

    /// Find entries in `model_id` similar to `embedding` at or above
    /// `threshold`, remove them, and return the evicted UUIDs (M26).
    /// Used by `/admin/invalidate`. Skips entries already evicted or
    /// expired. Single locked operation.
    pub fn invalidate_similar(
        &mut self,
        embedding: &[f32],
        threshold: f32,
        model_id: &str,
        now: u64,
    ) -> Vec<String> {
        let mut removed: Vec<String> = Vec::new();
        if let Some(ns) = self.namespaces.get_mut(model_id) {
            let matches = ns.find_similar(embedding, threshold, now);
            for (id, _uuid) in matches {
                if let Some(uuid) = ns.remove_internal(id) {
                    removed.push(uuid);
                }
            }
        }
        for u in &removed {
            self.uuid_to_namespace.remove(u);
        }
        removed
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
                    tombstone: false,
                    expires_at: entry.expires_at,
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
        self.query_with_text(embedding, threshold, model_id, None, now_unix_secs())
    }

    /// Full M27 query path with exact-match pre-filter. The handler
    /// passes `now` once so the same instant evaluates expiry on every
    /// candidate.
    pub fn query_with_text(
        &self,
        embedding: &[f32],
        threshold: f32,
        model_id: &str,
        query_text: Option<&str>,
        now: u64,
    ) -> Result<Option<QueryHit>> {
        match self.namespaces.get(model_id) {
            Some(ns) => ns.query_with_text(embedding, threshold, query_text, now),
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
                        evicted_ghost_count: v.evicted_ghost_count(),
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
            entries.sort_by_key(|b| std::cmp::Reverse(b.access_count));
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

/// Standalone cosine similarity (M26). Used by `find_similar` for the
/// admin invalidation path — kept separate from the HNSW distance metric
/// so the sweep is explicit and easy to test. Returns 0 if either vector
/// is the zero vector. Mismatched lengths yield 0 (caller should validate
/// dimension first).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
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
            tombstone: false,
            expires_at: None,
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
            tombstone: false,
            expires_at: None,
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
            tombstone: false,
            expires_at: None,
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
            tombstone: false,
            expires_at: None,
        })
        .unwrap();
        let entry = idx.get_entry_by_uuid("u1").unwrap();
        assert_eq!(entry.inserted_at, 5000);
        assert_eq!(entry.last_accessed_at, 6000);
        assert_eq!(entry.access_count, 42);
    }

    // --- M25: LRU eviction + lazy deletion + rebuild ---------------------

    /// Helper: stamp `last_accessed_at` directly on the entry so eviction
    /// ordering tests don't depend on `now_unix_secs()` time resolution.
    fn set_access_ts(idx: &mut SemanticIndex, model_id: &str, uuid: &str, ts: u64) {
        let ns = idx.namespaces.get_mut(model_id).unwrap();
        let internal_id = *ns.uuid_to_internal.get(uuid).unwrap();
        let entry = ns.entries.get_mut(&internal_id).unwrap();
        entry.last_accessed_at = ts;
    }

    #[test]
    fn test_evict_lru_removes_oldest() {
        let mut idx = test_index();
        let u1 = idx
            .insert(vec![1.0, 0.0, 0.0], "r1".into(), "q1".into(), M)
            .unwrap();
        let u2 = idx
            .insert(vec![0.0, 1.0, 0.0], "r2".into(), "q2".into(), M)
            .unwrap();
        let u3 = idx
            .insert(vec![0.0, 0.0, 1.0], "r3".into(), "q3".into(), M)
            .unwrap();
        // u1 oldest, u2 middle, u3 newest by access time.
        set_access_ts(&mut idx, M, &u1, 100);
        set_access_ts(&mut idx, M, &u2, 200);
        set_access_ts(&mut idx, M, &u3, 300);

        let ns = idx.namespaces.get_mut(M).unwrap();
        let evicted = ns.evict_lru(M).unwrap();
        assert_eq!(evicted.uuid, u1);
        assert_eq!(evicted.model_id, M);
        assert!(!ns.uuid_to_internal.contains_key(&u1));
        // The internal id for u1 was 0 (insertion order), now in evicted_ids.
        assert!(ns.evicted_ids.contains(&0));
    }

    #[test]
    fn test_evict_lru_empty_namespace() {
        let mut idx = test_index();
        // Force the namespace to exist but be empty by inserting then removing.
        let u = idx
            .insert(vec![1.0, 0.0, 0.0], "r".into(), "q".into(), M)
            .unwrap();
        idx.remove_by_uuid(&u);
        let ns = idx.namespaces.get_mut(M).unwrap();
        assert!(ns.evict_lru(M).is_none());
    }

    #[test]
    fn test_query_filters_evicted() {
        let mut idx = test_index();
        let v = vec![1.0_f32, 0.0, 0.0];
        let uuid = idx.insert(v.clone(), "r".into(), "q".into(), M).unwrap();
        // Evict via top-level remove (same effect as evict_lru for this test).
        assert!(idx.remove_by_uuid(&uuid));
        let result = idx.query(&v, 0.9, M).unwrap();
        assert!(
            result.is_none(),
            "query must not return ghost: got {result:?}"
        );
    }

    #[test]
    fn test_query_finds_next_best_after_eviction() {
        let mut idx = test_index();
        // A and B both unit-length; A is identical to the probe.
        let a_vec = vec![1.0_f32, 0.0, 0.0];
        let b_vec = {
            // 70/30 mix with A so cosine similarity is ~0.92 (above 0.90 threshold)
            let v = [0.9_f32, 0.43589, 0.0];
            let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            vec![v[0] / n, v[1] / n, v[2] / n]
        };
        let a = idx
            .insert(a_vec.clone(), "ra".into(), "qa".into(), M)
            .unwrap();
        let _b = idx
            .insert(b_vec.clone(), "rb".into(), "qb".into(), M)
            .unwrap();
        // Evict A.
        idx.remove_by_uuid(&a);
        // Probe at A's vector with a threshold B can clear.
        let hit = idx
            .query(&a_vec, 0.85, M)
            .unwrap()
            .expect("B should be the next-best non-ghost match");
        assert_eq!(hit.response, "rb");
    }

    #[test]
    fn test_needs_rebuild_threshold() {
        let mut idx = test_index();
        // 10 inserts; HNSW dimension auto-detects from the first.
        for i in 0..10u32 {
            let mut v = vec![0.0_f32; 3];
            v[(i % 3) as usize] = 1.0 + (i as f32) * 0.001;
            idx.insert(v, format!("r{i}"), format!("q{i}"), M).unwrap();
        }
        let live_uuids: Vec<String> = idx
            .namespaces
            .get(M)
            .unwrap()
            .entries
            .values()
            .map(|e| e.uuid.clone())
            .collect();

        // Evict 1: ratio = 1/10 = 0.10 → not over.
        idx.remove_by_uuid(&live_uuids[0]);
        assert!(!idx.namespaces.get(M).unwrap().needs_rebuild());
        // Evict 2: ratio = 2/10 = 0.20 (not strictly over).
        idx.remove_by_uuid(&live_uuids[1]);
        assert!(!idx.namespaces.get(M).unwrap().needs_rebuild());
        // Evict 3: ratio = 3/10 = 0.30 → over.
        idx.remove_by_uuid(&live_uuids[2]);
        assert!(idx.namespaces.get(M).unwrap().needs_rebuild());
    }

    #[test]
    fn test_rebuild_clears_evicted_ids() {
        let mut idx = test_index();
        for i in 0..5u32 {
            let mut v = vec![0.0_f32; 3];
            v[(i % 3) as usize] = 1.0 + (i as f32) * 0.01;
            idx.insert(v, format!("r{i}"), format!("q{i}"), M).unwrap();
        }
        let to_evict: Vec<String> = idx
            .namespaces
            .get(M)
            .unwrap()
            .entries
            .values()
            .take(2)
            .map(|e| e.uuid.clone())
            .collect();
        for u in &to_evict {
            idx.remove_by_uuid(u);
        }
        assert_eq!(idx.namespaces.get(M).unwrap().evicted_ids.len(), 2);
        idx.rebuild_dirty_namespaces(&HnswConfig::default());
        // Even though the threshold may not have been crossed, calling
        // rebuild explicitly should clear ghosts. Force it:
        idx.namespaces
            .get_mut(M)
            .unwrap()
            .rebuild(&HnswConfig::default());
        let ns = idx.namespaces.get(M).unwrap();
        assert!(ns.evicted_ids.is_empty());
        assert_eq!(ns.entries.len(), 3);
        // Live entries remain queryable.
        let any_live_uuid = ns.entries.values().next().unwrap().uuid.clone();
        let entry = idx.get_entry_by_uuid(&any_live_uuid).unwrap();
        assert_eq!(entry.uuid, any_live_uuid);
    }

    #[test]
    fn test_rebuild_reassigns_internal_ids() {
        let mut idx = test_index();
        for i in 0..5u32 {
            let mut v = vec![0.0_f32; 3];
            v[(i % 3) as usize] = 1.0 + (i as f32) * 0.01;
            idx.insert(v, format!("r{i}"), format!("q{i}"), M).unwrap();
        }
        let evict_uuid = idx
            .namespaces
            .get(M)
            .unwrap()
            .entries
            .values()
            .next()
            .unwrap()
            .uuid
            .clone();
        idx.remove_by_uuid(&evict_uuid);

        idx.namespaces
            .get_mut(M)
            .unwrap()
            .rebuild(&HnswConfig::default());

        let ns = idx.namespaces.get(M).unwrap();
        // After rebuild, internal IDs are 0..live_count.
        let mut ids: Vec<usize> = ns.entries.keys().copied().collect();
        ids.sort();
        assert_eq!(ids, vec![0, 1, 2, 3]);
        assert_eq!(ns.next_id, 4);
        // uuid_to_internal mappings match entries.
        for (uuid, &id) in &ns.uuid_to_internal {
            assert_eq!(ns.entries.get(&id).unwrap().uuid, *uuid);
        }
    }

    #[test]
    fn test_remove_by_uuid() {
        let mut idx = test_index();
        let u = idx
            .insert(vec![1.0, 0.0, 0.0], "r".into(), "q".into(), M)
            .unwrap();
        let prev_id = *idx
            .namespaces
            .get(M)
            .unwrap()
            .uuid_to_internal
            .get(&u)
            .unwrap();
        assert!(idx.remove_by_uuid(&u));
        let ns = idx.namespaces.get(M).unwrap();
        assert!(!ns.uuid_to_internal.contains_key(&u));
        assert!(!ns.entries.contains_key(&prev_id));
        assert!(ns.evicted_ids.contains(&prev_id));
        assert!(idx.get_entry_by_uuid(&u).is_none());
        // Idempotent — second call returns false.
        assert!(!idx.remove_by_uuid(&u));
    }

    // --- M26: TTL + invalidation -----------------------------------------

    /// Helper: directly stamp `expires_at` on the entry under M.
    fn set_expires_at(idx: &mut SemanticIndex, model_id: &str, uuid: &str, ts: Option<u64>) {
        let ns = idx.namespaces.get_mut(model_id).unwrap();
        let id = *ns.uuid_to_internal.get(uuid).unwrap();
        ns.entries.get_mut(&id).unwrap().expires_at = ts;
    }

    #[test]
    fn test_expired_entry_query_miss() {
        let mut idx = test_index();
        let v = vec![1.0_f32, 0.0, 0.0];
        let u = idx
            .insert(v.clone(), "ephemeral".into(), "q".into(), M)
            .unwrap();
        // Force-expire it 10s in the past.
        let past = now_unix_secs().saturating_sub(10);
        set_expires_at(&mut idx, M, &u, Some(past));
        // Query at the entry's own vector, threshold 0.9 — would hit but
        // for the expiry check, which converts it to a miss.
        let res = idx.query(&v, 0.9, M).unwrap();
        assert!(res.is_none(), "expired entry must miss: {res:?}");
    }

    #[test]
    fn test_non_expired_entry_query_hit() {
        let mut idx = test_index();
        let v = vec![1.0_f32, 0.0, 0.0];
        let u = idx
            .insert(v.clone(), "fresh".into(), "q".into(), M)
            .unwrap();
        let future = now_unix_secs() + 3600;
        set_expires_at(&mut idx, M, &u, Some(future));
        let hit = idx.query(&v, 0.9, M).unwrap().expect("must hit");
        assert_eq!(hit.response, "fresh");
    }

    #[test]
    fn test_no_expiry_entry_never_expires() {
        let mut idx = test_index();
        let v = vec![1.0_f32, 0.0, 0.0];
        idx.insert(v.clone(), "forever".into(), "q".into(), M)
            .unwrap();
        // expires_at defaults to None on insert; querying always hits.
        let hit = idx.query(&v, 0.9, M).unwrap().expect("must hit");
        assert_eq!(hit.response, "forever");
    }

    #[test]
    fn test_collect_expired() {
        let mut idx = test_index();
        let u_expired = idx
            .insert(vec![1.0, 0.0, 0.0], "expired".into(), "q".into(), M)
            .unwrap();
        let u_future = idx
            .insert(vec![0.0, 1.0, 0.0], "future".into(), "q".into(), M)
            .unwrap();
        let u_eternal = idx
            .insert(vec![0.0, 0.0, 1.0], "eternal".into(), "q".into(), M)
            .unwrap();
        let now = now_unix_secs();
        set_expires_at(&mut idx, M, &u_expired, Some(now.saturating_sub(10)));
        set_expires_at(&mut idx, M, &u_future, Some(now + 3600));
        set_expires_at(&mut idx, M, &u_eternal, None);

        let evicted = idx.collect_expired(now);
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].0, u_expired);
        assert_eq!(evicted[0].1, M);
        // Survivors still queryable.
        assert!(idx.get_entry_by_uuid(&u_future).is_some());
        assert!(idx.get_entry_by_uuid(&u_eternal).is_some());
        assert!(idx.get_entry_by_uuid(&u_expired).is_none());
    }

    #[test]
    fn test_find_similar() {
        let mut idx = test_index();
        // Probe vector along +x. A is identical, B is nearby (~0.92 sim),
        // C is orthogonal (sim 0).
        idx.insert(vec![1.0_f32, 0.0, 0.0], "A".into(), "qa".into(), M)
            .unwrap();
        let r2 = std::f32::consts::FRAC_1_SQRT_2;
        idx.insert(vec![r2, r2, 0.0], "B".into(), "qb".into(), M)
            .unwrap();
        idx.insert(vec![0.0_f32, 0.0, 1.0], "C".into(), "qc".into(), M)
            .unwrap();
        let now = now_unix_secs();
        let probe = vec![1.0_f32, 0.0, 0.0];
        let matches = idx.find_similar(&probe, 0.70, M, now);
        // A (1.0) and B (~0.707) — 0.70 picks both, C (0) is excluded.
        let responses: Vec<String> = matches
            .iter()
            .map(|(_, uuid)| idx.get_entry_by_uuid(uuid).unwrap().response)
            .collect();
        assert_eq!(matches.len(), 2);
        assert!(responses.contains(&"A".to_string()));
        assert!(responses.contains(&"B".to_string()));
    }

    #[test]
    fn test_find_similar_filters_evicted() {
        let mut idx = test_index();
        let v = vec![1.0_f32, 0.0, 0.0];
        let u = idx
            .insert(v.clone(), "doomed".into(), "q".into(), M)
            .unwrap();
        idx.remove_by_uuid(&u);
        let now = now_unix_secs();
        let matches = idx.find_similar(&v, 0.5, M, now);
        assert!(matches.is_empty(), "evicted entries must be skipped");
    }

    #[test]
    fn test_cosine_similarity_known_values() {
        // Identical → 1.0
        let a = [1.0_f32, 0.0, 0.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
        // Orthogonal → 0.0
        let b = [0.0_f32, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
        // Mismatched lengths → 0
        let c = [1.0_f32, 0.0];
        assert_eq!(cosine_similarity(&a, &c), 0.0);
        // Zero vector → 0
        let z = [0.0_f32, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &z), 0.0);
    }

    #[test]
    fn test_invalidate_similar() {
        let mut idx = test_index();
        let r2 = std::f32::consts::FRAC_1_SQRT_2;
        let _a = idx
            .insert(vec![1.0_f32, 0.0, 0.0], "A".into(), "qa".into(), M)
            .unwrap();
        let _b = idx
            .insert(vec![r2, r2, 0.0], "B".into(), "qb".into(), M)
            .unwrap();
        let c = idx
            .insert(vec![0.0_f32, 0.0, 1.0], "C".into(), "qc".into(), M)
            .unwrap();
        let now = now_unix_secs();
        let removed = idx.invalidate_similar(&[1.0, 0.0, 0.0], 0.70, M, now);
        assert_eq!(removed.len(), 2);
        // C survives.
        assert!(idx.get_entry_by_uuid(&c).is_some());
        for u in &removed {
            assert!(idx.get_entry_by_uuid(u).is_none());
        }
    }

    #[test]
    fn test_eviction_preserves_most_accessed() {
        let mut idx = test_index();
        let mut uuids: Vec<String> = Vec::new();
        for i in 0..5u32 {
            let mut v = vec![0.0_f32; 3];
            v[(i % 3) as usize] = 1.0 + (i as f32) * 0.01;
            uuids.push(idx.insert(v, format!("r{i}"), format!("q{i}"), M).unwrap());
        }
        // Make uuids[2] the most-recently-accessed; uuids[0] and [1] oldest.
        for (i, u) in uuids.iter().enumerate() {
            set_access_ts(&mut idx, M, u, 100 + i as u64 * 10);
        }
        set_access_ts(&mut idx, M, &uuids[2], 9_999);

        let evicted = idx.evict_to_cap(3);
        assert_eq!(evicted.len(), 2);
        let evicted_uuids: HashSet<String> = evicted.iter().map(|e| e.uuid.clone()).collect();
        // uuids[2] survives; the two oldest (uuids[0], uuids[1]) are evicted.
        assert!(idx.get_entry_by_uuid(&uuids[2]).is_some());
        assert!(evicted_uuids.contains(&uuids[0]));
        assert!(evicted_uuids.contains(&uuids[1]));
        assert!(idx.get_entry_by_uuid(&uuids[0]).is_none());
        assert!(idx.get_entry_by_uuid(&uuids[1]).is_none());
    }

    // --- M27: exact-match pre-filter ---------------------------------------

    #[test]
    fn test_normalize_query_text() {
        // Lowercase + trim + collapse whitespace runs.
        assert_eq!(normalize_query_text("Hello"), "hello");
        assert_eq!(
            normalize_query_text("  What is\tthe  Capital   of  France?  "),
            "what is the capital of france?"
        );
        assert_eq!(normalize_query_text(""), "");
        assert_eq!(normalize_query_text("   "), "");
        assert_eq!(normalize_query_text("a\nb\nc"), "a b c");
    }

    #[test]
    fn test_exact_match_hit() {
        let mut idx = test_index();
        let v = vec![1.0_f32, 0.0, 0.0];
        idx.insert(
            v.clone(),
            "Paris".into(),
            "What is the capital of France?".into(),
            M,
        )
        .unwrap();
        // Query with a totally different embedding — the exact-match
        // pre-filter should still hit because the text matches.
        let now = now_unix_secs();
        let hit = idx
            .query_with_text(
                &[0.0_f32, 0.0, 1.0], // orthogonal — would miss in HNSW
                0.99,
                M,
                Some("What is the capital of France?"),
                now,
            )
            .unwrap()
            .expect("exact-match hit");
        assert!(hit.exact_match);
        assert_eq!(hit.similarity, 1.0);
        assert_eq!(hit.response, "Paris");
    }

    #[test]
    fn test_exact_match_normalization() {
        let mut idx = test_index();
        idx.insert(
            vec![1.0_f32, 0.0, 0.0],
            "Paris".into(),
            "What is the capital of France?".into(),
            M,
        )
        .unwrap();
        let now = now_unix_secs();
        // Different case + extra whitespace + missing punctuation in the
        // wrong place — case + space normalisation should still hit, but
        // missing punctuation alone shouldn't (we don't strip punctuation).
        let hit = idx
            .query_with_text(
                &[0.0_f32, 0.0, 1.0],
                0.99,
                M,
                Some("  WHAT  IS\t the capital   of  france?  "),
                now,
            )
            .unwrap()
            .expect("normalized exact-match");
        assert!(hit.exact_match);
        assert_eq!(hit.response, "Paris");
    }

    #[test]
    fn test_exact_match_miss_falls_through_to_ann() {
        let mut idx = test_index();
        let v = vec![1.0_f32, 0.0, 0.0];
        idx.insert(v.clone(), "answer".into(), "X".into(), M)
            .unwrap();
        let now = now_unix_secs();
        // query_text "Y" doesn't match stored "X", but the embedding
        // matches → ANN hit, exact_match=false.
        let hit = idx
            .query_with_text(&v, 0.9, M, Some("Y"), now)
            .unwrap()
            .expect("ANN hit");
        assert!(!hit.exact_match);
        assert!(hit.similarity > 0.99);
    }

    #[test]
    fn test_exact_match_evicted_falls_through() {
        let mut idx = test_index();
        let v = vec![1.0_f32, 0.0, 0.0];
        let u = idx
            .insert(v.clone(), "r".into(), "evict-me".into(), M)
            .unwrap();
        idx.remove_by_uuid(&u);
        let now = now_unix_secs();
        let result = idx
            .query_with_text(&v, 0.9, M, Some("evict-me"), now)
            .unwrap();
        assert!(result.is_none(), "evicted entry must not match");
    }

    #[test]
    fn test_exact_match_expired_falls_through() {
        let mut idx = test_index();
        let v = vec![1.0_f32, 0.0, 0.0];
        let u = idx
            .insert(v.clone(), "r".into(), "ttl-text".into(), M)
            .unwrap();
        // Force expiry into the past.
        let past = now_unix_secs().saturating_sub(10);
        set_expires_at(&mut idx, M, &u, Some(past));
        let now = now_unix_secs();
        let result = idx
            .query_with_text(&v, 0.9, M, Some("ttl-text"), now)
            .unwrap();
        assert!(result.is_none(), "expired entry must not match");
    }

    #[test]
    fn test_query_without_query_text_uses_ann() {
        let mut idx = test_index();
        let v = vec![1.0_f32, 0.0, 0.0];
        idx.insert(v.clone(), "r".into(), "some text".into(), M)
            .unwrap();
        let now = now_unix_secs();
        // No query_text → exact-match pre-filter is skipped entirely.
        let hit = idx
            .query_with_text(&v, 0.9, M, None, now)
            .unwrap()
            .expect("ANN hit");
        assert!(!hit.exact_match);
    }

    #[test]
    fn test_exact_match_index_cleaned_on_delete() {
        let mut idx = test_index();
        let v = vec![1.0_f32, 0.0, 0.0];
        let u = idx
            .insert(v.clone(), "r".into(), "delete-me".into(), M)
            .unwrap();
        // Sanity: pre-filter hits before delete.
        let now = now_unix_secs();
        assert!(
            idx.query_with_text(&[0.0_f32, 0.0, 1.0], 0.99, M, Some("delete-me"), now)
                .unwrap()
                .is_some()
        );
        // Delete via the user-facing path (writes a tombstone in prod;
        // tests just call the index method directly).
        assert!(idx.remove_by_uuid(&u));
        // After delete, the same probe must miss — exact-match index
        // was cleaned up by `drop_entry`.
        assert!(
            idx.query_with_text(&[0.0_f32, 0.0, 1.0], 0.99, M, Some("delete-me"), now)
                .unwrap()
                .is_none()
        );
        // And the inner map no longer has the key.
        let ns = idx.namespaces.get(M).unwrap();
        assert!(!ns.exact_match_index.contains_key("delete-me"));
    }

    #[test]
    fn test_exact_match_index_ownership_check() {
        // Insert A with text "test", delete A, insert B with same text.
        // The exact-match index must point to B's UUID, not the dead one.
        let mut idx = test_index();
        let a = idx
            .insert(vec![1.0_f32, 0.0, 0.0], "A".into(), "test".into(), M)
            .unwrap();
        idx.remove_by_uuid(&a);
        let b = idx
            .insert(vec![0.0_f32, 1.0, 0.0], "B".into(), "test".into(), M)
            .unwrap();
        let now = now_unix_secs();
        let hit = idx
            .query_with_text(&[0.0_f32, 0.0, 1.0], 0.99, M, Some("test"), now)
            .unwrap()
            .expect("must hit B, not the deleted A");
        assert_eq!(hit.id, b);
        assert_eq!(hit.response, "B");
    }

    #[test]
    fn test_exact_match_survives_multi_path_removal() {
        // M25 evict_lru, M26 collect_expired, and M26 invalidate_similar
        // all need to clean up the exact-match index. Verify each.
        let mut idx = test_index();
        let v1 = vec![1.0_f32, 0.0, 0.0];
        let u_evict = idx
            .insert(v1.clone(), "evict".into(), "evict-text".into(), M)
            .unwrap();
        let u_expire = idx
            .insert(
                vec![0.0_f32, 1.0, 0.0],
                "expire".into(),
                "expire-text".into(),
                M,
            )
            .unwrap();
        let u_inv = idx
            .insert(vec![0.0_f32, 0.0, 1.0], "inv".into(), "inv-text".into(), M)
            .unwrap();

        // Path 1: evict_lru — make u_evict the LRU and call evict_lru.
        set_access_ts(&mut idx, M, &u_evict, 100);
        set_access_ts(&mut idx, M, &u_expire, 200);
        set_access_ts(&mut idx, M, &u_inv, 300);
        idx.namespaces.get_mut(M).unwrap().evict_lru(M).unwrap();

        // Path 2: collect_expired — force u_expire into the past.
        let past = now_unix_secs().saturating_sub(10);
        set_expires_at(&mut idx, M, &u_expire, Some(past));
        idx.collect_expired(now_unix_secs());

        // Path 3: invalidate_similar — match u_inv exactly.
        idx.invalidate_similar(&[0.0_f32, 0.0, 1.0], 0.99, M, now_unix_secs());

        // None of the three texts should be in the exact-match index.
        let ns = idx.namespaces.get(M).unwrap();
        assert!(!ns.exact_match_index.contains_key("evict-text"));
        assert!(!ns.exact_match_index.contains_key("expire-text"));
        assert!(!ns.exact_match_index.contains_key("inv-text"));
    }

    // -- M28: cache_scope + tenant isolation -----------------------------------

    #[test]
    fn test_effective_namespace_function() {
        assert_eq!(effective_namespace("m", None), "m");
        assert_eq!(effective_namespace("m", Some("s")), "m::s");
        // Empty / whitespace-only scope is treated as absent — defensive
        // against callers that pass `cache_scope: ""`.
        assert_eq!(effective_namespace("m", Some("")), "m");
        assert_eq!(effective_namespace("m", Some("  ")), "m");
        assert_eq!(effective_namespace("m", Some("\t\n")), "m");
        // Multi-segment model_id stays intact.
        assert_eq!(
            effective_namespace("all-MiniLM-L6-v2::384", Some("tenant_abc")),
            "all-MiniLM-L6-v2::384::tenant_abc"
        );
    }

    #[test]
    fn test_scoped_insert_creates_separate_namespace() {
        let mut idx = test_index();
        let ns_a = effective_namespace("m::3", Some("tenant_a"));
        let ns_b = effective_namespace("m::3", Some("tenant_b"));
        idx.insert(vec![1.0_f32, 0.0, 0.0], "ra".into(), "qa".into(), &ns_a)
            .unwrap();
        idx.insert(vec![0.0_f32, 1.0, 0.0], "rb".into(), "qb".into(), &ns_b)
            .unwrap();
        assert!(idx.namespaces.contains_key("m::3::tenant_a"));
        assert!(idx.namespaces.contains_key("m::3::tenant_b"));
        assert_eq!(idx.namespaces.len(), 2);
    }

    #[test]
    fn test_scoped_query_isolates() {
        let mut idx = test_index();
        let ns_a = effective_namespace("m::3", Some("tenant_a"));
        let ns_b = effective_namespace("m::3", Some("tenant_b"));
        let v = vec![1.0_f32, 0.0, 0.0];
        idx.insert(v.clone(), "tenant_a's answer".into(), "q".into(), &ns_a)
            .unwrap();
        // tenant_b queries with the same vector — different namespace,
        // namespace doesn't even exist yet, so it's a miss.
        let result = idx.query(&v, 0.50, &ns_b).unwrap();
        assert!(result.is_none(), "cross-tenant leakage: {result:?}");
    }

    #[test]
    fn test_scoped_query_same_scope_hits() {
        let mut idx = test_index();
        let ns_a = effective_namespace("m::3", Some("tenant_a"));
        let v = vec![1.0_f32, 0.0, 0.0];
        let uuid = idx
            .insert(v.clone(), "answer".into(), "q".into(), &ns_a)
            .unwrap();
        let hit = idx.query(&v, 0.50, &ns_a).unwrap().expect("should hit");
        assert_eq!(hit.id, uuid);
        assert_eq!(hit.response, "answer");
    }

    #[test]
    fn test_unscoped_backward_compat() {
        let mut idx = test_index();
        // No scope → effective namespace = model_id verbatim, no `::` suffix.
        let ns = effective_namespace("m::3", None);
        assert_eq!(ns, "m::3");
        let v = vec![1.0_f32, 0.0, 0.0];
        idx.insert(v.clone(), "r".into(), "q".into(), &ns).unwrap();
        let hit = idx.query(&v, 0.50, &ns).unwrap().expect("hit");
        assert_eq!(hit.response, "r");
    }

    #[test]
    fn test_scoped_does_not_see_unscoped() {
        // Inserting WITHOUT scope creates the unscoped namespace. A query
        // WITH scope looks at a different namespace and must miss — the
        // absence of a scope IS a distinct scope.
        let mut idx = test_index();
        let unscoped = effective_namespace("m::3", None);
        let scoped = effective_namespace("m::3", Some("any"));
        let v = vec![1.0_f32, 0.0, 0.0];
        idx.insert(v.clone(), "shared?".into(), "q".into(), &unscoped)
            .unwrap();
        let result = idx.query(&v, 0.50, &scoped).unwrap();
        assert!(result.is_none(), "scoped query saw unscoped entry");
    }
}
