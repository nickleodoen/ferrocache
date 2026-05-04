use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRequest {
    pub embedding: Vec<f32>,
    pub threshold: f32,
    /// Required since M14 — queries only search the namespace matching this
    /// `model_id`. Modeled as `Option` so missing values produce a 400 error
    /// rather than a serde rejection.
    #[serde(default)]
    pub model_id: Option<String>,
    /// Optional original query text (M27). When present, the server tries
    /// an O(1) exact-match lookup against normalized stored `query_text`
    /// values BEFORE running the HNSW ANN search. Backward-compatible:
    /// pre-M27 callers omit this field and behaviour is unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_text: Option<String>,
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
    /// `Some(true)` if the hit came from the M27 exact-match pre-filter;
    /// `Some(false)` for an HNSW ANN hit; omitted from JSON on miss.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exact_match: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsertRequest {
    pub embedding: Vec<f32>,
    pub response: String,
    pub query_text: String,
    /// Required since M14 — namespace partition for the entry.
    #[serde(default)]
    pub model_id: Option<String>,
    /// When set on a `local=true` request, the receiving node uses this UUID
    /// instead of generating a new one. Used by the coordinator so all
    /// replicas store the same id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uuid: Option<String>,
    /// Per-entry TTL in seconds (M26). When set, the entry expires
    /// `ttl_seconds` after its `inserted_at` timestamp; query hits past
    /// the deadline return miss and a background reaper writes a tombstone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl_seconds: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsertResponse {
    pub id: String,
    pub status: String,
}

/// Wire format for `GET /internal/entry/{uuid}` and what
/// `forward_get_entry` returns. Carries the embedding so the coordinator
/// can re-insert the entry locally during read repair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullEntryResponse {
    pub uuid: String,
    pub embedding: Vec<f32>,
    pub response: String,
    pub query_text: String,
    pub model_id: String,
    /// Access metadata travels with the entry on read repair (M24) so a
    /// re-joined node receives the entry with the replica's access counts.
    /// Old peers that don't send these fields default them to 0.
    #[serde(default)]
    pub inserted_at: u64,
    #[serde(default)]
    pub last_accessed_at: u64,
    #[serde(default)]
    pub access_count: u64,
    /// TTL deadline (M26) — `None` means no expiry. Carried over the wire
    /// so read-repair preserves expiry on the receiving node.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<u64>,
}

/// Wire format for `POST /internal/read-repair`. Same shape as a normal
/// insert but the `uuid` is mandatory — read-repair never invents UUIDs,
/// it copies the existing one from the replica that had the entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadRepairRequest {
    pub uuid: String,
    pub embedding: Vec<f32>,
    pub response: String,
    pub query_text: String,
    pub model_id: String,
}

#[derive(Debug, Serialize)]
pub struct ReadRepairResponse {
    pub status: String,
    pub repaired: bool,
}

/// `DELETE /entry/:uuid` response (M26). `deleted` is `true` for any 200
/// OK; 404 carries an `ErrorResponse` instead.
#[derive(Debug, Serialize)]
pub struct DeleteEntryResponse {
    pub deleted: bool,
}

/// `POST /admin/invalidate` request (M26). Same shape as a query — a
/// vector + threshold + namespace — but evicts every entry whose cosine
/// similarity is at or above the threshold instead of returning the best
/// match.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvalidateRequest {
    pub embedding: Vec<f32>,
    pub threshold: f32,
    #[serde(default)]
    pub model_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InvalidateResponse {
    pub invalidated_count: usize,
    pub uuids: Vec<String>,
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
    pub namespaces: HashMap<String, NamespaceStatsEntry>,
    pub counters: CountersResponse,
}

#[derive(Debug, Serialize)]
pub struct CountersResponse {
    pub queries_total: u64,
    pub queries_hit: u64,
    pub queries_miss: u64,
    pub hit_rate: f64,
    pub inserts_total: u64,
    pub replication_forwards: u64,
    pub replication_failures: u64,
    pub replication_retries: u64,
    pub compactions: u64,
    pub read_repairs: u64,
    pub read_repair_failures: u64,
    /// LRU evictions across all namespaces (M25).
    pub evictions_total: u64,
    /// HNSW rebuilds (M25).
    pub index_rebuilds: u64,
    /// TTL expirations (M26).
    pub expirations_total: u64,
    /// Explicit deletions via `DELETE /entry/:uuid` (M26).
    pub deletions_total: u64,
    /// Semantic invalidations via `POST /admin/invalidate` (M26).
    pub invalidations_total: u64,
    /// M27: queries resolved by the exact-match pre-filter. Additive
    /// to `queries_hit` — every exact-match is also a hit.
    pub exact_match_hits_total: u64,
}

#[derive(Debug, Serialize)]
pub struct StatsHnsw {
    pub max_nb_connection: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub dimension: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct NamespaceStatsEntry {
    pub entry_count: usize,
    pub dimension: Option<usize>,
    /// Access tracking aggregates (M24). All zero when the namespace is empty.
    pub oldest_entry_ts: u64,
    pub newest_entry_ts: u64,
    pub total_accesses: u64,
    /// HNSW ghost nodes (evicted but still in the graph) — M25.
    pub evicted_ghost_count: usize,
}

/// `/admin/entry-stats` per-namespace top-N rollup (M24).
#[derive(Debug, Serialize)]
pub struct EntryStatsResponse {
    pub namespaces: HashMap<String, NamespaceTopEntries>,
}

#[derive(Debug, Serialize)]
pub struct NamespaceTopEntries {
    pub top_entries: Vec<TopEntryRow>,
}

#[derive(Debug, Serialize)]
pub struct TopEntryRow {
    pub uuid: String,
    pub access_count: u64,
    pub last_accessed_at: u64,
    pub query_text_preview: String,
}

#[derive(Debug, Serialize)]
pub struct CompactResponse {
    pub status: String,
    pub entries_snapshotted: usize,
    pub wal_sequence: u64,
}

#[derive(Debug, Serialize)]
pub struct ClusterStatusResponse {
    pub mode: &'static str,
    pub self_node_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gossip_addr: Option<String>,
    pub nodes: Vec<String>,
    pub node_count: usize,
    /// Phi accrual failure detector state (M21). Empty when cluster is
    /// disabled or no peers have been observed yet.
    #[serde(skip_serializing_if = "HashMap::is_empty", default)]
    pub peer_health: HashMap<String, PeerHealth>,
    /// Nodes the failure detector confirmed dead and the reconciler removed
    /// from the ring (M22). They re-enter `nodes` automatically on re-join.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub dead_nodes: Vec<String>,
    /// Whether read-repair (M23) is active for this node. Reflects
    /// `cluster.read_repair_enabled` at config-load time.
    pub read_repair_enabled: bool,
}

#[derive(Debug, Serialize)]
pub struct PeerHealth {
    pub status: crate::failure_detector::PeerStatus,
    pub phi: f64,
}
