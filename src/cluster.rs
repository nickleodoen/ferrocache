use std::collections::{BTreeSet, HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use chitchat::transport::UdpTransport;
use chitchat::{ChitchatConfig, ChitchatHandle, ChitchatId, FailureDetectorConfig, spawn_chitchat};
use tokio::sync::RwLock;

use crate::config::ClusterConfig;
use crate::failure_detector::{PeerStatus, PhiAccrualDetector};
use crate::metrics::Metrics;
use crate::ring::HashRing;

const RING_RECONCILE_INTERVAL: Duration = Duration::from_secs(2);
const CLUSTER_ID: &str = "ferrocache";
const API_ADDR_KEY: &str = "api_addr";

pub struct ClusterState {
    #[allow(dead_code)] // kept alive so the gossip task isn't dropped
    chitchat_handle: Arc<ChitchatHandle>,
    ring: Arc<RwLock<HashRing>>,
    addrs: Arc<RwLock<HashMap<String, String>>>,
    /// Nodes the failure detector has confirmed `Dead` and the reconciler
    /// has removed from the ring (M22). They stay tracked here so we can
    /// (a) report them via `/cluster/status` and (b) recognize a re-join
    /// when chitchat reports them again.
    dead_nodes: Arc<RwLock<HashSet<String>>>,
    self_node_id: String,
    self_api_addr: String,
    gossip_addr: SocketAddr,
    failure_detector: Arc<PhiAccrualDetector>,
}

impl ClusterState {
    /// `forward_addr` is the addr peers should use to reach this node for
    /// replication forwarding. With TLS off, it's `config.api_addr`; with
    /// TLS on, it's `host:internal_port`.
    pub async fn new(
        node_id: &str,
        config: &ClusterConfig,
        forward_addr: &str,
        metrics: Arc<Metrics>,
    ) -> Result<Self> {
        let gossip_addr: SocketAddr = config
            .gossip_addr
            .parse()
            .with_context(|| format!("invalid cluster.gossip_addr: {}", config.gossip_addr))?;

        let generation = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let chitchat_id = ChitchatId::new(node_id.to_string(), generation, gossip_addr);
        let chitchat_config = ChitchatConfig {
            chitchat_id,
            cluster_id: CLUSTER_ID.to_string(),
            gossip_interval: Duration::from_secs(1),
            listen_addr: gossip_addr,
            seed_nodes: config.seed_nodes.clone(),
            failure_detector_config: FailureDetectorConfig::default(),
            marked_for_deletion_grace_period: Duration::from_secs(3_600),
            catchup_callback: None,
            extra_liveness_predicate: None,
        };

        let initial_kvs = vec![(API_ADDR_KEY.to_string(), forward_addr.to_string())];
        let handle = spawn_chitchat(chitchat_config, initial_kvs, &UdpTransport)
            .await
            .context("spawn_chitchat failed")?;
        let handle = Arc::new(handle);

        let mut ring = HashRing::new(config.virtual_nodes);
        ring.add_node(node_id);
        let ring = Arc::new(RwLock::new(ring));

        let mut addrs = HashMap::new();
        addrs.insert(node_id.to_string(), forward_addr.to_string());
        let addrs = Arc::new(RwLock::new(addrs));

        let dead_nodes = Arc::new(RwLock::new(HashSet::new()));

        let failure_detector = Arc::new(PhiAccrualDetector::new(
            config.phi_threshold,
            config.phi_window_size,
            config.phi_min_std_dev_ms,
        ));

        spawn_ring_reconciler(
            handle.clone(),
            ring.clone(),
            addrs.clone(),
            dead_nodes.clone(),
            node_id.to_string(),
            forward_addr.to_string(),
            failure_detector.clone(),
            metrics,
            config.dead_node_removal_enabled,
        );

        Ok(Self {
            chitchat_handle: handle,
            ring,
            addrs,
            dead_nodes,
            self_node_id: node_id.to_string(),
            self_api_addr: forward_addr.to_string(),
            gossip_addr,
            failure_detector,
        })
    }

    pub fn failure_detector(&self) -> &Arc<PhiAccrualDetector> {
        &self.failure_detector
    }

    /// Convenience: status snapshot for a peer. Returns `Alive` for peers
    /// not yet tracked by the detector (no heartbeats observed) so callers
    /// don't need a special "unknown" branch.
    pub async fn peer_status(&self, node_id: &str) -> PeerStatus {
        self.failure_detector
            .peer_statuses()
            .await
            .get(node_id)
            .copied()
            .unwrap_or(PeerStatus::Alive)
    }

    pub fn self_node_id(&self) -> &str {
        &self.self_node_id
    }

    pub fn gossip_addr(&self) -> SocketAddr {
        self.gossip_addr
    }

    pub async fn get_target_addr(&self, embedding: &[f32]) -> Option<(String, String)> {
        let node_id = self
            .ring
            .read()
            .await
            .get_node_for_embedding(embedding)?
            .to_string();
        let addr = self.addrs.read().await.get(&node_id).cloned()?;
        Some((node_id, addr))
    }

    pub async fn get_replica_addrs(
        &self,
        embedding: &[f32],
        replication_factor: usize,
    ) -> Vec<(String, String)> {
        let node_ids = self
            .ring
            .read()
            .await
            .get_n_nodes_for_embedding(embedding, replication_factor);
        let addrs = self.addrs.read().await;
        node_ids
            .into_iter()
            .filter_map(|id| {
                let addr = if id == self.self_node_id {
                    Some(self.self_api_addr.clone())
                } else {
                    addrs.get(&id).cloned()
                };
                addr.map(|a| (id, a))
            })
            .collect()
    }

    pub async fn live_nodes(&self) -> Vec<String> {
        self.ring.read().await.nodes()
    }

    pub async fn ring_node_count(&self) -> usize {
        self.ring.read().await.node_count()
    }

    pub async fn dead_nodes(&self) -> Vec<String> {
        let mut v: Vec<String> = self.dead_nodes.read().await.iter().cloned().collect();
        v.sort();
        v
    }

    /// Snapshot of every peer (`(node_id, api_addr)`) currently in the
    /// ring. Used by admin operations like `DELETE /entry/:uuid` and
    /// `/admin/invalidate`, which fan out to all peers (no embedding to
    /// hash against the ring).
    pub async fn all_peer_addrs(&self) -> Vec<(String, String)> {
        let addrs = self.addrs.read().await;
        addrs
            .iter()
            .filter(|(id, _)| id.as_str() != self.self_node_id())
            .map(|(id, addr)| (id.clone(), addr.clone()))
            .collect()
    }
}

/// One pass of the ring/detector reconciliation. Extracted from the spawned
/// task so tests can drive it with a synthetic chitchat snapshot — no UDP,
/// no gossip handshake, no real time.
///
/// Behaviour (M22):
/// 1. Update the addrs map from the snapshot (peers may re-advertise).
/// 2. Record a heartbeat in the detector for every peer in the snapshot
///    (excluding self). This resets dead peers back to `Alive` if they
///    have re-joined.
/// 3. Add to the ring any peer that's in the snapshot but not yet in the
///    ring. If that peer was previously in `dead_nodes`, treat it as a
///    re-join and remove it from the dead set. Either case bumps the
///    `ring_changes_total` metric.
/// 4. Run `check_all` on the detector. Log peers transitioning to
///    `Suspected` (warn) or `Dead` (error).
/// 5. If `dead_node_removal_enabled`, remove from the ring every peer the
///    detector now reports as `Dead`. Move them into `dead_nodes` and drop
///    them from `addrs`. Bump `ring_changes_total` per removal.
///
/// Notes:
/// - We deliberately do NOT remove peers from the ring just because chitchat
///   dropped them from `live_nodes()`. The detector drives ring membership;
///   chitchat is for discovery + KV propagation only.
/// - We deliberately do NOT call `failure_detector.remove_peer` here. The
///   detector keeps its history so phi can rise on subsequent silence and
///   so a re-joining node's history is preserved (one outlier inter-arrival
///   from the gap, then heartbeats resume).
#[allow(clippy::too_many_arguments)]
pub(crate) async fn reconcile_step(
    snapshot: &HashMap<String, String>,
    self_node_id: &str,
    self_api_addr: &str,
    ring: &Arc<RwLock<HashRing>>,
    addrs: &Arc<RwLock<HashMap<String, String>>>,
    dead_nodes: &Arc<RwLock<HashSet<String>>>,
    failure_detector: &Arc<PhiAccrualDetector>,
    metrics: Option<&Arc<Metrics>>,
    dead_node_removal_enabled: bool,
) {
    // Step 1: addrs map. Always reflect the latest snapshot, including self.
    {
        let mut a = addrs.write().await;
        for (id, addr) in snapshot {
            a.insert(id.clone(), addr.clone());
        }
        a.insert(self_node_id.to_string(), self_api_addr.to_string());
    }

    // Step 2: heartbeat every non-self peer in the snapshot. Resets Dead
    // peers to Alive in the detector (re-join); creates entries for new
    // peers (with `Alive` and no inter-arrival history yet).
    for id in snapshot.keys() {
        if id != self_node_id {
            failure_detector.record_heartbeat(id).await;
        }
    }

    // Step 3: discover. Anything in snapshot not in ring → add. If it was
    // in dead_nodes, treat as a re-join.
    let to_add: Vec<String> = {
        let r = ring.read().await;
        let current: BTreeSet<String> = r.nodes().into_iter().collect();
        snapshot
            .keys()
            .filter(|id| !current.contains(id.as_str()))
            .cloned()
            .collect()
    };
    if !to_add.is_empty() {
        {
            let mut w = ring.write().await;
            for id in &to_add {
                w.add_node(id);
            }
        }
        let mut rejoined: Vec<String> = Vec::new();
        {
            let mut d = dead_nodes.write().await;
            for id in &to_add {
                if d.remove(id) {
                    rejoined.push(id.clone());
                }
            }
        }
        for id in &rejoined {
            tracing::info!(peer = %id, "node re-joined cluster, added back to ring");
        }
        let new_only: Vec<&String> = to_add.iter().filter(|i| !rejoined.contains(i)).collect();
        if !new_only.is_empty() {
            tracing::info!(?new_only, "new nodes added to ring");
        }
        if let Some(m) = metrics {
            for _ in 0..to_add.len() {
                m.record_ring_change();
            }
        }
    }

    // Step 4: detector tick. Logs only — ring removals happen below.
    let statuses = failure_detector.check_all().await;
    for (id, (status, phi)) in &statuses {
        match status {
            PeerStatus::Suspected => {
                tracing::warn!(peer = %id, phi, "peer suspected down");
            }
            PeerStatus::Dead => {
                tracing::error!(peer = %id, phi, "peer confirmed dead");
            }
            PeerStatus::Alive => {}
        }
    }

    // Step 5: dead-driven ring removal. Suspected peers stay — only Dead.
    if dead_node_removal_enabled {
        let to_remove: Vec<String> = {
            let r = ring.read().await;
            let current: BTreeSet<String> = r.nodes().into_iter().collect();
            statuses
                .iter()
                .filter(|(id, (status, _))| {
                    *status == PeerStatus::Dead && current.contains(id.as_str())
                })
                .map(|(id, _)| id.clone())
                .collect()
        };
        if !to_remove.is_empty() {
            {
                let mut w = ring.write().await;
                for id in &to_remove {
                    w.remove_node(id);
                }
            }
            {
                let mut a = addrs.write().await;
                for id in &to_remove {
                    a.remove(id);
                }
            }
            {
                let mut d = dead_nodes.write().await;
                for id in &to_remove {
                    d.insert(id.clone());
                }
            }
            for id in &to_remove {
                tracing::error!(peer = %id, "node declared dead, removed from ring");
            }
            if let Some(m) = metrics {
                for _ in 0..to_remove.len() {
                    m.record_ring_change();
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_ring_reconciler(
    handle: Arc<ChitchatHandle>,
    ring: Arc<RwLock<HashRing>>,
    addrs: Arc<RwLock<HashMap<String, String>>>,
    dead_nodes: Arc<RwLock<HashSet<String>>>,
    self_node_id: String,
    self_api_addr: String,
    failure_detector: Arc<PhiAccrualDetector>,
    metrics: Arc<Metrics>,
    dead_node_removal_enabled: bool,
) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(RING_RECONCILE_INTERVAL);
        interval.tick().await; // first tick fires immediately; skip
        loop {
            interval.tick().await;

            // Snapshot live members + their api_addrs from chitchat.
            let snapshot: HashMap<String, String> = handle
                .with_chitchat(|cc| {
                    let mut m = HashMap::new();
                    for chat_id in cc.live_nodes() {
                        if let Some(state) = cc.node_state(chat_id)
                            && let Some(addr) = state.get(API_ADDR_KEY)
                        {
                            m.insert(chat_id.node_id.clone(), addr.to_string());
                        }
                    }
                    m.insert(self_node_id.clone(), self_api_addr.clone());
                    m
                })
                .await;

            reconcile_step(
                &snapshot,
                &self_node_id,
                &self_api_addr,
                &ring,
                &addrs,
                &dead_nodes,
                &failure_detector,
                Some(&metrics),
                dead_node_removal_enabled,
            )
            .await;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    /// Build the routing-side data structures directly. The chitchat handle
    /// isn't needed for replica lookups, so tests bypass `ClusterState::new`
    /// (which would require a UDP socket + gossip handshake) and exercise
    /// the lookup logic against the raw maps.
    type SharedRing = Arc<RwLock<HashRing>>;
    type SharedAddrs = Arc<RwLock<HashMap<String, String>>>;

    fn ring_and_addrs(
        self_id: &str,
        self_addr: &str,
        peers: &[(&str, &str)],
    ) -> (SharedRing, SharedAddrs) {
        let mut ring = HashRing::new(64);
        let mut addrs = HashMap::new();
        ring.add_node(self_id);
        addrs.insert(self_id.to_string(), self_addr.to_string());
        for (id, addr) in peers {
            ring.add_node(id);
            addrs.insert((*id).to_string(), (*addr).to_string());
        }
        (Arc::new(RwLock::new(ring)), Arc::new(RwLock::new(addrs)))
    }

    async fn replica_lookup(
        ring: &SharedRing,
        addrs: &SharedAddrs,
        embedding: &[f32],
        n: usize,
    ) -> Vec<(String, String)> {
        let ids = ring.read().await.get_n_nodes_for_embedding(embedding, n);
        let a = addrs.read().await;
        ids.into_iter()
            .filter_map(|id| a.get(&id).cloned().map(|addr| (id, addr)))
            .collect()
    }

    #[tokio::test]
    async fn test_get_replica_addrs_deduplicates() {
        let (ring, addrs) = ring_and_addrs("A", "127.0.0.1:1", &[("B", "127.0.0.1:2")]);
        let result = replica_lookup(&ring, &addrs, &[0.5_f32, 0.5], 2).await;
        assert_eq!(result.len(), 2);
        let ids: BTreeSet<&str> = result.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ids.len(), 2, "distinct node_ids: {result:?}");
    }

    #[tokio::test]
    async fn test_get_replica_addrs_wraps_around() {
        let (ring, addrs) = ring_and_addrs(
            "A",
            "127.0.0.1:1",
            &[("B", "127.0.0.1:2"), ("C", "127.0.0.1:3")],
        );
        for k in 0..30u64 {
            let emb = [k as f32 * 0.7, (k as f32) * -0.3];
            let result = replica_lookup(&ring, &addrs, &emb, 2).await;
            assert_eq!(result.len(), 2, "k={k}");
            assert_ne!(result[0].0, result[1].0);
        }
    }

    // --- M22: reconcile_step state machine ---------------------------------

    type SharedDead = Arc<RwLock<HashSet<String>>>;
    type SharedDetector = Arc<PhiAccrualDetector>;
    type SharedMetrics = Arc<Metrics>;
    type ReconcileFixture = (
        SharedRing,
        SharedAddrs,
        SharedDead,
        SharedDetector,
        SharedMetrics,
    );

    fn build_state() -> ReconcileFixture {
        let ring = Arc::new(RwLock::new(HashRing::new(64)));
        let addrs = Arc::new(RwLock::new(HashMap::new()));
        let dead = Arc::new(RwLock::new(HashSet::new()));
        let det = Arc::new(PhiAccrualDetector::new(8.0, 100, 100.0));
        let m = Arc::new(Metrics::new());
        (ring, addrs, dead, det, m)
    }

    fn snap(items: &[(&str, &str)]) -> HashMap<String, String> {
        items
            .iter()
            .map(|(id, addr)| (id.to_string(), addr.to_string()))
            .collect()
    }

    #[tokio::test]
    async fn test_reconcile_adds_new_nodes() {
        let (ring, addrs, dead, det, m) = build_state();
        let s = snap(&[("A", "127.0.0.1:1"), ("B", "127.0.0.1:2")]);
        reconcile_step(
            &s,
            "A",
            "127.0.0.1:1",
            &ring,
            &addrs,
            &dead,
            &det,
            Some(&m),
            true,
        )
        .await;
        let nodes = ring.read().await.nodes();
        assert!(nodes.contains(&"A".to_string()));
        assert!(nodes.contains(&"B".to_string()));
        assert_eq!(
            addrs.read().await.get("B").map(String::as_str),
            Some("127.0.0.1:2")
        );
        assert_eq!(
            m.ring_changes_total
                .load(std::sync::atomic::Ordering::Relaxed),
            2
        );
    }

    /// Drive a peer's phi accrual from healthy to Dead by recording N
    /// regular heartbeats and then a long silence in the detector before
    /// calling `check_all` twice (Alive→Suspected, Suspected→Dead).
    async fn drive_to_dead(det: &PhiAccrualDetector, peer: &str) {
        let base = Instant::now();
        for i in 0..30u32 {
            det.record_heartbeat_at(peer, base + Duration::from_millis(1000 * i as u64))
                .await;
        }
        let silence = base + Duration::from_secs(120);
        let _ = det.check_all_at(silence).await;
        let _ = det.check_all_at(silence).await;
    }

    #[tokio::test]
    async fn test_dead_node_removed_from_ring() {
        let (ring, addrs, dead, det, m) = build_state();
        // Seed a 3-node ring directly so we can isolate the removal logic.
        {
            let mut w = ring.write().await;
            for id in ["A", "B", "C"] {
                w.add_node(id);
            }
        }
        {
            let mut a = addrs.write().await;
            a.insert("A".into(), "127.0.0.1:1".into());
            a.insert("B".into(), "127.0.0.1:2".into());
            a.insert("C".into(), "127.0.0.1:3".into());
        }
        // Push C to Dead in the detector.
        drive_to_dead(&det, "C").await;
        // Reconciler tick: chitchat reports only A and B (C dropped from gossip).
        let s = snap(&[("A", "127.0.0.1:1"), ("B", "127.0.0.1:2")]);
        reconcile_step(
            &s,
            "A",
            "127.0.0.1:1",
            &ring,
            &addrs,
            &dead,
            &det,
            Some(&m),
            true,
        )
        .await;

        let nodes = ring.read().await.nodes();
        assert!(!nodes.contains(&"C".to_string()), "C must be removed");
        assert!(nodes.contains(&"A".to_string()));
        assert!(nodes.contains(&"B".to_string()));
        assert!(dead.read().await.contains("C"));
        assert!(!addrs.read().await.contains_key("C"), "addr scrubbed");
        assert!(
            m.ring_changes_total
                .load(std::sync::atomic::Ordering::Relaxed)
                >= 1,
            "ring change recorded"
        );
    }

    #[tokio::test]
    async fn test_suspected_stays_in_ring() {
        // Override the default detector with a wider std-dev floor so phi
        // settles in the (threshold, 2× threshold) band rather than
        // saturating past 2×.
        let (ring, addrs, dead, _default_det, m) = build_state();
        {
            let mut w = ring.write().await;
            for id in ["A", "B", "C"] {
                w.add_node(id);
            }
        }
        let det = Arc::new(PhiAccrualDetector::new(8.0, 100, 1000.0));
        let base = Instant::now();
        for i in 0..10u32 {
            det.record_heartbeat_at("C", base + Duration::from_millis(1000 * i as u64))
                .await;
        }
        let _ = det.check_all_at(base + Duration::from_secs(7)).await;
        let s = snap(&[("A", "127.0.0.1:1"), ("B", "127.0.0.1:2")]);
        reconcile_step(
            &s,
            "A",
            "127.0.0.1:1",
            &ring,
            &addrs,
            &dead,
            &det,
            Some(&m),
            true,
        )
        .await;

        let nodes = ring.read().await.nodes();
        assert!(nodes.contains(&"C".to_string()), "Suspected stays in ring");
        assert!(!dead.read().await.contains("C"));
    }

    #[tokio::test]
    async fn test_dead_node_rejoin() {
        // The detector starts with no entry for C — `record_heartbeat`
        // inside reconcile_step is what creates it and sets it to Alive.
        let (ring, addrs, dead, det, m) = build_state();
        // Pre-state: C was removed because it died.
        {
            let mut w = ring.write().await;
            w.add_node("A");
            w.add_node("B");
        }
        dead.write().await.insert("C".to_string());

        // C re-appears in chitchat's snapshot (process restarted).
        let s = snap(&[
            ("A", "127.0.0.1:1"),
            ("B", "127.0.0.1:2"),
            ("C", "127.0.0.1:3"),
        ]);
        reconcile_step(
            &s,
            "A",
            "127.0.0.1:1",
            &ring,
            &addrs,
            &dead,
            &det,
            Some(&m),
            true,
        )
        .await;

        let nodes = ring.read().await.nodes();
        assert!(nodes.contains(&"C".to_string()), "C must rejoin ring");
        assert!(!dead.read().await.contains("C"), "C cleared from dead set");
    }

    #[tokio::test]
    async fn test_dead_node_removal_disabled() {
        let (ring, addrs, dead, det, m) = build_state();
        {
            let mut w = ring.write().await;
            for id in ["A", "B", "C"] {
                w.add_node(id);
            }
        }
        drive_to_dead(&det, "C").await;
        // dead_node_removal_enabled = false: monitoring works, ring stays static.
        let s = snap(&[("A", "127.0.0.1:1"), ("B", "127.0.0.1:2")]);
        reconcile_step(
            &s,
            "A",
            "127.0.0.1:1",
            &ring,
            &addrs,
            &dead,
            &det,
            Some(&m),
            false,
        )
        .await;

        let nodes = ring.read().await.nodes();
        assert!(
            nodes.contains(&"C".to_string()),
            "C stays under monitoring-only"
        );
        assert!(
            !dead.read().await.contains("C"),
            "no entry in dead_nodes when removal disabled"
        );
        // Phi state still observable via the detector.
        let statuses = det.peer_statuses().await;
        assert_eq!(statuses.get("C"), Some(&PeerStatus::Dead));
    }

    #[tokio::test]
    async fn test_ring_change_metric() {
        let (ring, addrs, dead, det, m) = build_state();
        // First tick: ring is empty, so self + A are both adds (2 changes).
        let s1 = snap(&[("self", "127.0.0.1:0"), ("A", "127.0.0.1:1")]);
        reconcile_step(
            &s1,
            "self",
            "127.0.0.1:0",
            &ring,
            &addrs,
            &dead,
            &det,
            Some(&m),
            true,
        )
        .await;
        // Second tick: B is the only new node (+1 change).
        let s2 = snap(&[
            ("self", "127.0.0.1:0"),
            ("A", "127.0.0.1:1"),
            ("B", "127.0.0.1:2"),
        ]);
        reconcile_step(
            &s2,
            "self",
            "127.0.0.1:0",
            &ring,
            &addrs,
            &dead,
            &det,
            Some(&m),
            true,
        )
        .await;
        assert_eq!(
            m.ring_changes_total
                .load(std::sync::atomic::Ordering::Relaxed),
            3,
            "self + A + B = 3 adds"
        );

        // Drive-to-Dead removes A (+1 change).
        drive_to_dead(&det, "A").await;
        let s3 = snap(&[("self", "127.0.0.1:0"), ("B", "127.0.0.1:2")]);
        reconcile_step(
            &s3,
            "self",
            "127.0.0.1:0",
            &ring,
            &addrs,
            &dead,
            &det,
            Some(&m),
            true,
        )
        .await;
        assert_eq!(
            m.ring_changes_total
                .load(std::sync::atomic::Ordering::Relaxed),
            4,
            "increment for the removal"
        );
    }

    #[tokio::test]
    async fn test_query_routes_to_replica_after_failover() {
        // Pick an embedding such that with replication factor 2, the
        // primary is whichever node the ring assigns first. Walk the ring
        // BEFORE removing C to record (primary, replica). Remove C. Walk
        // again and assert the new primary is the original replica.
        let mut r = HashRing::new(64);
        for id in ["A", "B", "C"] {
            r.add_node(id);
        }
        // Find an embedding whose primary is C.
        let mut target = None;
        let mut secondary = None;
        for k in 0..2000u32 {
            let emb = [k as f32 * 0.013, (k as f32) * -0.027];
            let walk = r.get_n_nodes_for_embedding(&emb, 2);
            if walk.first().map(String::as_str) == Some("C") {
                target = Some(emb);
                secondary = walk.get(1).cloned();
                break;
            }
        }
        let target = target.expect("found an embedding routing to C as primary");
        let secondary = secondary.expect("replica walk produces a 2nd node");
        // Remove C — its arc folds into the clockwise successor.
        r.remove_node("C");
        let new_primary = r.get_node_for_embedding(&target).unwrap();
        assert_ne!(new_primary, "C");
        assert_eq!(
            new_primary, secondary,
            "post-failover primary must equal the pre-failover replica (where the data lives)"
        );
    }
}
