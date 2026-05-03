use std::collections::{BTreeSet, HashMap};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use chitchat::transport::UdpTransport;
use chitchat::{ChitchatConfig, ChitchatHandle, ChitchatId, FailureDetectorConfig, spawn_chitchat};
use tokio::sync::RwLock;

use crate::config::ClusterConfig;
use crate::failure_detector::{PeerStatus, PhiAccrualDetector};
use crate::ring::HashRing;

const RING_RECONCILE_INTERVAL: Duration = Duration::from_secs(2);
const CLUSTER_ID: &str = "ferrocache";
const API_ADDR_KEY: &str = "api_addr";

pub struct ClusterState {
    #[allow(dead_code)] // kept alive so the gossip task isn't dropped
    chitchat_handle: Arc<ChitchatHandle>,
    ring: Arc<RwLock<HashRing>>,
    addrs: Arc<RwLock<HashMap<String, String>>>,
    self_node_id: String,
    self_api_addr: String,
    gossip_addr: SocketAddr,
    failure_detector: Arc<PhiAccrualDetector>,
}

impl ClusterState {
    /// `forward_addr` is the addr peers should use to reach this node for
    /// replication forwarding. With TLS off, it's `config.api_addr`; with
    /// TLS on, it's `host:internal_port`. The chitchat KV key stays
    /// `api_addr` regardless — semantically it's "addr for peer forwards",
    /// and a homogeneous cluster (all-TLS or none-TLS) means peers consume
    /// it the same way.
    pub async fn new(node_id: &str, config: &ClusterConfig, forward_addr: &str) -> Result<Self> {
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

        let failure_detector = Arc::new(PhiAccrualDetector::new(
            config.phi_threshold,
            config.phi_window_size,
            config.phi_min_std_dev_ms,
        ));

        spawn_ring_reconciler(
            handle.clone(),
            ring.clone(),
            addrs.clone(),
            node_id.to_string(),
            forward_addr.to_string(),
            failure_detector.clone(),
        );

        Ok(Self {
            chitchat_handle: handle,
            ring,
            addrs,
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
}

fn spawn_ring_reconciler(
    handle: Arc<ChitchatHandle>,
    ring: Arc<RwLock<HashRing>>,
    addrs: Arc<RwLock<HashMap<String, String>>>,
    self_node_id: String,
    self_api_addr: String,
    failure_detector: Arc<PhiAccrualDetector>,
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

            // Heartbeat on every gossip tick: each peer chitchat reports as
            // live counts as one heartbeat. The phi calculation uses these
            // inter-arrivals so a missed tick climbs phi at the rate
            // observed jitter says is unusual.
            for node_id in snapshot.keys() {
                if node_id != &self_node_id {
                    failure_detector.record_heartbeat(node_id).await;
                }
            }

            // Recompute phi for every tracked peer AFTER recording — peers
            // that just heartbeated have phi reset; peers that didn't are
            // the ones that may transition to Suspected/Dead.
            let statuses = failure_detector.check_all().await;
            for (node_id, (status, phi)) in &statuses {
                match status {
                    PeerStatus::Suspected => {
                        tracing::warn!(peer = %node_id, phi = phi, "peer suspected down");
                    }
                    PeerStatus::Dead => {
                        tracing::error!(peer = %node_id, phi = phi, "peer confirmed dead");
                    }
                    PeerStatus::Alive => {}
                }
            }

            let live: BTreeSet<String> = snapshot.keys().cloned().collect();
            let current: BTreeSet<String> = ring.read().await.nodes().into_iter().collect();
            let added: Vec<String> = live.difference(&current).cloned().collect();
            let removed: Vec<String> = current
                .difference(&live)
                .filter(|n| **n != self_node_id)
                .cloned()
                .collect();

            // Always update the addrs map (peers may have re-advertised).
            {
                let mut a = addrs.write().await;
                for (id, addr) in &snapshot {
                    a.insert(id.clone(), addr.clone());
                }
                for id in &removed {
                    a.remove(id);
                }
            }

            // Stop tracking peers chitchat dropped from the ring (M22 will
            // handle the reverse: keeping them in the ring while marking
            // them dead until reassignment).
            for id in &removed {
                failure_detector.remove_peer(id).await;
            }

            if added.is_empty() && removed.is_empty() {
                continue;
            }

            {
                let mut w = ring.write().await;
                for n in &added {
                    w.add_node(n);
                }
                for n in &removed {
                    w.remove_node(n);
                }
            }

            tracing::info!(?added, ?removed, "ring updated");
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
