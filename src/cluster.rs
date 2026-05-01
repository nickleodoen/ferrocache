use std::collections::BTreeSet;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use chitchat::transport::UdpTransport;
use chitchat::{ChitchatConfig, ChitchatHandle, ChitchatId, FailureDetectorConfig, spawn_chitchat};
use tokio::sync::RwLock;

use crate::config::ClusterConfig;
use crate::ring::HashRing;

const RING_RECONCILE_INTERVAL: Duration = Duration::from_secs(2);
const CLUSTER_ID: &str = "ferrocache";

pub struct ClusterState {
    #[allow(dead_code)] // kept alive so the gossip task isn't dropped
    chitchat_handle: Arc<ChitchatHandle>,
    ring: Arc<RwLock<HashRing>>,
    self_node_id: String,
    gossip_addr: SocketAddr,
}

impl ClusterState {
    pub async fn new(node_id: &str, config: &ClusterConfig) -> Result<Self> {
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

        let handle = spawn_chitchat(chitchat_config, Vec::new(), &UdpTransport)
            .await
            .context("spawn_chitchat failed")?;
        let handle = Arc::new(handle);

        let mut ring = HashRing::new(config.virtual_nodes);
        ring.add_node(node_id);
        let ring = Arc::new(RwLock::new(ring));

        spawn_ring_reconciler(handle.clone(), ring.clone(), node_id.to_string());

        Ok(Self {
            chitchat_handle: handle,
            ring,
            self_node_id: node_id.to_string(),
            gossip_addr,
        })
    }

    pub fn self_node_id(&self) -> &str {
        &self.self_node_id
    }

    pub fn gossip_addr(&self) -> SocketAddr {
        self.gossip_addr
    }

    // M6 will route through these.
    #[allow(dead_code)]
    pub async fn get_target_node(&self, embedding: &[f32]) -> Option<String> {
        self.ring
            .read()
            .await
            .get_node_for_embedding(embedding)
            .map(|s| s.to_string())
    }

    #[allow(dead_code)]
    pub async fn is_local(&self, embedding: &[f32]) -> bool {
        match self.get_target_node(embedding).await {
            Some(target) => target == self.self_node_id,
            None => true,
        }
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
    self_node_id: String,
) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(RING_RECONCILE_INTERVAL);
        interval.tick().await; // first tick fires immediately; skip
        loop {
            interval.tick().await;

            let live: BTreeSet<String> = handle
                .with_chitchat(|cc| {
                    let mut s: BTreeSet<String> =
                        cc.live_nodes().map(|id| id.node_id.clone()).collect();
                    s.insert(cc.self_chitchat_id().node_id.clone());
                    s
                })
                .await;

            let current: BTreeSet<String> = ring.read().await.nodes().into_iter().collect();
            let added: Vec<String> = live.difference(&current).cloned().collect();
            let removed: Vec<String> = current
                .difference(&live)
                .filter(|n| **n != self_node_id)
                .cloned()
                .collect();

            if added.is_empty() && removed.is_empty() {
                continue;
            }

            let mut w = ring.write().await;
            for n in &added {
                w.add_node(n);
            }
            for n in &removed {
                w.remove_node(n);
            }
            drop(w);

            tracing::info!(?added, ?removed, "ring updated");
        }
    });
}
