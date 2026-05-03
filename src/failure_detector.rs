//! Phi accrual failure detector — same algorithm Cassandra and Akka use.
//!
//! Maintains a sliding window of heartbeat inter-arrival times per peer.
//! Phi at time `t` is `-log10(1 - F(t - t_last))` where `F` is the CDF of
//! a normal distribution fit to the observed inter-arrivals. Higher phi
//! means the heartbeat is overdue relative to history; the detector
//! adapts to actual network jitter rather than relying on a fixed timeout.
//!
//! Threshold conventions (Cassandra/Akka default 8.0):
//! - phi ≈ 1  → ~10% chance the peer is down
//! - phi ≈ 3  → ~99.9% chance
//! - phi ≈ 8  → ~99.999999% chance
//!
//! State machine: `Alive → Suspected (phi ≥ threshold) → Dead (phi ≥ 2×threshold)`.
//! Any heartbeat returns the peer to `Alive` — recovery from `Dead` is
//! handled here; ring removal is M22's problem.

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use serde::Serialize;
use tokio::sync::RwLock;

/// Default phi threshold — matches Cassandra/Akka defaults; tuned for LAN
/// gossip at ~1s intervals.
pub const DEFAULT_PHI_THRESHOLD: f64 = 8.0;
pub const DEFAULT_WINDOW_SIZE: usize = 100;
/// Floor on standard deviation. Without it, a perfectly regular heartbeat
/// stream collapses std_dev to 0 and the CDF degenerates to a step function.
pub const DEFAULT_MIN_STD_DEV_MS: f64 = 100.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PeerStatus {
    Alive,
    Suspected,
    Dead,
}

#[derive(Debug)]
struct PeerState {
    /// Inter-arrival times in milliseconds, oldest first.
    heartbeat_history: VecDeque<f64>,
    last_heartbeat: Instant,
    status: PeerStatus,
}

pub struct PhiAccrualDetector {
    peers: RwLock<HashMap<String, PeerState>>,
    phi_threshold: f64,
    window_size: usize,
    min_std_dev_ms: f64,
}

impl PhiAccrualDetector {
    pub fn new(phi_threshold: f64, window_size: usize, min_std_dev_ms: f64) -> Self {
        Self {
            peers: RwLock::new(HashMap::new()),
            phi_threshold,
            window_size: window_size.max(2),
            min_std_dev_ms: min_std_dev_ms.max(1.0),
        }
    }

    pub fn phi_threshold(&self) -> f64 {
        self.phi_threshold
    }

    /// Record a heartbeat at the current instant.
    pub async fn record_heartbeat(&self, node_id: &str) {
        self.record_heartbeat_at(node_id, Instant::now()).await;
    }

    /// Record a heartbeat at an explicit instant. Used by tests to control
    /// the time axis without relying on real wall-clock delays.
    pub async fn record_heartbeat_at(&self, node_id: &str, at: Instant) {
        let mut peers = self.peers.write().await;
        match peers.get_mut(node_id) {
            None => {
                peers.insert(
                    node_id.to_string(),
                    PeerState {
                        heartbeat_history: VecDeque::with_capacity(self.window_size),
                        last_heartbeat: at,
                        status: PeerStatus::Alive,
                    },
                );
            }
            Some(state) => {
                let elapsed = at
                    .saturating_duration_since(state.last_heartbeat)
                    .as_secs_f64()
                    * 1000.0;
                if elapsed > 0.0 {
                    if state.heartbeat_history.len() >= self.window_size {
                        state.heartbeat_history.pop_front();
                    }
                    state.heartbeat_history.push_back(elapsed);
                }
                state.last_heartbeat = at;
                state.status = PeerStatus::Alive;
            }
        }
    }

    /// Compute phi for a peer at the current instant.
    pub async fn phi(&self, node_id: &str) -> Option<f64> {
        self.phi_at(node_id, Instant::now()).await
    }

    /// Compute phi for a peer at an explicit instant. Returns `None` for
    /// unknown peers and for peers with no inter-arrival samples yet
    /// (one heartbeat doesn't establish a distribution).
    pub async fn phi_at(&self, node_id: &str, at: Instant) -> Option<f64> {
        let peers = self.peers.read().await;
        let state = peers.get(node_id)?;
        if state.heartbeat_history.is_empty() {
            return None;
        }
        let elapsed_ms = at
            .saturating_duration_since(state.last_heartbeat)
            .as_secs_f64()
            * 1000.0;
        Some(self.compute_phi(elapsed_ms, &state.heartbeat_history))
    }

    fn compute_phi(&self, elapsed_ms: f64, history: &VecDeque<f64>) -> f64 {
        let n = history.len() as f64;
        let mean: f64 = history.iter().sum::<f64>() / n;
        let variance: f64 = if n > 1.0 {
            history.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n
        } else {
            0.0
        };
        let std_dev = variance.sqrt().max(self.min_std_dev_ms);
        // Phi = -log10(1 - F(elapsed)). When `elapsed` is many σ away from
        // the mean, the erf-based CDF saturates at 1.0 and `1 - cdf` goes
        // to literal zero; floor at MIN_POSITIVE so phi stays finite (caps
        // around 307, far above any realistic threshold).
        let cdf = normal_cdf(elapsed_ms, mean, std_dev);
        let p = (1.0 - cdf).max(f64::MIN_POSITIVE);
        -p.log10()
    }

    /// Recompute phi for every tracked peer at `at`, transitioning each
    /// peer's status if the threshold is crossed.
    ///
    /// State machine:
    /// - `Alive` + phi ≥ threshold       → `Suspected`
    /// - `Suspected` + phi ≥ 2×threshold → `Dead`
    /// - `Suspected`/`Dead` + phi falls below threshold   → stays put until
    ///   a heartbeat arrives (recovery happens in `record_heartbeat_at`)
    ///
    /// Returns the `(status, phi)` pair for each peer. Peers with no
    /// inter-arrival history yet are reported with phi = 0.0.
    pub async fn check_all(&self) -> HashMap<String, (PeerStatus, f64)> {
        self.check_all_at(Instant::now()).await
    }

    pub async fn check_all_at(&self, at: Instant) -> HashMap<String, (PeerStatus, f64)> {
        let mut out = HashMap::new();
        let mut peers = self.peers.write().await;
        for (id, state) in peers.iter_mut() {
            let phi = if state.heartbeat_history.is_empty() {
                0.0
            } else {
                let elapsed_ms = at
                    .saturating_duration_since(state.last_heartbeat)
                    .as_secs_f64()
                    * 1000.0;
                self.compute_phi(elapsed_ms, &state.heartbeat_history)
            };

            match state.status {
                PeerStatus::Alive => {
                    if phi >= self.phi_threshold {
                        state.status = PeerStatus::Suspected;
                    }
                }
                PeerStatus::Suspected => {
                    if phi >= 2.0 * self.phi_threshold {
                        state.status = PeerStatus::Dead;
                    }
                }
                PeerStatus::Dead => {}
            }
            out.insert(id.clone(), (state.status, phi));
        }
        out
    }

    pub async fn remove_peer(&self, node_id: &str) {
        self.peers.write().await.remove(node_id);
    }

    pub async fn peer_statuses(&self) -> HashMap<String, PeerStatus> {
        self.peers
            .read()
            .await
            .iter()
            .map(|(k, v)| (k.clone(), v.status))
            .collect()
    }

    pub async fn alive_peers(&self) -> Vec<String> {
        self.peers
            .read()
            .await
            .iter()
            .filter(|(_, v)| v.status == PeerStatus::Alive)
            .map(|(k, _)| k.clone())
            .collect()
    }

    pub async fn suspected_peers(&self) -> Vec<String> {
        self.peers
            .read()
            .await
            .iter()
            .filter(|(_, v)| v.status == PeerStatus::Suspected)
            .map(|(k, _)| k.clone())
            .collect()
    }

    pub async fn dead_peers(&self) -> Vec<String> {
        self.peers
            .read()
            .await
            .iter()
            .filter(|(_, v)| v.status == PeerStatus::Dead)
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// Snapshot of the detector state suitable for /metrics + /cluster/status.
    /// Returns `(node_id, status, phi)` triples computed at `at`.
    pub async fn snapshot_at(&self, at: Instant) -> Vec<(String, PeerStatus, f64)> {
        let peers = self.peers.read().await;
        let mut out = Vec::with_capacity(peers.len());
        for (id, state) in peers.iter() {
            let phi = if state.heartbeat_history.is_empty() {
                0.0
            } else {
                let elapsed_ms = at
                    .saturating_duration_since(state.last_heartbeat)
                    .as_secs_f64()
                    * 1000.0;
                self.compute_phi(elapsed_ms, &state.heartbeat_history)
            };
            out.push((id.clone(), state.status, phi));
        }
        out
    }

    pub async fn snapshot(&self) -> Vec<(String, PeerStatus, f64)> {
        self.snapshot_at(Instant::now()).await
    }
}

/// CDF of N(mean, std_dev^2) at x.
fn normal_cdf(x: f64, mean: f64, std_dev: f64) -> f64 {
    let z = (x - mean) / std_dev;
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

/// Abramowitz & Stegun 7.1.26 approximation. Max error ≈ 1.5e-7.
fn erf(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn detector() -> PhiAccrualDetector {
        PhiAccrualDetector::new(
            DEFAULT_PHI_THRESHOLD,
            DEFAULT_WINDOW_SIZE,
            DEFAULT_MIN_STD_DEV_MS,
        )
    }

    #[tokio::test]
    async fn test_initial_phi_is_none() {
        let d = detector();
        assert!(d.phi("nobody").await.is_none());
    }

    #[tokio::test]
    async fn test_heartbeat_registers_peer() {
        let d = detector();
        d.record_heartbeat("n1").await;
        let statuses = d.peer_statuses().await;
        assert_eq!(statuses.get("n1"), Some(&PeerStatus::Alive));
    }

    /// Helper: seed a peer with a regular 1s heartbeat history of `n` samples
    /// ending at `last`. Returns `last`, the most recent heartbeat instant.
    async fn seed_regular_heartbeats(
        d: &PhiAccrualDetector,
        node_id: &str,
        n: usize,
        interval: Duration,
        end_at: Instant,
    ) {
        let start = end_at - interval * (n as u32);
        for i in 0..=n {
            d.record_heartbeat_at(node_id, start + interval * (i as u32))
                .await;
        }
    }

    #[tokio::test]
    async fn test_phi_increases_with_silence() {
        let d = detector();
        let now = Instant::now();
        seed_regular_heartbeats(&d, "n1", 10, Duration::from_millis(1000), now).await;

        // Phi just after the last heartbeat: should be near zero.
        let phi_now = d.phi_at("n1", now).await.unwrap();
        // Phi 5 seconds later: should be much higher (5σ-ish given 100ms floor).
        let phi_later = d.phi_at("n1", now + Duration::from_secs(5)).await.unwrap();
        assert!(
            phi_later > phi_now,
            "phi should grow with silence: now={phi_now}, later={phi_later}"
        );
    }

    #[tokio::test]
    async fn test_alive_to_suspected_transition() {
        let d = detector();
        let now = Instant::now();
        seed_regular_heartbeats(&d, "n1", 30, Duration::from_millis(1000), now).await;
        // 30s of silence should easily push phi past 8.0 with 1s mean inter-arrival.
        let statuses = d.check_all_at(now + Duration::from_secs(30)).await;
        let (status, phi) = statuses.get("n1").unwrap();
        assert!(
            *phi >= DEFAULT_PHI_THRESHOLD,
            "phi={phi} should exceed threshold"
        );
        assert_eq!(*status, PeerStatus::Suspected);
    }

    #[tokio::test]
    async fn test_suspected_to_dead_transition() {
        let d = detector();
        let now = Instant::now();
        seed_regular_heartbeats(&d, "n1", 30, Duration::from_millis(1000), now).await;
        // First check pushes Alive → Suspected.
        let _ = d.check_all_at(now + Duration::from_secs(30)).await;
        // Continued silence past 2× threshold pushes Suspected → Dead.
        let statuses = d.check_all_at(now + Duration::from_secs(120)).await;
        let (status, phi) = statuses.get("n1").unwrap();
        assert!(
            *phi >= 2.0 * DEFAULT_PHI_THRESHOLD,
            "phi={phi} should exceed 2× threshold"
        );
        assert_eq!(*status, PeerStatus::Dead);
    }

    #[tokio::test]
    async fn test_recovery_from_suspected() {
        let d = detector();
        let now = Instant::now();
        seed_regular_heartbeats(&d, "n1", 30, Duration::from_millis(1000), now).await;
        let _ = d.check_all_at(now + Duration::from_secs(30)).await;
        assert_eq!(
            d.peer_statuses().await.get("n1"),
            Some(&PeerStatus::Suspected)
        );
        // A new heartbeat snaps the peer back to Alive.
        d.record_heartbeat_at("n1", now + Duration::from_secs(30))
            .await;
        assert_eq!(d.peer_statuses().await.get("n1"), Some(&PeerStatus::Alive));
    }

    #[tokio::test]
    async fn test_recovery_from_dead() {
        let d = detector();
        let now = Instant::now();
        seed_regular_heartbeats(&d, "n1", 30, Duration::from_millis(1000), now).await;
        let _ = d.check_all_at(now + Duration::from_secs(30)).await;
        let _ = d.check_all_at(now + Duration::from_secs(120)).await;
        assert_eq!(d.peer_statuses().await.get("n1"), Some(&PeerStatus::Dead));
        // Even Dead recovers on heartbeat — ring removal is M22's job, not ours.
        d.record_heartbeat_at("n1", now + Duration::from_secs(120))
            .await;
        assert_eq!(d.peer_statuses().await.get("n1"), Some(&PeerStatus::Alive));
    }

    #[tokio::test]
    async fn test_check_all_returns_correct_statuses() {
        // Use a 1s std_dev floor so phi has a measurable gradient between
        // the threshold (8) and 2× threshold (16). With the default 100ms
        // floor, even a 5s silence saturates phi at ~307, jumping over the
        // "Suspected" zone in a single tick.
        let d = PhiAccrualDetector::new(8.0, 100, 1000.0);
        let now = Instant::now();
        // n1: alive, fresh heartbeats ending at `now`.
        seed_regular_heartbeats(&d, "n1", 10, Duration::from_millis(1000), now).await;
        // n2: suspected zone — silent 7s, phi ≈ 9 (between threshold and 2×).
        seed_regular_heartbeats(
            &d,
            "n2",
            10,
            Duration::from_millis(1000),
            now - Duration::from_secs(7),
        )
        .await;
        // n3: dead zone — silent 60s, phi saturates well past 2×.
        seed_regular_heartbeats(
            &d,
            "n3",
            10,
            Duration::from_millis(1000),
            now - Duration::from_secs(60),
        )
        .await;
        // First pass: Alive → Suspected for n2 and n3.
        let _ = d.check_all_at(now).await;
        // Second pass: n3's phi >> 2× threshold so it advances to Dead;
        // n2 stays Suspected because its phi sits in the (threshold, 2×) band.
        let statuses = d.check_all_at(now).await;
        assert_eq!(statuses.get("n1").unwrap().0, PeerStatus::Alive);
        assert_eq!(statuses.get("n2").unwrap().0, PeerStatus::Suspected);
        assert_eq!(statuses.get("n3").unwrap().0, PeerStatus::Dead);
    }

    #[tokio::test]
    async fn test_window_size_bounds() {
        let d = PhiAccrualDetector::new(DEFAULT_PHI_THRESHOLD, 5, DEFAULT_MIN_STD_DEV_MS);
        let now = Instant::now();
        // 20 heartbeats — far more than the 5-sample window.
        for i in 0..20u32 {
            d.record_heartbeat_at("n1", now + Duration::from_millis(1000 * i as u64))
                .await;
        }
        let peers = d.peers.read().await;
        let state = peers.get("n1").unwrap();
        assert!(
            state.heartbeat_history.len() <= 5,
            "history grew beyond window: {}",
            state.heartbeat_history.len()
        );
    }

    #[tokio::test]
    async fn test_remove_peer() {
        let d = detector();
        d.record_heartbeat("n1").await;
        d.remove_peer("n1").await;
        assert!(d.phi("n1").await.is_none());
        assert!(d.peer_statuses().await.is_empty());
    }

    #[test]
    fn test_erf_approximation() {
        assert!((erf(0.0)).abs() < 1e-6);
        assert!((erf(1.0) - 0.8427).abs() < 1e-3);
        assert!((erf(2.0) - 0.9953).abs() < 1e-3);
        assert!((erf(-1.0) + 0.8427).abs() < 1e-3);
    }

    #[test]
    fn test_normal_cdf_known_values() {
        // CDF(mean) = 0.5 exactly.
        assert!((normal_cdf(100.0, 100.0, 10.0) - 0.5).abs() < 1e-6);
        // CDF(mean + 3σ) ≈ 0.9987.
        let v = normal_cdf(130.0, 100.0, 10.0);
        assert!((v - 0.9987).abs() < 1e-3, "cdf+3σ={v}");
    }

    #[tokio::test]
    async fn test_phi_at_unknown_returns_none() {
        let d = detector();
        let now = Instant::now();
        assert!(d.phi_at("ghost", now).await.is_none());
    }

    #[tokio::test]
    async fn test_alive_peers_filters_correctly() {
        let d = detector();
        let now = Instant::now();
        seed_regular_heartbeats(&d, "n1", 5, Duration::from_millis(1000), now).await;
        seed_regular_heartbeats(
            &d,
            "n2",
            5,
            Duration::from_millis(1000),
            now - Duration::from_secs(60),
        )
        .await;
        let _ = d.check_all_at(now).await;
        let alive = d.alive_peers().await;
        let suspected = d.suspected_peers().await;
        assert!(alive.contains(&"n1".to_string()));
        assert!(!alive.contains(&"n2".to_string()));
        assert!(suspected.contains(&"n2".to_string()));
    }
}
