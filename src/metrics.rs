use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::failure_detector::PeerStatus;
use crate::index::NamespaceStats;

/// Per-peer phi accrual snapshot for /metrics rendering. Sourced from
/// `PhiAccrualDetector::snapshot()` in the metrics handler.
pub type PeerPhiSnapshot = Vec<(String, PeerStatus, f64)>;

/// Histogram bucket upper bounds (seconds). 100µs → 10s.
pub const BUCKET_BOUNDS: &[f64] = &[
    0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
    5.0, 10.0,
];

pub const METRICS_CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

/// Fixed-bucket histogram with relaxed atomic counters. Cheap to observe;
/// reading is also relaxed and may briefly disagree with `_sum`/`_count`
/// during render, which is fine for a metrics endpoint.
pub struct LatencyHistogram {
    buckets: Vec<(f64, AtomicU64)>,
    /// Sum of observations expressed in microseconds (so a u64 can hold
    /// >500 thousand years of single-second observations).
    sum_micros: AtomicU64,
    count: AtomicU64,
}

impl LatencyHistogram {
    pub fn new() -> Self {
        Self {
            buckets: BUCKET_BOUNDS
                .iter()
                .map(|b| (*b, AtomicU64::new(0)))
                .collect(),
            sum_micros: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    pub fn observe(&self, duration_seconds: f64) {
        let d = if duration_seconds.is_nan() || duration_seconds < 0.0 {
            0.0
        } else {
            duration_seconds
        };
        for (bound, count) in &self.buckets {
            if d <= *bound {
                count.fetch_add(1, Ordering::Relaxed);
            }
        }
        // Microsecond resolution is plenty; saturate at u64::MAX rather than overflow.
        let micros = (d * 1_000_000.0).round() as u64;
        self.sum_micros.fetch_add(micros, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    fn sum_seconds(&self) -> f64 {
        self.sum_micros.load(Ordering::Relaxed) as f64 / 1_000_000.0
    }
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Default)]
pub struct NamespaceMetrics {
    pub queries_hit: AtomicU64,
    pub queries_miss: AtomicU64,
    pub inserts: AtomicU64,
}

pub struct Metrics {
    pub queries_total: AtomicU64,
    pub queries_hit: AtomicU64,
    pub queries_miss: AtomicU64,
    pub inserts_total: AtomicU64,
    pub replication_forwards_total: AtomicU64,
    pub replication_failures_total: AtomicU64,
    pub replication_retries_total: AtomicU64,
    pub compactions_total: AtomicU64,
    /// Cumulative count of ring membership mutations (M22): each add or
    /// remove from `HashRing` driven by the reconciler bumps this once.
    pub ring_changes_total: AtomicU64,
    /// Read repairs completed (M23): an entry was fetched from a replica
    /// after a local miss and successfully re-inserted on this node.
    pub read_repairs_total: AtomicU64,
    /// Read repair attempts that didn't complete (network error, replica
    /// returned 404 for the UUID, WAL channel closed, etc.). Tracked
    /// separately so it can be alerted on without polluting `repairs_total`.
    pub read_repair_failures_total: AtomicU64,
    pub namespace_metrics: RwLock<HashMap<String, NamespaceMetrics>>,
    pub query_duration: LatencyHistogram,
    pub insert_duration: LatencyHistogram,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            queries_total: AtomicU64::new(0),
            queries_hit: AtomicU64::new(0),
            queries_miss: AtomicU64::new(0),
            inserts_total: AtomicU64::new(0),
            replication_forwards_total: AtomicU64::new(0),
            replication_failures_total: AtomicU64::new(0),
            replication_retries_total: AtomicU64::new(0),
            compactions_total: AtomicU64::new(0),
            ring_changes_total: AtomicU64::new(0),
            read_repairs_total: AtomicU64::new(0),
            read_repair_failures_total: AtomicU64::new(0),
            namespace_metrics: RwLock::new(HashMap::new()),
            query_duration: LatencyHistogram::new(),
            insert_duration: LatencyHistogram::new(),
        }
    }

    fn ns_query_hit(&self, namespace: &str) {
        if let Some(ns) = self.namespace_metrics.read().unwrap().get(namespace) {
            ns.queries_hit.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let mut w = self.namespace_metrics.write().unwrap();
        w.entry(namespace.to_string())
            .or_default()
            .queries_hit
            .fetch_add(1, Ordering::Relaxed);
    }

    fn ns_query_miss(&self, namespace: &str) {
        if let Some(ns) = self.namespace_metrics.read().unwrap().get(namespace) {
            ns.queries_miss.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let mut w = self.namespace_metrics.write().unwrap();
        w.entry(namespace.to_string())
            .or_default()
            .queries_miss
            .fetch_add(1, Ordering::Relaxed);
    }

    fn ns_insert(&self, namespace: &str) {
        if let Some(ns) = self.namespace_metrics.read().unwrap().get(namespace) {
            ns.inserts.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let mut w = self.namespace_metrics.write().unwrap();
        w.entry(namespace.to_string())
            .or_default()
            .inserts
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_query_hit(&self, namespace: &str, duration_secs: f64) {
        self.queries_total.fetch_add(1, Ordering::Relaxed);
        self.queries_hit.fetch_add(1, Ordering::Relaxed);
        self.ns_query_hit(namespace);
        self.query_duration.observe(duration_secs);
    }

    pub fn record_query_miss(&self, namespace: &str, duration_secs: f64) {
        self.queries_total.fetch_add(1, Ordering::Relaxed);
        self.queries_miss.fetch_add(1, Ordering::Relaxed);
        self.ns_query_miss(namespace);
        self.query_duration.observe(duration_secs);
    }

    pub fn record_insert(&self, namespace: &str, duration_secs: f64) {
        self.inserts_total.fetch_add(1, Ordering::Relaxed);
        self.ns_insert(namespace);
        self.insert_duration.observe(duration_secs);
    }

    pub fn record_replication_forward(&self) {
        self.replication_forwards_total
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_replication_failure(&self) {
        self.replication_failures_total
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_replication_retry(&self) {
        self.replication_retries_total
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_compaction(&self) {
        self.compactions_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_ring_change(&self) {
        self.ring_changes_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_read_repair(&self) {
        self.read_repairs_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_read_repair_failure(&self) {
        self.read_repair_failures_total
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.queries_total.load(Ordering::Relaxed);
        if total == 0 {
            0.0
        } else {
            self.queries_hit.load(Ordering::Relaxed) as f64 / total as f64
        }
    }

    /// Render Prometheus text-exposition output. `index_stats` carries
    /// per-namespace entry counts (the index is the source of truth, not
    /// our counters, since entries can be inserted via WAL replay).
    /// `cluster_nodes` is the live node count (1 if cluster disabled).
    /// `peer_phi` is the optional per-peer failure-detector snapshot
    /// (M21); empty when cluster is disabled.
    pub fn render(
        &self,
        index_stats: &HashMap<String, NamespaceStats>,
        cluster_nodes: usize,
        peer_phi: &PeerPhiSnapshot,
    ) -> String {
        let mut out = String::with_capacity(4096);

        let queries_total = self.queries_total.load(Ordering::Relaxed);
        let queries_hit = self.queries_hit.load(Ordering::Relaxed);
        let queries_miss = self.queries_miss.load(Ordering::Relaxed);
        let inserts_total = self.inserts_total.load(Ordering::Relaxed);
        let repl_forwards = self.replication_forwards_total.load(Ordering::Relaxed);
        let repl_failures = self.replication_failures_total.load(Ordering::Relaxed);
        let repl_retries = self.replication_retries_total.load(Ordering::Relaxed);
        let compactions = self.compactions_total.load(Ordering::Relaxed);
        let hit_rate = self.hit_rate();

        write_counter(
            &mut out,
            "ferrocache_queries_total",
            "Total number of cache queries.",
            queries_total,
        );
        write_counter(
            &mut out,
            "ferrocache_queries_hit_total",
            "Total cache hits.",
            queries_hit,
        );
        write_counter(
            &mut out,
            "ferrocache_queries_miss_total",
            "Total cache misses.",
            queries_miss,
        );
        write_gauge_f64(
            &mut out,
            "ferrocache_hit_rate",
            "Current cache hit rate (hits / total queries).",
            hit_rate,
        );
        write_counter(
            &mut out,
            "ferrocache_inserts_total",
            "Total number of cache inserts.",
            inserts_total,
        );
        write_counter(
            &mut out,
            "ferrocache_replication_forwards_total",
            "Total replication forwards to peers.",
            repl_forwards,
        );
        write_counter(
            &mut out,
            "ferrocache_replication_failures_total",
            "Total replication failures.",
            repl_failures,
        );
        write_counter(
            &mut out,
            "ferrocache_replication_retries_total",
            "Total replication retry attempts (high values indicate flaky peers).",
            repl_retries,
        );
        write_counter(
            &mut out,
            "ferrocache_compactions_total",
            "Total compaction cycles completed.",
            compactions,
        );
        let ring_changes = self.ring_changes_total.load(Ordering::Relaxed);
        write_counter(
            &mut out,
            "ferrocache_ring_changes_total",
            "Total ring membership changes (adds + removes).",
            ring_changes,
        );
        let read_repairs = self.read_repairs_total.load(Ordering::Relaxed);
        write_counter(
            &mut out,
            "ferrocache_read_repairs_total",
            "Total read repairs completed (entries copied from replica to local).",
            read_repairs,
        );
        let read_repair_failures = self.read_repair_failures_total.load(Ordering::Relaxed);
        write_counter(
            &mut out,
            "ferrocache_read_repair_failures_total",
            "Read repair attempts that failed (replica unreachable, entry missing, etc).",
            read_repair_failures,
        );

        let total_entries: usize = index_stats.values().map(|s| s.entry_count).sum();
        write_gauge_u64(
            &mut out,
            "ferrocache_index_entries",
            "Total entries in the index.",
            total_entries as u64,
        );

        // Sorted output so /metrics is deterministic.
        let mut ns_keys: Vec<&String> = index_stats.keys().collect();
        ns_keys.sort();

        // Per-namespace entry count (gauge from the index)
        let _ = writeln!(
            out,
            "# HELP ferrocache_namespace_entries Entries per namespace."
        );
        let _ = writeln!(out, "# TYPE ferrocache_namespace_entries gauge");
        for k in &ns_keys {
            let stats = &index_stats[*k];
            let _ = writeln!(
                out,
                "ferrocache_namespace_entries{{namespace=\"{}\"}} {}",
                escape_label(k),
                stats.entry_count
            );
        }
        out.push('\n');

        // Per-namespace counters; merge keys we've seen with index_stats keys.
        let ns_metrics = self.namespace_metrics.read().unwrap();
        let mut all_ns: std::collections::BTreeSet<String> = ns_metrics.keys().cloned().collect();
        for k in &ns_keys {
            all_ns.insert((*k).clone());
        }

        let _ = writeln!(
            out,
            "# HELP ferrocache_namespace_queries_hit Cache hits per namespace."
        );
        let _ = writeln!(out, "# TYPE ferrocache_namespace_queries_hit counter");
        for k in &all_ns {
            let v = ns_metrics
                .get(k)
                .map(|n| n.queries_hit.load(Ordering::Relaxed))
                .unwrap_or(0);
            let _ = writeln!(
                out,
                "ferrocache_namespace_queries_hit{{namespace=\"{}\"}} {}",
                escape_label(k),
                v
            );
        }
        out.push('\n');

        let _ = writeln!(
            out,
            "# HELP ferrocache_namespace_queries_miss Cache misses per namespace."
        );
        let _ = writeln!(out, "# TYPE ferrocache_namespace_queries_miss counter");
        for k in &all_ns {
            let v = ns_metrics
                .get(k)
                .map(|n| n.queries_miss.load(Ordering::Relaxed))
                .unwrap_or(0);
            let _ = writeln!(
                out,
                "ferrocache_namespace_queries_miss{{namespace=\"{}\"}} {}",
                escape_label(k),
                v
            );
        }
        out.push('\n');

        let _ = writeln!(
            out,
            "# HELP ferrocache_namespace_inserts Inserts per namespace."
        );
        let _ = writeln!(out, "# TYPE ferrocache_namespace_inserts counter");
        for k in &all_ns {
            let v = ns_metrics
                .get(k)
                .map(|n| n.inserts.load(Ordering::Relaxed))
                .unwrap_or(0);
            let _ = writeln!(
                out,
                "ferrocache_namespace_inserts{{namespace=\"{}\"}} {}",
                escape_label(k),
                v
            );
        }
        out.push('\n');
        drop(ns_metrics);

        write_histogram(
            &mut out,
            "ferrocache_query_duration_seconds",
            "Query latency histogram.",
            &self.query_duration,
        );
        write_histogram(
            &mut out,
            "ferrocache_insert_duration_seconds",
            "Insert latency histogram (includes WAL fsync).",
            &self.insert_duration,
        );

        write_gauge_u64(
            &mut out,
            "ferrocache_cluster_nodes",
            "Number of nodes in the cluster.",
            cluster_nodes as u64,
        );
        write_gauge_u64(
            &mut out,
            "ferrocache_ring_members",
            "Current number of nodes in the hash ring (matches cluster_nodes; \
             distinct gauge name kept for clarity in dashboards).",
            cluster_nodes as u64,
        );

        // Per-peer phi gauge + suspected/dead peer count rollups (M21).
        // Sorted output keeps /metrics deterministic across scrapes.
        let mut peers_sorted: Vec<&(String, PeerStatus, f64)> = peer_phi.iter().collect();
        peers_sorted.sort_by(|a, b| a.0.cmp(&b.0));
        let _ = writeln!(
            out,
            "# HELP ferrocache_peer_phi Phi accrual value per peer (higher = more likely down)."
        );
        let _ = writeln!(out, "# TYPE ferrocache_peer_phi gauge");
        for (id, _, phi) in &peers_sorted {
            let _ = writeln!(
                out,
                "ferrocache_peer_phi{{peer=\"{}\"}} {phi:.6}",
                escape_label(id)
            );
        }
        out.push('\n');

        let suspected = peers_sorted
            .iter()
            .filter(|(_, s, _)| *s == PeerStatus::Suspected)
            .count() as u64;
        let dead = peers_sorted
            .iter()
            .filter(|(_, s, _)| *s == PeerStatus::Dead)
            .count() as u64;
        write_gauge_u64(
            &mut out,
            "ferrocache_peers_suspected",
            "Number of peers currently in suspected state.",
            suspected,
        );
        write_gauge_u64(
            &mut out,
            "ferrocache_peers_dead",
            "Number of peers currently confirmed dead.",
            dead,
        );

        out
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

fn write_counter(out: &mut String, name: &str, help: &str, value: u64) {
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} counter");
    let _ = writeln!(out, "{name} {value}");
    out.push('\n');
}

fn write_gauge_u64(out: &mut String, name: &str, help: &str, value: u64) {
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} gauge");
    let _ = writeln!(out, "{name} {value}");
    out.push('\n');
}

fn write_gauge_f64(out: &mut String, name: &str, help: &str, value: f64) {
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} gauge");
    // Always print with full precision so 0.0 doesn't become an integer.
    let _ = writeln!(out, "{name} {value:.6}");
    out.push('\n');
}

fn write_histogram(out: &mut String, name: &str, help: &str, h: &LatencyHistogram) {
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} histogram");
    for (bound, count) in &h.buckets {
        let _ = writeln!(
            out,
            "{name}_bucket{{le=\"{bound}\"}} {}",
            count.load(Ordering::Relaxed)
        );
    }
    let total = h.count();
    let _ = writeln!(out, "{name}_bucket{{le=\"+Inf\"}} {total}");
    let _ = writeln!(out, "{name}_sum {:.6}", h.sum_seconds());
    let _ = writeln!(out, "{name}_count {total}");
    out.push('\n');
}

fn escape_label(s: &str) -> String {
    // Prometheus label values escape \, ", and \n.
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_stats() -> HashMap<String, NamespaceStats> {
        HashMap::new()
    }

    #[test]
    fn test_record_query_hit_increments() {
        let m = Metrics::new();
        for _ in 0..3 {
            m.record_query_hit("ns::3", 0.001);
        }
        assert_eq!(m.queries_total.load(Ordering::Relaxed), 3);
        assert_eq!(m.queries_hit.load(Ordering::Relaxed), 3);
        assert_eq!(m.queries_miss.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_record_query_miss_increments() {
        let m = Metrics::new();
        m.record_query_miss("ns::3", 0.001);
        m.record_query_miss("ns::3", 0.002);
        assert_eq!(m.queries_total.load(Ordering::Relaxed), 2);
        assert_eq!(m.queries_miss.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_namespace_counters() {
        let m = Metrics::new();
        m.record_query_hit("A::3", 0.001);
        m.record_query_hit("A::3", 0.001);
        m.record_query_miss("B::3", 0.002);
        m.record_insert("A::3", 0.003);
        let r = m.namespace_metrics.read().unwrap();
        assert_eq!(r["A::3"].queries_hit.load(Ordering::Relaxed), 2);
        assert_eq!(r["A::3"].queries_miss.load(Ordering::Relaxed), 0);
        assert_eq!(r["A::3"].inserts.load(Ordering::Relaxed), 1);
        assert_eq!(r["B::3"].queries_miss.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_histogram_observe() {
        let h = LatencyHistogram::new();
        h.observe(0.001); // exactly 1ms
        // Buckets <= 1ms get incremented; lower-than-observation buckets do not.
        // 100µs (0.0001) and 250µs (0.00025) and 500µs (0.0005) are below 1ms — NOT incremented.
        // 1ms (0.001) and above ARE incremented (le-style).
        assert_eq!(h.buckets[0].1.load(Ordering::Relaxed), 0); // 100µs
        assert_eq!(h.buckets[1].1.load(Ordering::Relaxed), 0); // 250µs
        assert_eq!(h.buckets[2].1.load(Ordering::Relaxed), 0); // 500µs
        assert_eq!(h.buckets[3].1.load(Ordering::Relaxed), 1); // 1ms
        assert_eq!(h.buckets[4].1.load(Ordering::Relaxed), 1); // 2.5ms
        assert_eq!(h.buckets.last().unwrap().1.load(Ordering::Relaxed), 1); // 10s
    }

    #[test]
    fn test_histogram_sum_and_count() {
        let h = LatencyHistogram::new();
        h.observe(0.001);
        h.observe(0.002);
        h.observe(0.005);
        assert_eq!(h.count(), 3);
        // Sum is approximately 0.008s, allow some rounding from microsecond storage.
        assert!(
            (h.sum_seconds() - 0.008).abs() < 1e-5,
            "sum={}",
            h.sum_seconds()
        );
    }

    #[test]
    fn test_render_contains_expected_metrics() {
        let m = Metrics::new();
        m.record_query_hit("A::3", 0.001);
        m.record_query_hit("A::3", 0.002);
        m.record_query_miss("A::3", 0.003);
        m.record_insert("A::3", 0.0008);

        let mut stats = HashMap::new();
        stats.insert(
            "A::3".to_string(),
            NamespaceStats {
                entry_count: 7,
                dimension: Some(3),
                oldest_entry_ts: 0,
                newest_entry_ts: 0,
                total_accesses: 0,
            },
        );
        let body = m.render(&stats, 1, &Vec::new());

        assert!(body.contains("ferrocache_queries_total 3"), "{body}");
        assert!(body.contains("ferrocache_queries_hit_total 2"));
        assert!(body.contains("ferrocache_queries_miss_total 1"));
        assert!(body.contains("ferrocache_hit_rate 0.666"));
        assert!(body.contains("ferrocache_inserts_total 1"));
        assert!(
            body.contains("ferrocache_namespace_entries{namespace=\"A::3\"} 7"),
            "namespace gauge missing in:\n{body}"
        );
        assert!(
            body.contains("ferrocache_namespace_queries_hit{namespace=\"A::3\"} 2"),
            "namespace hit missing in:\n{body}"
        );
        assert!(body.contains("ferrocache_query_duration_seconds_bucket"));
        assert!(body.contains("ferrocache_query_duration_seconds_bucket{le=\"+Inf\"} 3"));
        assert!(body.contains("ferrocache_insert_duration_seconds_count 1"));
        assert!(body.contains("ferrocache_cluster_nodes 1"));
    }

    #[test]
    fn test_hit_rate_zero_queries() {
        let m = Metrics::new();
        assert_eq!(m.hit_rate(), 0.0);
        let body = m.render(&empty_stats(), 1, &Vec::new());
        assert!(body.contains("ferrocache_hit_rate 0.000000"));
    }

    #[test]
    fn test_replication_and_compaction_counters() {
        let m = Metrics::new();
        m.record_replication_forward();
        m.record_replication_forward();
        m.record_replication_failure();
        m.record_compaction();
        let body = m.render(&empty_stats(), 1, &Vec::new());
        assert!(body.contains("ferrocache_replication_forwards_total 2"));
        assert!(body.contains("ferrocache_replication_failures_total 1"));
        assert!(body.contains("ferrocache_compactions_total 1"));
    }
}
