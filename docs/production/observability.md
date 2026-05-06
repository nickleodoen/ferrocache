# Observability

FerroCache exposes a hand-written Prometheus `/metrics` endpoint (no `prometheus` crate dependency) and ships a Grafana dashboard.

## Scraping

```yaml
# prometheus.yml
scrape_configs:
  - job_name: ferrocache
    metrics_path: /metrics
    static_configs:
      - targets:
          - ferrocache-1:3000
          - ferrocache-2:3000
          - ferrocache-3:3000
```

`/metrics` and `/health` stay open even with auth enabled — no bearer token required.

## Counters

| Metric | Meaning |
|---|---|
| `ferrocache_queries_total` | All `/query` requests received |
| `ferrocache_hits_total` | Queries that returned a cached entry |
| `ferrocache_misses_total` | Queries that fell through |
| `ferrocache_exact_match_hits_total` | Hits via the M27 pre-filter |
| `ferrocache_inserts_total` | All `/insert` requests received |
| `ferrocache_evictions_total` | LRU evictions |
| `ferrocache_expirations_total` | TTL expiries |
| `ferrocache_deletions_total` | Explicit `DELETE /entry/:uuid` |
| `ferrocache_invalidations_total` | Radius invalidations |
| `ferrocache_index_rebuilds_total` | HNSW rebuilds (ghost-ratio threshold) |
| `ferrocache_replication_retries_total` | Inter-node forward retries |
| `ferrocache_replication_failures_total` | Inter-node forward final failures |
| `ferrocache_read_repair_inserts_total` | Background read-repair re-inserts |

Each counter exists top-level **and** per-namespace (label `namespace="..."`).

## Gauges

| Metric | Meaning |
|---|---|
| `ferrocache_entry_count` | Total entries across namespaces |
| `ferrocache_namespace_entry_count{namespace="..."}` | Entries per namespace |
| `ferrocache_peer_phi{peer="..."}` | Phi-accrual score per peer |
| `ferrocache_peers_alive` | Count of `Alive` peers |
| `ferrocache_peers_suspected` | Count of `Suspected` peers |
| `ferrocache_peers_dead` | Count of `Dead` peers |
| `ferrocache_ring_size` | Ring entries (`node_count × virtual_nodes`) |

## Histograms

`ferrocache_query_latency_seconds` and `ferrocache_insert_latency_seconds` are exposed as 16-bucket fixed-bucket histograms covering 100µs → 10s.

## What to alert on

| Alert | Condition | Why it matters |
|---|---|---|
| **Hit rate dropped** | `rate(hits[5m]) / rate(queries[5m]) < 0.5` for 10m | Cache thresholds may be too strict, or workload shifted |
| **Replication failures** | `rate(replication_failures_total[5m]) > 0` | Cluster degraded — entries may be under-replicated |
| **Dead peers** | `ferrocache_peers_dead > 0` | A node is offline; `replication_factor - 1` failures away from data loss |
| **Eviction storm** | `rate(evictions_total[1m]) > 100` | Working set exceeds `max_entries_per_namespace`; raise the cap or shed load |
| **Suspected stuck** | `ferrocache_peers_suspected > 0` for 5m | Network blip didn't clear; investigate |

## Grafana dashboard

The repo's `monitoring/` overlay starts Prometheus + Grafana alongside a 3-node FerroCache cluster:

```bash
docker compose -f docker-compose.yml -f monitoring/compose.overlay.yml up -d
# Grafana at http://localhost:3030 (admin/admin)
```

The dashboard has 8 panels:

1. Hit rate (last 5m)
2. Query throughput (queries/s, hits/s, misses/s)
3. Insert throughput
4. Per-namespace entry counts
5. Cluster: peer phi values
6. Cluster: alive / suspected / dead counts
7. Eviction + expiration rates
8. Replication: retries vs failures

Edit `monitoring/grafana/dashboards/ferrocache.json` to customize.

## Per-namespace insight

`/admin/entry-stats` returns the top-10 most-accessed entries per namespace:

```bash
curl http://localhost:3000/admin/entry-stats | python3 -m json.tool
```

```json
{
  "namespaces": {
    "all-MiniLM-L6-v2::384": [
      { "uuid": "...", "access_count": 1284, "last_accessed_at": 1714857600 },
      { "uuid": "...", "access_count": 942,  "last_accessed_at": 1714857512 }
    ]
  }
}
```

Use this to identify hot keys, validate that high-traffic queries are actually cached, and find candidates for explicit pinning (via TTL or per-entry warmup).

## Logging

FerroCache uses `tracing` for structured logging. Configure via `RUST_LOG`:

```bash
export RUST_LOG=ferrocache=info,chitchat=warn,tower_http=warn
```

Useful events to watch:

- `cluster: peer transition Alive → Suspected → Dead` — failure detector firing.
- `replication: degraded` — write fan-out lost a replica; reports `effective_replicas`.
- `index: rebuilding namespace` — ghost ratio crossed 20%.
