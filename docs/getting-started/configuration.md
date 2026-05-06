# Configuration

FerroCache reads configuration from three sources, in order of priority (highest wins):

1. **Environment variables** with prefix `FERROCACHE_`
2. **`ferrocache.toml`** in the working directory
3. **Defaults**

Nested keys use `__` as a section separator. Lists are comma-separated.

## Core

| Key | Type | Default | Env var |
|---|---|---|---|
| `port` | u16 | `3000` | `FERROCACHE_PORT` |
| `node_id` | string? | random UUID | `FERROCACHE_NODE_ID` |
| `wal_path` | string | `./ferrocache.wal` | `FERROCACHE_WAL_PATH` |
| `auth_token` | string? | `None` (auth off) | `FERROCACHE_AUTH_TOKEN` |

## HNSW (vector index)

| Key | Type | Default | Env var |
|---|---|---|---|
| `hnsw.max_nb_connection` | usize | `16` | `FERROCACHE_HNSW__MAX_NB_CONNECTION` |
| `hnsw.max_elements` | usize | `100000` | `FERROCACHE_HNSW__MAX_ELEMENTS` |
| `hnsw.ef_construction` | usize | `200` | `FERROCACHE_HNSW__EF_CONSTRUCTION` |
| `hnsw.ef_search` | usize | `32` | `FERROCACHE_HNSW__EF_SEARCH` |
| `hnsw.default_threshold` | f32 | `0.92` | `FERROCACHE_HNSW__DEFAULT_THRESHOLD` |
| `hnsw.max_entries_per_namespace` | usize? | `None` (unlimited) | `FERROCACHE_HNSW__MAX_ENTRIES_PER_NAMESPACE` |

`default_threshold` is used when a `/query` request omits its own `threshold`. `max_entries_per_namespace` enables LRU eviction on a per-namespace basis.

## Eviction & TTL

| Key | Type | Default | Env var |
|---|---|---|---|
| `expire_scan_interval_secs` | u64 | `60` | `FERROCACHE_EXPIRE_SCAN_INTERVAL_SECS` |
| `conversation_ttl_seconds` | u64? | `None` (no auto-TTL) | `FERROCACHE_CONVERSATION_TTL_SECONDS` |

The reaper sweeps expired entries on this interval. `conversation_ttl_seconds` stamps an automatic TTL on conversation-scoped inserts when no explicit `ttl_seconds` is provided; explicit always wins.

## Cluster

| Key | Type | Default | Env var |
|---|---|---|---|
| `cluster.enabled` | bool | `false` | `FERROCACHE_CLUSTER__ENABLED` |
| `cluster.gossip_addr` | string | `0.0.0.0:4000` | `FERROCACHE_CLUSTER__GOSSIP_ADDR` |
| `cluster.api_addr` | string | `0.0.0.0:3000` | `FERROCACHE_CLUSTER__API_ADDR` |
| `cluster.seed_nodes` | list<string> | `[]` | `FERROCACHE_CLUSTER__SEED_NODES` (CSV) |
| `cluster.virtual_nodes` | usize | `64` | `FERROCACHE_CLUSTER__VIRTUAL_NODES` |
| `cluster.replication_factor` | usize | `2` | `FERROCACHE_CLUSTER__REPLICATION_FACTOR` |
| `cluster.max_replication_retries` | usize | `3` | `FERROCACHE_CLUSTER__MAX_REPLICATION_RETRIES` |
| `cluster.phi_threshold` | f64 | `8.0` | `FERROCACHE_CLUSTER__PHI_THRESHOLD` |
| `cluster.dead_node_removal_enabled` | bool | `true` | `FERROCACHE_CLUSTER__DEAD_NODE_REMOVAL_ENABLED` |
| `cluster.read_repair_enabled` | bool | `true` | `FERROCACHE_CLUSTER__READ_REPAIR_ENABLED` |

`phi_threshold` controls how aggressive failure detection is — lower values fail nodes faster but raise false-positive risk. `dead_node_removal_enabled = false` runs in monitoring-only canary mode.

## Cluster TLS

| Key | Type | Default | Env var |
|---|---|---|---|
| `cluster.tls.enabled` | bool | `false` | `FERROCACHE_CLUSTER__TLS__ENABLED` |
| `cluster.tls.ca_cert_path` | string? | auto-generated | `FERROCACHE_CLUSTER__TLS__CA_CERT_PATH` |
| `cluster.tls.node_cert_path` | string? | auto-generated | `FERROCACHE_CLUSTER__TLS__NODE_CERT_PATH` |
| `cluster.tls.node_key_path` | string? | auto-generated | `FERROCACHE_CLUSTER__TLS__NODE_KEY_PATH` |
| `cluster.tls.internal_port` | u16 | `port + 1000` | `FERROCACHE_CLUSTER__TLS__INTERNAL_PORT` |

Missing cert paths fall back to in-memory self-signed certs (single-node smoke testing only — peers won't trust each other).

## WAL & performance

| Key | Type | Default | Env var |
|---|---|---|---|
| `wal_batch_size` | usize | `256` | `FERROCACHE_WAL_BATCH_SIZE` |
| `wal_batch_timeout_ms` | u64 | `1` | `FERROCACHE_WAL_BATCH_TIMEOUT_MS` |
| `compact_interval_inserts` | u64 | `10000` | `FERROCACHE_COMPACT_INTERVAL_INSERTS` |

`wal_batch_size = 1` degrades to per-insert fsync (used in tests). The default `256` plus `1ms` batch timeout coalesces concurrent inserts into one fsync per batch.

## Example `ferrocache.toml`

```toml
port = 3000
wal_path = "/data/ferrocache.wal"
expire_scan_interval_secs = 30
conversation_ttl_seconds = 86400

[hnsw]
default_threshold = 0.90
max_entries_per_namespace = 50000

[cluster]
enabled = true
gossip_addr = "0.0.0.0:4000"
api_addr = "node1.cluster.local:3000"
seed_nodes = ["node2.cluster.local:4000", "node3.cluster.local:4000"]
replication_factor = 3
read_repair_enabled = true

[cluster.tls]
enabled = true
ca_cert_path = "/certs/ca.pem"
node_cert_path = "/certs/node1/cert.pem"
node_key_path = "/certs/node1/key.pem"
```

## Equivalent env vars

```bash
export FERROCACHE_PORT=3000
export FERROCACHE_WAL_PATH=/data/ferrocache.wal
export FERROCACHE_HNSW__DEFAULT_THRESHOLD=0.90
export FERROCACHE_HNSW__MAX_ENTRIES_PER_NAMESPACE=50000
export FERROCACHE_CLUSTER__ENABLED=true
export FERROCACHE_CLUSTER__SEED_NODES=node2:4000,node3:4000
export FERROCACHE_CLUSTER__REPLICATION_FACTOR=3
export FERROCACHE_CLUSTER__TLS__ENABLED=true
```
