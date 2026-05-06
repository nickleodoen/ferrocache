# Cluster Setup

Run a 3-node cluster when you need the cache to survive a node failure without any application-side changes. Cluster mode is opt-in (`cluster.enabled = true`); single-node behaviour is bit-identical when it's off.

## What you get

- **Consistent hashing** — keys route to a deterministic owner; adding/removing a node only remaps `1/N` of keys.
- **Synchronous replication** — `replication_factor` (default 2) copies of every entry; coordinator stamps the UUID and fans out.
- **Gossip discovery** (chitchat) — no Zookeeper, no etcd, no coordinator. Ring updates propagate in ~2s.
- **Phi-accrual failure detection** — Cassandra-style. Failed nodes' ring arcs fold to their replica neighbour automatically.
- **Read repair** — coordinator misses fan out to replicas in parallel; the first hit is returned and durably re-inserted.

## Quick start (docker-compose)

The repo ships a 3-node `docker-compose.yml`:

```bash
git clone https://github.com/nickleodoen/ferrocache
cd ferrocache
docker compose up -d --build
sleep 5
curl http://localhost:3001/cluster/status
docker compose down -v
```

External ports `3001` / `3002` / `3003` map to the three nodes. An insert sent to any node is replicated to `replication_factor` owners along the ring; a query sent to any node is forwarded to the owning shard.

## docker-compose walkthrough

The service definition (one of three identical-shaped entries):

```yaml
services:
  ferrocache-1:
    build: .
    ports: ["3001:3000"]
    environment:
      FERROCACHE_NODE_ID: node1
      FERROCACHE_CLUSTER__ENABLED: "true"
      FERROCACHE_CLUSTER__GOSSIP_ADDR: "0.0.0.0:4000"
      FERROCACHE_CLUSTER__API_ADDR: "ferrocache-1:3000"
      FERROCACHE_CLUSTER__SEED_NODES: "ferrocache-2:4000,ferrocache-3:4000"
      FERROCACHE_CLUSTER__REPLICATION_FACTOR: "2"
    volumes:
      - ./data/node1:/data
```

Each node:

- Advertises its `api_addr` via chitchat KV.
- Listens for gossip on `gossip_addr` (UDP).
- Connects to `seed_nodes` (one or two peers) to bootstrap.
- Serves the public HTTP API on `port`.

## Adding a node

There's no rebalance — consistent hashing means adding a fourth node only remaps `1/4` of the ring. Existing replicas keep serving until their position in the ring is taken over by the new node.

```bash
# Start a fourth node, seeded by the existing cluster
docker run -d \
  --network ferrocache-net \
  -e FERROCACHE_NODE_ID=node4 \
  -e FERROCACHE_CLUSTER__ENABLED=true \
  -e FERROCACHE_CLUSTER__GOSSIP_ADDR=0.0.0.0:4000 \
  -e FERROCACHE_CLUSTER__API_ADDR=ferrocache-4:3000 \
  -e FERROCACHE_CLUSTER__SEED_NODES=ferrocache-1:4000,ferrocache-2:4000 \
  ghcr.io/nickleodoen/ferrocache:latest
```

Verify the cluster picked it up:

```bash
curl http://localhost:3001/cluster/status | python3 -m json.tool
```

You should see `node_count: 4` and the new node in the peer list within ~2s.

## `replication_factor` and durability

| Setting | Effect |
|---|---|
| `replication_factor = 1` | One copy. Node failure = data loss for the keys it owned. |
| `replication_factor = 2` | Two copies (default). Survives one node failure with zero ring movement. |
| `replication_factor = 3` | Three copies. Survives two node failures. |

`replication_factor` must be `<= cluster_size`. A 3-node cluster with `replication_factor = 3` writes every entry to every node — every read is local, every write is global.

## Failure detection (phi accrual)

Each node feeds peer heartbeats into a phi-accrual detector. `phi_threshold` (default `8.0`) controls the failure threshold:

- Lower threshold → faster failure detection, higher false-positive rate.
- Higher threshold → slower detection, fewer false positives.

State machine: `Alive → Suspected → Dead`. Only `Dead` triggers ring removal:

- `Alive`: normal traffic.
- `Suspected`: still in the ring; queries still routed to it.
- `Dead`: removed from the ring; queries skip it (immediate miss instead of retry-then-502); insert fan-out logs `effective_replicas` and `dead_peers` and continues degraded.

`dead_node_removal_enabled = false` runs in monitoring-only canary mode — phi values are reported but the ring is not modified.

## Read repair

When a coordinator's local query misses, it queries non-dead replicas in parallel. The first hit:

1. Is returned to the client immediately.
2. Triggers a background `/internal/entry/{uuid}` fetch.
3. Re-inserts via the WAL group-commit channel (durable).

`read_repair_enabled = true` (default) gates the entire fan-out.

## Verifying cluster health

```bash
curl http://localhost:3001/cluster/status | python3 -m json.tool
```

Healthy output:

```json
{
  "node_id": "node1",
  "peers": [
    { "node_id": "node2", "api_addr": "ferrocache-2:3000", "phi": 0.4, "status": "Alive" },
    { "node_id": "node3", "api_addr": "ferrocache-3:3000", "phi": 0.3, "status": "Alive" }
  ],
  "dead_nodes": [],
  "ring_size": 192,
  "read_repair_enabled": true
}
```

`ring_size = node_count × virtual_nodes` (default `64`). A 3-node cluster ⇒ `192`.

## What gossips, what doesn't

| Channel | Carries | Encryption |
|---|---|---|
| Gossip UDP (`gossip_addr`) | Ring metadata: node IDs, generation numbers, `api_addr` | None — restrict via firewall |
| HTTP forwards (`api_addr`) | Insert / query / delete / read repair | Optional [mTLS](security.md) |

**No cached data flows over gossip.** Only ring membership.

## See also

- [Security](security.md) — bearer auth, mTLS, threat model.
- [Observability](observability.md) — `/metrics` for cluster-aware monitoring.
- [HTTP API](../integrations/http-api.md) — `/cluster/status` reference.
