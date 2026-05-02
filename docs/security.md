# ferrocache Security Guide

This document describes ferrocache's threat model, the defenses
currently implemented, deployment recipes for various scenarios, and the
known limitations operators need to handle out-of-band.

## Threat Model

ferrocache exposes four distinct attack surfaces. Each row below states the
threat, the in-tree defense, and what an operator still has to do.

### Surface 1 — Public HTTP API (port 3000)

| | |
|---|---|
| **Threat** | Unauthorized read or write access to cached data over the wire. |
| **Defense** | Bearer token authentication (M17). |
| **Implementation** | `FERROCACHE_AUTH_TOKEN` enables `Authorization: Bearer <token>` checks on `/query`, `/insert`, `/stats`, `/cluster/status`, `/admin/compact`. `/health` and `/metrics` stay open so load balancers and Prometheus can scrape unauthenticated. Tokens are compared with `subtle::ConstantTimeEq`. |
| **Status** | Implemented. **Opt-in** — disabled by default, identical to pre-M17 when unset. |
| **Operator action** | Set `FERROCACHE_AUTH_TOKEN` to a long random string in production. Terminate public TLS at a reverse proxy. Consider rate-limiting failed auth attempts at the proxy. |

### Surface 2 — Inter-node Cluster Traffic (TCP)

| | |
|---|---|
| **Threat** | A rogue process joins the cluster as a peer (and receives forwarded inserts), or eavesdrops on replication traffic in flight. |
| **Defense** | Mutual TLS between cluster nodes (M18). |
| **Implementation** | `FERROCACHE_CLUSTER__TLS__ENABLED=true` makes ferrocache bind a second listener on `internal_port` (default `port + 1000`). The TLS server requires a client cert chained to the cluster CA (`WebPkiClientVerifier::builder().build()`); anonymous clients are rejected at the handshake. The forwarding `reqwest::Client` disables system roots and trusts only the cluster CA. |
| **Status** | Implemented. **Opt-in** — disabled by default. |
| **Operator action** | Generate a cluster CA + per-node leaf certs. All nodes must share the same CA. Distribute certs out-of-band (Vault, sealed secrets, baked images). Plan for cert rotation (see below). |

### Surface 3 — Filesystem (WAL + Snapshot)

| | |
|---|---|
| **Threat** | Anyone who reads `/data/ferrocache.wal` and `/data/ferrocache.wal.snap` recovers every cached query/response pair. |
| **Defense** | None in-tree. |
| **Mitigation** | OS-level disk encryption (LUKS, FileVault, BitLocker) or cloud-provider EBS/Disk encryption. Restrict the WAL volume to the ferrocache container's user. |
| **Operator action** | Enable disk encryption in the host or mounted volume. Don't snapshot or back up the WAL volume to unencrypted storage. |

### Surface 4 — Gossip Protocol (UDP)

| | |
|---|---|
| **Threat** | Eavesdropping on UDP gossip reveals node IDs, generation numbers, and forwarding addresses. **No cached data flows over gossip** — only ring metadata. |
| **Defense** | None in-tree. chitchat does not support TLS. |
| **Mitigation** | Restrict the gossip UDP port to cluster-internal traffic via firewall / security group. |
| **Operator action** | Block the gossip port at the network edge. Consider running the cluster inside a service mesh / overlay network if your threat model includes lateral movement. |

## Deployment Recipes

### Single Node (Development)

```bash
cargo run --release
# or
docker run -p 3000:3000 ghcr.io/nickleodoen/ferrocache:latest
```

No auth, no TLS. Fine for local development. **Do not expose this port to the public internet.**

### Single Node (Production, behind a reverse proxy)

The reverse proxy (nginx, caddy, ALB, Cloud Run, …) handles public TLS
termination. ferrocache validates the bearer token; the proxy adds it to every
request from authenticated upstream clients, or proxies straight through if
clients carry the header themselves.

```bash
# On the ferrocache host:
export FERROCACHE_AUTH_TOKEN="$(openssl rand -hex 32)"
export FERROCACHE_PORT=3000
ferrocache  # listens on plain HTTP at :3000, only accessible from the proxy
```

```nginx
# nginx snippet
server {
    listen 443 ssl http2;
    server_name cache.example.com;
    ssl_certificate     /etc/letsencrypt/live/cache.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/cache.example.com/privkey.pem;

    # Optional: rate-limit failed auth attempts
    limit_req zone=auth burst=10 nodelay;

    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_set_header Host $host;
    }
}
```

### Multi-Node Cluster (Production)

Pre-flight, on a trusted machine (ideally not one of the cluster nodes):

```bash
# 1. Generate certs
git clone <ferrocache repo> && cd ferrocache
cargo run --bin gen_certs node1 node2 node3
# → ./certs/ca.pem
# → ./certs/node{1,2,3}/{cert.pem,key.pem}

# 2. Distribute. Each node gets the CA + its own cert/key. NOT other nodes' keys.
scp certs/ca.pem certs/node1/* node1:/etc/ferrocache/certs/
scp certs/ca.pem certs/node2/* node2:/etc/ferrocache/certs/
scp certs/ca.pem certs/node3/* node3:/etc/ferrocache/certs/

# 3. Wipe the local copy of the keys.
shred -u certs/node*/key.pem
```

On each node, run with auth + mTLS enabled:

```bash
export FERROCACHE_AUTH_TOKEN="$(cat /etc/ferrocache/auth-token)"  # same on every node

export FERROCACHE_CLUSTER__ENABLED=true
export FERROCACHE_CLUSTER__GOSSIP_ADDR=0.0.0.0:4000
export FERROCACHE_CLUSTER__API_ADDR=node1:3000  # change per node
export FERROCACHE_CLUSTER__SEED_NODES=node2:4000,node3:4000  # change per node

export FERROCACHE_CLUSTER__TLS__ENABLED=true
export FERROCACHE_CLUSTER__TLS__CA_CERT_PATH=/etc/ferrocache/certs/ca.pem
export FERROCACHE_CLUSTER__TLS__NODE_CERT_PATH=/etc/ferrocache/certs/cert.pem
export FERROCACHE_CLUSTER__TLS__NODE_KEY_PATH=/etc/ferrocache/certs/key.pem
export FERROCACHE_CLUSTER__TLS__INTERNAL_PORT=4443

ferrocache
```

Verify the cluster converged: `curl -H "Authorization: Bearer $TOKEN" http://node1:3000/cluster/status` should report `node_count: 3`.

### Certificate Management

The `gen_certs` binary produces certs valid until year 4096 — fine for
development, **not appropriate for production**. For production:

- Use your existing PKI (HashiCorp Vault, AWS PCA, internal corporate CA).
- The cluster CA can be a standalone offline CA whose only job is signing
  ferrocache leaf certs. Its private key never needs to live on a ferrocache node.
- Each node's leaf cert needs `subjectAltName` covering the hostname/IP
  peers will dial it by, plus `extKeyUsage = serverAuth, clientAuth` since
  the same cert plays both roles in mTLS.
- ferrocache loads certs at startup from the configured paths. Rotation is a
  rolling restart: replace the files on disk, then restart one node at a time.
- ferrocache does **not** support CRLs or OCSP. If a node's key leaks, you
  must rotate the entire CA and reissue all node certs.

### Firewall Rules

| Port | Protocol | Source | Purpose |
|---|---|---|---|
| 3000 | TCP | clients, load balancer, Prometheus | Public API + `/metrics` (always allowed) |
| 4000 | UDP | cluster nodes only | Chitchat gossip (ring membership) |
| 4443 (or `internal_port`) | TCP | cluster nodes only | mTLS replication forwards |

Public clients should never be able to reach the gossip or internal ports. If they can, they can spoof gossip membership or attempt mTLS handshakes (which they will fail without a valid client cert, but the noise is unnecessary).

## Configuration Reference

| Env var | Purpose |
|---|---|
| `FERROCACHE_AUTH_TOKEN` | Bearer token for the public API. Empty / unset disables auth. **Never log or check into VCS.** |
| `FERROCACHE_CLUSTER__TLS__ENABLED` | `true` to enable mTLS on the internal listener. |
| `FERROCACHE_CLUSTER__TLS__CA_CERT_PATH` | PEM file with the cluster CA cert(s). |
| `FERROCACHE_CLUSTER__TLS__NODE_CERT_PATH` | PEM file with this node's leaf cert. |
| `FERROCACHE_CLUSTER__TLS__NODE_KEY_PATH` | PEM file with this node's private key. PKCS#8 format. |
| `FERROCACHE_CLUSTER__TLS__INTERNAL_PORT` | TCP port for the mTLS listener. Default = `port + 1000`. |
| `FERROCACHE_CLUSTER__MAX_REPLICATION_RETRIES` | Retry attempts for replication forwards (M19). Default = 3. |

Any of the cert path fields missing → ferrocache generates self-signed certs
in memory and logs a warning. Each node's auto-generated CA is unique, so
peers do not trust each other — useful only for single-node smoke tests.

## Known Limitations

- **No at-rest encryption.** WAL and snapshot files are plaintext on disk. Mitigate with full-disk encryption.
- **No per-client ACLs.** Authentication is binary: any caller with the token can do anything (read, write, compact). A future "read-only token" / "admin token" split is reasonable but not implemented.
- **No certificate revocation.** Compromised node keys require rotating the entire CA.
- **No rate limiting on auth failures.** Even with constant-time comparison, an attacker can grind tokens at line-rate. Handle at the reverse proxy.
- **Token rotation requires a restart.** Hot-reloading tokens isn't supported. Plan rotations during a maintenance window or rely on the rolling restart that mTLS rotation already requires.
- **Gossip UDP is unencrypted.** Ring metadata (node IDs, addresses) leaks to anyone with read access on the gossip port. Restrict via firewall.
- **Tokens are loaded into memory and never zeroized.** A core dump or `/proc/<pid>/mem` read by a privileged user reveals the token. Guard the host accordingly.
- **Public-port TLS is out of scope.** ferrocache itself only speaks plain HTTP on the public port. Use a reverse proxy for public TLS termination.
