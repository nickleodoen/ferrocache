# Security

FerroCache exposes four distinct attack surfaces. This page describes the threat model, the in-tree defenses, deployment recipes, and known limitations operators handle out-of-band.

## Threat model

### Surface 1 — Public HTTP API (port 3000)

| | |
|---|---|
| **Threat** | Unauthorized read/write access to cached data over the wire. |
| **Defense** | Bearer token authentication. |
| **Implementation** | `FERROCACHE_AUTH_TOKEN` enables `Authorization: Bearer <token>` checks on `/query`, `/insert`, `/stats`, `/cluster/status`, `/admin/*`. `/health` and `/metrics` stay open so load balancers and Prometheus can scrape unauthenticated. Tokens compared with `subtle::ConstantTimeEq`. |
| **Status** | Implemented. **Opt-in** — disabled by default. |
| **Operator action** | Set `FERROCACHE_AUTH_TOKEN` to a long random string in production. Terminate public TLS at a reverse proxy. Rate-limit failed auth attempts at the proxy. |

### Surface 2 — Inter-node Cluster Traffic (TCP)

| | |
|---|---|
| **Threat** | Rogue process joins the cluster as a peer (and receives forwarded inserts), or eavesdrops on replication traffic. |
| **Defense** | Mutual TLS between cluster nodes. |
| **Implementation** | `FERROCACHE_CLUSTER__TLS__ENABLED=true` makes FerroCache bind a second listener on `internal_port` (default `port + 1000`). The TLS server requires a client cert chained to the cluster CA (`WebPkiClientVerifier`); anonymous clients are rejected at the handshake. The forwarding `reqwest::Client` disables system roots and trusts only the cluster CA. |
| **Status** | Implemented. **Opt-in** — disabled by default. |
| **Operator action** | Generate a cluster CA + per-node leaf certs. All nodes must share the same CA. Distribute certs out-of-band (Vault, sealed secrets, baked images). |

### Surface 3 — Filesystem (WAL + Snapshot)

| | |
|---|---|
| **Threat** | Anyone reading `/data/ferrocache.wal` recovers every cached query/response pair. |
| **Defense** | None in-tree. |
| **Mitigation** | OS-level disk encryption (LUKS, FileVault, BitLocker) or cloud-provider EBS/Disk encryption. Restrict the WAL volume to FerroCache's user. |
| **Operator action** | Enable disk encryption on the host or mounted volume. |

### Surface 4 — Gossip Protocol (UDP)

| | |
|---|---|
| **Threat** | Eavesdropping on UDP gossip reveals node IDs, generation numbers, and forwarding addresses. **No cached data flows over gossip** — only ring metadata. |
| **Defense** | None in-tree. chitchat does not support TLS. |
| **Mitigation** | Restrict the gossip UDP port to cluster-internal traffic via firewall / security group. |
| **Operator action** | Block the gossip port at the network edge. |

## Deployment recipes

### Single node (development)

```bash
cargo run --release
# or
docker run -p 3000:3000 ghcr.io/nickleodoen/ferrocache:latest
```

No auth, no TLS. Fine for local development. **Do not expose this port to the public internet.**

### Single node (production, behind a reverse proxy)

```bash
export FERROCACHE_AUTH_TOKEN="$(openssl rand -hex 32)"
export FERROCACHE_PORT=3000
ferrocache  # plain HTTP at :3000, only accessible from the proxy
```

```nginx
server {
    listen 443 ssl http2;
    server_name cache.example.com;
    ssl_certificate     /etc/letsencrypt/live/cache.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/cache.example.com/privkey.pem;

    limit_req zone=auth burst=10 nodelay;

    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_set_header Host $host;
    }
}
```

### Multi-node cluster (production)

Pre-flight, on a trusted machine (ideally not one of the cluster nodes):

```bash
git clone https://github.com/nickleodoen/ferrocache && cd ferrocache
cargo run --bin gen_certs node1 node2 node3
# → ./certs/ca.pem
# → ./certs/node{1,2,3}/{cert.pem,key.pem}

scp certs/ca.pem certs/node1/* node1:/etc/ferrocache/certs/
scp certs/ca.pem certs/node2/* node2:/etc/ferrocache/certs/
scp certs/ca.pem certs/node3/* node3:/etc/ferrocache/certs/

shred -u certs/node*/key.pem
```

On each node:

```bash
export FERROCACHE_AUTH_TOKEN="$(cat /etc/ferrocache/auth-token)"  # same on every node

export FERROCACHE_CLUSTER__ENABLED=true
export FERROCACHE_CLUSTER__GOSSIP_ADDR=0.0.0.0:4000
export FERROCACHE_CLUSTER__API_ADDR=node1:3000              # change per node
export FERROCACHE_CLUSTER__SEED_NODES=node2:4000,node3:4000 # change per node

export FERROCACHE_CLUSTER__TLS__ENABLED=true
export FERROCACHE_CLUSTER__TLS__CA_CERT_PATH=/etc/ferrocache/certs/ca.pem
export FERROCACHE_CLUSTER__TLS__NODE_CERT_PATH=/etc/ferrocache/certs/cert.pem
export FERROCACHE_CLUSTER__TLS__NODE_KEY_PATH=/etc/ferrocache/certs/key.pem
export FERROCACHE_CLUSTER__TLS__INTERNAL_PORT=4443

ferrocache
```

Verify: `curl -H "Authorization: Bearer $TOKEN" http://node1:3000/cluster/status` should report `node_count: 3`.

## Certificate management

The `gen_certs` binary produces certs valid until year 4096 — fine for development, **not appropriate for production**. For production:

- Use your existing PKI (HashiCorp Vault, AWS PCA, internal corporate CA).
- The cluster CA can be a standalone offline CA whose only job is signing FerroCache leaf certs. Its private key never needs to live on a FerroCache node.
- Each node's leaf cert needs `subjectAltName` covering the hostname/IP peers will dial it by, plus `extKeyUsage = serverAuth, clientAuth` since the same cert plays both roles in mTLS.
- FerroCache loads certs at startup. Rotation = rolling restart: replace files on disk, restart one node at a time.
- FerroCache does **not** support CRLs or OCSP. Compromised key → rotate the entire CA.

## Firewall rules

| Port | Protocol | Source | Purpose |
|---|---|---|---|
| 3000 | TCP | clients, load balancer, Prometheus | Public API + `/metrics` |
| 4000 | UDP | cluster nodes only | Chitchat gossip (ring membership) |
| 4443 (or `internal_port`) | TCP | cluster nodes only | mTLS replication forwards |

Public clients should never be able to reach the gossip or internal ports.

## Configuration reference

| Env var | Purpose |
|---|---|
| `FERROCACHE_AUTH_TOKEN` | Bearer token for the public API. **Never log or check into VCS.** |
| `FERROCACHE_CLUSTER__TLS__ENABLED` | `true` enables mTLS on the internal listener. |
| `FERROCACHE_CLUSTER__TLS__CA_CERT_PATH` | PEM file with cluster CA cert(s). |
| `FERROCACHE_CLUSTER__TLS__NODE_CERT_PATH` | PEM file with this node's leaf cert. |
| `FERROCACHE_CLUSTER__TLS__NODE_KEY_PATH` | PEM file with this node's private key. PKCS#8. |
| `FERROCACHE_CLUSTER__TLS__INTERNAL_PORT` | TCP port for the mTLS listener. Default = `port + 1000`. |
| `FERROCACHE_CLUSTER__MAX_REPLICATION_RETRIES` | Retry attempts for replication forwards. Default = 3. |

Missing cert paths → FerroCache generates self-signed certs in memory and logs a warning. Each auto-generated CA is unique, so peers don't trust each other — useful only for single-node smoke tests.

## Known limitations

- **No at-rest encryption.** WAL and snapshot files are plaintext on disk. Mitigate with full-disk encryption.
- **No per-client ACLs.** Authentication is binary: any caller with the token can do anything.
- **No certificate revocation.** Compromised node keys require rotating the entire CA.
- **No rate limiting on auth failures.** Even with constant-time comparison, an attacker can grind tokens at line rate. Handle at the reverse proxy.
- **Token rotation requires a restart.** Hot-reloading isn't supported. Plan rotations during a maintenance window.
- **Gossip UDP is unencrypted.** Ring metadata leaks to anyone with read access on the gossip port. Restrict via firewall.
- **Tokens loaded into memory and not zeroized.** A core dump or `/proc/<pid>/mem` read by a privileged user reveals the token.
- **Public-port TLS is out of scope.** FerroCache speaks plain HTTP on the public port; use a reverse proxy.
