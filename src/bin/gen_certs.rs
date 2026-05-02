//! Generate a CA + per-node leaf certs for a ferrocache cluster (M18).
//!
//! Usage:
//!   cargo run --bin gen_certs                   # default: node1,node2,node3 → ./certs
//!   cargo run --bin gen_certs -- node-a node-b  # custom node list
//!   FERROCACHE_CERT_DIR=/etc/ferrocache cargo run --bin gen_certs
//!
//! Output layout:
//!   ./certs/ca.pem
//!   ./certs/<node_id>/cert.pem
//!   ./certs/<node_id>/key.pem
//!
//! Each leaf cert carries SANs `[<node_id>, localhost, 127.0.0.1]` so it
//! validates whether peers dial it by service name (Docker), hostname, or
//! loopback. The certs are dev-only — long validity, no revocation, no key
//! protection — and the binary is intentionally NOT shipped in the
//! production Docker image.

use std::env;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use ferrocache::tls::{generate_ca, generate_node_cert};

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    let nodes: Vec<String> = if args.is_empty() {
        vec!["node1".into(), "node2".into(), "node3".into()]
    } else {
        args
    };
    let output_dir = env::var("FERROCACHE_CERT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("./certs"));
    fs::create_dir_all(&output_dir).with_context(|| format!("create {}", output_dir.display()))?;

    let (ca_cert, ca_key) = generate_ca().context("generate CA")?;
    let ca_path = output_dir.join("ca.pem");
    fs::write(&ca_path, ca_cert.pem()).with_context(|| format!("write {}", ca_path.display()))?;
    println!("wrote {}", ca_path.display());

    for node in &nodes {
        let sans = vec![
            node.clone(),
            "localhost".to_string(),
            "127.0.0.1".to_string(),
        ];
        let (cert, key) = generate_node_cert(&ca_cert, &ca_key, node, &sans)
            .with_context(|| format!("generate cert for {node}"))?;
        let node_dir = output_dir.join(node);
        fs::create_dir_all(&node_dir).with_context(|| format!("create {}", node_dir.display()))?;
        let cert_path = node_dir.join("cert.pem");
        let key_path = node_dir.join("key.pem");
        fs::write(&cert_path, cert.pem())
            .with_context(|| format!("write {}", cert_path.display()))?;
        fs::write(&key_path, key.serialize_pem())
            .with_context(|| format!("write {}", key_path.display()))?;
        println!("wrote {} + {}", cert_path.display(), key_path.display());
    }

    println!(
        "\ndone — {} CA + {} node cert(s) under {}",
        1,
        nodes.len(),
        output_dir.display()
    );
    println!("mount these into your containers and set:");
    println!("  FERROCACHE_CLUSTER__TLS__ENABLED=true");
    println!("  FERROCACHE_CLUSTER__TLS__CA_CERT_PATH=/certs/ca.pem");
    println!("  FERROCACHE_CLUSTER__TLS__NODE_CERT_PATH=/certs/<node_id>/cert.pem");
    println!("  FERROCACHE_CLUSTER__TLS__NODE_KEY_PATH=/certs/<node_id>/key.pem");
    Ok(())
}
