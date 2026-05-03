use std::collections::BTreeMap;

const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

fn fnv1a_hash(data: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// Routing surface used by ClusterState and (in M6) the request-routing layer.
#[allow(dead_code)]
pub fn embedding_to_key(embedding: &[f32]) -> u64 {
    let mut bytes = [0u8; 8];
    for (i, val) in embedding.iter().take(2).enumerate() {
        let b = val.to_be_bytes();
        bytes[i * 4..(i + 1) * 4].copy_from_slice(&b);
    }
    u64::from_be_bytes(bytes)
}

pub struct HashRing {
    ring: BTreeMap<u64, String>,
    virtual_nodes: usize,
}

impl HashRing {
    pub fn new(virtual_nodes: usize) -> Self {
        Self {
            ring: BTreeMap::new(),
            virtual_nodes: virtual_nodes.max(1),
        }
    }

    pub fn add_node(&mut self, node_id: &str) {
        for i in 0..self.virtual_nodes {
            let label = format!("{node_id}-{i}");
            let pos = fnv1a_hash(label.as_bytes());
            self.ring.insert(pos, node_id.to_string());
        }
    }

    pub fn remove_node(&mut self, node_id: &str) {
        for i in 0..self.virtual_nodes {
            let label = format!("{node_id}-{i}");
            let pos = fnv1a_hash(label.as_bytes());
            self.ring.remove(&pos);
        }
    }

    #[allow(dead_code)] // consumed by ClusterState; direct callers added in M6
    pub fn get_node(&self, key: u64) -> Option<&str> {
        if self.ring.is_empty() {
            return None;
        }
        let next = self.ring.range(key..).next();
        let (_, node) = next.or_else(|| self.ring.iter().next())?;
        Some(node.as_str())
    }

    #[allow(dead_code)] // consumed by ClusterState; M6 will route through this
    pub fn get_node_for_embedding(&self, embedding: &[f32]) -> Option<&str> {
        let key = embedding_to_key(embedding);
        self.get_node(key)
    }

    /// Walk the ring clockwise from `key`, collect up to `n` distinct
    /// physical nodes. Wraps around. Order matches walk order.
    pub fn get_n_nodes(&self, key: u64, n: usize) -> Vec<String> {
        if self.ring.is_empty() || n == 0 {
            return Vec::new();
        }
        let mut out: Vec<String> = Vec::with_capacity(n);
        let mut seen = std::collections::HashSet::new();
        let walker = self.ring.range(key..).chain(self.ring.range(..key));
        for (_, node) in walker {
            if seen.insert(node.clone()) {
                out.push(node.clone());
                if out.len() == n {
                    break;
                }
            }
        }
        out
    }

    pub fn get_n_nodes_for_embedding(&self, embedding: &[f32], n: usize) -> Vec<String> {
        let key = embedding_to_key(embedding);
        self.get_n_nodes(key, n)
    }

    pub fn nodes(&self) -> Vec<String> {
        let mut seen = std::collections::BTreeSet::new();
        for n in self.ring.values() {
            seen.insert(n.clone());
        }
        seen.into_iter().collect()
    }

    pub fn node_count(&self) -> usize {
        self.nodes().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_lookup() {
        let mut r = HashRing::new(64);
        r.add_node("A");
        for k in [0u64, 1, 42, u64::MAX, u64::MAX / 2] {
            assert_eq!(r.get_node(k), Some("A"));
        }
    }

    #[test]
    fn test_two_nodes_distribute() {
        let mut r = HashRing::new(64);
        r.add_node("A");
        r.add_node("B");
        let mut a = 0;
        let mut b = 0;
        for k in 0..100u64 {
            // Use varied string keys so FNV-1a fully avalanches; raw integer
            // bytes that differ only in the LSB cluster on the ring.
            let mixed = fnv1a_hash(format!("key-{k}").as_bytes());
            match r.get_node(mixed).unwrap() {
                "A" => a += 1,
                "B" => b += 1,
                _ => unreachable!(),
            }
        }
        assert!(a > 0 && b > 0, "distribution: a={a} b={b}");
    }

    #[test]
    fn test_remove_node() {
        let mut r = HashRing::new(64);
        r.add_node("A");
        r.add_node("B");
        r.remove_node("B");
        for k in 0..50u64 {
            let mixed = fnv1a_hash(format!("key-{k}").as_bytes());
            assert_eq!(r.get_node(mixed), Some("A"));
        }
        assert_eq!(r.node_count(), 1);
    }

    #[test]
    fn test_consistency_after_add() {
        let mut r = HashRing::new(64);
        r.add_node("A");
        r.add_node("B");
        let keys: Vec<u64> = (0..100u64)
            .map(|k| fnv1a_hash(format!("key-{k}").as_bytes()))
            .collect();
        let before: Vec<String> = keys
            .iter()
            .map(|k| r.get_node(*k).unwrap().to_string())
            .collect();
        r.add_node("C");
        let after: Vec<String> = keys
            .iter()
            .map(|k| r.get_node(*k).unwrap().to_string())
            .collect();
        let same = before.iter().zip(&after).filter(|(a, b)| a == b).count();
        assert!(same >= 60, "consistency: {same}/100 stable after adding C");
    }

    #[test]
    fn test_empty_ring() {
        let r = HashRing::new(64);
        assert!(r.get_node(0).is_none());
        assert!(r.get_node(42).is_none());
        assert_eq!(r.node_count(), 0);
    }

    #[test]
    fn test_embedding_to_key() {
        let v1 = [1.0_f32, 0.0, 0.5, 0.25];
        let v2 = [1.0_f32, 0.0, 0.5, 0.25];
        let v3 = [0.5_f32, 0.5, 0.5, 0.25];
        assert_eq!(embedding_to_key(&v1), embedding_to_key(&v2));
        assert_ne!(embedding_to_key(&v1), embedding_to_key(&v3));
    }

    #[test]
    fn test_get_n_nodes_distinct() {
        let mut r = HashRing::new(64);
        r.add_node("A");
        r.add_node("B");
        let walked = r.get_n_nodes(0, 2);
        assert_eq!(walked.len(), 2);
        assert_ne!(walked[0], walked[1]);
    }

    #[test]
    fn test_get_n_nodes_wraps() {
        let mut r = HashRing::new(64);
        r.add_node("A");
        r.add_node("B");
        r.add_node("C");
        for k in 0..30u64 {
            let key = fnv1a_hash(format!("k-{k}").as_bytes());
            let walked = r.get_n_nodes(key, 2);
            assert_eq!(walked.len(), 2);
            assert_ne!(walked[0], walked[1]);
        }
    }

    #[test]
    fn test_get_node_for_embedding() {
        let mut r = HashRing::new(64);
        r.add_node("only");
        assert_eq!(r.get_node_for_embedding(&[1.0_f32, 0.0, 0.0]), Some("only"));
    }

    #[test]
    fn test_remove_redistributes_keys() {
        // Consistent-hashing invariant: removing a node only remaps the
        // keys it owned. Keys owned by other nodes keep their owner.
        let mut r = HashRing::new(64);
        for id in ["A", "B", "C"] {
            r.add_node(id);
        }
        let keys: Vec<u64> = (0..200u64)
            .map(|k| fnv1a_hash(format!("k-{k}").as_bytes()))
            .collect();
        let before: Vec<String> = keys
            .iter()
            .map(|k| r.get_node(*k).unwrap().to_string())
            .collect();

        r.remove_node("C");

        let after: Vec<String> = keys
            .iter()
            .map(|k| r.get_node(*k).unwrap().to_string())
            .collect();

        let mut owned_by_c = 0usize;
        let mut not_c_changed = 0usize;
        for (b, a) in before.iter().zip(&after) {
            assert_ne!(a, "C", "removed node still owns a key");
            if b == "C" {
                owned_by_c += 1;
            } else if b != a {
                not_c_changed += 1;
            }
        }
        assert!(owned_by_c > 0, "test sanity: C owned at least one key");
        assert_eq!(not_c_changed, 0, "non-C keys must keep their owner");
    }
}
