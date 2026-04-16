/// Initial node feature vector construction for Cleora.
///
/// Each node gets a D-dimensional float32 vector built from:
///  [0..HASH_DIMS)   — FNV-1a name hash bucketed into HASH_DIMS bins
///  [HASH_DIMS..HASH_DIMS+LABEL_DIMS) — label one-hot
///  [HASH_DIMS+LABEL_DIMS..D)         — log-normalised in/out degree per edge type
use crate::graph::schema::{EdgeType, Node, NodeLabel};
use fnv::FnvHasher;
use ndarray::Array1;
use std::collections::HashMap;
use std::hash::Hasher;

pub const HASH_DIMS: usize = 64;
pub const LABEL_DIMS: usize = NodeLabel::COUNT;
pub const DEGREE_DIMS: usize = EdgeType::ALL.len() * 2; // in + out per type
pub const FEATURE_DIM: usize = HASH_DIMS + LABEL_DIMS + DEGREE_DIMS;

/// Degree counts per node, split by edge type and direction.
pub struct DegreeTable {
    /// node_id → (edge_type_index → (in_degree, out_degree))
    inner: HashMap<i64, HashMap<usize, (u32, u32)>>,
}

impl DegreeTable {
    pub fn new() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    pub fn add_edge(&mut self, source_id: i64, target_id: i64, edge_type: EdgeType) {
        let et_idx = edge_type_index(edge_type);
        // out-degree for source
        self.inner
            .entry(source_id)
            .or_default()
            .entry(et_idx)
            .or_insert((0, 0))
            .1 += 1;
        // in-degree for target
        self.inner
            .entry(target_id)
            .or_default()
            .entry(et_idx)
            .or_insert((0, 0))
            .0 += 1;
    }

    pub fn degrees_for(&self, node_id: i64) -> Option<&HashMap<usize, (u32, u32)>> {
        self.inner.get(&node_id)
    }
}

fn edge_type_index(et: EdgeType) -> usize {
    EdgeType::ALL.iter().position(|&t| t == et).unwrap_or(0)
}

/// Build the initial feature vector for a single node.
pub fn build_feature_vector(node: &Node, degree_table: &DegreeTable) -> Array1<f32> {
    let mut v = Array1::<f32>::zeros(FEATURE_DIM);

    // ── Hash dims: name tokens ─────────────────────────────────────────────
    // Split qualified name on "::" and hash each token component
    for token in node.qualified_name.split("::") {
        let bucket = fnv1a_bucket(token, HASH_DIMS);
        v[bucket] += 1.0;
    }

    // ── Label one-hot ──────────────────────────────────────────────────────
    let label_offset = HASH_DIMS;
    v[label_offset + node.label.one_hot_index()] = 1.0;

    // ── Degree features ────────────────────────────────────────────────────
    let degree_offset = HASH_DIMS + LABEL_DIMS;
    if let Some(degrees) = degree_table.degrees_for(node.id) {
        for (&et_idx, &(in_deg, out_deg)) in degrees {
            let base = degree_offset + et_idx * 2;
            if base + 1 < FEATURE_DIM {
                v[base] = log_norm(in_deg);
                v[base + 1] = log_norm(out_deg);
            }
        }
    }

    l2_normalize(&mut v);
    v
}

/// FNV-1a hash of `s`, bucketed into `dims` bins.
fn fnv1a_bucket(s: &str, dims: usize) -> usize {
    let mut hasher = FnvHasher::default();
    hasher.write(s.as_bytes());
    (hasher.finish() as usize) % dims
}

/// Log-normalise a degree count: log2(1 + d).
fn log_norm(d: u32) -> f32 {
    (1.0 + d as f32).ln()
}

/// In-place L2 normalisation; leaves zero vectors unchanged.
pub fn l2_normalize(v: &mut Array1<f32>) {
    let norm = v.dot(v).sqrt();
    if norm > 1e-9 {
        *v /= norm;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::schema::{Node, NodeLabel};

    fn dummy_node(id: i64, label: NodeLabel, qname: &str) -> Node {
        Node {
            id,
            project: "test".into(),
            label,
            qualified_name: qname.into(),
            file_path: None,
            line_start: None,
            line_end: None,
            properties: serde_json::Value::Null,
        }
    }

    #[test]
    fn feature_dim_matches_constants() {
        assert_eq!(FEATURE_DIM, HASH_DIMS + LABEL_DIMS + DEGREE_DIMS);
        assert_eq!(LABEL_DIMS, NodeLabel::COUNT);
        assert_eq!(DEGREE_DIMS, EdgeType::ALL.len() * 2);
    }

    #[test]
    fn build_feature_vector_correct_length() {
        let node = dummy_node(1, NodeLabel::Function, "ns::foo");
        let degree_table = DegreeTable::new();
        let fv = build_feature_vector(&node, &degree_table);
        assert_eq!(fv.len(), FEATURE_DIM);
    }

    #[test]
    fn build_feature_vector_is_unit_length() {
        let node = dummy_node(1, NodeLabel::Method, "ns::MyClass::bar");
        let degree_table = DegreeTable::new();
        let fv = build_feature_vector(&node, &degree_table);
        let norm: f32 = fv.dot(&fv).sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm={}", norm);
    }

    #[test]
    fn l2_normalize_unit_vector() {
        let mut v = Array1::from_vec(vec![3.0_f32, 4.0]);
        l2_normalize(&mut v);
        let norm: f32 = v.dot(&v).sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_zero_vector_unchanged() {
        let mut v = Array1::from_vec(vec![0.0_f32, 0.0, 0.0]);
        l2_normalize(&mut v);
        assert!(v.iter().all(|&x| x == 0.0));
    }
}
