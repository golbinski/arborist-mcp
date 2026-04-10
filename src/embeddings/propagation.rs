/// Cleora iterative propagation + L2 re-normalisation.
///
/// Algorithm:
///   for iter in 0..NUM_ITERS:
///     for each node v:
///       new_v = alpha * v + (1-alpha) * mean_pool(neighbours weighted by edge_weight)
///       new_v = L2_normalise(new_v)
use ndarray::{Array1, Array2};
use std::collections::HashMap;

use super::features::{l2_normalize, FEATURE_DIM};
use crate::graph::schema::EdgeType;

pub const NUM_ITERS: usize = 4;
const ALPHA: f32 = 0.5; // self-loop weight

/// Adjacency entry: (neighbor_idx, edge weight).
pub type AdjList = Vec<Vec<(usize, f32)>>;

/// Build a symmetric weighted adjacency list from raw edge data.
///
/// `node_ids` is the ordered list of node IDs; `edges` is `(source_id, target_id, edge_type)`.
pub fn build_adjacency(
    node_ids: &[i64],
    edges: &[(i64, i64, EdgeType)],
) -> AdjList {
    let id_to_idx: HashMap<i64, usize> = node_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    let mut adj: AdjList = vec![Vec::new(); node_ids.len()];

    for &(src, tgt, et) in edges {
        let w = et.weight();
        if let (Some(&si), Some(&ti)) = (id_to_idx.get(&src), id_to_idx.get(&tgt)) {
            adj[si].push((ti, w));
            adj[ti].push((si, w)); // undirected propagation
        }
    }

    // Deduplicate: sum weights for parallel edges
    for list in &mut adj {
        list.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        let mut deduped: Vec<(usize, f32)> = Vec::with_capacity(list.len());
        for &(idx, w) in list.iter() {
            if let Some(last) = deduped.last_mut() {
                if last.0 == idx {
                    last.1 += w;
                    continue;
                }
            }
            deduped.push((idx, w));
        }
        *list = deduped;
    }

    adj
}

/// Run Cleora propagation and return the final embedding matrix.
///
/// `init` is shaped `(N, FEATURE_DIM)` with the initial feature vectors.
pub fn propagate(init: Array2<f32>, adj: &AdjList) -> Array2<f32> {
    let n = init.nrows();
    let mut current = init;

    for _iter in 0..NUM_ITERS {
        let mut next = Array2::<f32>::zeros((n, FEATURE_DIM));

        for (i, neighbors) in adj.iter().enumerate() {
            let self_row = current.row(i).to_owned();

            if neighbors.is_empty() {
                // Isolated node: keep self embedding
                next.row_mut(i).assign(&self_row);
                continue;
            }

            // Weighted mean of neighbours
            let mut total_weight = 0.0f32;
            let mut agg = Array1::<f32>::zeros(FEATURE_DIM);
            for &(j, w) in neighbors {
                let nb_row = current.row(j).to_owned();
                agg = agg + nb_row * w;
                total_weight += w;
            }
            if total_weight > 0.0 {
                agg /= total_weight;
            }

            // Combine with self (Array * scalar to avoid requiring Mul<Array> for f32)
            let mut combined = self_row * ALPHA + agg * (1.0 - ALPHA);
            l2_normalize(&mut combined);
            next.row_mut(i).assign(&combined);
        }

        current = next;
    }

    current
}

/// Compute cosine similarity between two unit vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::schema::EdgeType;

    fn unit_vec(vals: &[f32]) -> Array1<f32> {
        let mut v = Array1::from_vec(vals.to_vec());
        let norm: f32 = v.dot(&v).sqrt();
        if norm > 1e-9 { v /= norm; }
        v
    }

    #[test]
    fn build_adjacency_symmetric() {
        let node_ids = vec![10i64, 20, 30];
        let edges = vec![(10, 20, EdgeType::Calls)];
        let adj = build_adjacency(&node_ids, &edges);
        assert_eq!(adj.len(), 3);
        assert!(adj[0].iter().any(|&(nb, _)| nb == 1), "adj[0] missing nb 1");
        assert!(adj[1].iter().any(|&(nb, _)| nb == 0), "adj[1] missing nb 0");
        assert!(adj[2].is_empty(), "adj[2] should be empty (isolated)");
    }

    #[test]
    fn build_adjacency_deduplicates_parallel_edges() {
        let node_ids = vec![1i64, 2];
        let edges = vec![
            (1, 2, EdgeType::Calls),
            (1, 2, EdgeType::Calls),
        ];
        let adj = build_adjacency(&node_ids, &edges);
        assert_eq!(adj[0].len(), 1, "parallel edges should be merged");
        let (_, w) = adj[0][0];
        assert!((w - EdgeType::Calls.weight() * 2.0).abs() < 1e-6, "w={}", w);
    }

    #[test]
    fn propagate_isolated_node_stable() {
        let n = 1;
        let mut init = Array2::<f32>::zeros((n, FEATURE_DIM));
        let row = unit_vec(&vec![1.0; FEATURE_DIM]);
        init.row_mut(0).assign(&row);
        let adj: AdjList = vec![vec![]];
        let result = propagate(init.clone(), &adj);
        for i in 0..FEATURE_DIM {
            assert!((result[[0, i]] - init[[0, i]]).abs() < 1e-6, "dim {} changed", i);
        }
    }

    #[test]
    fn propagate_output_is_unit_length() {
        let n = 3;
        let mut init = Array2::<f32>::zeros((n, FEATURE_DIM));
        for i in 0..n {
            let v = unit_vec(&vec![i as f32 + 1.0; FEATURE_DIM]);
            init.row_mut(i).assign(&v);
        }
        let node_ids = vec![0i64, 1, 2];
        let edges = vec![(0, 1, EdgeType::Calls), (1, 2, EdgeType::Calls)];
        let adj = build_adjacency(&node_ids, &edges);
        let result = propagate(init, &adj);
        for i in 0..n {
            let row = result.row(i);
            let norm: f32 = row.dot(&row).sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "row {} norm={}", i, norm);
        }
    }

    #[test]
    fn cosine_similarity_identical() {
        let v = vec![1.0f32, 0.0, 0.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }
}
