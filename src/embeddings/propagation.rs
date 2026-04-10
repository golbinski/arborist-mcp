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
