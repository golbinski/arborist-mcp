pub mod features;
pub mod propagation;

use anyhow::Result;
use ndarray::Array2;
use std::sync::{Arc, Mutex};

use crate::graph::schema::EdgeType;
use crate::graph::store::GraphStore;
use features::{build_feature_vector, DegreeTable, FEATURE_DIM};
use propagation::{build_adjacency, cosine_similarity, propagate};

/// Compute Cleora embeddings for all nodes in `project` and store them.
pub fn compute_and_store(store: Arc<Mutex<GraphStore>>, project: &str) -> Result<()> {
    let locked = store.lock().unwrap();

    let nodes = locked.get_all_nodes(project)?;
    if nodes.is_empty() {
        return Ok(());
    }

    let edges = locked.get_all_edges(project)?;

    // Build degree table
    let mut degree_table = DegreeTable::new();
    for edge in &edges {
        degree_table.add_edge(edge.source_id, edge.target_id, edge.edge_type);
    }

    // Build initial feature matrix
    let n = nodes.len();
    let mut init = Array2::<f32>::zeros((n, FEATURE_DIM));
    for (i, node) in nodes.iter().enumerate() {
        let fv = build_feature_vector(node, &degree_table);
        init.row_mut(i).assign(&fv);
    }

    // Build adjacency list
    let node_ids: Vec<i64> = nodes.iter().map(|n| n.id).collect();
    let raw_edges: Vec<(i64, i64, EdgeType)> = edges
        .iter()
        .map(|e| (e.source_id, e.target_id, e.edge_type))
        .collect();
    let adj = build_adjacency(&node_ids, &raw_edges);

    // Drop the lock before the expensive computation
    drop(locked);

    // Propagate
    let embeddings = propagate(init, &adj);

    // Store results
    let locked = store.lock().unwrap();
    for (i, node) in nodes.iter().enumerate() {
        let row = embeddings.row(i);
        let vec: Vec<f32> = row.iter().copied().collect();
        locked.upsert_embedding(node.id, &vec)?;
    }

    Ok(())
}

/// Find the top-k most similar nodes to `anchor_id` by cosine similarity.
pub fn find_similar(
    store: Arc<Mutex<GraphStore>>,
    project: &str,
    anchor_id: i64,
    top_k: usize,
) -> Result<Vec<(i64, f32)>> {
    let locked = store.lock().unwrap();

    let anchor_vec = match locked.get_embedding(anchor_id)? {
        Some(v) => v,
        None => return Ok(vec![]),
    };

    let all_embeddings = locked.get_all_embeddings(project)?;

    let mut scored: Vec<(i64, f32)> = all_embeddings
        .iter()
        .filter(|(id, _)| *id != anchor_id)
        .map(|(id, vec)| (*id, cosine_similarity(&anchor_vec, vec)))
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(top_k);
    Ok(scored)
}
