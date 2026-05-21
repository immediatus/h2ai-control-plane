use std::collections::HashMap;

use crate::types::KnowledgeNode;

/// Weighted directed edge in the constraint graph.
struct Edge {
    /// Index of the destination node in `node_ids`.
    dst: usize,
    /// Transition weight (after normalisation this is weight / sum-of-out-weights).
    weight: f32,
}

/// Sparse adjacency list representation of the constraint graph.
///
/// Edges are bidirectional: if A lists B in `related` or `cross_references`, both
/// A→B and B→A edges are added. Edges to unknown nodes and duplicate edges (same
/// neighbour appearing via both `related` and `cross_references`) are silently dropped.
pub struct ConstraintGraph {
    /// Canonical ordered list of node IDs.
    node_ids: Vec<String>,
    /// Map from ID string to index in `node_ids`.
    id_to_idx: HashMap<String, usize>,
    /// Normalised adjacency list. `adj[i]` is the list of outgoing edges from node `i`.
    /// Weights are already divided by the total outgoing weight so they sum to 1.0.
    adj: Vec<Vec<Edge>>,
}

impl ConstraintGraph {
    /// Build the graph from a slice of [`KnowledgeNode`]s.
    #[allow(clippy::cast_precision_loss)]
    pub fn build(nodes: &[KnowledgeNode]) -> Self {
        // --- index ---
        let node_ids: Vec<String> = nodes.iter().map(|n| n.id.clone()).collect();
        let id_to_idx: HashMap<String, usize> = node_ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.clone(), i))
            .collect();

        // raw adjacency: adj_raw[src] = set of dst indices (deduped)
        let n = node_ids.len();
        let mut adj_raw: Vec<std::collections::HashSet<usize>> =
            (0..n).map(|_| std::collections::HashSet::new()).collect();

        for node in nodes {
            let Some(&src) = id_to_idx.get(&node.id) else {
                continue;
            };

            // collect neighbour IDs from both `related` and `cross_references`
            let neighbours = node
                .related
                .iter()
                .map(String::as_str)
                .chain(node.cross_references.iter().map(|cr| cr.id.as_str()));

            for nb_id in neighbours {
                let Some(&dst) = id_to_idx.get(nb_id) else {
                    continue; // skip unknown nodes
                };
                if dst == src {
                    continue; // skip self-loops
                }
                // bidirectional
                adj_raw[src].insert(dst);
                adj_raw[dst].insert(src);
            }
        }

        // normalise weights: uniform weight 1.0 per edge, then divide by out-degree
        let adj: Vec<Vec<Edge>> = adj_raw
            .into_iter()
            .map(|neighbours| {
                let degree = neighbours.len();
                if degree == 0 {
                    return vec![];
                }
                let w = 1.0_f32 / degree as f32;
                neighbours
                    .into_iter()
                    .map(|dst| Edge { dst, weight: w })
                    .collect()
            })
            .collect();

        Self {
            node_ids,
            id_to_idx,
            adj,
        }
    }

    /// Personalised `PageRank`.
    ///
    /// Returns `(constraint_id, ppr_score)` sorted descending, with seed nodes excluded.
    /// Only entries with score > 0 are returned, truncated to `top_k`.
    ///
    /// # Parameters
    /// - `seed_ids`: starting nodes for the personalisation vector.
    /// - `alpha`: teleportation probability (restart probability).
    /// - `top_k`: maximum number of results.
    /// - `max_iter`: number of power-iteration steps.
    ///
    /// Personalised `PageRank`.
    ///
    /// Returns `(constraint_id, ppr_score)` sorted descending, with seed nodes excluded.
    /// Only entries with score > 0 are returned, truncated to `top_k`.
    ///
    /// # Parameters
    /// - `seed_ids`: starting nodes for the personalisation vector.
    /// - `alpha`: teleportation probability in (0, 1); clamped to [0.01, 0.99]. Standard: 0.15.
    /// - `top_k`: maximum number of results.
    /// - `max_iter`: power-iteration steps. Convergence is typically reached in 20 iterations
    ///   for graphs with hundreds of nodes. Increase for denser graphs.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn ppr(
        &self,
        seed_ids: &[&str],
        alpha: f32,
        top_k: usize,
        max_iter: usize,
    ) -> Vec<(String, f32)> {
        let alpha = alpha.clamp(0.01, 0.99);
        let n = self.node_ids.len();
        if n == 0 {
            return vec![];
        }

        // --- resolve valid seed indices ---
        let seed_indices: Vec<usize> = seed_ids
            .iter()
            .filter_map(|&id| self.id_to_idx.get(id).copied())
            .collect();

        if seed_indices.is_empty() {
            return vec![];
        }

        // --- personalisation vector ---
        let mut personal = vec![0.0_f32; n];
        let seed_weight = 1.0 / seed_indices.len() as f32;
        for &s in &seed_indices {
            personal[s] = seed_weight;
        }

        // --- power iteration ---
        let mut r = personal.clone();
        let mut new_r = vec![0.0_f32; n];

        for _ in 0..max_iter {
            // zero out accumulator
            new_r.fill(0.0);

            // propagate: new_r[dst] += (1 - alpha) * r[src] * w
            for (src, &r_src) in r.iter().enumerate() {
                if r_src == 0.0 {
                    // r[src] is exactly 0.0 (e.g. isolated non-seed); skip propagation
                    continue;
                }
                let mass = (1.0 - alpha) * r_src;
                for edge in &self.adj[src] {
                    new_r[edge.dst] += mass * edge.weight;
                }
            }

            // teleport: new_r[i] += alpha * personal[i]
            for i in 0..n {
                new_r[i] += alpha * personal[i];
            }

            std::mem::swap(&mut r, &mut new_r);
        }

        // --- collect results (exclude seeds) ---
        let seed_set: std::collections::HashSet<usize> = seed_indices.into_iter().collect();

        let mut results: Vec<(String, f32)> = r
            .into_iter()
            .enumerate()
            .filter(|&(i, score)| !seed_set.contains(&i) && score > 0.0)
            .map(|(i, score)| (self.node_ids[i].clone(), score))
            .collect();

        // sort descending by score
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }
}
