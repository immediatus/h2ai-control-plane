use crate::types::ConstraintDoc;
use std::collections::{HashMap, HashSet, VecDeque};

/// Groups `docs` into connected components via BFS on their `related_to` edges.
/// References to IDs not present in `docs` are ignored (no phantom nodes).
/// Returns a `Vec` of clusters; each cluster is a `Vec<String>` of constraint IDs.
pub fn build_clusters(docs: &[ConstraintDoc]) -> Vec<Vec<String>> {
    let known_ids: HashSet<&str> = docs.iter().map(|d| d.id.as_str()).collect();

    let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
    for doc in docs {
        adj.entry(doc.id.as_str()).or_default();
        for rel in &doc.related_to {
            let rel_str = rel.as_str();
            if known_ids.contains(rel_str) {
                adj.entry(doc.id.as_str()).or_default().push(rel_str);
                adj.entry(rel_str).or_default().push(doc.id.as_str());
            }
        }
    }

    let all_ids: Vec<&str> = docs.iter().map(|d| d.id.as_str()).collect();
    let mut visited: HashSet<&str> = HashSet::new();
    let mut clusters: Vec<Vec<String>> = Vec::new();

    for &id in &all_ids {
        if visited.contains(id) {
            continue;
        }
        let mut cluster: Vec<String> = Vec::new();
        let mut queue: VecDeque<&str> = VecDeque::new();
        queue.push_back(id);
        visited.insert(id);
        while let Some(node) = queue.pop_front() {
            cluster.push(node.to_owned());
            if let Some(neighbors) = adj.get(node) {
                for &n in neighbors {
                    if !visited.contains(n) {
                        visited.insert(n);
                        queue.push_back(n);
                    }
                }
            }
        }
        clusters.push(cluster);
    }

    clusters
}

/// Returns the global check indices (positions in `all_check_ids`) whose prefix
/// matches any constraint ID in `cluster_ids`.
/// Convention: check IDs are formatted as `"<constraint_id>:<check_text>"`.
pub fn cluster_check_indices(cluster_ids: &[String], all_check_ids: &[String]) -> Vec<usize> {
    let cluster_set: HashSet<&str> = cluster_ids.iter().map(|s| s.as_str()).collect();
    all_check_ids
        .iter()
        .enumerate()
        .filter_map(|(i, check)| {
            let prefix = check.split(':').next().unwrap_or("");
            if cluster_set.contains(prefix) {
                Some(i)
            } else {
                None
            }
        })
        .collect()
}
