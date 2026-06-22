use crate::gap_checkers::Gap;
use std::collections::{HashMap, HashSet, VecDeque};

/// Returned when `dispatch_batches` detects a cycle in the gap dependency DAG.
#[derive(Debug)]
pub struct CycleError(pub String);

impl std::fmt::Display for CycleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cycle detected in gap dependency DAG: {}", self.0)
    }
}

/// Holds a set of gaps for a task run and provides topological batch ordering.
pub struct GapRegistry {
    gaps: Vec<Gap>,
}

impl GapRegistry {
    pub fn new(gaps: Vec<Gap>) -> Self {
        Self { gaps }
    }

    pub fn gaps(&self) -> &[Gap] {
        &self.gaps
    }

    /// Returns gaps as concurrent dispatch batches via Kahn's algorithm.
    ///
    /// Gaps with no `depends_on` (or whose dependencies are in earlier batches) are
    /// placed in the same batch and can be dispatched concurrently. Each batch must
    /// complete before the next begins.
    ///
    /// Returns `Err(CycleError)` when the dependency graph contains a cycle.
    pub fn dispatch_batches(&self) -> Result<Vec<Vec<String>>, CycleError> {
        // Build adjacency and in-degree maps.
        let all_ids: HashSet<&str> = self.gaps.iter().map(|g| g.id.as_str()).collect();
        let mut in_degree: HashMap<&str, usize> =
            self.gaps.iter().map(|g| (g.id.as_str(), 0)).collect();
        let mut dependents: HashMap<&str, Vec<&str>> = HashMap::new();

        for gap in &self.gaps {
            if let Some(deps) = &gap.depends_on {
                for dep in deps {
                    if all_ids.contains(dep.as_str()) {
                        *in_degree.entry(gap.id.as_str()).or_insert(0) += 1;
                        dependents
                            .entry(dep.as_str())
                            .or_default()
                            .push(gap.id.as_str());
                    }
                }
            }
        }

        let mut queue: VecDeque<&str> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut batches: Vec<Vec<String>> = Vec::new();
        let mut visited = 0usize;

        while !queue.is_empty() {
            let batch_size = queue.len();
            let mut batch = Vec::with_capacity(batch_size);
            for _ in 0..batch_size {
                let node = queue.pop_front().unwrap();
                batch.push(node.to_string());
                visited += 1;
                if let Some(deps) = dependents.get(node) {
                    for &dep in deps {
                        let deg = in_degree.get_mut(dep).unwrap();
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(dep);
                        }
                    }
                }
            }
            batches.push(batch);
        }

        if visited != self.gaps.len() {
            let remaining: Vec<&str> = in_degree
                .iter()
                .filter(|(_, &deg)| deg > 0)
                .map(|(&id, _)| id)
                .collect();
            return Err(CycleError(remaining.join(", ")));
        }

        Ok(batches)
    }
}
