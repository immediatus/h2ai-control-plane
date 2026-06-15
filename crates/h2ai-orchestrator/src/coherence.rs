use h2ai_constraints::types::ConstraintDoc;
use h2ai_types::events::BranchPrunedEvent;
use h2ai_types::identity::ExplorerId;
use std::collections::{HashMap, HashSet};

/// Domain-level epistemic closure state for a MAPE-K wave.
///
/// A domain is "uncovered" when any of its constraints appear in a pruned proposal's
/// `violated_constraints` list AND no surviving proposal adequately covers that domain
/// (i.e., scores ≥ 0.5 on at least one constraint in the domain). A contradiction exists
/// when two surviving proposals score on opposite sides of the pass threshold for the same
/// constraint domain. When `is_closed()` is true, both fields are empty.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CoherenceState {
    /// Constraint domains that had violations in pruned proposals and are NOT covered by
    /// any surviving proposal. Sorted alphabetically.
    pub uncovered_domains: Vec<String>,
    /// Pairs of surviving proposals that contradict on the same constraint domain.
    /// Each entry is (`explorer_a`, `explorer_b`, domain). Sorted by domain then explorer ID.
    pub active_contradictions: Vec<(ExplorerId, ExplorerId, String)>,
}

impl CoherenceState {
    /// Compute uncovered domains from a constraint corpus and all pruned proposals.
    #[must_use]
    pub fn from_pruned(corpus: &[ConstraintDoc], all_pruned: &[BranchPrunedEvent]) -> Self {
        let violated: HashSet<String> = all_pruned
            .iter()
            .flat_map(|p| {
                p.violated_constraints
                    .iter()
                    .map(|v| v.constraint_id.clone())
            })
            .collect();

        if violated.is_empty() {
            return Self::default();
        }

        let mut id_to_domains: HashMap<String, Vec<String>> = HashMap::new();
        for doc in corpus {
            if !doc.domains.is_empty() {
                id_to_domains.insert(doc.id.clone(), doc.domains.clone());
            }
        }

        let mut uncovered: HashSet<String> = HashSet::new();
        for id in &violated {
            if let Some(domains) = id_to_domains.get(id) {
                for d in domains {
                    uncovered.insert(d.clone());
                }
            }
        }

        let mut uncovered_domains: Vec<String> = uncovered.into_iter().collect();
        uncovered_domains.sort();

        Self {
            uncovered_domains,
            active_contradictions: vec![],
        }
    }

    /// Augment with contradiction pairs detected from the constraint satisfaction matrix.
    ///
    /// A contradiction is when proposals i and j score on opposite sides of 0.5 on any
    /// constraint in the same domain. Only one entry per (pair, domain) is recorded.
    #[must_use]
    pub fn with_contradictions(
        mut self,
        corpus: &[ConstraintDoc],
        explorer_ids: &[ExplorerId],
        satisfaction_matrix: &[Vec<f64>],
        constraint_ids: &[String],
    ) -> Self {
        const THRESHOLD: f64 = 0.5;

        let id_to_domains: HashMap<&str, &[String]> = corpus
            .iter()
            .filter(|d| !d.domains.is_empty())
            .map(|d| (d.id.as_str(), d.domains.as_slice()))
            .collect();

        let mut domain_to_indices: HashMap<String, Vec<usize>> = HashMap::new();
        for (j, cid) in constraint_ids.iter().enumerate() {
            if let Some(domains) = id_to_domains.get(cid.as_str()) {
                for d in *domains {
                    domain_to_indices.entry(d.clone()).or_default().push(j);
                }
            }
        }

        if domain_to_indices.is_empty() {
            return self;
        }

        let n = explorer_ids.len();
        let mut seen: HashSet<(usize, usize, String)> = HashSet::new();

        for (domain, indices) in &domain_to_indices {
            for i in 0..n {
                for j in (i + 1)..n {
                    let contradicts = indices.iter().any(|&k| {
                        let si = satisfaction_matrix
                            .get(i)
                            .and_then(|row| row.get(k))
                            .copied()
                            .unwrap_or(0.0);
                        let sj = satisfaction_matrix
                            .get(j)
                            .and_then(|row| row.get(k))
                            .copied()
                            .unwrap_or(0.0);
                        (si >= THRESHOLD && sj < THRESHOLD) || (sj >= THRESHOLD && si < THRESHOLD)
                    });

                    if contradicts && seen.insert((i, j, domain.clone())) {
                        self.active_contradictions.push((
                            explorer_ids[i].clone(),
                            explorer_ids[j].clone(),
                            domain.clone(),
                        ));
                    }
                }
            }
        }

        self.active_contradictions.sort_by(|a, b| {
            a.2.cmp(&b.2)
                .then_with(|| a.0.to_string().cmp(&b.0.to_string()))
        });

        self
    }

    /// Remove domains from `uncovered_domains` that are adequately covered by at least
    /// one surviving proposal (score ≥ 0.5 on any constraint in the domain).
    ///
    /// A domain flagged by `from_pruned` is a false-positive uncovered signal when the
    /// winning proposal correctly satisfies the same domain. This method filters those
    /// out using the satisfaction matrix from the `ConstraintFrontierEvent`.
    #[must_use]
    pub fn filter_covered_by_survivors(
        mut self,
        corpus: &[ConstraintDoc],
        satisfaction_matrix: &[Vec<f64>],
        constraint_ids: &[String],
    ) -> Self {
        const PASS_THRESHOLD: f64 = 0.5;

        if self.uncovered_domains.is_empty() || satisfaction_matrix.is_empty() {
            return self;
        }

        let id_to_domains: HashMap<&str, &[String]> = corpus
            .iter()
            .filter(|d| !d.domains.is_empty())
            .map(|d| (d.id.as_str(), d.domains.as_slice()))
            .collect();

        let mut covered: HashSet<String> = HashSet::new();
        for (j, cid) in constraint_ids.iter().enumerate() {
            let any_survivor_passes = satisfaction_matrix
                .iter()
                .any(|row| row.get(j).copied().unwrap_or(0.0) >= PASS_THRESHOLD);
            if any_survivor_passes {
                if let Some(domains) = id_to_domains.get(cid.as_str()) {
                    for d in *domains {
                        covered.insert(d.clone());
                    }
                }
            }
        }

        self.uncovered_domains.retain(|d| !covered.contains(d));
        self
    }

    /// Remove domains from `uncovered_domains` that are provably covered by surviving proposals.
    ///
    /// Unlike `filter_covered_by_survivors` (which re-derives coverage from a satisfaction
    /// matrix), this method accepts a pre-computed `covered` set produced by the
    /// `llm_coverage` phase. Any domain present in `covered` is dropped from
    /// `uncovered_domains`. `active_contradictions` is left untouched.
    #[must_use]
    pub fn subtract_covered_domains(mut self, covered: &HashSet<String>) -> Self {
        self.uncovered_domains.retain(|d| !covered.contains(d));
        self
    }

    #[must_use]
    pub const fn is_closed(&self) -> bool {
        self.uncovered_domains.is_empty() && self.active_contradictions.is_empty()
    }
}
