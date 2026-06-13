use crate::types::{CompositeOp, ConstraintDoc, ConstraintPredicate, NumericOp};
use std::collections::{HashMap, HashSet};

/// Static conflict graph built once from a constraint corpus.
///
/// A conflict pair (A, B) means satisfying A and B simultaneously may be impossible
/// under the detected predicate structure. CSPR-v2 uses this to emit `MetaRepair`
/// instructions instead of contradictory per-constraint hints.
pub struct ConstraintConflictGraph {
    conflict_pairs: HashSet<(String, String)>,
}

impl ConstraintConflictGraph {
    #[must_use]
    pub fn build(docs: &[ConstraintDoc]) -> Self {
        let mut conflict_pairs = HashSet::new();

        let mut orderings: Vec<(String, String, String)> = Vec::new();
        let mut numerics: HashMap<String, Vec<(String, NumericOp, f64)>> = HashMap::new();

        for doc in docs {
            collect_predicates(&doc.id, &doc.predicate, &mut orderings, &mut numerics);
        }

        // SemanticOrdering conflict: A(first=X, then=Y) vs B(first=Y, then=X)
        for i in 0..orderings.len() {
            for j in (i + 1)..orderings.len() {
                let (id_a, first_a, then_a) = &orderings[i];
                let (id_b, first_b, then_b) = &orderings[j];
                if first_a == then_b && then_a == first_b {
                    conflict_pairs.insert(canonical_pair(id_a, id_b));
                }
            }
        }

        // NumericThreshold conflict: same field, Le(v1) and Ge(v2) where v2 > v1
        for entries in numerics.values() {
            let les: Vec<_> = entries
                .iter()
                .filter(|(_, op, _)| matches!(op, NumericOp::Le))
                .collect();
            let ges: Vec<_> = entries
                .iter()
                .filter(|(_, op, _)| matches!(op, NumericOp::Ge))
                .collect();
            for (id_le, _, v_le) in &les {
                for (id_ge, _, v_ge) in &ges {
                    if v_ge > v_le {
                        conflict_pairs.insert(canonical_pair(id_le, id_ge));
                    }
                }
            }
        }

        // Seed coupling pairs from explicit `related_to` cross-references.
        for doc in docs {
            for related_id in &doc.related_to {
                conflict_pairs.insert(canonical_pair(&doc.id, related_id));
            }
        }

        Self { conflict_pairs }
    }

    #[must_use]
    pub fn are_conflicting(&self, id_a: &str, id_b: &str) -> bool {
        self.conflict_pairs.contains(&canonical_pair(id_a, id_b))
    }

    #[must_use]
    pub fn conflicts_for(&self, id: &str) -> Vec<&str> {
        self.conflict_pairs
            .iter()
            .filter_map(|(a, b)| {
                if a == id {
                    Some(b.as_str())
                } else if b == id {
                    Some(a.as_str())
                } else {
                    None
                }
            })
            .collect()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.conflict_pairs.is_empty()
    }
}

/// Returns a canonical (lexicographically ordered) pair of constraint IDs,
/// used as the key for the conflict set so insertion order doesn't matter.
fn canonical_pair(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_owned(), b.to_owned())
    } else {
        (b.to_owned(), a.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ConstraintDoc;

    #[test]
    fn seeds_coupling_from_related_to() {
        let make_doc = |id: &str, related: &[&str]| -> ConstraintDoc {
            ConstraintDoc {
                related_to: related.iter().map(|s| s.to_string()).collect(),
                ..ConstraintDoc::new_llm_judge(id, "")
            }
        };
        let docs = vec![
            make_doc("C-005", &["C-TAU-2"]),
            make_doc("C-TAU-2", &["C-005"]),
        ];
        let graph = ConstraintConflictGraph::build(&docs);
        assert!(
            graph.are_conflicting("C-005", "C-TAU-2"),
            "related_to cross-reference must produce a coupling pair in the graph"
        );
        assert!(graph.conflicts_for("C-005").contains(&"C-TAU-2"));
    }

    #[test]
    fn related_to_only_on_one_side_still_seeds() {
        let docs = vec![
            ConstraintDoc {
                related_to: vec!["B".to_string()],
                ..ConstraintDoc::new_llm_judge("A", "")
            },
            ConstraintDoc::new_llm_judge("B", ""),
        ];
        let graph = ConstraintConflictGraph::build(&docs);
        assert!(graph.are_conflicting("A", "B"));
    }
}

fn collect_predicates(
    id: &str,
    pred: &ConstraintPredicate,
    orderings: &mut Vec<(String, String, String)>,
    numerics: &mut HashMap<String, Vec<(String, NumericOp, f64)>>,
) {
    match pred {
        ConstraintPredicate::SemanticOrdering { first, then, .. } => {
            orderings.push((id.to_owned(), first.clone(), then.clone()));
        }
        ConstraintPredicate::NumericThreshold {
            field_pattern,
            op,
            value,
        } => {
            // Lt, Gt, Eq variants are not analysed — feasibility analysis is only
            // implemented for Le/Ge range conflicts.
            numerics.entry(field_pattern.clone()).or_default().push((
                id.to_owned(),
                op.clone(),
                *value,
            ));
        }
        // Only recurse into And composites: an Or or Not composite does not
        // guarantee any individual child predicate holds, so treating its
        // children as enforced constraints would produce false positives.
        ConstraintPredicate::Composite {
            op: CompositeOp::And,
            children,
        } => {
            for child in children {
                collect_predicates(id, child, orderings, numerics);
            }
        }
        _ => {}
    }
}
