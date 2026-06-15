use h2ai_constraints::types::{ConstraintDoc, ConstraintSeverity};
use std::collections::HashSet;

pub struct Input<'a> {
    pub corpus: &'a [ConstraintDoc],
    pub survivor_count: usize,
    pub bypassed_ids: &'a HashSet<String>,
}

pub struct Output {
    pub covered_domains: Vec<String>,
}

/// Returns the sorted, deduplicated set of domains provably covered by the surviving proposals.
///
/// If `survivor_count == 0`, returns an empty vec. Otherwise, collects every domain from
/// Hard-severity constraints in the corpus, excluding any constraint whose ID appears in
/// `bypassed_ids`. Soft and Advisory constraints are excluded — coverage for non-Hard
/// constraints cannot be asserted from call-site data available here.
///
/// Bypassed Hard constraints are excluded because the MAPE-K bypass mechanism allows a
/// proposal to survive without satisfying that constraint, so its domain cannot be
/// declared covered.
///
/// The caller is responsible for ensuring that `survivor_count` reflects proposals that
/// genuinely satisfied all non-bypassed Hard constraints (e.g., proposals that cleared
/// `VerificationPhase`).
#[must_use]
pub fn run(input: Input<'_>) -> Output {
    if input.survivor_count == 0 {
        return Output {
            covered_domains: vec![],
        };
    }
    let mut covered: HashSet<String> = HashSet::new();
    for doc in input.corpus {
        if matches!(doc.severity, ConstraintSeverity::Hard { .. })
            && !input.bypassed_ids.contains(&doc.id)
        {
            for d in &doc.domains {
                covered.insert(d.clone());
            }
        }
    }
    let mut covered_domains: Vec<String> = covered.into_iter().collect();
    covered_domains.sort();
    Output { covered_domains }
}
