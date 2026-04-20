use std::collections::HashSet;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum VocabularyMode {
    AllOf,
    AnyOf,
    NoneOf,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum NumericOp {
    Lt,
    Le,
    Eq,
    Ge,
    Gt,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum CompositeOp {
    And,
    Or,
    Not,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum ConstraintPredicate {
    VocabularyPresence {
        mode: VocabularyMode,
        terms: Vec<String>,
    },
    NegativeKeyword {
        terms: Vec<String>,
    },
    RegexMatch {
        pattern: String,
        must_match: bool,
    },
    NumericThreshold {
        field_pattern: String,
        op: NumericOp,
        value: f64,
    },
    LlmJudge {
        rubric: String,
    },
    Composite {
        op: CompositeOp,
        children: Vec<ConstraintPredicate>,
    },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum ConstraintSeverity {
    Hard { threshold: f64 },
    Soft { weight: f64 },
    Advisory,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConstraintDoc {
    pub id: String,
    pub source_file: String,
    pub description: String,
    pub severity: ConstraintSeverity,
    pub predicate: ConstraintPredicate,
    pub remediation_hint: Option<String>,
}

impl ConstraintDoc {
    /// All vocabulary terms from the predicate tree (positive and negative combined).
    /// Used for system context construction and keyword preservation in compaction.
    pub fn vocabulary(&self) -> HashSet<String> {
        let mut v = self.positive_vocabulary();
        v.extend(self.negative_vocabulary());
        v
    }

    /// Terms that a compliant proposal SHOULD contain (AllOf / AnyOf predicates).
    pub fn positive_vocabulary(&self) -> HashSet<String> {
        collect_positive_vocabulary(&self.predicate)
    }

    /// Terms that a compliant proposal MUST NOT contain (NoneOf / NegativeKeyword predicates).
    /// A task manifest that uses these terms is likely proposing constraint-violating behaviour.
    pub fn negative_vocabulary(&self) -> HashSet<String> {
        collect_negative_vocabulary(&self.predicate)
    }
}

fn collect_positive_vocabulary(pred: &ConstraintPredicate) -> HashSet<String> {
    match pred {
        ConstraintPredicate::VocabularyPresence {
            mode: VocabularyMode::AllOf | VocabularyMode::AnyOf,
            terms,
        } => terms.iter().cloned().collect(),
        ConstraintPredicate::Composite { children, .. } => {
            children.iter().flat_map(collect_positive_vocabulary).collect()
        }
        _ => HashSet::new(),
    }
}

fn collect_negative_vocabulary(pred: &ConstraintPredicate) -> HashSet<String> {
    match pred {
        ConstraintPredicate::VocabularyPresence {
            mode: VocabularyMode::NoneOf,
            terms,
        } => terms.iter().cloned().collect(),
        ConstraintPredicate::NegativeKeyword { terms } => terms.iter().cloned().collect(),
        ConstraintPredicate::Composite { children, .. } => {
            children.iter().flat_map(collect_negative_vocabulary).collect()
        }
        _ => HashSet::new(),
    }
}

#[derive(Debug, Clone)]
pub struct ComplianceResult {
    pub constraint_id: String,
    pub score: f64,
    pub severity: ConstraintSeverity,
    pub remediation_hint: Option<String>,
}

impl ComplianceResult {
    /// Returns true if this result does not block the hard gate.
    pub fn hard_passes(&self) -> bool {
        match &self.severity {
            ConstraintSeverity::Hard { threshold } => self.score >= *threshold,
            _ => true,
        }
    }
}

/// Weighted average score over Soft constraints. Returns 1.0 if no Soft constraints exist.
pub fn aggregate_compliance_score(results: &[ComplianceResult]) -> f64 {
    let soft: Vec<_> = results
        .iter()
        .filter(|r| matches!(r.severity, ConstraintSeverity::Soft { .. }))
        .collect();
    if soft.is_empty() {
        return 1.0;
    }
    let (weighted_sum, total_weight): (f64, f64) = soft.iter().fold((0.0, 0.0), |(ws, tw), r| {
        let w = match r.severity {
            ConstraintSeverity::Soft { weight } => weight,
            _ => unreachable!(),
        };
        (ws + w * r.score, tw + w)
    });
    if total_weight == 0.0 {
        return 1.0;
    }
    weighted_sum / total_weight
}
