use std::collections::HashSet;

/// Execution tier for a constraint predicate, determining probe eligibility in Phase 1.5.
///
/// Static constraints (pure-Rust, microseconds) are the only tier eligible for the
/// N-probe satisfaction matrix. Heavy constraints (subprocess/oracle) are excluded to
/// avoid spiking coordination cost α during Phase 1.5 routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConstraintTier {
    /// Pure-Rust evaluation: `VocabularyPresence`, `NegativeKeyword`, `RegexMatch`,
    /// `NumericThreshold`, `JsonSchema`, `LengthRange`, Composite (when all children Static).
    Static,
    /// Single LLM call: `LlmJudge`. Acceptable probe cost but excluded for safety.
    Light,
    /// Subprocess / oracle: `OracleExecution`. Excluded from Phase 1.5 probing.
    Heavy,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum VocabularyMode {
    AllOf,
    AnyOf,
    NoneOf,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum NumericOp {
    Lt,
    Le,
    Eq,
    Ge,
    Gt,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum CompositeOp {
    And,
    Or,
    Not,
}

const fn default_oracle_timeout_secs() -> u64 {
    30
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
        children: Vec<Self>,
    },
    /// Tier 1: calls an external HTTP test runner for binary pass/fail oracle evaluation.
    OracleExecution {
        /// POST endpoint. Request: `{output, test_suite}`. Response: `{passed, failure_count, output_text, duration_ms}`.
        test_runner_uri: String,
        /// Test suite identifier passed to the runner (e.g., test file path or suite name).
        test_suite: String,
        /// Request timeout in seconds. Default 30.
        #[serde(default = "default_oracle_timeout_secs")]
        timeout_secs: u64,
    },
    /// Tier 2: validates that the output is valid JSON conforming to the given JSON Schema.
    JsonSchema {
        schema: serde_json::Value,
    },
    /// Tier 2: validates that the output character count falls within the given range.
    LengthRange {
        min_chars: Option<usize>,
        max_chars: Option<usize>,
    },
    /// Binary gate: does the response contain evidence of concept X?
    /// Async-only — `eval_sync` returns 1.0 (pass-through) so the Composite And engine defers it.
    SemanticPresence {
        concept: String,
        #[serde(default = "default_binary_passes")]
        passes: u8,
    },
    /// Binary gate: does `first` occur before `then` in the response?
    /// Async-only — `eval_sync` returns 1.0 (pass-through).
    SemanticOrdering {
        first: String,
        then: String,
        #[serde(default = "default_binary_passes")]
        passes: u8,
    },
    /// Binary gate: is `pattern` absent from the response?
    /// Async-only — `eval_sync` returns 1.0 (pass-through). Result is inverted: YES found → 0.0.
    SemanticExclusion {
        pattern: String,
        #[serde(default = "default_binary_passes")]
        passes: u8,
    },
}

const fn default_binary_passes() -> u8 {
    3
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
    /// Domain tags for wiki context routing (e.g. "`eu_data`", "`financial_report`").
    /// Parsed from YAML frontmatter in the constraint .md file.
    #[serde(default)]
    pub domains: Vec<String>,
    /// Force-inject this constraint when `task_tags` contain any of these values.
    /// Parsed from YAML frontmatter in the constraint .md file.
    #[serde(default)]
    pub mandatory_for_tags: Vec<String>,
    /// Explicit cross-references to related constraint IDs.
    /// Used for wiki graph navigation and retrieval context expansion.
    #[serde(default)]
    pub related_to: Vec<String>,
}

impl ConstraintDoc {
    /// Execution tier of this constraint's predicate.
    ///
    /// Returns the highest-cost tier among all predicates in the tree.
    /// Composite predicates propagate the maximum tier of their children.
    #[must_use]
    pub fn tier(&self) -> ConstraintTier {
        predicate_tier(&self.predicate)
    }

    /// All vocabulary terms from the predicate tree (positive and negative combined).
    /// Used for system context construction and keyword preservation in compaction.
    #[must_use]
    pub fn vocabulary(&self) -> HashSet<String> {
        let mut v = self.positive_vocabulary();
        v.extend(self.negative_vocabulary());
        v
    }

    /// Terms that a compliant proposal SHOULD contain (`AllOf` / `AnyOf` predicates).
    #[must_use]
    pub fn positive_vocabulary(&self) -> HashSet<String> {
        collect_positive_vocabulary(&self.predicate)
    }

    /// Terms that a compliant proposal MUST NOT contain (`NoneOf` / `NegativeKeyword` predicates).
    /// A task manifest that uses these terms is likely proposing constraint-violating behaviour.
    #[must_use]
    pub fn negative_vocabulary(&self) -> HashSet<String> {
        collect_negative_vocabulary(&self.predicate)
    }

    /// Build a minimal Hard `LlmJudge` constraint — use in tests instead of markdown parsing.
    #[must_use]
    pub fn new_llm_judge(id: &str, rubric: &str) -> Self {
        Self {
            id: id.to_owned(),
            source_file: format!("{id}.yaml"),
            description: String::new(),
            severity: ConstraintSeverity::Hard { threshold: 0.45 },
            predicate: ConstraintPredicate::Composite {
                op: CompositeOp::And,
                children: vec![ConstraintPredicate::LlmJudge {
                    rubric: rubric.to_owned(),
                }],
            },
            remediation_hint: None,
            domains: vec![],
            mandatory_for_tags: vec![],
            related_to: vec![],
        }
    }

    /// Build a Soft `LlmJudge` constraint — use in tests for soft-gate scenarios.
    #[must_use]
    pub fn new_soft_llm_judge(id: &str, rubric: &str) -> Self {
        Self {
            id: id.to_owned(),
            source_file: format!("{id}.yaml"),
            description: String::new(),
            severity: ConstraintSeverity::Soft { weight: 1.0 },
            predicate: ConstraintPredicate::Composite {
                op: CompositeOp::And,
                children: vec![ConstraintPredicate::LlmJudge {
                    rubric: rubric.to_owned(),
                }],
            },
            remediation_hint: None,
            domains: vec![],
            mandatory_for_tags: vec![],
            related_to: vec![],
        }
    }
}

fn predicate_tier(pred: &ConstraintPredicate) -> ConstraintTier {
    match pred {
        ConstraintPredicate::OracleExecution { .. } => ConstraintTier::Heavy,
        ConstraintPredicate::LlmJudge { .. }
        | ConstraintPredicate::SemanticPresence { .. }
        | ConstraintPredicate::SemanticOrdering { .. }
        | ConstraintPredicate::SemanticExclusion { .. } => ConstraintTier::Light,
        ConstraintPredicate::Composite { children, .. } => children
            .iter()
            .map(predicate_tier)
            .max()
            .unwrap_or(ConstraintTier::Static),
        _ => ConstraintTier::Static,
    }
}

fn collect_positive_vocabulary(pred: &ConstraintPredicate) -> HashSet<String> {
    match pred {
        ConstraintPredicate::VocabularyPresence {
            mode: VocabularyMode::AllOf | VocabularyMode::AnyOf,
            terms,
        } => terms.iter().cloned().collect(),
        ConstraintPredicate::Composite { children, .. } => children
            .iter()
            .flat_map(collect_positive_vocabulary)
            .collect(),
        _ => HashSet::new(),
    }
}

fn collect_negative_vocabulary(pred: &ConstraintPredicate) -> HashSet<String> {
    match pred {
        ConstraintPredicate::VocabularyPresence {
            mode: VocabularyMode::NoneOf,
            terms,
        }
        | ConstraintPredicate::NegativeKeyword { terms } => terms.iter().cloned().collect(),
        ConstraintPredicate::Composite { children, .. } => children
            .iter()
            .flat_map(collect_negative_vocabulary)
            .collect(),
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
    #[must_use]
    pub fn hard_passes(&self) -> bool {
        match &self.severity {
            ConstraintSeverity::Hard { threshold } => self.score >= *threshold,
            _ => true,
        }
    }
}

/// Weighted average score over Soft constraints. Returns 1.0 if no Soft constraints exist.
#[must_use]
pub fn aggregate_compliance_score(results: &[ComplianceResult]) -> f64 {
    let soft: Vec<_> = results
        .iter()
        .filter(|r| matches!(r.severity, ConstraintSeverity::Soft { .. }))
        .collect();
    if soft.is_empty() {
        return 1.0;
    }
    let (weighted_sum, total_weight): (f64, f64) = soft.iter().fold((0.0, 0.0), |(ws, tw), r| {
        let ConstraintSeverity::Soft { weight: w } = r.severity else {
            unreachable!()
        };
        (w.mul_add(r.score, ws), tw + w)
    });
    if total_weight == 0.0 {
        return 1.0;
    }
    weighted_sum / total_weight
}

/// Evaluation tier for Phase 4 lazy loading — determines whether a payload fetch is needed.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PredicateKind {
    /// Pure-Rust evaluation; predicate is inlined in `ConstraintMeta`.
    Static,
    /// Requires an LLM call with the rubric text; payload fetched from Predicate Store.
    LlmJudge,
    /// Requires an HTTP call to a test runner; payload fetched from Predicate Store.
    Oracle,
}

/// Lightweight constraint descriptor — loaded at Phase 1 Bootstrap.
///
/// ~300 bytes per entry; the entire wiki index fits in memory (30 MB for 100K constraints).
/// Used for: `system_context` injection (summary), Phase 4 routing (`predicate_kind`),
/// tag-based applicability resolution (`mandatory_for_tags`, domains).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConstraintMeta {
    pub id: String,
    /// 2–3 sentence synthesis of regulatory intent; injected into `system_context` (~50 tokens).
    pub summary: String,
    pub severity: ConstraintSeverity,
    pub predicate_kind: PredicateKind,
    pub domains: Vec<String>,
    /// Force-inject this constraint when any of these tags appear in `task_tags`.
    pub mandatory_for_tags: Vec<String>,
    /// Explicit cross-references to related constraint IDs.
    /// Populated from the YAML `related_to` field; used for wiki graph traversal.
    #[serde(default)]
    pub related_to: Vec<String>,
    /// Version pin for the Predicate Store entry; stored in `ConstraintSnapshot` for audit.
    pub payload_version: String,
    /// For Static predicates: full predicate inlined here; no Predicate Store fetch needed.
    #[serde(default)]
    pub inline_predicate: Option<ConstraintPredicate>,
    /// Provenance: source document path or URI (e.g. "nist-800-53/AC-2", "internal/policy-42").
    /// Used by the synthesis agent for staleness detection and audit.
    #[serde(default)]
    pub source: Option<String>,
    /// Unix epoch ms when this wiki entry was last synthesized/updated.
    /// Set by the synthesis agent; used for cache freshness and audit trails.
    #[serde(default)]
    pub last_updated_ms: Option<u64>,
}

impl ConstraintMeta {
    /// Build a `ConstraintMeta` from a `ConstraintDoc` for backward compatibility.
    ///
    /// Static predicates are inlined; `LlmJudge` and Oracle are left for lazy fetch.
    #[must_use]
    pub fn from_doc(doc: &ConstraintDoc) -> Self {
        let kind = match doc.tier() {
            ConstraintTier::Heavy => PredicateKind::Oracle,
            ConstraintTier::Light => PredicateKind::LlmJudge,
            ConstraintTier::Static => PredicateKind::Static,
        };
        let inline = if kind == PredicateKind::Static {
            Some(doc.predicate.clone())
        } else {
            None
        };
        Self {
            id: doc.id.clone(),
            summary: if doc.description.is_empty() {
                format!("Constraint {}: enforce compliance", doc.id)
            } else {
                doc.description.clone()
            },
            severity: doc.severity.clone(),
            predicate_kind: kind,
            domains: doc.domains.clone(),
            mandatory_for_tags: doc.mandatory_for_tags.clone(),
            related_to: doc.related_to.clone(),
            payload_version: "v1".to_string(),
            inline_predicate: inline,
            source: Some(doc.source_file.clone()),
            last_updated_ms: None,
        }
    }
}

/// Full constraint descriptor — fetched from the Predicate Store on demand during Phase 4.
///
/// Never preloaded. Fetched at most once per (constraint, proposal) pair.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConstraintPayload {
    pub id: String,
    pub version: String,
    pub predicate: ConstraintPredicate,
}
