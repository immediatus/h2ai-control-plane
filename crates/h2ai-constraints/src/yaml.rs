use crate::types::{
    CompositeOp, ConstraintDoc, ConstraintPredicate, ConstraintSeverity, NumericOp,
};
use serde::Deserialize;
use std::path::Path;

/// A single deterministic numeric pre-condition. The regex must contain exactly one capture
/// group that yields a float-parseable string. Evaluated via `eval_sync` before the LlmJudge.
#[derive(Debug, Deserialize)]
pub struct NumericCheck {
    /// Regex with one capture group matching the numeric value to extract from the proposal.
    pub pattern: String,
    /// Comparison operator: "lt" | "le" | "eq" | "ge" | "gt"
    pub op: String,
    /// Threshold value to compare against.
    pub value: f64,
}

/// A declarative structural predicate for binary (pass/fail) gate evaluation.
/// Evaluated async via majority vote before LlmJudge in the And chain.
#[derive(Debug, Deserialize)]
pub struct StructuredPredicate {
    #[serde(rename = "type")]
    pub predicate_type: String,
    /// Required for semantic_presence
    pub concept: Option<String>,
    /// Required for semantic_ordering: the event that must come first
    pub first: Option<String>,
    /// Required for semantic_ordering: the event that must come after `first`
    pub then: Option<String>,
    /// Required for semantic_exclusion
    pub pattern: Option<String>,
    /// Number of independent LLM passes for majority vote. Default 3.
    #[serde(default = "default_binary_passes_yaml")]
    pub passes: u8,
}

fn default_binary_passes_yaml() -> u8 {
    3
}

/// A failure mode entry — enriches the LlmJudge rubric with known failure patterns.
/// Accepts both (trigger/consequence) and (name/description/impact) field variants
/// so the struct is compatible with existing constraint files.
#[derive(Debug, Deserialize)]
pub struct FailureModeYaml {
    pub id: String,
    /// Short label for the failure mode (alias: trigger)
    #[serde(alias = "trigger")]
    pub name: Option<String>,
    /// Full description of what goes wrong (alias: consequence)
    #[serde(alias = "consequence")]
    pub description: Option<String>,
    /// Impact of the failure (optional — not all existing files provide this)
    pub impact: Option<String>,
}

/// An example entry (positive or negative) — enriches the LlmJudge rubric.
/// Accepts both `scenario` and `description` field names for compatibility with existing files.
#[derive(Debug, Deserialize)]
pub struct ExampleYaml {
    /// Scenario label (alias: description — used by existing constraint files)
    #[serde(alias = "description")]
    pub scenario: Option<String>,
    pub code: Option<String>,
    pub why_wrong: Option<String>,
    pub why_correct: Option<String>,
}

impl ExampleYaml {
    fn label(&self) -> &str {
        self.scenario.as_deref().unwrap_or("")
    }

    fn rationale(&self) -> &str {
        self.why_wrong
            .as_deref()
            .or(self.why_correct.as_deref())
            .unwrap_or("")
    }
}

/// Structured YAML constraint file — the canonical format for new constraints.
///
/// Replaces the markdown heuristic (## Key Terms / ## Semantic Rules parsing) with
/// explicit typed fields. The framework assembles the LlmJudge rubric from `criteria`;
/// constraint authors never write JSON format instructions.
///
/// Minimal example:
/// ```yaml
/// id: CONSTRAINT-004
/// title: "Budget Pacing — Idempotency Protection"
/// severity: hard
/// criteria:
///   pass: "Idempotency key + atomic check-and-deduct."
///   fail: "No idempotency, or non-atomic check-then-act."
/// ```
#[derive(Debug, Deserialize)]
pub struct ConstraintYaml {
    pub id: String,
    pub title: String,

    /// hard | soft | advisory
    #[serde(default = "default_severity")]
    pub severity: SeverityKind,

    /// Threshold for Hard severity. Defaults by severity: hard→0.45, soft→ignored, advisory→ignored.
    pub threshold: Option<f64>,

    #[serde(default)]
    pub domains: Vec<String>,

    #[serde(default)]
    pub mandatory_for_tags: Vec<String>,

    /// Explicit cross-references to related constraint IDs for wiki graph navigation.
    #[serde(default)]
    pub related_to: Vec<String>,

    /// Shown in violation events to guide remediation.
    pub remediation_hint: Option<String>,

    /// Deterministic numeric pre-conditions that run before the LlmJudge rubric.
    /// When present, generates Composite(And([NumericThreshold..., LlmJudge])).
    #[serde(default)]
    pub numeric_checks: Vec<NumericCheck>,

    pub criteria: Criteria,

    /// Declarative binary structural predicates — evaluated async before LlmJudge.
    /// When non-empty, builds Composite(And([...structural, LlmJudge])).
    #[serde(default)]
    pub predicates: Vec<StructuredPredicate>,

    /// Failure modes appended to the LlmJudge rubric for richer evaluator context.
    #[serde(default)]
    pub failure_modes: Vec<FailureModeYaml>,

    /// Negative examples appended to the rubric to steer the model away from bad patterns.
    #[serde(default)]
    pub negative_examples: Vec<ExampleYaml>,

    /// Positive examples appended to the rubric to reinforce correct patterns.
    #[serde(default)]
    pub positive_examples: Vec<ExampleYaml>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum SeverityKind {
    #[default]
    Hard,
    Soft,
    Advisory,
}

/// Pass/partial/fail behavioral descriptions — no JSON format boilerplate.
/// The framework assembles these into the LlmJudge rubric.
#[derive(Debug, Deserialize)]
pub struct Criteria {
    /// What the proposal must do to score 1.0.
    pub pass: String,
    /// What scores 0.5 (optional — defaults to "partially satisfies the pass criteria").
    pub partial: Option<String>,
    /// What causes a 0.0 score.
    pub fail: String,
    /// Binary yes/no questions for Anchored CoT scoring.
    /// Score = count(PRESENT) / count(total). Overrides the 1.0/0.5/0.0 guide.
    #[serde(default)]
    pub checks: Vec<String>,
}

fn default_severity() -> SeverityKind {
    SeverityKind::Hard
}

impl ConstraintYaml {
    /// Assemble a LlmJudge rubric from structured criteria.
    ///
    /// The JSON response format lives in EVALUATOR_SYSTEM_PROMPT — not here.
    /// Domain context, remediation guidance, failure modes, and examples are appended
    /// so the evaluator can recognise compliant solutions without guessing the intent.
    pub fn build_rubric(&self) -> String {
        let partial = self.criteria.partial.as_deref().unwrap_or(
            "Partially satisfies the pass criteria, or intent is correct but a key detail is missing or unclear.",
        );
        let mut rubric = format!(
            "{title}\n\nPass (1.0): {pass}\n\nPartial (0.5): {partial}\n\nFail (0.0): {fail}",
            title = self.title,
            pass = self.criteria.pass.trim(),
            fail = self.criteria.fail.trim(),
        );
        if !self.domains.is_empty() {
            rubric.push_str(&format!("\n\nDomain: {}", self.domains.join(", ")));
        }
        if let Some(hint) = &self.remediation_hint {
            rubric.push_str(&format!("\n\nRemediation hint: {hint}"));
        }
        if !self.criteria.checks.is_empty() {
            rubric.push_str("\n\nBinary compliance checks — evaluate each in order:");
            for (i, check) in self.criteria.checks.iter().enumerate() {
                rubric.push_str(&format!("\n{}. {}", i + 1, check));
            }
            rubric.push_str(&format!(
                "\n\nScore = number of checks marked PRESENT divided by {} (total checks). Ignore the Pass/Partial/Fail guide above when binary checks are listed — compute score arithmetically.",
                self.criteria.checks.len()
            ));
        }
        if !self.failure_modes.is_empty() {
            rubric.push_str("\n\n--- Failure Modes ---");
            for fm in &self.failure_modes {
                let name = fm.name.as_deref().unwrap_or("");
                let desc = fm.description.as_deref().unwrap_or("");
                let impact_str = fm
                    .impact
                    .as_deref()
                    .map(|i| format!(" Impact: {i}"))
                    .unwrap_or_default();
                rubric.push_str(&format!("\n{} ({}): {}{}", fm.id, name, desc, impact_str));
            }
        }
        if !self.negative_examples.is_empty() {
            rubric.push_str("\n\n--- Negative Examples (DO NOT generate) ---");
            for ex in &self.negative_examples {
                let label = ex.label();
                if !label.is_empty() {
                    rubric.push_str(&format!("\nScenario: {label}"));
                }
                if let Some(code) = &ex.code {
                    rubric.push_str(&format!("\n```\n{code}\n```"));
                }
                let rationale = ex.rationale();
                if !rationale.is_empty() {
                    rubric.push_str(&format!("\nWhy wrong: {rationale}"));
                }
            }
        }
        if !self.positive_examples.is_empty() {
            rubric.push_str("\n\n--- Positive Examples (generate patterns like these) ---");
            for ex in &self.positive_examples {
                let label = ex.label();
                if !label.is_empty() {
                    rubric.push_str(&format!("\nScenario: {label}"));
                }
                if let Some(code) = &ex.code {
                    rubric.push_str(&format!("\n```\n{code}\n```"));
                }
                let rationale = ex.rationale();
                if !rationale.is_empty() {
                    rubric.push_str(&format!("\nWhy correct: {rationale}"));
                }
            }
        }
        rubric
    }

    pub fn into_constraint_doc(self) -> ConstraintDoc {
        let constraint_id = self.id.clone();
        let rubric = self.build_rubric();
        let severity = match self.severity {
            SeverityKind::Hard => ConstraintSeverity::Hard {
                threshold: self.threshold.unwrap_or(0.45),
            },
            SeverityKind::Soft => ConstraintSeverity::Soft {
                weight: self.threshold.unwrap_or(1.0),
            },
            SeverityKind::Advisory => ConstraintSeverity::Advisory,
        };

        let structural_predicates: Vec<ConstraintPredicate> = self
            .predicates
            .into_iter()
            .filter_map(|p| match p.predicate_type.as_str() {
                "semantic_ordering" => {
                    let first = p.first.filter(|s| !s.is_empty())?;
                    let then = p.then.filter(|s| !s.is_empty())?;
                    Some(ConstraintPredicate::SemanticOrdering {
                        first,
                        then,
                        passes: p.passes,
                    })
                }
                "semantic_presence" => {
                    let concept = p.concept.filter(|s| !s.is_empty())?;
                    Some(ConstraintPredicate::SemanticPresence {
                        concept,
                        passes: p.passes,
                    })
                }
                "semantic_exclusion" => {
                    let pattern = p.pattern.filter(|s| !s.is_empty())?;
                    Some(ConstraintPredicate::SemanticExclusion {
                        pattern,
                        passes: p.passes,
                    })
                }
                other => {
                    tracing::warn!(
                        constraint_id = %constraint_id,
                        predicate_type = other,
                        "unknown predicate type; skipping"
                    );
                    None
                }
            })
            .collect();

        let predicate = if self.numeric_checks.is_empty() && structural_predicates.is_empty() {
            ConstraintPredicate::LlmJudge { rubric }
        } else {
            let mut children: Vec<ConstraintPredicate> = self
                .numeric_checks
                .iter()
                .map(|nc| {
                    let op = match nc.op.as_str() {
                        "lt" => NumericOp::Lt,
                        "le" => NumericOp::Le,
                        "eq" => NumericOp::Eq,
                        "ge" => NumericOp::Ge,
                        "gt" => NumericOp::Gt,
                        other => {
                            tracing::warn!(
                                "unknown numeric_check op '{}'; defaulting to le",
                                other
                            );
                            NumericOp::Le
                        }
                    };
                    ConstraintPredicate::NumericThreshold {
                        field_pattern: nc.pattern.clone(),
                        op,
                        value: nc.value,
                    }
                })
                .collect();
            // Structural predicates run before LlmJudge — a 0.0 gate short-circuits it
            children.extend(structural_predicates);
            children.push(ConstraintPredicate::LlmJudge { rubric });
            ConstraintPredicate::Composite {
                op: CompositeOp::And,
                children,
            }
        };

        ConstraintDoc {
            id: self.id.clone(),
            source_file: self.id.clone(),
            description: self.title,
            severity,
            predicate,
            remediation_hint: self.remediation_hint,
            domains: self.domains,
            mandatory_for_tags: self.mandatory_for_tags,
            related_to: self.related_to,
        }
    }
}

/// Parse a single `.yaml` constraint file. Returns `None` on parse error (logged as warning).
pub fn parse_yaml_constraint(path: &Path, content: &str) -> Option<ConstraintDoc> {
    match serde_yaml::from_str::<ConstraintYaml>(content) {
        Ok(y) => Some(y.into_constraint_doc()),
        Err(e) => {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "failed to parse YAML constraint file; skipping"
            );
            None
        }
    }
}
