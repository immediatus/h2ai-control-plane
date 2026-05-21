use std::fmt::Write as _;
use std::path::Path;

use serde::Deserialize;

use crate::types::{
    CompositeOp, ConstraintDoc, ConstraintPredicate, ConstraintSeverity, NumericOp,
};

/// A single deterministic numeric pre-condition. The regex must contain exactly one capture
/// group that yields a float-parseable string. Evaluated via `eval_sync` before the `LlmJudge`.
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
/// Evaluated async via majority vote before `LlmJudge` in the And chain.
#[derive(Debug, Deserialize)]
pub struct StructuredPredicate {
    #[serde(rename = "type")]
    pub predicate_type: String,
    /// Required for `semantic_presence`
    pub concept: Option<String>,
    /// Required for `semantic_ordering`: the event that must come first
    pub first: Option<String>,
    /// Required for `semantic_ordering`: the event that must come after `first`
    pub then: Option<String>,
    /// Required for `semantic_exclusion`
    pub pattern: Option<String>,
    /// Number of independent LLM passes for majority vote. Default 3.
    #[serde(default = "default_binary_passes_yaml")]
    pub passes: u8,
}

const fn default_binary_passes_yaml() -> u8 {
    3
}

#[derive(Debug, Deserialize, Default)]
pub struct SemanticSectionYaml {
    #[serde(default)]
    pub exclusions: Vec<ExclusionYaml>,
    #[serde(default)]
    pub requirements: Vec<RequirementYaml>,
    #[serde(default)]
    pub orderings: Vec<OrderingYaml>,
}

#[derive(Debug, Deserialize)]
pub struct ExclusionYaml {
    pub pattern: String,
    #[serde(default = "default_binary_passes_yaml")]
    pub passes: u8,
}

#[derive(Debug, Deserialize)]
pub struct RequirementYaml {
    pub concept: String,
    #[serde(default = "default_binary_passes_yaml")]
    pub passes: u8,
}

#[derive(Debug, Deserialize)]
pub struct OrderingYaml {
    pub first: String,
    pub then: String,
    #[serde(default = "default_binary_passes_yaml")]
    pub passes: u8,
}

/// A failure mode entry — enriches the `LlmJudge` rubric with known failure patterns.
///
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

/// An example entry (positive or negative) — enriches the `LlmJudge` rubric.
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
/// explicit typed fields. The framework assembles the `LlmJudge` rubric from `criteria`;
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

    /// Deterministic numeric pre-conditions that run before the `LlmJudge` rubric.
    /// When present, generates `Composite(And([NumericThreshold..., LlmJudge]))`.
    #[serde(default)]
    pub numeric_checks: Vec<NumericCheck>,

    pub criteria: Criteria,

    /// Declarative binary structural predicates — evaluated async before `LlmJudge`.
    /// When non-empty, builds `Composite(And([...structural, LlmJudge]))`.
    #[serde(default)]
    pub predicates: Vec<StructuredPredicate>,

    /// Typed semantic facets — preferred over the legacy `predicates:` array.
    #[serde(default)]
    pub semantic: Option<SemanticSectionYaml>,

    /// Failure modes appended to the `LlmJudge` rubric for richer evaluator context.
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
/// The framework assembles these into the `LlmJudge` rubric.
#[derive(Debug, Deserialize)]
pub struct Criteria {
    /// What the proposal must do to score 1.0.
    pub pass: String,
    /// What scores 0.5 (optional — defaults to "partially satisfies the pass criteria").
    pub partial: Option<String>,
    /// What causes a 0.0 score.
    pub fail: String,
    /// Binary yes/no questions for Anchored `CoT` scoring.
    /// Score = count(PRESENT) / count(total). Overrides the 1.0/0.5/0.0 guide.
    #[serde(default)]
    pub checks: Vec<String>,
}

const fn default_severity() -> SeverityKind {
    SeverityKind::Hard
}

impl ConstraintYaml {
    /// Assemble a `LlmJudge` rubric from structured criteria.
    ///
    /// The JSON response format lives in `EVALUATOR_SYSTEM_PROMPT` — not here.
    /// Domain context, remediation guidance, failure modes, and examples are appended
    /// so the evaluator can recognise compliant solutions without guessing the intent.
    #[must_use]
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
            let _ = write!(rubric, "\n\nDomain: {}", self.domains.join(", "));
        }
        if let Some(hint) = &self.remediation_hint {
            let _ = write!(rubric, "\n\nRemediation hint: {hint}");
        }
        if !self.criteria.checks.is_empty() {
            rubric.push_str("\n\nBinary compliance checks — evaluate each in order:");
            for (i, check) in self.criteria.checks.iter().enumerate() {
                let _ = write!(rubric, "\n{}. {}", i + 1, check);
            }
            let _ = write!(
                rubric,
                "\n\nScore = number of checks marked PRESENT divided by {} (total checks). Ignore the Pass/Partial/Fail guide above when binary checks are listed — compute score arithmetically.",
                self.criteria.checks.len()
            );
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
                let _ = write!(rubric, "\n{} ({}): {}{}", fm.id, name, desc, impact_str);
            }
        }
        if !self.negative_examples.is_empty() {
            rubric.push_str("\n\n--- Negative Examples (DO NOT generate) ---");
            for ex in &self.negative_examples {
                let label = ex.label();
                if !label.is_empty() {
                    let _ = write!(rubric, "\nScenario: {label}");
                }
                if let Some(code) = &ex.code {
                    let _ = write!(rubric, "\n```\n{code}\n```");
                }
                let rationale = ex.rationale();
                if !rationale.is_empty() {
                    let _ = write!(rubric, "\nWhy wrong: {rationale}");
                }
            }
        }
        if !self.positive_examples.is_empty() {
            rubric.push_str("\n\n--- Positive Examples (generate patterns like these) ---");
            for ex in &self.positive_examples {
                let label = ex.label();
                if !label.is_empty() {
                    let _ = write!(rubric, "\nScenario: {label}");
                }
                if let Some(code) = &ex.code {
                    let _ = write!(rubric, "\n```\n{code}\n```");
                }
                let rationale = ex.rationale();
                if !rationale.is_empty() {
                    let _ = write!(rubric, "\nWhy correct: {rationale}");
                }
            }
        }
        rubric
    }

    #[must_use]
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

        let structural_predicates: Vec<ConstraintPredicate> =
            if self.semantic.is_some() && !self.predicates.is_empty() {
                tracing::warn!(
                    constraint_id = %constraint_id,
                    "contains both 'semantic:' and 'predicates:' — ignoring structural predicates"
                );
                vec![]
            } else {
                let section = match self.semantic {
                    Some(s) => s,
                    None => map_legacy_predicates(&self.predicates),
                };
                let mut children = Vec::new();
                for e in section.exclusions {
                    children.push(ConstraintPredicate::SemanticExclusion {
                        pattern: e.pattern,
                        passes: e.passes,
                    });
                }
                for r in section.requirements {
                    children.push(ConstraintPredicate::SemanticPresence {
                        concept: r.concept,
                        passes: r.passes,
                    });
                }
                for o in section.orderings {
                    children.push(ConstraintPredicate::SemanticOrdering {
                        first: o.first,
                        then: o.then,
                        passes: o.passes,
                    });
                }
                children
            };

        let mut children: Vec<ConstraintPredicate> = self
            .numeric_checks
            .iter()
            .map(|nc| {
                let op = match nc.op.as_str() {
                    "lt" => NumericOp::Lt,
                    "eq" => NumericOp::Eq,
                    "ge" => NumericOp::Ge,
                    "gt" => NumericOp::Gt,
                    "le" => NumericOp::Le,
                    other => {
                        tracing::warn!("unknown numeric_check op '{}'; defaulting to le", other);
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
        children.extend(structural_predicates);
        children.push(ConstraintPredicate::LlmJudge { rubric });

        let predicate = ConstraintPredicate::Composite {
            op: CompositeOp::And,
            children,
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

    /// Convert to `SemanticSpec` IR. Returns `Err(message)` on key collision (Fix #4).
    ///
    /// # Errors
    /// Returns an error string when both `semantic:` and the deprecated `predicates:` keys are present.
    pub fn into_semantic_spec(self) -> Result<crate::spec::SemanticSpec, String> {
        use crate::spec::{
            Example, Exclusion, FailureMode, Ordering, QualityRubric, Requirement, SemanticSpec,
        };

        if self.semantic.is_some() && !self.predicates.is_empty() {
            return Err(format!(
                "Constraint '{}' contains both 'semantic:' and 'predicates:'. \
                 Remove the deprecated 'predicates:' key before loading.",
                self.id
            ));
        }

        let section = match self.semantic {
            Some(s) => s,
            None => map_legacy_predicates(&self.predicates),
        };

        let severity = match self.severity {
            SeverityKind::Hard => crate::types::ConstraintSeverity::Hard {
                threshold: self.threshold.unwrap_or(0.45),
            },
            SeverityKind::Soft => crate::types::ConstraintSeverity::Soft {
                weight: self.threshold.unwrap_or(1.0),
            },
            SeverityKind::Advisory => crate::types::ConstraintSeverity::Advisory,
        };

        let failure_modes: Vec<FailureMode> = self
            .failure_modes
            .into_iter()
            .map(|fm| FailureMode {
                id: fm.id,
                name: fm.name.unwrap_or_default(),
                description: fm.description.unwrap_or_default(),
                impact: fm.impact,
            })
            .collect();

        let negative_examples: Vec<Example> = self
            .negative_examples
            .into_iter()
            .map(|ex| Example {
                label: ex.scenario.unwrap_or_default(),
                code: ex.code,
                rationale: ex.why_wrong.unwrap_or_default(),
            })
            .collect();

        let positive_examples: Vec<Example> = self
            .positive_examples
            .into_iter()
            .map(|ex| Example {
                label: ex.scenario.unwrap_or_default(),
                code: ex.code,
                rationale: ex.why_correct.unwrap_or_default(),
            })
            .collect();

        Ok(SemanticSpec {
            id: self.id.clone(),
            title: self.title,
            source_file: self.id,
            severity,
            domains: self.domains,
            mandatory_for_tags: self.mandatory_for_tags,
            related_to: self.related_to,
            remediation_hint: self.remediation_hint,
            exclusions: section
                .exclusions
                .into_iter()
                .map(|e| Exclusion {
                    pattern: e.pattern,
                    passes: e.passes,
                })
                .collect(),
            requirements: section
                .requirements
                .into_iter()
                .map(|r| Requirement {
                    concept: r.concept,
                    passes: r.passes,
                })
                .collect(),
            orderings: section
                .orderings
                .into_iter()
                .map(|o| Ordering {
                    first: o.first,
                    then: o.then,
                    passes: o.passes,
                })
                .collect(),
            rubric: QualityRubric {
                pass: self.criteria.pass,
                partial: self.criteria.partial,
                fail: self.criteria.fail,
                checks: self.criteria.checks,
                failure_modes,
                negative_examples,
                positive_examples,
            },
        })
    }
}

fn map_legacy_predicates(predicates: &[StructuredPredicate]) -> SemanticSectionYaml {
    let mut section = SemanticSectionYaml::default();
    for p in predicates {
        match p.predicate_type.as_str() {
            "semantic_ordering" => {
                if let (Some(first), Some(then)) = (
                    p.first.as_ref().filter(|s| !s.is_empty()),
                    p.then.as_ref().filter(|s| !s.is_empty()),
                ) {
                    section.orderings.push(OrderingYaml {
                        first: first.clone(),
                        then: then.clone(),
                        passes: p.passes,
                    });
                }
            }
            "semantic_presence" => {
                if let Some(concept) = p.concept.as_ref().filter(|s| !s.is_empty()) {
                    section.requirements.push(RequirementYaml {
                        concept: concept.clone(),
                        passes: p.passes,
                    });
                }
            }
            "semantic_exclusion" => {
                if let Some(pattern) = p.pattern.as_ref().filter(|s| !s.is_empty()) {
                    section.exclusions.push(ExclusionYaml {
                        pattern: pattern.clone(),
                        passes: p.passes,
                    });
                }
            }
            _ => {}
        }
    }
    section
}

/// Parse a single `.yaml` constraint file. Returns `None` on parse error or key collision
/// (both logged as warnings).
pub fn parse_yaml_constraint(path: &Path, content: &str) -> Option<ConstraintDoc> {
    match serde_yaml::from_str::<ConstraintYaml>(content) {
        Ok(y) => {
            if y.semantic.is_some() && !y.predicates.is_empty() {
                tracing::warn!(
                    path = %path.display(),
                    id = %y.id,
                    "constraint contains both 'semantic:' and 'predicates:' — skipping"
                );
                return None;
            }
            Some(y.into_constraint_doc())
        }
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
