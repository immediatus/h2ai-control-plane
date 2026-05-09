use crate::types::{ConstraintDoc, ConstraintPredicate, ConstraintSeverity};
use serde::Deserialize;
use std::path::Path;

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

    pub criteria: Criteria,
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
    /// Domain context and remediation guidance are appended when present so the
    /// evaluator can recognise compliant solutions without guessing the intent.
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
        rubric
    }

    pub fn into_constraint_doc(self) -> ConstraintDoc {
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
        ConstraintDoc {
            id: self.id.clone(),
            source_file: self.id.clone(),
            description: self.title,
            severity,
            predicate: ConstraintPredicate::LlmJudge { rubric },
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
