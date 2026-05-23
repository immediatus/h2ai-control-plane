use std::fmt::Write as _;

use crate::types::{CompositeOp, ConstraintDoc, ConstraintPredicate, ConstraintSeverity};

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SemanticSpec {
    pub id: String,
    pub title: String,
    pub source_file: String,
    pub severity: ConstraintSeverity,
    pub domains: Vec<String>,
    pub mandatory_for_tags: Vec<String>,
    pub related_to: Vec<String>,
    pub remediation_hint: Option<String>,
    pub exclusions: Vec<Exclusion>,
    pub requirements: Vec<Requirement>,
    pub orderings: Vec<Ordering>,
    pub rubric: QualityRubric,
    #[serde(default = "default_spec_version")]
    pub version: u64,
    #[serde(default)]
    pub repair_provenance: Option<crate::versioned::RepairProvenance>,
}

fn default_spec_version() -> u64 {
    1
}

#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct Exclusion {
    pub pattern: String,
    pub passes: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct Requirement {
    pub concept: String,
    pub passes: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct Ordering {
    pub first: String,
    pub then: String,
    pub passes: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct QualityRubric {
    pub pass: String,
    pub partial: Option<String>,
    pub fail: String,
    pub checks: Vec<String>,
    pub failure_modes: Vec<FailureMode>,
    pub negative_examples: Vec<Example>,
    pub positive_examples: Vec<Example>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct FailureMode {
    pub id: String,
    pub name: String,
    pub description: String,
    pub impact: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct Example {
    pub label: String,
    pub code: Option<String>,
    pub rationale: String,
}

impl SemanticSpec {
    pub fn builder(id: impl Into<String>) -> SemanticSpecBuilder {
        SemanticSpecBuilder {
            id: id.into(),
            title: String::new(),
            source_file: String::new(),
            severity: ConstraintSeverity::Hard { threshold: 0.5 },
            domains: vec![],
            mandatory_for_tags: vec![],
            related_to: vec![],
            remediation_hint: None,
            exclusions: vec![],
            requirements: vec![],
            orderings: vec![],
            rubric: QualityRubric::default(),
        }
    }

    /// Build the `LlmJudge` rubric string from all `SemanticSpec` fields.
    /// Mirrors `ConstraintYaml::build_rubric` — keeps <Domain:/Remediation> hint: format.
    #[must_use]
    pub fn build_rubric_text(&self) -> String {
        let partial = self.rubric.partial.as_deref().unwrap_or(
            "Partially satisfies the pass criteria, or intent is correct but a key detail is missing or unclear.",
        );
        let mut rubric = format!(
            "{title}\n\nPass (1.0): {pass}\n\nPartial (0.5): {partial}\n\nFail (0.0): {fail}",
            title = self.title,
            pass = self.rubric.pass.trim(),
            fail = self.rubric.fail.trim(),
        );
        if !self.domains.is_empty() {
            let _ = write!(rubric, "\n\nDomain: {}", self.domains.join(", "));
        }
        if let Some(hint) = &self.remediation_hint {
            let _ = write!(rubric, "\n\nRemediation hint: {hint}");
        }
        if !self.rubric.checks.is_empty() {
            rubric.push_str("\n\nBinary compliance checks — evaluate each in order:");
            for (i, check) in self.rubric.checks.iter().enumerate() {
                let _ = write!(rubric, "\n{}. {}", i + 1, check);
            }
            let _ = write!(
                rubric,
                "\n\nScore = number of checks marked PRESENT divided by {} (total checks). Ignore the Pass/Partial/Fail guide above when binary checks are listed — compute score arithmetically.",
                self.rubric.checks.len()
            );
        }
        if !self.rubric.failure_modes.is_empty() {
            rubric.push_str("\n\n--- Failure Modes ---");
            for fm in &self.rubric.failure_modes {
                let impact_str = fm
                    .impact
                    .as_deref()
                    .map(|i| format!(" Impact: {i}"))
                    .unwrap_or_default();
                let _ = write!(
                    rubric,
                    "\n{} ({}): {}{}",
                    fm.id, fm.name, fm.description, impact_str
                );
            }
        }
        if !self.rubric.negative_examples.is_empty() {
            rubric.push_str("\n\n--- Negative Examples (DO NOT generate) ---");
            for ex in &self.rubric.negative_examples {
                if !ex.label.is_empty() {
                    let _ = write!(rubric, "\nScenario: {}", ex.label);
                }
                if let Some(code) = &ex.code {
                    let _ = write!(rubric, "\n```\n{code}\n```");
                }
                if !ex.rationale.is_empty() {
                    let _ = write!(rubric, "\nWhy wrong: {}", ex.rationale);
                }
            }
        }
        if !self.rubric.positive_examples.is_empty() {
            rubric.push_str("\n\n--- Positive Examples (generate patterns like these) ---");
            for ex in &self.rubric.positive_examples {
                if !ex.label.is_empty() {
                    let _ = write!(rubric, "\nScenario: {}", ex.label);
                }
                if let Some(code) = &ex.code {
                    let _ = write!(rubric, "\n```\n{code}\n```");
                }
                if !ex.rationale.is_empty() {
                    let _ = write!(rubric, "\nWhy correct: {}", ex.rationale);
                }
            }
        }
        rubric
    }

    /// Compile to bytecode. Always produces `Composite(And([exclusions..., requirements..., orderings..., LlmJudge]))`.
    /// When all facets are empty → `Composite(And([LlmJudge]))`, behaviorally identical to old bare `LlmJudge`.
    #[must_use]
    pub fn into_constraint_doc(self) -> ConstraintDoc {
        let rubric_text = self.build_rubric_text();
        let mut children: Vec<ConstraintPredicate> = Vec::new();
        for e in self.exclusions {
            children.push(ConstraintPredicate::SemanticExclusion {
                pattern: e.pattern,
                passes: e.passes,
            });
        }
        for r in self.requirements {
            children.push(ConstraintPredicate::SemanticPresence {
                concept: r.concept,
                passes: r.passes,
            });
        }
        for o in self.orderings {
            children.push(ConstraintPredicate::SemanticOrdering {
                first: o.first,
                then: o.then,
                passes: o.passes,
            });
        }
        children.push(ConstraintPredicate::LlmJudge {
            rubric: rubric_text,
        });
        let binary_checks = self.rubric.checks;
        let pass_criteria = if self.rubric.pass.is_empty() {
            None
        } else {
            Some(self.rubric.pass.clone())
        };
        ConstraintDoc {
            id: self.id,
            source_file: self.source_file,
            description: self.title,
            severity: self.severity,
            predicate: ConstraintPredicate::Composite {
                op: CompositeOp::And,
                children,
            },
            remediation_hint: self.remediation_hint,
            domains: self.domains,
            mandatory_for_tags: self.mandatory_for_tags,
            related_to: self.related_to,
            binary_checks,
            version: self.version,
            repair_provenance: self.repair_provenance,
            pass_criteria,
        }
    }

    #[doc(hidden)]
    pub fn default_for_test(id: &str) -> Self {
        Self {
            id: id.to_owned(),
            title: id.to_owned(),
            source_file: format!("{id}.yaml"),
            severity: crate::types::ConstraintSeverity::Hard { threshold: 0.5 },
            domains: vec![],
            mandatory_for_tags: vec![],
            related_to: vec![],
            remediation_hint: None,
            exclusions: vec![],
            requirements: vec![],
            orderings: vec![],
            rubric: QualityRubric {
                pass: "pass".into(),
                partial: None,
                fail: "fail".into(),
                checks: vec![],
                failure_modes: vec![],
                negative_examples: vec![],
                positive_examples: vec![],
            },
            version: 1,
            repair_provenance: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SemanticSpecBuilder {
    id: String,
    title: String,
    source_file: String,
    severity: ConstraintSeverity,
    domains: Vec<String>,
    mandatory_for_tags: Vec<String>,
    related_to: Vec<String>,
    remediation_hint: Option<String>,
    exclusions: Vec<Exclusion>,
    requirements: Vec<Requirement>,
    orderings: Vec<Ordering>,
    rubric: QualityRubric,
}

impl SemanticSpecBuilder {
    #[must_use]
    pub fn title(mut self, t: impl Into<String>) -> Self {
        self.title = t.into();
        self
    }
    #[must_use]
    pub fn source_file(mut self, f: impl Into<String>) -> Self {
        self.source_file = f.into();
        self
    }
    #[must_use]
    pub const fn severity_hard(mut self, threshold: f64) -> Self {
        self.severity = ConstraintSeverity::Hard { threshold };
        self
    }
    #[must_use]
    pub const fn severity_soft(mut self, weight: f64) -> Self {
        self.severity = ConstraintSeverity::Soft { weight };
        self
    }
    #[must_use]
    pub fn domain(mut self, d: impl Into<String>) -> Self {
        self.domains.push(d.into());
        self
    }
    #[must_use]
    pub fn remediation_hint(mut self, h: impl Into<String>) -> Self {
        self.remediation_hint = Some(h.into());
        self
    }
    #[must_use]
    pub fn exclude(mut self, pattern: impl Into<String>) -> Self {
        self.exclusions.push(Exclusion {
            pattern: pattern.into(),
            passes: 3,
        });
        self
    }
    #[must_use]
    pub fn require(mut self, concept: impl Into<String>) -> Self {
        self.requirements.push(Requirement {
            concept: concept.into(),
            passes: 3,
        });
        self
    }
    #[must_use]
    pub fn order(mut self, first: impl Into<String>, then: impl Into<String>) -> Self {
        self.orderings.push(Ordering {
            first: first.into(),
            then: then.into(),
            passes: 3,
        });
        self
    }
    #[must_use]
    pub fn rubric_pass(mut self, p: impl Into<String>) -> Self {
        self.rubric.pass = p.into();
        self
    }
    #[must_use]
    pub fn rubric_partial(mut self, p: impl Into<String>) -> Self {
        self.rubric.partial = Some(p.into());
        self
    }
    #[must_use]
    pub fn rubric_fail(mut self, f: impl Into<String>) -> Self {
        self.rubric.fail = f.into();
        self
    }
    #[must_use]
    pub fn rubric_check(mut self, c: impl Into<String>) -> Self {
        self.rubric.checks.push(c.into());
        self
    }
    #[must_use]
    pub fn failure_mode(
        mut self,
        id: impl Into<String>,
        name: impl Into<String>,
        desc: impl Into<String>,
    ) -> Self {
        self.rubric.failure_modes.push(FailureMode {
            id: id.into(),
            name: name.into(),
            description: desc.into(),
            impact: None,
        });
        self
    }
    #[must_use]
    pub fn negative_example(mut self, ex: Example) -> Self {
        self.rubric.negative_examples.push(ex);
        self
    }
    #[must_use]
    pub fn positive_example(mut self, ex: Example) -> Self {
        self.rubric.positive_examples.push(ex);
        self
    }
    #[must_use]
    pub fn mandatory_for_tag(mut self, tag: impl Into<String>) -> Self {
        self.mandatory_for_tags.push(tag.into());
        self
    }
    #[must_use]
    pub fn related_to(mut self, id: impl Into<String>) -> Self {
        self.related_to.push(id.into());
        self
    }
    #[must_use]
    pub fn build(self) -> SemanticSpec {
        SemanticSpec {
            id: self.id,
            title: self.title,
            source_file: self.source_file,
            severity: self.severity,
            domains: self.domains,
            mandatory_for_tags: self.mandatory_for_tags,
            related_to: self.related_to,
            remediation_hint: self.remediation_hint,
            exclusions: self.exclusions,
            requirements: self.requirements,
            orderings: self.orderings,
            rubric: self.rubric,
            version: 1,
            repair_provenance: None,
        }
    }
}
