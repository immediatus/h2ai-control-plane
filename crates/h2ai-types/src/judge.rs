use serde::{Deserialize, Serialize};

/// Interpretive stance a judge takes when evaluating a proposal against a constraint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum JudgePersona {
    Literal,
    Contextual,
    Skeptical,
}

impl JudgePersona {
    #[must_use]
    pub const fn system_prompt_prefix(&self) -> &'static str {
        match self {
            Self::Literal => {
                "Evaluate this proposal against each constraint exactly as written. \
                Apply no benefit of the doubt; the proposal must satisfy the literal \
                text of each constraint."
            }
            Self::Contextual => {
                "Evaluate this proposal against each constraint in the spirit of its \
                intent. Tolerate reasonable variation in form as long as the underlying \
                requirement is satisfied."
            }
            Self::Skeptical => {
                "Evaluate this proposal against each constraint with a high bar for \
                compliance. Only mark a constraint as satisfied when the proposal provides \
                explicit, unambiguous evidence of compliance."
            }
        }
    }
}

/// Panel composition type — determines verdict aggregation rule.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PanelDiversityKind {
    /// ≥2 distinct model families; errors uncorrelated; supermajority vote applies.
    CrossFamily,
    /// Single model family with persona diversity; errors may be correlated; unanimous rule applies.
    PersonaOnly,
}
