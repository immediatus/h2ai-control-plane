use h2ai_config::JudgePanelConfig;
use h2ai_types::adapter::IComputeAdapter;
use h2ai_types::config::AdapterKind;
use h2ai_types::judge::{JudgePersona, PanelDiversityKind};

pub struct RuntimeJudgeVariant<'a> {
    pub adapter: &'a dyn IComputeAdapter,
    pub persona: JudgePersona,
    /// `None` = use VerificationConfig tau; `Some(t)` = override for persona diversity.
    pub temperature_override: Option<f32>,
}

pub struct JudgePanel<'a> {
    pub variants: Vec<RuntimeJudgeVariant<'a>>,
    pub diversity_kind: PanelDiversityKind,
}

impl<'a> JudgePanel<'a> {
    /// Build a panel from the primary verification adapter plus optional cross-family adapters.
    ///
    /// `additional` is a slice of `(adapter, adapter_kind)` pairs from the explorer pool.
    /// Deduplicates by family, selects cross-family first (cap 2 additional),
    /// falls back to 3-persona PersonaOnly when no cross-family adapters available.
    pub fn build(
        primary: &'a dyn IComputeAdapter,
        additional: &[(&'a dyn IComputeAdapter, &AdapterKind)],
        cfg: &JudgePanelConfig,
    ) -> Self {
        let primary_family = primary.kind().family();
        let mut seen_families = std::collections::HashSet::new();
        seen_families.insert(primary_family);
        let cross: Vec<&'a dyn IComputeAdapter> = additional
            .iter()
            .filter_map(|(adapter, kind)| {
                let fam = kind.family();
                if seen_families.insert(fam) {
                    Some(*adapter)
                } else {
                    None
                }
            })
            .take(2)
            .collect();

        if !cross.is_empty() {
            let mut variants = vec![RuntimeJudgeVariant {
                adapter: primary,
                persona: JudgePersona::Literal,
                temperature_override: None,
            }];
            for adapter in &cross {
                variants.push(RuntimeJudgeVariant {
                    adapter: *adapter,
                    persona: JudgePersona::Literal,
                    temperature_override: None,
                });
            }
            JudgePanel {
                variants,
                diversity_kind: PanelDiversityKind::CrossFamily,
            }
        } else {
            let temps = cfg.persona_temperatures;
            JudgePanel {
                variants: vec![
                    RuntimeJudgeVariant {
                        adapter: primary,
                        persona: JudgePersona::Literal,
                        temperature_override: Some(temps[0]),
                    },
                    RuntimeJudgeVariant {
                        adapter: primary,
                        persona: JudgePersona::Contextual,
                        temperature_override: Some(temps[1]),
                    },
                    RuntimeJudgeVariant {
                        adapter: primary,
                        persona: JudgePersona::Skeptical,
                        temperature_override: Some(temps[2]),
                    },
                ],
                diversity_kind: PanelDiversityKind::PersonaOnly,
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum ConstraintVerdict {
    Pass,
    Fail,
    Uncertain { votes_pass: usize, votes_fail: usize },
}

/// Aggregate per-constraint votes into a `ConstraintVerdict`.
///
/// CrossFamily: supermajority (quorum_fraction) required for confident verdict.
/// PersonaOnly: unanimous agreement required; any split → Uncertain.
pub fn aggregate_votes(
    votes_pass: usize,
    votes_fail: usize,
    diversity_kind: &PanelDiversityKind,
    quorum_fraction: f64,
) -> ConstraintVerdict {
    let total = votes_pass + votes_fail;
    if total == 0 {
        return ConstraintVerdict::Uncertain {
            votes_pass: 0,
            votes_fail: 0,
        };
    }
    match diversity_kind {
        PanelDiversityKind::CrossFamily => {
            let quorum = (total as f64 * quorum_fraction).ceil() as usize;
            if votes_pass >= quorum {
                ConstraintVerdict::Pass
            } else if votes_fail >= quorum {
                ConstraintVerdict::Fail
            } else {
                ConstraintVerdict::Uncertain {
                    votes_pass,
                    votes_fail,
                }
            }
        }
        PanelDiversityKind::PersonaOnly => {
            if votes_fail == 0 {
                ConstraintVerdict::Pass
            } else if votes_pass == 0 {
                ConstraintVerdict::Fail
            } else {
                ConstraintVerdict::Uncertain {
                    votes_pass,
                    votes_fail,
                }
            }
        }
    }
}
