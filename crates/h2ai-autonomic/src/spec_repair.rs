use crate::coherence_probe::CoherenceProbe;
use h2ai_config::GapK1Config;
use h2ai_constraints::versioned::{RepairProvenance, VersionedConstraintSource};
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::sizing::TauValue;
use std::sync::Arc;

pub struct RepairInput {
    pub task_id: String,
    pub constraint_id: String,
    pub check_index: usize,
    pub original_check_text: String,
    pub divergent_reasons: Vec<String>,
    pub should_pass_example: String,
    pub should_prune_example: Option<String>,
    pub current_version: u64,
}

#[derive(Debug)]
pub enum RepairOutcome {
    Repaired { new_version: u64 },
    Failed { best_score: f64 },
}

pub struct SpecRepairAdvisor {
    cfg: GapK1Config,
}

impl SpecRepairAdvisor {
    pub fn new(cfg: GapK1Config) -> Self {
        Self { cfg }
    }

    pub async fn run(
        &self,
        input: RepairInput,
        source: Arc<impl VersionedConstraintSource>,
        adapter: &dyn IComputeAdapter,
    ) -> RepairOutcome {
        // 1. Generate candidate rewrites
        let candidates = self.generate_candidates(&input, adapter).await;
        if candidates.is_empty() {
            return RepairOutcome::Failed { best_score: 0.0 };
        }

        // 2. Probe each candidate
        let probe = CoherenceProbe::new(self.cfg.clone());
        let mut best_score = 0.0_f64;
        let mut best_candidate: Option<String> = None;

        for candidate in &candidates {
            let result = probe
                .run(candidate, &input.should_pass_example, adapter)
                .await;
            if result.consistency > best_score {
                best_score = result.consistency;
                best_candidate = Some(candidate.clone());
            }
        }

        // 3. Accept or reject
        if best_score < self.cfg.repair_acceptance_threshold {
            return RepairOutcome::Failed { best_score };
        }

        let best_text = match best_candidate {
            Some(t) => t,
            None => return RepairOutcome::Failed { best_score },
        };

        // 4. Build repaired spec
        let vs = match source.load_latest_versioned(&input.constraint_id).await {
            Ok(vs) => vs,
            Err(_) => return RepairOutcome::Failed { best_score },
        };

        let mut repaired_spec = vs.spec.clone();
        if let Some(check) = repaired_spec.rubric.checks.get_mut(input.check_index) {
            *check = best_text.clone();
        } else {
            return RepairOutcome::Failed { best_score };
        }

        let provenance = RepairProvenance {
            triggered_by_task: input.task_id.clone(),
            triggered_at_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            instability_score: mean_jaccard(&input.divergent_reasons),
            original_check_index: input.check_index,
            original_check_text: input.original_check_text.clone(),
            simplified_check_text: best_text,
            validation_consistency: best_score,
        };

        // 5. CAS write — retry once on conflict
        match source
            .create_next_version(
                &input.constraint_id,
                input.current_version,
                repaired_spec.clone(),
                provenance.clone(),
            )
            .await
        {
            Ok(new_version) => RepairOutcome::Repaired { new_version },
            Err(_conflict) => {
                // Reload and retry once
                match source.load_latest_versioned(&input.constraint_id).await {
                    Ok(fresh) => {
                        if fresh.spec.version > input.current_version {
                            return RepairOutcome::Repaired {
                                new_version: fresh.spec.version,
                            };
                        }
                        match source
                            .create_next_version(
                                &input.constraint_id,
                                fresh.spec.version,
                                repaired_spec,
                                provenance,
                            )
                            .await
                        {
                            Ok(v) => RepairOutcome::Repaired { new_version: v },
                            Err(_) => RepairOutcome::Failed { best_score },
                        }
                    }
                    Err(_) => RepairOutcome::Failed { best_score },
                }
            }
        }
    }

    async fn generate_candidates(
        &self,
        input: &RepairInput,
        adapter: &dyn IComputeAdapter,
    ) -> Vec<String> {
        let prune_section = match &input.should_prune_example {
            Some(p) => format!("\nSHOULD_PRUNE EXAMPLE:\n{p}"),
            None => String::new(),
        };
        let reasons = input.divergent_reasons.join("\n");
        let system = "You are a constraint specification engineer. \
            Rewrite an ambiguous binary compliance check into a single unambiguous assertion.\n\
            RULES:\n\
            - One clear pass/fail criterion only — no OR branches\n\
            - No conditional \"acceptable if...\" logic\n\
            - No multi-part \"all of the following\" lists\n\
            - The rewritten check MUST pass the SHOULD_PASS example";
        let task = format!(
            "ORIGINAL CHECK:\n{original}\n\n\
             EVIDENCE OF AMBIGUITY:\n{reasons}\n\n\
             SHOULD_PASS EXAMPLE:\n{pass}{prune}\n\n\
             OUTPUT: {n} candidate rewrites, one per line, no numbering.",
            original = input.original_check_text,
            pass = input.should_pass_example,
            prune = prune_section,
            n = self.cfg.repair_candidates,
        );

        let tau = TauValue::new(0.7).unwrap_or_else(|_| TauValue::new(0.5).expect("0.5 is valid"));
        let req = ComputeRequest {
            system_context: system.into(),
            task,
            tau,
            max_tokens: 512,
        };

        let Ok(resp) = adapter.execute(req).await else {
            return vec![];
        };

        resp.output
            .lines()
            .map(|l| l.trim().to_owned())
            .filter(|l| !l.is_empty())
            .take(self.cfg.repair_candidates)
            .collect()
    }
}

fn mean_jaccard(reasons: &[String]) -> f64 {
    if reasons.len() < 2 {
        return 1.0;
    }
    let mut sum = 0.0_f64;
    let mut count = 0usize;
    for i in 0..reasons.len() {
        for j in (i + 1)..reasons.len() {
            sum += jaccard_words(&reasons[i], &reasons[j]);
            count += 1;
        }
    }
    if count == 0 {
        1.0
    } else {
        sum / count as f64
    }
}

fn jaccard_words(a: &str, b: &str) -> f64 {
    use std::collections::HashSet;
    let a: HashSet<&str> = a.split_whitespace().collect();
    let b: HashSet<&str> = b.split_whitespace().collect();
    let union = a.union(&b).count();
    if union == 0 {
        return 1.0;
    }
    a.intersection(&b).count() as f64 / union as f64
}
