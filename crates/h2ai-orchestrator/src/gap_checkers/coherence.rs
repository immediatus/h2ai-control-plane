use super::{Gap, GapCheckContext, GapChecker, GapKind, GapSeverity, GapSource};
use async_trait::async_trait;
use h2ai_config::prompts::{COHERENCE_CHECK_SYSTEM, COHERENCE_CHECK_TASK};
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::sizing::TauValue;
use serde::Deserialize;
use std::sync::Arc;

#[derive(Deserialize)]
struct CoherenceConflict {
    provision_a: String,
    provision_b: String,
    risk: String,
    severity: String,
}

fn severity_rank(s: &str) -> u8 {
    match s {
        "high" => 2,
        "medium" => 1,
        _ => 0,
    }
}

fn parse_severity(s: &str) -> GapSeverity {
    match s {
        "high" => GapSeverity::High,
        "medium" => GapSeverity::Medium,
        _ => GapSeverity::Low,
    }
}

/// Calls one LLM round-trip to detect inter-provision contradictions in the resolved output.
pub struct CoherenceChecker {
    adapter: Arc<dyn IComputeAdapter>,
    min_severity: String,
}

impl CoherenceChecker {
    pub fn new(adapter: Arc<dyn IComputeAdapter>, min_severity: String) -> Self {
        Self {
            adapter,
            min_severity,
        }
    }
}

#[async_trait]
impl GapChecker for CoherenceChecker {
    async fn check(&self, document: &str, context: &GapCheckContext) -> Vec<Gap> {
        let provisions_text = if context.verified_provision_list.is_empty() {
            document.to_string()
        } else {
            context.verified_provision_list.join("\n")
        };
        let task_text = COHERENCE_CHECK_TASK.replace("{provisions}", &provisions_text);

        let request = ComputeRequest {
            system_context: COHERENCE_CHECK_SYSTEM.to_string(),
            task: task_text,
            tau: TauValue::new(0.7).unwrap(),
            max_tokens: 1024,
        };

        let response = match self.adapter.execute(request).await {
            Ok(r) => r,
            Err(_) => return vec![],
        };

        let conflicts: Vec<CoherenceConflict> = match serde_json::from_str(&response.output) {
            Ok(v) => v,
            Err(_) => return vec![],
        };

        let min_rank = severity_rank(&self.min_severity);

        conflicts
            .into_iter()
            .filter(|c| severity_rank(&c.severity) >= min_rank)
            .enumerate()
            .map(|(i, c)| Gap {
                id: format!("coh-{}", i + 1),
                kind: GapKind::InterProvisionConflict,
                severity: parse_severity(&c.severity),
                description: format!(
                    "Conflict between '{}' and '{}': {}",
                    c.provision_a, c.provision_b, c.risk
                ),
                affected_provisions: vec![c.provision_a, c.provision_b],
                depends_on: None,
                source: GapSource::CoherenceCheck,
            })
            .collect()
    }
}
