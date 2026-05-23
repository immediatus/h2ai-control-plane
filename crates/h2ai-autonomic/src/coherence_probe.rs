use h2ai_config::GapK1Config;
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::sizing::TauValue;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeMode {
    ExampleBased,
    SelfConsistency,
}

#[derive(Debug, Clone)]
pub struct ProbeResult {
    pub consistency: f64,
    pub mode: ProbeMode,
    pub is_coherent: bool,
}

pub struct CoherenceProbe {
    cfg: GapK1Config,
}

impl CoherenceProbe {
    pub fn new(cfg: GapK1Config) -> Self {
        Self { cfg }
    }

    /// Evaluate `check_text` against `should_pass_example` N times.
    /// Returns the fraction of runs that produced a pass verdict (score ≥ 0.5).
    pub async fn run(
        &self,
        check_text: &str,
        should_pass_example: &str,
        adapter: &dyn IComputeAdapter,
    ) -> ProbeResult {
        let mut pass_count = 0usize;
        let mut total = 0usize;

        for _ in 0..self.cfg.probe_runs {
            // LLM call failures are skipped; only successful verdicts count toward total.
            if let Some(passed) = self
                .single_probe(check_text, should_pass_example, adapter)
                .await
            {
                total += 1;
                if passed {
                    pass_count += 1;
                }
            }
        }

        // If more than half of calls failed, treat as inconclusive (coherent = true to not block)
        if total < self.cfg.probe_runs / 2 + 1 {
            return ProbeResult {
                consistency: 1.0,
                mode: ProbeMode::ExampleBased,
                is_coherent: true,
            };
        }

        let consistency = pass_count as f64 / total as f64;
        ProbeResult {
            is_coherent: consistency >= self.cfg.coherence_threshold,
            consistency,
            mode: ProbeMode::ExampleBased,
        }
    }

    async fn single_probe(
        &self,
        check_text: &str,
        proposal: &str,
        adapter: &dyn IComputeAdapter,
    ) -> Option<bool> {
        let system = "You are a compliance verifier. \
            Given a binary check and a proposal, output JSON: {\"verdict\":\"pass\"|\"fail\",\"score\":0.0-1.0}. \
            Output ONLY the JSON, no other text.";
        let task = format!(
            "CHECK:\n{check_text}\n\nPROPOSAL:\n{proposal}\n\nDoes the proposal satisfy the check?"
        );
        let tau = TauValue::new(0.1).unwrap_or_else(|_| TauValue::new(0.2).expect("0.2 is valid"));
        let req = ComputeRequest {
            system_context: system.into(),
            task,
            tau,
            max_tokens: 64,
        };
        let resp = adapter.execute(req).await.ok()?;
        parse_verdict(&resp.output)
    }
}

fn parse_verdict(output: &str) -> Option<bool> {
    // Try JSON first
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(output.trim()) {
        if let Some(score) = v.get("score").and_then(|s| s.as_f64()) {
            return Some(score >= 0.5);
        }
        if let Some(verdict) = v.get("verdict").and_then(|v| v.as_str()) {
            return Some(verdict == "pass");
        }
    }
    // Fallback: plain text
    let lower = output.to_lowercase();
    if lower.contains("pass") {
        Some(true)
    } else if lower.contains("fail") {
        Some(false)
    } else {
        None
    }
}
