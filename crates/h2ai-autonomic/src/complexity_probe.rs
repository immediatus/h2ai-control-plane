use h2ai_config::ComplexityRoutingConfig;
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::sizing::TauValue;

/// Result of a pre-dispatch complexity probe.
#[derive(Debug, Clone)]
pub struct ComplexityProbeResult {
    /// Rated complexity on 1–5 scale. 2 = conservative safe default on failure.
    pub complexity: u8,
    /// One-sentence rationale from the probe model.
    pub rationale: String,
    /// Whether the probe model recommends decomposition.
    pub decompose_recommended: bool,
}

impl Default for ComplexityProbeResult {
    fn default() -> Self {
        Self {
            complexity: 2,
            rationale: "probe unavailable; defaulting to complexity 2".into(),
            decompose_recommended: false,
        }
    }
}

/// Lightweight pre-dispatch probe that rates task complexity 1–5.
///
/// One cheap LLM call. On timeout or parse failure, defaults to `complexity = 2`
/// (conservative — never misroutes easy tasks to decomposition).
pub struct ComplexityProbe {
    cfg: ComplexityRoutingConfig,
}

const PROBE_SYSTEM: &str = "You are a task complexity evaluator. \
    Rate the computational complexity of the given task on a 1–5 scale and return ONLY JSON.";

const PROBE_SCALE: &str = "\
1 = factual lookup or direct retrieval\n\
2 = multi-step reasoning with clear structure (algorithm trace, code explanation)\n\
3 = constructive proof or algorithm design (invent a correct solution)\n\
4 = formal proof with multiple dependent sub-claims (liveness + safety + adversarial correctness)\n\
5 = requires verification beyond a single reasoning pass (NP-hard verification, cross-domain proof synthesis)";

impl ComplexityProbe {
    pub fn new(cfg: ComplexityRoutingConfig) -> Self {
        Self { cfg }
    }

    /// Run the probe against `task_description`. Returns `ComplexityProbeResult::default()`
    /// on any failure (adapter error, timeout, parse error).
    pub async fn run(
        &self,
        task_description: &str,
        adapter: &dyn IComputeAdapter,
    ) -> ComplexityProbeResult {
        let task = format!(
            "Given this task description, rate its computational complexity:\n\n\
            Scale:\n{PROBE_SCALE}\n\n\
            Task:\n{task_description}\n\n\
            Return JSON only: {{\"complexity\": <1-5>, \"rationale\": \"<one sentence>\", \"decompose_recommended\": <true|false>}}"
        );
        let tau = TauValue::new(0.1).unwrap_or_else(|_| TauValue::new(0.2).expect("0.2 is valid"));
        let req = ComputeRequest {
            system_context: PROBE_SYSTEM.into(),
            task,
            tau,
            max_tokens: 128,
        };

        let timeout = std::time::Duration::from_secs(self.cfg.complexity_probe_timeout_secs);
        match tokio::time::timeout(timeout, adapter.execute(req)).await {
            Ok(Ok(resp)) => parse_probe_result(&resp.output).unwrap_or_default(),
            Ok(Err(_)) | Err(_) => ComplexityProbeResult::default(),
        }
    }
}

/// Extract the first JSON object from `output` (model may add preamble text).
fn parse_probe_result(output: &str) -> Option<ComplexityProbeResult> {
    let start = output.find('{')?;
    let end = output.rfind('}')?;
    if end < start {
        return None;
    }
    let json_str = &output[start..=end];
    let v: serde_json::Value = serde_json::from_str(json_str).ok()?;
    let complexity = v.get("complexity")?.as_u64()? as u8;
    if !(1..=5).contains(&complexity) {
        return None;
    }
    Some(ComplexityProbeResult {
        complexity,
        rationale: v
            .get("rationale")
            .and_then(|r| r.as_str())
            .unwrap_or("")
            .to_string(),
        decompose_recommended: v
            .get("decompose_recommended")
            .and_then(|d| d.as_bool())
            .unwrap_or(false),
    })
}
