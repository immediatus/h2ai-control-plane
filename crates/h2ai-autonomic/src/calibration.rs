use chrono::Utc;
use h2ai_config::H2AIConfig;
use h2ai_context::jaccard::{jaccard, tokenize};
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::events::CalibrationCompletedEvent;
use h2ai_types::identity::TaskId;
use h2ai_types::physics::{CoherencyCoefficients, CoordinationThreshold, PhysicsError, TauValue};
use std::time::Instant;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CalibrationError {
    #[error("adapter error: {0}")]
    Adapter(String),
    #[error("physics error: {0}")]
    Physics(#[from] PhysicsError),
}

pub struct CalibrationInput<'a> {
    pub calibration_id: TaskId,
    pub task_prompts: Vec<String>,
    pub adapters: Vec<&'a dyn IComputeAdapter>,
    pub cfg: &'a H2AIConfig,
}

pub struct CalibrationHarness;

impl CalibrationHarness {
    pub async fn run(
        input: CalibrationInput<'_>,
    ) -> Result<CalibrationCompletedEvent, CalibrationError> {
        let tau = TauValue::new(input.cfg.calibration_tau).expect("calibration_tau must be in (0,1]");

        let mut adapter_outputs: Vec<Vec<String>> = Vec::new();
        let mut _total_sequential_ms = 0u128;

        for adapter in &input.adapters {
            let mut outputs = Vec::new();
            for prompt in &input.task_prompts {
                let req = ComputeRequest {
                    system_context: String::new(),
                    task: prompt.clone(),
                    tau,
                    max_tokens: input.cfg.calibration_max_tokens,
                };
                let t0 = Instant::now();
                let resp = adapter
                    .execute(req)
                    .await
                    .map_err(|e| CalibrationError::Adapter(e.to_string()))?;
                _total_sequential_ms += t0.elapsed().as_millis();
                outputs.push(resp.output);
            }
            adapter_outputs.push(outputs);
        }

        let alpha = if input.adapters.len() < 2 {
            0.12_f64
        } else {
            let n = input.adapters.len() as f64;
            (1.0 - 1.0 / n).clamp(0.05, 0.30)
        };

        let cg_samples: Vec<f64> = if input.adapters.len() < 2 {
            vec![0.7]
        } else {
            let mut pairs = Vec::new();
            for i in 0..adapter_outputs.len() {
                for j in (i + 1)..adapter_outputs.len() {
                    let outputs_i = adapter_outputs[i].join(" ");
                    let outputs_j = adapter_outputs[j].join(" ");
                    let ki = tokenize(&outputs_i);
                    let kj = tokenize(&outputs_j);
                    pairs.push(jaccard(&ki, &kj));
                }
            }
            pairs
        };

        let cg_mean: f64 = cg_samples.iter().sum::<f64>() / cg_samples.len() as f64;
        let kappa_base = (0.019 * cg_mean).max(0.010);

        let cc = CoherencyCoefficients::new(alpha, kappa_base, cg_samples)?;
        let coordination_threshold =
            CoordinationThreshold::from_calibration(&cc, input.cfg.coordination_threshold_max);

        Ok(CalibrationCompletedEvent {
            calibration_id: input.calibration_id,
            coefficients: cc,
            coordination_threshold,
            timestamp: Utc::now(),
        })
    }
}
