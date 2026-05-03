use h2ai_config::H2AIConfig;
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::events::ProposalEvent;
use h2ai_types::sizing::TauValue;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Per-proposal verdict from the critique stage.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum CritiqueVerdict {
    Strong,
    Partial,
    Weak,
}

/// Critique of a single proposal produced by Stage 1.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposalCritique {
    pub proposal_id: String,
    pub strengths: Vec<String>,
    pub weaknesses: Vec<String>,
    pub verdict: CritiqueVerdict,
}

/// A contradiction between two or more proposals and its resolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContradictionRecord {
    pub proposals: Vec<String>,
    pub conflict_description: String,
    pub resolution: String,
}

/// Structured critique document produced by Stage 1 (LLM call).
/// Serialised to JSON for the `SynthesisCritiqueEvent` audit log and fed into Stage 2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CritiqueDocument {
    pub proposal_critiques: Vec<ProposalCritique>,
    pub contradictions: Vec<ContradictionRecord>,
    pub synthesis_guidance: String,
}

/// JSON schema description injected into Stage 1 prompt as `{critique_schema}`.
const CRITIQUE_SCHEMA: &str = r#"{
  "proposal_critiques": [
    {"proposal_id": "string", "strengths": ["string"], "weaknesses": ["string"], "verdict": "strong|partial|weak"}
  ],
  "contradictions": [
    {"proposals": ["string"], "conflict_description": "string", "resolution": "string"}
  ],
  "synthesis_guidance": "string"
}"#;

/// Input to `SynthesisPhase::run`.
pub struct SynthesisInput<'a> {
    pub task_description: &'a str,
    pub constraint_list: &'a str,
    pub proposals: &'a [ProposalEvent],
    pub adapter: &'a dyn IComputeAdapter,
    pub cfg: &'a H2AIConfig,
}

/// Successful output from `SynthesisPhase::run`.
#[derive(Debug)]
pub struct SynthesisOutput {
    pub critique_doc: CritiqueDocument,
    pub critique_doc_json: String,
    pub synthesis_text: String,
    pub critique_tokens: u64,
    pub synthesis_tokens: u64,
}

/// Errors produced by the synthesis phase.
#[derive(Debug, Error)]
pub enum SynthesisError {
    #[error("critique JSON parse failed after retry: {0}")]
    CritiqueFailed(String),
    #[error("synthesis adapter error: {0}")]
    AdapterError(String),
}

pub struct SynthesisPhase;

impl SynthesisPhase {
    /// Run the two-stage critique-then-write pipeline.
    ///
    /// Stage 1: call `adapter` with the critique prompt; parse → `CritiqueDocument`.
    /// Retries once on JSON parse failure with a stricter prompt prefix.
    /// Returns `SynthesisError::CritiqueFailed` if both attempts fail.
    ///
    /// Stage 2: call `adapter` with the synthesis prompt and the critique document;
    /// return the raw synthesis text for the caller to re-verify.
    pub async fn run(input: SynthesisInput<'_>) -> Result<SynthesisOutput, SynthesisError> {
        let proposals_block = Self::format_proposals(input.proposals);

        let tau = TauValue::new(input.cfg.synthesis_tau)
            .map_err(|e| SynthesisError::AdapterError(format!("invalid synthesis_tau: {e}")))?;

        // ── Stage 1: Critique ────────────────────────────────────────────────
        let critique_prompt = h2ai_types::prompts::SYNTHESIS_CRITIQUE_PROMPT
            .replace("{task_description}", input.task_description)
            .replace("{constraint_list}", input.constraint_list)
            .replace("{proposals_block}", &proposals_block)
            .replace("{critique_schema}", CRITIQUE_SCHEMA);

        let (critique_doc, critique_doc_json, critique_tokens) =
            Self::run_critique(&critique_prompt, input.adapter, tau, input.cfg).await?;

        // ── Stage 2: Synthesis ───────────────────────────────────────────────
        let synthesis_prompt = h2ai_types::prompts::SYNTHESIS_WRITE_PROMPT
            .replace("{task_description}", input.task_description)
            .replace("{constraint_list}", input.constraint_list)
            .replace("{proposals_block}", &proposals_block)
            .replace("{critique_document}", &critique_doc_json);

        let synthesis_req = ComputeRequest {
            system_context: String::new(),
            task: synthesis_prompt,
            tau,
            max_tokens: input.cfg.synthesis_max_tokens,
        };

        let synthesis_resp = input
            .adapter
            .execute(synthesis_req)
            .await
            .map_err(|e| SynthesisError::AdapterError(e.to_string()))?;

        Ok(SynthesisOutput {
            critique_doc,
            critique_doc_json,
            synthesis_text: synthesis_resp.output,
            critique_tokens,
            synthesis_tokens: synthesis_resp.token_cost,
        })
    }

    async fn run_critique(
        critique_prompt: &str,
        adapter: &dyn IComputeAdapter,
        tau: TauValue,
        cfg: &H2AIConfig,
    ) -> Result<(CritiqueDocument, String, u64), SynthesisError> {
        let req = ComputeRequest {
            system_context: String::new(),
            task: critique_prompt.to_string(),
            tau,
            max_tokens: cfg.synthesis_critique_max_tokens,
        };

        let resp = adapter
            .execute(req.clone())
            .await
            .map_err(|e| SynthesisError::AdapterError(e.to_string()))?;

        match serde_json::from_str::<CritiqueDocument>(&resp.output) {
            Ok(doc) => Ok((doc, resp.output, resp.token_cost)),
            Err(_) => {
                // One retry with a stricter instruction prepended
                let stricter = format!(
                    "You MUST respond with ONLY valid JSON matching the schema. No prose.\n\n{}",
                    critique_prompt
                );
                let retry_req = ComputeRequest {
                    task: stricter,
                    ..req
                };
                let retry_resp = adapter
                    .execute(retry_req)
                    .await
                    .map_err(|e| SynthesisError::AdapterError(e.to_string()))?;
                let doc = serde_json::from_str::<CritiqueDocument>(&retry_resp.output)
                    .map_err(|e| SynthesisError::CritiqueFailed(e.to_string()))?;
                Ok((doc, retry_resp.output, retry_resp.token_cost))
            }
        }
    }

    fn format_proposals(proposals: &[ProposalEvent]) -> String {
        proposals
            .iter()
            .enumerate()
            .map(|(i, p)| format!("--- Proposal {} ---\n{}", i + 1, p.raw_output))
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}
