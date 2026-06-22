use crate::gap_checkers::{GapKind, GapResolveContext, GapResolver, ResolutionResult};
use async_trait::async_trait;
use h2ai_config::prompts::{RECOVERY_SYSTEM, RECOVERY_TASK};
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::sizing::TauValue;
use std::sync::Arc;

/// Resolves `MissingProvision` and `IncompleteProvision` gaps via a single focused LLM call.
///
/// The resolver injects the full constraint text and the verified-provision list into the
/// RECOVERY_TASK prompt so the LLM cannot accidentally overwrite already-passing provisions.
///
/// Acceptance is binary: if a non-empty patch is returned it is accepted (`score_delta = 1.0`);
/// if the adapter returns nothing or fails, no patch is produced (`score_delta = 0.0`).
/// The resolver cannot compute a real verification delta without a second LLM pass.
pub struct MicroExplorerResolver {
    adapter: Arc<dyn IComputeAdapter>,
}

impl MicroExplorerResolver {
    pub fn new(adapter: Arc<dyn IComputeAdapter>) -> Self {
        Self { adapter }
    }
}

#[async_trait]
impl GapResolver for MicroExplorerResolver {
    fn handles(&self, kind: &GapKind) -> bool {
        matches!(
            kind,
            GapKind::MissingProvision | GapKind::IncompleteProvision
        )
    }

    async fn resolve(&self, context: GapResolveContext) -> ResolutionResult {
        let verified_list = context.verified_provision_list.join("\n- ");
        let verified_block = if verified_list.is_empty() {
            "(none)".to_string()
        } else {
            format!("- {}", verified_list)
        };

        let task_text = RECOVERY_TASK
            .replace("{gap_description}", &context.gap.description)
            .replace("{constraint_text}", &context.constraint_text)
            .replace("{verified_provision_list}", &verified_block)
            .replace("{draft_section}", context.resolved_output.as_str());

        let request = ComputeRequest {
            system_context: RECOVERY_SYSTEM.to_string(),
            task: task_text,
            tau: TauValue::new(0.7).unwrap(),
            max_tokens: 2048,
        };

        let patched = match self.adapter.execute(request).await {
            Ok(r) if !r.output.trim().is_empty() => r.output,
            _ => {
                return ResolutionResult {
                    gap_id: context.gap.id.clone(),
                    patched_text: None,
                    score_delta: 0.0,
                }
            }
        };

        ResolutionResult {
            gap_id: context.gap.id.clone(),
            patched_text: Some(patched),
            // Binary: patch produced → accepted (1.0). The resolver cannot compute a real
            // verification delta without re-running the full verifier.
            score_delta: 1.0,
        }
    }
}
