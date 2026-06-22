use crate::gap_checkers::{
    Gap, GapCheckContext, GapChecker, GapKind, GapResolveContext, GapResolver, ResolutionResult,
};
use async_trait::async_trait;
use std::sync::Arc;

/// A resolver that handles no gap kinds. Used when `recovery_enabled = false` so the
/// feedback loop can still run the CoherenceChecker per-pass without closing any gaps.
pub struct NullResolver;

#[async_trait]
impl GapResolver for NullResolver {
    fn handles(&self, _kind: &GapKind) -> bool {
        false
    }
    async fn resolve(&self, ctx: GapResolveContext) -> ResolutionResult {
        ResolutionResult {
            gap_id: ctx.gap.id.clone(),
            patched_text: None,
            score_delta: 0.0,
        }
    }
}

/// Parameters for the epistemic feedback loop.
pub struct FeedbackLoopParams {
    /// Gaps seeded before the loop (SelectionPruning + UncertainDomain). These are
    /// static — seeded once and carried across all passes; the loop only re-detects
    /// CoherenceChecker gaps per pass.
    pub static_gaps: Vec<Gap>,
    pub initial_output: String,
    /// Present when `coherence_check_enabled = true`. Re-run on `current_output` at
    /// the start of each pass so coherence detection reflects the latest patch.
    pub coherence_checker: Option<Arc<dyn GapChecker>>,
    pub resolver: Arc<dyn GapResolver>,
    pub verified_provision_list: Vec<String>,
    pub constraint_text: String,
    pub constraint_ids: Vec<String>,
    pub max_passes: usize,
}

/// Outcome of the feedback loop.
pub struct FeedbackLoopResult {
    pub final_output: String,
    pub closed_ids: Vec<String>,
    /// All gaps that remain open: unresolved static gaps + coherence gaps detected
    /// in the last pass that ran. Used by the caller to build the ProvenanceMap.
    pub open_gaps: Vec<Gap>,
    /// Number of times the CoherenceChecker was invoked across all passes.
    pub coherence_checks_run: usize,
}

/// Runs the epistemic quality feedback loop.
///
/// Each pass:
/// 1. Re-runs CoherenceChecker on `current_output` (if enabled).
/// 2. Collects all resolvable gaps (open static + new coherence gaps handled by `resolver`).
/// 3. Exits when no resolvable gaps remain **or** the resolver makes no progress.
/// 4. Resolves via DAG batch ordering; updates `current_output` on each accepted patch.
///
/// UncertainDomain and InterProvisionConflict gaps are never resolvable and always
/// propagate to `open_gaps`, ensuring `document_confidence` stays below `High`.
pub async fn run_epistemic_feedback_loop(params: FeedbackLoopParams) -> FeedbackLoopResult {
    let FeedbackLoopParams {
        static_gaps,
        initial_output,
        coherence_checker,
        resolver,
        verified_provision_list,
        constraint_text,
        constraint_ids,
        max_passes,
    } = params;

    let mut current_output = initial_output;
    let mut open_static = static_gaps.clone();
    let mut closed_ids: Vec<String> = Vec::new();
    let mut coherence_checks_run: usize = 0;
    let mut last_coherence_gaps: Vec<Gap> = Vec::new();

    let gap_check_ctx = GapCheckContext {
        verified_provision_list: verified_provision_list.clone(),
        constraint_text: constraint_text.clone(),
    };

    for _pass in 0..max_passes {
        // Step 1: Re-detect coherence gaps on the current (possibly patched) output.
        let coherence_gaps = if let Some(ref checker) = coherence_checker {
            coherence_checks_run += 1;
            checker.check(&current_output, &gap_check_ctx).await
        } else {
            vec![]
        };
        last_coherence_gaps = coherence_gaps.clone();

        // Step 2: Collect all resolvable gaps this pass.
        let resolvable: Vec<Gap> = open_static
            .iter()
            .filter(|g| resolver.handles(&g.kind))
            .chain(coherence_gaps.iter().filter(|g| resolver.handles(&g.kind)))
            .cloned()
            .collect();

        if resolvable.is_empty() {
            break;
        }

        // Step 3: DAG batch dispatch.
        let registry = crate::gap_registry::GapRegistry::new(resolvable.clone());
        let Ok(batches) = registry.dispatch_batches() else {
            break;
        };

        let mut closed_this_pass: Vec<String> = Vec::new();
        for batch in batches {
            let batch_gaps: Vec<&Gap> = resolvable
                .iter()
                .filter(|g| batch.contains(&g.id))
                .collect();
            for gap in batch_gaps {
                let resolve_ctx = GapResolveContext {
                    gap: gap.clone(),
                    resolved_output: Arc::new(current_output.clone()),
                    verified_provision_list: verified_provision_list.clone(),
                    constraint_text: constraint_text.clone(),
                    constraint_ids: constraint_ids.clone(),
                };
                let res = resolver.resolve(resolve_ctx).await;
                if let Some(patch) = res.patched_text {
                    current_output = patch;
                    closed_this_pass.push(res.gap_id);
                }
            }
        }

        if closed_this_pass.is_empty() {
            // Resolver made no progress — exit rather than spin.
            break;
        }

        open_static.retain(|g| !closed_this_pass.contains(&g.id));
        closed_ids.extend(closed_this_pass);
    }

    // open_gaps = remaining static gaps + any coherence gaps from the last pass.
    let mut open_gaps = open_static;
    open_gaps.extend(
        last_coherence_gaps
            .into_iter()
            .filter(|g| !closed_ids.contains(&g.id)),
    );

    FeedbackLoopResult {
        final_output: current_output,
        closed_ids,
        open_gaps,
        coherence_checks_run,
    }
}
