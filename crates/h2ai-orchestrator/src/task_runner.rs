use async_trait::async_trait;
use h2ai_config::ThinkingLoopConfig;
use h2ai_constraints::types::ConstraintDoc;
use h2ai_context::embedding::EmbeddingModel;
use h2ai_knowledge::provider::KnowledgeProvider;
use h2ai_state::backend::NatsBackend;
use h2ai_types::adapter::{AdapterRegistry, IComputeAdapter};
use h2ai_types::config::{AuditorConfig, ParetoWeights, TaoConfig, VerificationConfig};
use h2ai_types::events::CalibrationCompletedEvent;
use h2ai_types::identity::{TaskId, TenantId};
use h2ai_types::manifest::{ExplorerSlotConfig, TaskManifest};
use h2ai_types::thinking::ThinkingReport;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::bandit::BanditState;
use crate::context_assembler::stable_cache::StableContextCache;
use crate::context_assembler::AssembledContext;
use crate::decomposition::DecompositionError;
use crate::engine::{
    EngineError, EngineOutput, EngineRunContext, NatsDispatchConfig, ShadowAuditCtx,
};
use crate::induction::InductionScheduler;
use crate::induction_store::InductionStore;
use crate::srani_grounding::SraniGroundingChain;
use crate::tao_loop::TaoMultiplierEstimator;
use crate::task_store::TaskStore;

// ── Owned arg structs (no lifetimes — cross trait object boundaries) ──────────

pub struct ThinkingLoopArgs {
    pub task_description: String,
    pub constraint_ids: Vec<String>,
    pub constraint_tags: Vec<String>,
    /// Full constraint docs for the active task. Injected into the archetype selection
    /// prompt so the LLM can produce domain-scoped archetypes (one per constraint
    /// domain) rather than generic "distributed-systems-engineer" personas.
    pub constraint_corpus: Vec<ConstraintDoc>,
    pub knowledge_provider: Option<Arc<dyn KnowledgeProvider>>,
    pub n_archetypes: usize,
    pub cfg: ThinkingLoopConfig,
    pub adapter: Arc<dyn IComputeAdapter>,
    pub embedding_model: Option<Arc<dyn EmbeddingModel>>,
    pub nats_client: Option<async_nats::Client>,
    pub task_id: String,
    /// Hint text from the awareness probe re-iteration path.
    /// When `Some`, appended to `task_description` before the thinking loop runs.
    /// `None` on every first run and in shadow mode (byte-identical behaviour to today).
    pub awareness_hints: Option<String>,
    /// Tenant identifier passed to `load_priming_hints` to scope KV key lookups.
    pub tenant_id: TenantId,
    /// Optional induction scheduler. When `Some`, `DefaultThinkingLoopRunner` calls
    /// `load_priming_hints` and passes the result into `ThinkingLoopInput.retry_hint_priors`.
    pub induction_scheduler: Option<std::sync::Arc<dyn InductionScheduler>>,
}

pub struct DecompositionArgs {
    pub description: String,
    pub corpus: Vec<ConstraintDoc>,
    pub pareto_weights: ParetoWeights,
    pub n_target: usize,
    pub n_max: usize,
    pub adapter: Arc<dyn IComputeAdapter>,
    pub embedding_model: Option<Arc<dyn EmbeddingModel>>,
    pub step_max_tokens: u64,
    pub json_max_tokens: u64,
    pub thinking_context: String,
    /// Operator-specified extra slots appended after LLM decomposition, then re-pruned.
    pub extra_slots: Vec<ExplorerSlotConfig>,
}

/// `EngineInput<'a>` with all `&'a T` references replaced by `Arc<T>` or owned values.
pub struct OwnedEngineInput {
    pub task_id: TaskId,
    pub manifest: TaskManifest,
    pub calibration: CalibrationCompletedEvent,
    pub explorer_adapters: Vec<Arc<dyn IComputeAdapter>>,
    pub verification_adapter: Arc<dyn IComputeAdapter>,
    pub auditor_adapter: Arc<dyn IComputeAdapter>,
    pub auditor_config: AuditorConfig,
    pub tao_config: TaoConfig,
    pub verification_config: VerificationConfig,
    pub constraint_corpus: Vec<ConstraintDoc>,
    pub cfg: Arc<h2ai_config::H2AIConfig>,
    pub store: TaskStore,
    pub nats_dispatch: Option<NatsDispatchConfig>,
    /// Owned registry (cheap Clone); borrowed as `&registry` when converted to EngineInput.
    pub registry: AdapterRegistry,
    pub embedding_model: Option<Arc<dyn EmbeddingModel>>,
    pub tao_multiplier: f64,
    pub tao_estimator: Arc<RwLock<TaoMultiplierEstimator>>,
    pub synthesis_adapter: Option<Arc<dyn IComputeAdapter>>,
    pub bandit_state: Option<Arc<RwLock<BanditState>>>,
    pub shadow_audit_ctx: Option<ShadowAuditCtx>,
    pub researcher_adapter: Option<Arc<dyn IComputeAdapter>>,
    pub srani_ema_cfi: f64,
    pub srani_count: usize,
    pub srani_grounding_chain: Option<Arc<SraniGroundingChain>>,
    pub gap_research_chain: Option<Arc<SraniGroundingChain>>,
    pub nats_raw: Option<Arc<async_nats::Client>>,
    pub tenant_id: TenantId,
    pub nats: Option<Arc<dyn NatsBackend>>,
    pub prev_assembled_contexts: Vec<Option<AssembledContext>>,
    pub compression_adapter: Option<Arc<dyn IComputeAdapter>>,
    pub stable_cache: Option<Arc<StableContextCache>>,
    pub knowledge_provider: Option<Arc<dyn KnowledgeProvider + Send + Sync>>,
    pub induction_store: Option<Arc<InductionStore>>,
    /// Induction scheduler for MAPE-K retroactive trigger and `record_success` signalling.
    /// Distinct from `induction_store` (which handles `KnowledgeNodePattern`).
    pub induction_scheduler: Option<std::sync::Arc<dyn InductionScheduler>>,
    pub conformal_margin: f64,
}

// ── Traits ────────────────────────────────────────────────────────────────────

#[async_trait]
pub trait ThinkingLoopRunner: Send + Sync {
    async fn run(&self, args: ThinkingLoopArgs) -> ThinkingReport;
}

#[async_trait]
pub trait Decomposer: Send + Sync {
    async fn decompose(
        &self,
        args: DecompositionArgs,
    ) -> Result<Vec<ExplorerSlotConfig>, DecompositionError>;
}

#[async_trait]
pub trait EngineRunner: Send + Sync {
    async fn run(
        &self,
        input: OwnedEngineInput,
    ) -> Result<EngineOutput, (EngineError, EngineRunContext)>;
}

// ── Real (zero-field) implementations ────────────────────────────────────────

pub struct DefaultThinkingLoopRunner;

#[async_trait]
impl ThinkingLoopRunner for DefaultThinkingLoopRunner {
    async fn run(&self, args: ThinkingLoopArgs) -> ThinkingReport {
        use crate::induction::InductionContext;
        use crate::thinking_loop::{self, ThinkingLoopInput};
        use h2ai_types::memory::RetryHintPattern;
        // Append awareness hints to the task description when present.
        // No-op (None path) is byte-identical to the pre-probe behaviour.
        let effective_description = match &args.awareness_hints {
            Some(hints) => format!("{}\n\n{}", args.task_description, hints),
            None => args.task_description.clone(),
        };
        let retry_hint_priors: Vec<RetryHintPattern> = match &args.induction_scheduler {
            Some(sched) => {
                sched
                    .load_priming_hints(&InductionContext {
                        tenant_id: args.tenant_id.to_string(),
                        task_class_tags: args.constraint_tags.clone(),
                        violated_constraint_ids: vec![],
                    })
                    .await
            }
            None => vec![],
        };
        thinking_loop::run(ThinkingLoopInput {
            task_description: &effective_description,
            constraint_ids: &args.constraint_ids,
            constraint_tags: &args.constraint_tags,
            constraint_corpus: &args.constraint_corpus,
            research_context: "",
            knowledge_provider: args.knowledge_provider,
            n_archetypes: args.n_archetypes,
            cfg: &args.cfg,
            adapter: args.adapter.as_ref(),
            embedding_model: args.embedding_model.as_deref(),
            nats_client: args.nats_client,
            task_id: &args.task_id,
            induction_patterns: &[],
            retry_hint_priors: &retry_hint_priors,
        })
        .await
    }
}

pub struct DefaultDecomposer;

#[async_trait]
impl Decomposer for DefaultDecomposer {
    async fn decompose(
        &self,
        args: DecompositionArgs,
    ) -> Result<Vec<ExplorerSlotConfig>, DecompositionError> {
        use crate::decomposition::{prune_by_orthogonality, run_decomposition_agent};
        let mut slots = run_decomposition_agent(
            &args.description,
            &args.corpus,
            &args.pareto_weights,
            args.n_target,
            args.n_max,
            args.adapter.as_ref(),
            args.embedding_model.as_deref(),
            args.step_max_tokens,
            args.json_max_tokens,
            &args.thinking_context,
        )
        .await?;
        if !args.extra_slots.is_empty() {
            slots.extend(args.extra_slots);
            if let Some(model) = args.embedding_model.as_deref() {
                slots = prune_by_orthogonality(slots, args.n_max.max(1), model);
            } else {
                slots.truncate(args.n_max.max(1));
            }
        }
        Ok(slots)
    }
}

pub struct DefaultEngineRunner;

#[async_trait]
impl EngineRunner for DefaultEngineRunner {
    async fn run(
        &self,
        input: OwnedEngineInput,
    ) -> Result<EngineOutput, (EngineError, EngineRunContext)> {
        use crate::engine::{EngineInput, ExecutionEngine};
        // Destructure to own all fields before borrowing any.
        let OwnedEngineInput {
            task_id,
            manifest,
            calibration,
            explorer_adapters,
            verification_adapter,
            auditor_adapter,
            auditor_config,
            tao_config,
            verification_config,
            constraint_corpus,
            cfg,
            store,
            nats_dispatch,
            registry,
            embedding_model,
            tao_multiplier,
            tao_estimator,
            synthesis_adapter,
            bandit_state,
            shadow_audit_ctx,
            researcher_adapter,
            srani_ema_cfi,
            srani_count,
            srani_grounding_chain,
            gap_research_chain,
            nats_raw,
            tenant_id,
            nats,
            prev_assembled_contexts,
            compression_adapter,
            stable_cache,
            knowledge_provider,
            induction_store,
            induction_scheduler,
            conformal_margin,
        } = input;
        let explorer_refs: Vec<&dyn IComputeAdapter> =
            explorer_adapters.iter().map(|a| a.as_ref()).collect();
        let embedding_ref = embedding_model.as_deref();
        let synthesis_ref = synthesis_adapter.as_deref();
        ExecutionEngine::run_offline(EngineInput {
            task_id,
            manifest,
            calibration,
            explorer_adapters: explorer_refs,
            verification_adapter: verification_adapter.as_ref(),
            auditor_adapter: auditor_adapter.as_ref(),
            auditor_config,
            tao_config,
            verification_config,
            constraint_corpus,
            cfg: &cfg,
            store,
            nats_dispatch,
            registry: &registry,
            embedding_model: embedding_ref,
            tao_multiplier,
            tao_estimator,
            synthesis_adapter: synthesis_ref,
            bandit_state,
            shadow_audit_ctx,
            researcher_adapter,
            srani_ema_cfi,
            srani_count,
            srani_grounding_chain,
            gap_research_chain,
            nats_raw,
            tenant_id,
            nats,
            prev_assembled_contexts,
            compression_adapter,
            stable_cache,
            knowledge_provider,
            induction_store,
            induction_scheduler,
            conformal_margin,
        })
        .await
    }
}
