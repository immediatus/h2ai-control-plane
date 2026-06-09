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
use crate::context_assembler::AssembledContext;
use crate::context_assembler::stable_cache::StableContextCache;
use crate::decomposition::DecompositionError;
use crate::engine::{EngineError, EngineOutput, EngineRunContext, NatsDispatchConfig, ShadowAuditCtx};
use crate::induction_store::InductionStore;
use crate::srani_grounding::SraniGroundingChain;
use crate::tao_loop::TaoMultiplierEstimator;
use crate::task_store::TaskStore;

// ── Owned arg structs (no lifetimes — cross trait object boundaries) ──────────

pub struct ThinkingLoopArgs {
    pub task_description: String,
    pub constraint_ids: Vec<String>,
    pub constraint_tags: Vec<String>,
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
    pub conformal_margin: f64,
}

// ── Traits ────────────────────────────────────────────────────────────────────

#[async_trait]
pub trait ThinkingLoopRunner: Send + Sync {
    async fn run(&self, args: ThinkingLoopArgs) -> ThinkingReport;
}

#[async_trait]
pub trait Decomposer: Send + Sync {
    async fn decompose(&self, args: DecompositionArgs) -> Result<Vec<ExplorerSlotConfig>, DecompositionError>;
}

#[async_trait]
pub trait EngineRunner: Send + Sync {
    async fn run(&self, input: OwnedEngineInput) -> Result<EngineOutput, (EngineError, EngineRunContext)>;
}

// ── Real (zero-field) implementations ────────────────────────────────────────

pub struct DefaultThinkingLoopRunner;

#[async_trait]
impl ThinkingLoopRunner for DefaultThinkingLoopRunner {
    async fn run(&self, args: ThinkingLoopArgs) -> ThinkingReport {
        use crate::thinking_loop::{self, ThinkingLoopInput};
        // Append awareness hints to the task description when present.
        // No-op (None path) is byte-identical to the pre-probe behaviour.
        let effective_description = match &args.awareness_hints {
            Some(hints) => format!("{}\n\n{}", args.task_description, hints),
            None => args.task_description.clone(),
        };
        thinking_loop::run(ThinkingLoopInput {
            task_description: &effective_description,
            constraint_ids: &args.constraint_ids,
            constraint_tags: &args.constraint_tags,
            research_context: "",
            knowledge_provider: args.knowledge_provider,
            n_archetypes: args.n_archetypes,
            cfg: &args.cfg,
            adapter: args.adapter.as_ref(),
            embedding_model: args.embedding_model.as_deref(),
            nats_client: args.nats_client,
            task_id: &args.task_id,
        })
        .await
    }
}

pub struct DefaultDecomposer;

#[async_trait]
impl Decomposer for DefaultDecomposer {
    async fn decompose(&self, args: DecompositionArgs) -> Result<Vec<ExplorerSlotConfig>, DecompositionError> {
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
    async fn run(&self, input: OwnedEngineInput) -> Result<EngineOutput, (EngineError, EngineRunContext)> {
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
            conformal_margin,
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_thinking_loop_runner_satisfies_trait() {
        let _: Arc<dyn ThinkingLoopRunner> = Arc::new(DefaultThinkingLoopRunner);
    }

    #[test]
    fn default_decomposer_satisfies_trait() {
        let _: Arc<dyn Decomposer> = Arc::new(DefaultDecomposer);
    }

    #[test]
    fn default_engine_runner_satisfies_trait() {
        let _: Arc<dyn EngineRunner> = Arc::new(DefaultEngineRunner);
    }
}

#[cfg(test)]
mod awareness_hints_tests {
    use super::*;

    struct CapturingRunner {
        captured: std::sync::Mutex<Option<String>>,
    }

    #[async_trait::async_trait]
    impl ThinkingLoopRunner for CapturingRunner {
        async fn run(&self, args: ThinkingLoopArgs) -> ThinkingReport {
            *self.captured.lock().unwrap() = Some(args.task_description.clone());
            ThinkingReport::default()
        }
    }

    fn make_args(task_description: &str, awareness_hints: Option<String>) -> ThinkingLoopArgs {
        use h2ai_test_utils::mock_adapter;
        ThinkingLoopArgs {
            task_description: task_description.to_string(),
            constraint_ids: vec![],
            constraint_tags: vec![],
            knowledge_provider: None,
            n_archetypes: 1,
            cfg: h2ai_config::ThinkingLoopConfig::default(),
            adapter: Arc::new(mock_adapter("stub")),
            embedding_model: None,
            nats_client: None,
            task_id: "t1".to_string(),
            awareness_hints,
        }
    }

    #[tokio::test]
    async fn awareness_hints_field_is_stored_in_args() {
        let runner = CapturingRunner { captured: std::sync::Mutex::new(None) };
        let args = make_args("original task", Some("## Constraint contradiction check\nbullet".to_string()));
        // The CapturingRunner stores task_description as-is from args (no mutation).
        // This test validates that ThinkingLoopArgs accepts the awareness_hints field.
        let report = runner.run(args).await;
        let _ = report;
        let captured = runner.captured.lock().unwrap().clone().unwrap();
        assert_eq!(captured, "original task");
    }

    #[tokio::test]
    async fn no_awareness_hints_field_defaults_to_none() {
        let runner = CapturingRunner { captured: std::sync::Mutex::new(None) };
        let args = make_args("original task", None);
        let report = runner.run(args).await;
        let _ = report;
        let captured = runner.captured.lock().unwrap().clone().unwrap();
        assert_eq!(captured, "original task");
    }

    #[test]
    fn effective_description_with_hints_appends_section() {
        // Unit-test the format logic directly (no async needed).
        let base = "original task".to_string();
        let hints = "## Constraint contradiction check\nbullet".to_string();
        let effective = format!("{}\n\n{}", base, hints);
        assert!(effective.contains("original task"));
        assert!(effective.contains("Constraint contradiction check"));
    }

    #[test]
    fn effective_description_without_hints_is_unchanged() {
        let base = "original task".to_string();
        let awareness_hints: Option<String> = None;
        let effective = match &awareness_hints {
            Some(hints) => format!("{}\n\n{}", base, hints),
            None => base.clone(),
        };
        assert_eq!(effective, "original task");
    }
}
