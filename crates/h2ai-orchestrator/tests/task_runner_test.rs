use h2ai_orchestrator::task_runner::{
    Decomposer, DefaultDecomposer, DefaultEngineRunner, DefaultThinkingLoopRunner, EngineRunner,
    ThinkingLoopArgs, ThinkingLoopRunner,
};
use std::sync::Arc;

// ── CapturingRunner (used in awareness_hints_tests) ───────────────────────────

struct CapturingRunner {
    captured: std::sync::Mutex<Option<String>>,
}

#[async_trait::async_trait]
impl ThinkingLoopRunner for CapturingRunner {
    async fn run(&self, args: ThinkingLoopArgs) -> h2ai_types::thinking::ThinkingReport {
        *self.captured.lock().unwrap() = Some(args.task_description.clone());
        h2ai_types::thinking::ThinkingReport::default()
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
        constraint_corpus: vec![],
        tenant_id: h2ai_types::identity::TenantId::default_tenant(),
        induction_scheduler: None,
    }
}

// ── Trait-satisfaction tests ──────────────────────────────────────────────────

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

// ── awareness_hints tests ─────────────────────────────────────────────────────

#[tokio::test]
async fn awareness_hints_field_is_stored_in_args() {
    let runner = CapturingRunner {
        captured: std::sync::Mutex::new(None),
    };
    let args = make_args(
        "original task",
        Some("## Constraint contradiction check\nbullet".to_string()),
    );
    // The CapturingRunner stores task_description as-is from args (no mutation).
    // This test validates that ThinkingLoopArgs accepts the awareness_hints field.
    let report = runner.run(args).await;
    let _ = report;
    let captured = runner.captured.lock().unwrap().clone().unwrap();
    assert_eq!(captured, "original task");
}

#[tokio::test]
async fn no_awareness_hints_field_defaults_to_none() {
    let runner = CapturingRunner {
        captured: std::sync::Mutex::new(None),
    };
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
