#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::significant_drop_tightening,
    clippy::significant_drop_in_scrutinee,
    clippy::unused_async,
    clippy::default_trait_access,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::manual_let_else,
    clippy::match_wildcard_for_single_variants,
    clippy::similar_names,
    clippy::match_same_arms,
    clippy::literal_string_with_formatting_args,
    clippy::redundant_clone,
    clippy::redundant_closure_for_method_calls,
    clippy::useless_format,
    clippy::option_if_let_else,
    clippy::map_unwrap_or,
    clippy::cloned_instead_of_copied,
    clippy::trivially_copy_pass_by_ref,
    clippy::cast_lossless,
    clippy::uninlined_format_args,
    clippy::needless_pass_by_value,
    clippy::explicit_iter_loop,
    clippy::needless_borrow,
    clippy::large_futures,
    clippy::manual_string_new,
    clippy::needless_lifetimes,
    clippy::elidable_lifetime_names,
    clippy::redundant_else,
    clippy::stable_sort_primitive,
    clippy::type_complexity,
    clippy::wildcard_imports,
    clippy::single_match_else,
    clippy::missing_fields_in_debug,
    clippy::doc_link_with_quotes,
    clippy::implicit_hasher,
    clippy::needless_collect,
    clippy::suboptimal_flops,
    clippy::missing_const_for_fn,
    clippy::needless_type_cast,
    clippy::unreadable_literal,
    clippy::no_effect_underscore_binding
)]
use async_trait::async_trait;
use h2ai_config::ThinkingLoopConfig;
use h2ai_knowledge::factory::ProviderKind;
use h2ai_knowledge::provider::KnowledgeProvider;
use h2ai_knowledge::types::{KnowledgeQuery, KnowledgeResult};
use h2ai_orchestrator::thinking_loop::{run, ThinkingLoopInput};
use std::sync::{Arc, Mutex};

/// Spy provider: records explicit_ids from every query(), returns empty results.
struct SpyProvider {
    captured: Arc<Mutex<Vec<Vec<String>>>>,
}

impl SpyProvider {
    fn new() -> (Self, Arc<Mutex<Vec<Vec<String>>>>) {
        let captured = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                captured: Arc::clone(&captured),
            },
            captured,
        )
    }
}

#[async_trait]
impl KnowledgeProvider for SpyProvider {
    async fn query(&self, query: &KnowledgeQuery<'_>) -> KnowledgeResult {
        self.captured
            .lock()
            .unwrap()
            .push(query.explicit_ids.to_vec());
        KnowledgeResult {
            nodes: vec![],
            global_included: false,
            surfaced_tensions: vec![],
            ppr_expanded: false,
        }
    }

    async fn global_summary(&self) -> Option<h2ai_knowledge::types::KnowledgeNode> {
        None
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn kind(&self) -> &ProviderKind {
        &ProviderKind::Bm25Wiki
    }
}

#[allow(dead_code)]
fn cfg_enabled() -> ThinkingLoopConfig {
    ThinkingLoopConfig {
        enabled: true,
        max_iterations: 3,
        max_archetypes: 2,
        coverage_threshold: 0.75,
        convergence_threshold: 0.90,
        ..Default::default()
    }
}

fn cfg_disabled() -> ThinkingLoopConfig {
    ThinkingLoopConfig {
        enabled: false,
        ..Default::default()
    }
}

#[tokio::test]
async fn disabled_loop_returns_empty_report() {
    // When disabled, run() must return immediately with empty ThinkingReport.
    // We pass a None embedding_model since disabled path must not touch it.
    use h2ai_adapters::MockAdapter;
    let adapter = MockAdapter::new("irrelevant".into());
    let input = ThinkingLoopInput {
        task_description: "test task",
        constraint_ids: &[],
        constraint_tags: &[],
        research_context: "",
        knowledge_provider: None,
        n_archetypes: 2,
        cfg: &cfg_disabled(),
        adapter: &adapter,
        embedding_model: None,
        nats_client: None,
        task_id: "",
    };
    let report = run(input).await;
    assert!(report.shared_understanding.is_empty());
    assert_eq!(report.iteration, 0);
}

#[tokio::test]
async fn parse_archetypes_parses_valid_json() {
    use h2ai_orchestrator::thinking_loop::parse_archetypes;
    let json = r#"[{"name":"perf","persona":"You are a perf engineer.","scope":"latency","confidence":0.8,"tau":0.3,"model_tier":"capable","cot_style":"first_principles"}]"#;
    let archetypes = parse_archetypes(json).unwrap();
    assert_eq!(archetypes.len(), 1);
    assert_eq!(archetypes[0].name, "perf");
    assert!((archetypes[0].confidence - 0.8).abs() < 1e-9);
}

#[tokio::test]
async fn parse_archetypes_returns_none_on_invalid() {
    use h2ai_orchestrator::thinking_loop::parse_archetypes;
    assert!(parse_archetypes("not json").is_none());
    assert!(parse_archetypes("{}").is_none());
}

#[tokio::test]
async fn parse_thinking_report_parses_json() {
    use h2ai_orchestrator::thinking_loop::parse_thinking_report;
    let json = r#"{"shared_understanding":"use P95","tensions":["T1"],"coverage_score":0.85}"#;
    let report = parse_thinking_report(json);
    assert_eq!(report.shared_understanding, "use P95");
    assert_eq!(report.tensions, vec!["T1"]);
    assert!((report.coverage_score - 0.85).abs() < 1e-9);
}

#[tokio::test]
async fn parse_thinking_report_falls_back_on_plain_text() {
    use h2ai_orchestrator::thinking_loop::parse_thinking_report;
    let text = "use adaptive timeouts";
    let report = parse_thinking_report(text);
    assert_eq!(report.shared_understanding, text);
    assert!((report.coverage_score - 0.5).abs() < 1e-9);
}

#[test]
fn adaptive_n_contracts_with_coverage() {
    use h2ai_orchestrator::thinking_loop::{adaptive_n, adaptive_n_guarded};
    assert_eq!(adaptive_n(0, 4, 0.0), 4); // iter 0: always full
    assert_eq!(adaptive_n(1, 4, 0.5), 2); // ceil(4 * 0.5) = 2
    assert_eq!(adaptive_n(2, 4, 0.8), 2); // ceil(4 * 0.2) = 1, clamped to 2
    assert_eq!(adaptive_n(1, 4, 0.1), 4); // ceil(4 * 0.9) = 4

    // quality guard: filter_ratio 0.1 < floor 0.3 → don't contract, return max_n
    assert_eq!(adaptive_n_guarded(1, 4, 0.5, 0.1, 0.3), 4);
    // quality guard: filter_ratio 0.5 >= floor 0.3 → normal contraction
    assert_eq!(adaptive_n_guarded(1, 4, 0.5, 0.5, 0.3), 2);
}

#[test]
fn scheduled_tau_decreases_linearly() {
    use h2ai_orchestrator::thinking_loop::scheduled_tau;
    let t0 = scheduled_tau(0, 3, 0.85, 0.20);
    let t1 = scheduled_tau(1, 3, 0.85, 0.20);
    let t2 = scheduled_tau(2, 3, 0.85, 0.20);
    assert!((t0 - 0.85).abs() < 1e-9, "t0={t0}");
    assert!((t2 - 0.20).abs() < 1e-9, "t2={t2}");
    assert!(t0 > t1 && t1 > t2, "expected strict decrease");
    let t_single = scheduled_tau(0, 1, 0.85, 0.20);
    assert!((t_single - 0.85).abs() < 1e-9);
}

/// Regression: run() must forward ThinkingLoopInput::constraint_ids as explicit_ids
/// to every knowledge query. Before the fix, explicit_ids was hardcoded to &[] so the
/// provider never received the constraint IDs regardless of what the caller passed in.
#[tokio::test]
async fn run_forwards_constraint_ids_to_knowledge_query() {
    use h2ai_adapters::SequencedMockAdapter;

    let (spy, captured) = SpyProvider::new();
    let cfg = ThinkingLoopConfig {
        enabled: true,
        max_iterations: 1,
        max_archetypes: 1,
        ..Default::default()
    };

    // SequencedMockAdapter: archetype select → brainstorm → synthesis (one iteration, one archetype).
    let archetype_json = r#"[{"name":"audit","persona":"You are an audit engineer.","scope":"compliance","confidence":0.8,"tau":0.3,"model_tier":"capable","cot_style":"first_principles"}]"#;
    let brainstorm_text = "Use Kafka for audit trail.";
    let synthesis_json = r#"{"shared_understanding":"publish to Kafka before HTTP 200","tensions":[],"coverage_score":0.85}"#;
    let adapter = SequencedMockAdapter::new(vec![
        archetype_json.to_string(),
        brainstorm_text.to_string(),
        synthesis_json.to_string(),
    ]);

    let constraint_ids = vec!["CONSTRAINT-005".to_string(), "CONSTRAINT-004".to_string()];
    let input = ThinkingLoopInput {
        task_description: "budget enforcement system",
        constraint_ids: &constraint_ids,
        constraint_tags: &[],
        research_context: "",
        knowledge_provider: Some(Arc::new(spy)),
        n_archetypes: 1,
        cfg: &cfg,
        adapter: &adapter,
        embedding_model: None,
        nats_client: None,
        task_id: "",
    };

    run(input).await;

    let queries = captured.lock().unwrap();
    assert!(
        !queries.is_empty(),
        "knowledge provider must have been queried at least once"
    );
    for ids in queries.iter() {
        assert_eq!(
            ids, &constraint_ids,
            "explicit_ids passed to provider must equal ThinkingLoopInput::constraint_ids; \
             got {ids:?}, want {constraint_ids:?}"
        );
    }
}

// ── extract_candidate_solution ────────────────────────────────────────────────

#[test]
fn extract_candidate_solution_found() {
    use h2ai_orchestrator::thinking_loop::extract_candidate_solution;
    let text =
        r#"{"problem_analysis":"test","candidate_solution":"use async queue","confidence":0.8}"#;
    let result = extract_candidate_solution(text);
    assert_eq!(result, Some("use async queue".to_string()));
}

#[test]
fn extract_candidate_solution_uses_last_occurrence() {
    use h2ai_orchestrator::thinking_loop::extract_candidate_solution;
    let text = r#"{"candidate_solution":"first","candidate_solution":"last"}"#;
    let result = extract_candidate_solution(text);
    assert_eq!(result, Some("last".to_string()));
}

#[test]
fn extract_candidate_solution_not_present_returns_none() {
    use h2ai_orchestrator::thinking_loop::extract_candidate_solution;
    assert!(extract_candidate_solution("no marker here").is_none());
    assert!(extract_candidate_solution("").is_none());
}

#[test]
fn extract_candidate_solution_no_colon_after_marker_returns_none() {
    use h2ai_orchestrator::thinking_loop::extract_candidate_solution;
    // marker present but no colon following
    let text = "\"candidate_solution\" no colon";
    assert!(extract_candidate_solution(text).is_none());
}

#[test]
fn extract_candidate_solution_no_quote_after_colon_returns_none() {
    use h2ai_orchestrator::thinking_loop::extract_candidate_solution;
    // marker + colon but value is not a quoted string
    let text = r#""candidate_solution": 42"#;
    assert!(extract_candidate_solution(text).is_none());
}

// ── parse_archetypes — fenced JSON ───────────────────────────────────────────

#[test]
fn parse_archetypes_strips_json_fences() {
    use h2ai_orchestrator::thinking_loop::parse_archetypes;
    let fenced = "```json\n[{\"name\":\"perf\",\"persona\":\"Expert\",\"scope\":\"latency\",\"confidence\":0.8,\"tau\":0.3,\"model_tier\":\"capable\",\"cot_style\":\"step_by_step\"}]\n```";
    let result = parse_archetypes(fenced);
    assert!(result.is_some());
    assert_eq!(result.unwrap()[0].name, "perf");
}

#[test]
fn parse_archetypes_empty_array_returns_none() {
    use h2ai_orchestrator::thinking_loop::parse_archetypes;
    assert!(parse_archetypes("[]").is_none());
}

// ── parse_thinking_report — empty JSON fields fall back ───────────────────────

#[test]
fn parse_thinking_report_empty_shared_understanding_with_zero_coverage_falls_back() {
    use h2ai_orchestrator::thinking_loop::parse_thinking_report;
    // Both shared_understanding empty AND coverage_score=0.0 → plain text fallback
    let json = r#"{"shared_understanding":"","tensions":[],"coverage_score":0.0}"#;
    let report = parse_thinking_report(json);
    // Falls back to plain text path: full input becomes shared_understanding
    assert_eq!(report.shared_understanding, json);
    assert!((report.coverage_score - 0.5).abs() < 1e-9);
}

// ── run() — multi-iteration paths ────────────────────────────────────────────

#[tokio::test]
async fn run_with_two_iterations_covers_convergence_check() {
    use h2ai_adapters::SequencedMockAdapter;
    use h2ai_config::ThinkingLoopConfig;
    use h2ai_orchestrator::thinking_loop::{run, ThinkingLoopInput};

    let cfg = ThinkingLoopConfig {
        enabled: true,
        max_iterations: 2,
        max_archetypes: 1,
        coverage_threshold: 0.95, // high → first iter won't meet it
        convergence_threshold: 0.99,
        ..Default::default()
    };

    let archetype_json = r#"[{"name":"analyst","persona":"You are a systems analyst.","scope":"all","confidence":0.7,"tau":0.4,"model_tier":"standard","cot_style":"none"}]"#;
    let brainstorm1 = "SOLUTION SKETCH: Use distributed locking.\n{\"confidence\": 0.7}";
    let synthesis1 = r#"{"shared_understanding":"use distributed locking","tensions":["T1"],"coverage_score":0.5}"#;
    // Second iteration
    let archetype_json2 = r#"[{"name":"critic","persona":"You are a critic.","scope":"all","confidence":0.8,"tau":0.3,"model_tier":"standard","cot_style":"step_by_step"}]"#;
    let brainstorm2 = "SOLUTION SKETCH: Use optimistic locking instead.";
    let synthesis2 =
        r#"{"shared_understanding":"use optimistic locking","tensions":[],"coverage_score":0.8}"#;

    let adapter = SequencedMockAdapter::new(vec![
        archetype_json.to_string(),
        brainstorm1.to_string(),
        synthesis1.to_string(),
        archetype_json2.to_string(),
        brainstorm2.to_string(),
        synthesis2.to_string(),
    ]);

    let input = ThinkingLoopInput {
        task_description: "distributed locking problem",
        constraint_ids: &[],
        constraint_tags: &[],
        research_context: "",
        knowledge_provider: None,
        n_archetypes: 1,
        cfg: &cfg,
        adapter: &adapter,
        embedding_model: None,
        nats_client: None,
        task_id: "test-task",
    };

    let report = run(input).await;
    assert!(
        report.iteration >= 1,
        "must have completed at least one iteration"
    );
    assert!(!report.shared_understanding.is_empty());
}

#[tokio::test]
async fn run_with_low_coverage_continues_to_next_iteration() {
    use h2ai_adapters::SequencedMockAdapter;
    use h2ai_config::ThinkingLoopConfig;
    use h2ai_orchestrator::thinking_loop::{run, ThinkingLoopInput};

    let cfg = ThinkingLoopConfig {
        enabled: true,
        max_iterations: 2,
        max_archetypes: 1,
        coverage_threshold: 0.99, // always below → always continue
        convergence_threshold: 0.99,
        ..Default::default()
    };

    let archetype = r#"[{"name":"a","persona":"p","scope":"s","confidence":0.7,"tau":0.3,"model_tier":"standard","cot_style":"none"}]"#;
    let brainstorm = "simple brainstorm text";
    let synthesis_low = r#"{"shared_understanding":"low coverage result","tensions":["gap1"],"coverage_score":0.3}"#;

    let adapter = SequencedMockAdapter::new(vec![
        archetype.to_string(),
        brainstorm.to_string(),
        synthesis_low.to_string(),
        archetype.to_string(),
        brainstorm.to_string(),
        synthesis_low.to_string(),
    ]);

    let input = ThinkingLoopInput {
        task_description: "low coverage task",
        constraint_ids: &[],
        constraint_tags: &[],
        research_context: "",
        knowledge_provider: None,
        n_archetypes: 1,
        cfg: &cfg,
        adapter: &adapter,
        embedding_model: None,
        nats_client: None,
        task_id: "",
    };

    let report = run(input).await;
    // Both iterations ran; loop exits at is_last on iteration 1
    assert_eq!(report.iteration, 2);
}

#[tokio::test]
async fn run_breaks_early_on_empty_archetypes() {
    use h2ai_adapters::MockAdapter;
    use h2ai_config::ThinkingLoopConfig;
    use h2ai_orchestrator::thinking_loop::{run, ThinkingLoopInput};

    let cfg = ThinkingLoopConfig {
        enabled: true,
        max_iterations: 3,
        max_archetypes: 2,
        ..Default::default()
    };
    // Adapter returns non-JSON → parse_archetypes returns None → archetypes empty → break
    let adapter = MockAdapter::new("not valid json archetypes".into());

    let input = ThinkingLoopInput {
        task_description: "task",
        constraint_ids: &[],
        constraint_tags: &[],
        research_context: "",
        knowledge_provider: None,
        n_archetypes: 2,
        cfg: &cfg,
        adapter: &adapter,
        embedding_model: None,
        nats_client: None,
        task_id: "",
    };

    let report = run(input).await;
    // Loop breaks immediately on empty archetypes in iteration 0
    assert_eq!(
        report.iteration, 0,
        "no iteration completed when archetypes are empty"
    );
}
