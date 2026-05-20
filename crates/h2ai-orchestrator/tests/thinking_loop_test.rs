use h2ai_config::ThinkingLoopConfig;
use h2ai_orchestrator::thinking_loop::{run, ThinkingLoopInput};

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
