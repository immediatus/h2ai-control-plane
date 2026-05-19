use h2ai_adapters::mock::MockAdapter;
use h2ai_autonomic::calibration::{CalibrationHarness, CalibrationInput};
use h2ai_config::{H2AIConfig, SraniConfig};
use h2ai_orchestrator::engine::{EngineInput, ExecutionEngine};
use h2ai_orchestrator::srani_grounding::{
    GroundingContext, GroundingProvider, GroundingSource, LlmResearcherGrounder,
    SpecAnchorGrounder, SraniGroundingChain, WebSearchGrounder,
};
use h2ai_orchestrator::task_store::TaskStore;
use h2ai_tools::web_search::{
    GeminiSearchBackend, MockSearchBackend, StackOverflowSearchBackend, WebGroundingBackend,
    WebSearchBackend,
};
use h2ai_types::adapter::{AdapterRegistry, IComputeAdapter};
use h2ai_types::config::{AuditorConfig, ParetoWeights, TaoConfig, VerificationConfig};
use h2ai_types::identity::{TaskId, TenantId};
use h2ai_types::manifest::{ExplorerRequest, TaskManifest, TopologyRequest};
use std::sync::Arc;

async fn calibration() -> h2ai_types::events::CalibrationCompletedEvent {
    let adapter = MockAdapter::new("stateless JWT auth ADR-001".into());
    let cfg = H2AIConfig::default();
    CalibrationHarness::run(CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec!["Calibrate".into(), "Second task".into(), "Third".into()],
        adapters: vec![&adapter as &dyn IComputeAdapter],
        cfg: &cfg,
        constraint_corpus: &[],
        embedding_model: None,
    })
    .await
    .unwrap()
}

fn rate_limit_manifest() -> TaskManifest {
    TaskManifest {
        description: "Build a rate-limiting service using Redis sliding windows".into(),
        pareto_weights: ParetoWeights::new(0.2, 0.3, 0.5).unwrap(),
        topology: TopologyRequest {
            kind: "ensemble".into(),
            branching_factor: None,
        },
        explorers: ExplorerRequest {
            count: 2,
            tau_min: Some(0.3),
            tau_max: Some(0.8),
            roles: vec![],
            review_gates: vec![],
            slot_configs: vec![],
            diversity_ids: vec![],
        },
        constraints: vec![],
        context: None,
        oracle: None,
        require_approval: false,
        constraint_tags: vec![],
        measure_verifier_ab: false,
        tenant_id: h2ai_types::identity::TenantId::default_tenant(),
    }
}

/// Full pipeline with no grounding chain; must complete without panic.
#[tokio::test]
async fn system_chain_absent_task_completes_without_panic() {
    let explorer = MockAdapter::new(
        "I recommend CockroachDB for distributed rate-limiting consistency".into(),
    );
    let scorer = MockAdapter::new(r#"{"score": 0.9, "reason": "ok"}"#.into());
    let auditor = MockAdapter::new(r#"{"approved": true, "reason": "ok"}"#.into());
    let cal = calibration().await;
    let cfg = H2AIConfig::default();
    let registry =
        AdapterRegistry::new(Arc::new(MockAdapter::new("x".into())) as Arc<dyn IComputeAdapter>);
    let store = TaskStore::new();

    let input = EngineInput {
        task_id: TaskId::new(),
        manifest: rate_limit_manifest(),
        calibration: cal,
        explorer_adapters: vec![&explorer as _, &explorer as _],
        verification_adapter: &scorer as _,
        auditor_adapter: &auditor as _,
        auditor_config: AuditorConfig::default(),
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: vec![],
        embedding_model: None,
        cfg: &cfg,
        store: store.clone(),
        nats_dispatch: None,
        registry: &registry,
        tao_multiplier: 0.6,
        tao_estimator: Arc::new(tokio::sync::RwLock::new(
            h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
        )),
        synthesis_adapter: None,
        bandit_state: None,
        shadow_audit_ctx: None,
        researcher_adapter: None,
        srani_ema_cfi: 0.45,
        srani_count: 0,
        srani_grounding_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
        knowledge_provider: None,
        induction_store: None,
    };

    let output = ExecutionEngine::run_offline(input).await.unwrap();
    assert!(!output.resolved_output.is_empty());
}

/// Spec-anchor-only chain: SRANI grounding events carry SpecAnchor source.
#[tokio::test]
async fn system_spec_anchor_chain_emits_spec_anchor_source() {
    let explorer = MockAdapter::new(
        "I recommend CockroachDB for distributed rate-limiting consistency".into(),
    );
    let scorer = MockAdapter::new(r#"{"score": 0.9, "reason": "ok"}"#.into());
    let auditor = MockAdapter::new(r#"{"approved": true, "reason": "ok"}"#.into());
    let cal = calibration().await;
    let cfg = H2AIConfig::default();
    let registry =
        AdapterRegistry::new(Arc::new(MockAdapter::new("x".into())) as Arc<dyn IComputeAdapter>);
    let store = TaskStore::new();
    let chain = Arc::new(SraniGroundingChain::new(vec![Box::new(SpecAnchorGrounder)]));

    let input = EngineInput {
        task_id: TaskId::new(),
        manifest: rate_limit_manifest(),
        calibration: cal,
        explorer_adapters: vec![&explorer as _, &explorer as _],
        verification_adapter: &scorer as _,
        auditor_adapter: &auditor as _,
        auditor_config: AuditorConfig::default(),
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: vec![],
        embedding_model: None,
        cfg: &cfg,
        store: store.clone(),
        nats_dispatch: None,
        registry: &registry,
        tao_multiplier: 0.6,
        tao_estimator: Arc::new(tokio::sync::RwLock::new(
            h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
        )),
        synthesis_adapter: None,
        bandit_state: None,
        shadow_audit_ctx: None,
        researcher_adapter: None,
        srani_ema_cfi: 0.45,
        srani_count: 0,
        srani_grounding_chain: Some(chain),
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
        knowledge_provider: None,
        induction_store: None,
    };

    let output = ExecutionEngine::run_offline(input).await.unwrap();
    for ev in &output.researcher_grounding_events {
        if ev.slot.is_none() && !ev.shared_assumption.is_empty() {
            assert_eq!(
                ev.source,
                GroundingSource::SpecAnchor,
                "SRANI grounding event must carry SpecAnchor source"
            );
        }
    }
}

/// Web search chain: tier=0 uses LlmResearcher, tier=1 uses WebSearch.
/// Tests SraniGroundingChain::resolve directly.
#[tokio::test]
async fn system_web_search_chain_escalates_correctly_per_tier() {
    let web_snippet = "Redis sliding-window counter is the standard for rate limiting";
    let chain = SraniGroundingChain::new(vec![
        Box::new(SpecAnchorGrounder),
        Box::new(LlmResearcherGrounder::new(Arc::new(MockAdapter::new(
            r#"{"alternatives": ["x"], "statement": "should not appear at tier 1"}"#.into(),
        )))),
        Box::new(WebSearchGrounder::new(
            Arc::new(MockSearchBackend::new(web_snippet.to_string())),
            3,
        )),
    ]);

    let ctx = GroundingContext {
        fabricated_entities: vec!["CockroachDB".into(), "ClickHouse".into()],
        task_description: "Build a rate-limiting service using Redis sliding windows".into(),
    };

    let r0 = chain.resolve(&ctx, 0).await.unwrap();
    assert_eq!(
        r0.source,
        GroundingSource::LlmResearcher,
        "tier=0 must use LlmResearcher"
    );

    let r1 = chain.resolve(&ctx, 1).await.unwrap();
    assert_eq!(
        r1.source,
        GroundingSource::WebSearch,
        "tier=1 must use WebSearch"
    );
    assert!(r1.grounding_statement.contains("Redis"));
}

/// build_queries generates 3 targeted queries covering: domain implementation,
/// entity grounding, and alternatives — never the raw task description verbatim.
#[test]
fn grounder_build_queries_generates_targeted_queries() {
    let ctx = GroundingContext {
        fabricated_entities: vec!["CockroachDB".into(), "ClickHouse".into()],
        task_description: "Build a rate-limiting service using Redis sliding windows".into(),
    };
    let queries = WebSearchGrounder::build_queries(&ctx);

    assert_eq!(queries.len(), 3, "must generate exactly 3 queries");

    // Q1: domain implementation — must contain key domain words + "implementation"
    assert!(
        queries[0].contains("implementation"),
        "Q1 must target implementation; got: {}",
        queries[0]
    );
    assert!(
        queries[0].to_lowercase().contains("rate")
            || queries[0].to_lowercase().contains("limit")
            || queries[0].to_lowercase().contains("redis"),
        "Q1 must include domain terms; got: {}",
        queries[0]
    );

    // Q2: entity grounding — must reference the first fabricated entity
    assert!(
        queries[1].contains("CockroachDB"),
        "Q2 must reference the fabricated entity; got: {}",
        queries[1]
    );

    // Q3: alternatives / best practices
    assert!(
        queries[2].contains("alternatives") || queries[2].contains("best practices"),
        "Q3 must target alternatives; got: {}",
        queries[2]
    );

    // None of the queries should be the raw task description
    for (i, q) in queries.iter().enumerate() {
        assert!(
            !q.starts_with("Build a"),
            "Q{} must not be the raw task description; got: {}",
            i + 1,
            q
        );
    }
}

/// Live: StackOverflow backend — returns real Q+A bodies for a technical query.
/// Soft-skips on network errors or rate-limit.
#[tokio::test]
async fn live_stackoverflow_backend_returns_real_qa_content() {
    let backend = StackOverflowSearchBackend::new();
    match backend
        .search("rate limiting algorithms implementation", 3)
        .await
    {
        Err(e) => eprintln!("StackExchange API unreachable — skipping: {e}"),
        Ok(text) => {
            println!("StackOverflow response:\n{text}");
            if text == "No results found." {
                eprintln!("No results — API quota may be exhausted, soft skip");
                return;
            }
            assert!(!text.is_empty(), "response must not be empty");
            let lower = text.to_lowercase();
            // Real answers should contain at least one technical term
            assert!(
                lower.contains("rate")
                    || lower.contains("token")
                    || lower.contains("bucket")
                    || lower.contains("sliding")
                    || lower.contains("window")
                    || lower.contains("algorithm")
                    || lower.contains("redis"),
                "Expected technical content in answers; got:\n{text}"
            );
            // Must have structure (Answer markers from our formatter)
            assert!(
                text.contains("Answer") || text.contains("[1]"),
                "response must include formatted answer blocks; got:\n{text}"
            );
        }
    }
}

/// Live: full WebSearchGrounder pipeline with StackOverflow backend.
/// Runs 3 targeted queries, aggregates results, verifies multi-section output.
#[tokio::test]
async fn live_web_search_grounder_with_stackoverflow_aggregates_queries() {
    let backend = Arc::new(StackOverflowSearchBackend::new());
    let grounder = WebSearchGrounder::new(backend, 3);

    let ctx = GroundingContext {
        fabricated_entities: vec!["CockroachDB".into(), "ClickHouse".into()],
        task_description: "Build a rate-limiting service using Redis sliding windows".into(),
    };

    match grounder.ground(&ctx).await {
        None => eprintln!("WebSearchGrounder returned None — soft skip (quota exhausted?)"),
        Some(r) => {
            println!(
                "GroundingResult:\n  source={:?}\n  queries_run={}\n--- statement ---\n{}",
                r.source,
                r.grounding_statement.matches("=== Query").count(),
                r.grounding_statement
            );
            assert_eq!(r.source, GroundingSource::WebSearch);
            assert!(
                !r.grounding_statement.is_empty() && r.grounding_statement != "No results found.",
                "grounding_statement must not be empty"
            );
            // Multiple query sections should be present
            let query_sections = r.grounding_statement.matches("=== Query").count();
            assert!(
                query_sections >= 1,
                "expected at least 1 query section in output; got:\n{}",
                r.grounding_statement
            );
        }
    }
}

/// Live: Gemini Search backend — uses Google Search grounding for richer results.
/// Reads GEMINI_API_KEY from env; soft-skips on 429 or missing key.
#[tokio::test]
async fn live_gemini_search_backend_returns_real_grounded_text() {
    let api_key = match std::env::var("GEMINI_API_KEY") {
        Ok(k) if !k.is_empty() => k,
        _ => {
            eprintln!("GEMINI_API_KEY not set — skipping");
            return;
        }
    };

    let backend = GeminiSearchBackend::new(api_key);
    match backend
        .search("Redis sliding window rate limiting algorithm", 5)
        .await
    {
        Err(e) => eprintln!("Gemini API unavailable — skipping: {e}"),
        Ok(text) => {
            println!("Gemini grounded response:\n{text}");
            assert!(
                !text.is_empty() && text != "No results found.",
                "Expected real grounded text, got: {text}"
            );
            let lower = text.to_lowercase();
            assert!(
                lower.contains("redis")
                    || lower.contains("rate limit")
                    || lower.contains("sliding")
                    || lower.contains("window")
                    || lower.contains("token"),
                "Expected at least one relevant keyword;\ngot:\n{text}"
            );
        }
    }
}

/// Live: WebGroundingBackend — searches Reddit, HN, and GitHub then fetches pages.
/// Verifies diverse source types, real page content (not snippets), and scored output.
#[tokio::test]
async fn live_web_grounding_backend_fetches_real_page_content() {
    let backend = WebGroundingBackend::new();
    match backend
        .search("rate limiting Redis sliding window implementation", 4)
        .await
    {
        Err(e) => eprintln!("WebGroundingBackend error — skipping: {e}"),
        Ok(text) => {
            println!(
                "WebGroundingBackend response:\n{}",
                &text[..text.len().min(1200)]
            );
            if text == "No results found." {
                eprintln!("No fetchable links found — soft skip");
                return;
            }
            assert!(!text.is_empty(), "response must not be empty");
            assert!(
                text.len() > 200,
                "fetched content must be substantial (>200 chars), got {}",
                text.len()
            );
            let lower = text.to_lowercase();
            assert!(
                lower.contains("rate")
                    || lower.contains("limit")
                    || lower.contains("redis")
                    || lower.contains("token")
                    || lower.contains("bucket")
                    || lower.contains("sliding")
                    || lower.contains("window"),
                "expected technical content from fetched pages;\ngot:\n{text}"
            );
        }
    }
}

/// Live: WebGroundingBackend wired into WebSearchGrounder — 3 queries, multi-source.
/// Verifies: query generation → multi-source discovery → fetch → aggregation.
#[tokio::test]
async fn live_web_grounding_full_pipeline_aggregates_multi_source() {
    let backend = Arc::new(WebGroundingBackend::new());
    let grounder = WebSearchGrounder::new(backend, 3);

    let ctx = GroundingContext {
        fabricated_entities: vec!["CockroachDB".into(), "ClickHouse".into()],
        task_description: "Build a rate-limiting service using Redis sliding windows".into(),
    };

    println!(
        "Generated queries: {:?}",
        WebSearchGrounder::build_queries(&ctx)
    );

    match grounder.ground(&ctx).await {
        None => eprintln!("WebSearchGrounder returned None — no fetchable links"),
        Some(r) => {
            let sections = r.grounding_statement.matches("=== Query").count();
            println!(
                "\nGroundingResult:\n  source={:?}\n  query_sections={}\n  total_chars={}\n--- first 1500 ---\n{}",
                r.source, sections, r.grounding_statement.len(),
                &r.grounding_statement[..r.grounding_statement.len().min(1500)]
            );
            assert_eq!(r.source, GroundingSource::WebSearch);
            assert!(!r.grounding_statement.is_empty());
            assert!(sections >= 1, "expected ≥1 query section");
        }
    }
}

/// Pipeline SRANI force-fire: both explorers return an off-spec component ("CockroachDB"),
/// thresholds set to 0.0 so SRANI always warns AND injects, grounding chain wired.
/// Asserts: srani_events non-empty, researcher_grounding_events non-empty, source correct.
#[tokio::test]
async fn system_srani_force_fire_injects_grounding_in_pipeline() {
    // Both slots return the same off-spec component so CFI = 1.0.
    let explorer = MockAdapter::new(
        "I recommend CockroachDB for distributed rate-limiting consistency".into(),
    );
    let scorer = MockAdapter::new(r#"{"score": 0.9, "reason": "ok"}"#.into());
    let auditor = MockAdapter::new(r#"{"approved": true, "reason": "ok"}"#.into());
    // Researcher returns valid JSON with spec-compliant alternatives.
    let researcher = Arc::new(MockAdapter::new(
        r#"{"alternatives": ["Redis sliding window"], "statement": "Use Redis sorted sets as specified."}"#.into(),
    ));
    let cal = calibration().await;

    // Force SRANI to always warn and inject regardless of CFI value.
    let cfg = H2AIConfig {
        srani: SraniConfig {
            enabled: true,
            adaptive: false,
            warn_threshold: 0.0,
            inject_threshold: 0.0,
            gate_threshold: 0.0,
            ..SraniConfig::default()
        },
        ..H2AIConfig::default()
    };

    let chain = Arc::new(SraniGroundingChain::new(vec![
        Box::new(SpecAnchorGrounder),
        Box::new(LlmResearcherGrounder::new(
            Arc::clone(&researcher) as Arc<dyn IComputeAdapter>
        )),
    ]));

    let registry =
        AdapterRegistry::new(Arc::new(MockAdapter::new("x".into())) as Arc<dyn IComputeAdapter>);
    let store = TaskStore::new();

    let input = EngineInput {
        task_id: TaskId::new(),
        manifest: rate_limit_manifest(),
        calibration: cal,
        explorer_adapters: vec![&explorer as _, &explorer as _],
        verification_adapter: &scorer as _,
        auditor_adapter: &auditor as _,
        auditor_config: AuditorConfig::default(),
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: vec![],
        embedding_model: None,
        cfg: &cfg,
        store: store.clone(),
        nats_dispatch: None,
        registry: &registry,
        tao_multiplier: 0.6,
        tao_estimator: Arc::new(tokio::sync::RwLock::new(
            h2ai_orchestrator::tao_loop::TaoMultiplierEstimator::new_with_alpha(0.1),
        )),
        synthesis_adapter: None,
        bandit_state: None,
        shadow_audit_ctx: None,
        researcher_adapter: Some(Arc::clone(&researcher) as Arc<dyn IComputeAdapter>),
        srani_ema_cfi: 0.45,
        srani_count: 5,
        srani_grounding_chain: Some(chain),
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
        knowledge_provider: None,
        induction_store: None,
    };

    let output = ExecutionEngine::run_offline(input).await.unwrap();

    assert!(
        !output.srani_events.is_empty(),
        "SRANI must fire when warn_threshold=0 and CFI > 0"
    );
    assert!(
        output.srani_events.iter().all(|e| e.cfi > 0.0),
        "CFI must be > 0 when both proposals mention off-spec CockroachDB"
    );

    let grounding_events: Vec<_> = output
        .researcher_grounding_events
        .iter()
        .filter(|e| e.slot.is_none() && !e.shared_assumption.is_empty())
        .collect();
    assert!(
        !grounding_events.is_empty(),
        "grounding must be injected when inject_threshold=0; got events: {:?}",
        output
            .researcher_grounding_events
            .iter()
            .map(|e| (&e.source, e.shared_assumption.len()))
            .collect::<Vec<_>>()
    );

    // Chain resolves at SpecAnchor (tier 0) or LlmResearcher (tier 1) — never WebSearch.
    for ev in &grounding_events {
        assert!(
            matches!(
                ev.source,
                GroundingSource::SpecAnchor | GroundingSource::LlmResearcher
            ),
            "source must be SpecAnchor or LlmResearcher, got {:?}",
            ev.source
        );
    }
}
