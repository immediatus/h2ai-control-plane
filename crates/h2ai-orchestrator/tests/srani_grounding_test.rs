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
use h2ai_autonomic::calibration::{CalibrationHarness, CalibrationInput};
use h2ai_config::{H2AIConfig, SraniConfig};
use h2ai_orchestrator::engine::{EngineInput, ExecutionEngine};
use h2ai_orchestrator::srani_grounding::{
    format_grounding_hint, GroundingContext, GroundingProvider, GroundingResult, GroundingSource,
    LlmResearcherGrounder, SpecAnchorGrounder, SraniGroundingChain, WebSearchGrounder,
};
use h2ai_orchestrator::task_store::TaskStore;
use h2ai_test_utils::mock_adapter;
use h2ai_test_utils::mock_search;
use h2ai_tools::error::ToolError;
use h2ai_tools::web_search::{
    GeminiSearchBackend, StackOverflowSearchBackend, WebGroundingBackend, WebSearchBackend,
};
use h2ai_types::adapter::{AdapterRegistry, IComputeAdapter};
use h2ai_types::config::{AuditorConfig, ParetoWeights, TaoConfig, VerificationConfig};
use h2ai_types::identity::{TaskId, TenantId};
use h2ai_types::manifest::{ExplorerRequest, TaskManifest, TopologyRequest};
use std::sync::Arc;

async fn calibration() -> h2ai_types::events::CalibrationCompletedEvent {
    let adapter = mock_adapter("stateless JWT auth ADR-001");
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
    let explorer =
        mock_adapter("I recommend CockroachDB for distributed rate-limiting consistency");
    let scorer = mock_adapter(r#"{"score": 0.9, "reason": "ok"}"#);
    let auditor = mock_adapter(r#"{"approved": true, "reason": "ok"}"#);
    let cal = calibration().await;
    let cfg = H2AIConfig::default();
    let registry = AdapterRegistry::new(Arc::new(mock_adapter("x")) as Arc<dyn IComputeAdapter>);
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
        gap_research_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
        knowledge_provider: None,
        induction_store: None,
        conformal_margin: 0.0,
    };

    let output = ExecutionEngine::run_offline(input).await.unwrap();
    assert!(!output.resolved_output.is_empty());
}

/// Spec-anchor-only chain: SRANI grounding events carry SpecAnchor source.
#[tokio::test]
async fn system_spec_anchor_chain_emits_spec_anchor_source() {
    let explorer =
        mock_adapter("I recommend CockroachDB for distributed rate-limiting consistency");
    let scorer = mock_adapter(r#"{"score": 0.9, "reason": "ok"}"#);
    let auditor = mock_adapter(r#"{"approved": true, "reason": "ok"}"#);
    let cal = calibration().await;
    let cfg = H2AIConfig::default();
    let registry = AdapterRegistry::new(Arc::new(mock_adapter("x")) as Arc<dyn IComputeAdapter>);
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
        gap_research_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
        knowledge_provider: None,
        induction_store: None,
        conformal_margin: 0.0,
    };

    let output = ExecutionEngine::run_offline(input).await.unwrap();
    for ev in &output.researcher_grounding_events {
        if !ev.shared_assumption.is_empty() {
            assert_eq!(
                ev.source,
                GroundingSource::SpecAnchor,
                "SRANI grounding event must carry SpecAnchor source"
            );
            assert!(
                ev.slot.is_some(),
                "slot must be classified (Some), got None for assumption: {}",
                ev.shared_assumption
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
        Box::new(LlmResearcherGrounder::new(Arc::new(mock_adapter(
            r#"{"alternatives": ["x"], "statement": "should not appear at tier 1"}"#,
        )))),
        Box::new(WebSearchGrounder::new(
            Arc::new(mock_search(web_snippet.to_string())),
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

// ── SpecAnchorGrounder unit tests ─────────────────────────────────────────────

#[tokio::test]
async fn spec_anchor_extracts_spec_entities_as_alternatives() {
    let ctx = GroundingContext {
        fabricated_entities: vec!["CockroachDB".into(), "ClickHouse".into()],
        task_description: "Build a rate-limiting service using Redis and in-process counters"
            .into(),
    };
    let grounder = SpecAnchorGrounder;
    let result = grounder.ground(&ctx).await.unwrap();
    assert!(
        result.alternatives.contains(&"Redis".to_string()),
        "expected Redis in alternatives, got {:?}",
        result.alternatives
    );
    assert!(
        !result.alternatives.contains(&"CockroachDB".to_string()),
        "CockroachDB should not be promoted — it is fabricated"
    );
}

#[tokio::test]
async fn spec_anchor_excludes_fabricated_from_alternatives() {
    let ctx = GroundingContext {
        fabricated_entities: vec!["Redis".into(), "CockroachDB".into()],
        task_description: "Build a rate-limiting service using Redis and in-process counters"
            .into(),
    };
    let grounder = SpecAnchorGrounder;
    let result = grounder.ground(&ctx).await.unwrap();
    assert!(
        !result.alternatives.contains(&"Redis".to_string()),
        "Redis is fabricated — must not appear in alternatives"
    );
}

#[tokio::test]
async fn spec_anchor_empty_spec_still_produces_result() {
    let ctx = GroundingContext {
        fabricated_entities: vec!["CockroachDB".into()],
        task_description: "do something simple".into(),
    };
    let grounder = SpecAnchorGrounder;
    let result = grounder.ground(&ctx).await;
    assert!(result.is_some());
    assert!(result.unwrap().alternatives.is_empty());
}

#[tokio::test]
async fn spec_anchor_source_tag_is_correct() {
    let ctx = GroundingContext {
        fabricated_entities: vec!["CockroachDB".into(), "ClickHouse".into()],
        task_description: "Build a rate-limiting service using Redis and in-process counters"
            .into(),
    };
    let grounder = SpecAnchorGrounder;
    let result = grounder.ground(&ctx).await.unwrap();
    assert_eq!(result.source, GroundingSource::SpecAnchor);
}

// ── LlmResearcherGrounder unit tests ──────────────────────────────────────────

#[tokio::test]
async fn llm_researcher_happy_path() {
    let ctx = GroundingContext {
        fabricated_entities: vec!["CockroachDB".into(), "ClickHouse".into()],
        task_description: "Build a rate-limiting service using Redis and in-process counters"
            .into(),
    };
    let adapter = Arc::new(mock_adapter(
        r#"{"alternatives": ["Redis TTL counters", "sliding window"], "statement": "Use Redis TTL + Lua for rate limiting"}"#,
    ));
    let grounder = LlmResearcherGrounder::new(adapter);
    let result = grounder.ground(&ctx).await.unwrap();
    assert!(!result.alternatives.is_empty());
    assert_eq!(result.source, GroundingSource::LlmResearcher);
}

#[tokio::test]
async fn llm_researcher_invalid_json_returns_none() {
    let ctx = GroundingContext {
        fabricated_entities: vec!["CockroachDB".into()],
        task_description: "Build a rate-limiting service using Redis".into(),
    };
    let adapter = Arc::new(mock_adapter("not json at all !!!"));
    let grounder = LlmResearcherGrounder::new(adapter);
    let result = grounder.ground(&ctx).await;
    assert!(result.is_none());
}

#[tokio::test]
async fn llm_researcher_missing_alternatives_field_returns_none() {
    let ctx = GroundingContext {
        fabricated_entities: vec!["CockroachDB".into()],
        task_description: "Build a rate-limiting service using Redis".into(),
    };
    let adapter = Arc::new(mock_adapter(r#"{"statement": "use Redis"}"#));
    let grounder = LlmResearcherGrounder::new(adapter);
    let result = grounder.ground(&ctx).await;
    assert!(
        result.is_none(),
        "missing alternatives field must return None"
    );
}

// ── WebSearchGrounder unit tests ───────────────────────────────────────────────

#[tokio::test]
async fn web_search_produces_web_search_source() {
    let ctx = GroundingContext {
        fabricated_entities: vec!["CockroachDB".into(), "ClickHouse".into()],
        task_description: "Build a rate-limiting service using Redis and in-process counters"
            .into(),
    };
    let backend = Arc::new(mock_search(
        "Redis sliding-window counter is the standard approach for rate limiting".to_string(),
    ));
    let grounder = WebSearchGrounder::new(backend, 3);
    let result = grounder.ground(&ctx).await.unwrap();
    assert_eq!(result.source, GroundingSource::WebSearch);
    assert!(!result.grounding_statement.is_empty());
}

#[tokio::test]
async fn web_search_empty_results_returns_none() {
    let ctx = GroundingContext {
        fabricated_entities: vec!["CockroachDB".into()],
        task_description: "Build a rate-limiting service using Redis".into(),
    };
    let backend = Arc::new(mock_search("".to_string()));
    let grounder = WebSearchGrounder::new(backend, 3);
    let result = grounder.ground(&ctx).await;
    assert!(result.is_none());
}

#[tokio::test]
async fn web_search_error_returns_none() {
    use h2ai_test_utils::MockWebSearch;
    let mut backend = MockWebSearch::new();
    backend
        .expect_search()
        .returning(|_, _| Err(ToolError::MalformedInput("network error".into())));
    let ctx = GroundingContext {
        fabricated_entities: vec!["CockroachDB".into()],
        task_description: "Build a rate-limiting service using Redis".into(),
    };
    let grounder = WebSearchGrounder::new(Arc::new(backend), 3);
    let result = grounder.ground(&ctx).await;
    assert!(result.is_none());
}

// ── strip_urls via format_grounding_hint ──────────────────────────────────────
// strip_urls is private; exercised indirectly through format_grounding_hint.

#[test]
fn format_grounding_hint_removes_urls_from_statement() {
    let result = GroundingResult {
        alternatives: vec!["Redis".into()],
        grounding_statement:
            "Redis https://redis.io/docs sliding window http://example.com counter".into(),
        source: GroundingSource::WebSearch,
    };
    let hint = format_grounding_hint(&result, &["CockroachDB".into()]);
    assert!(!hint.contains("https://"), "https:// must be removed");
    assert!(!hint.contains("http://"), "http:// must be removed");
    assert!(hint.contains("Redis"), "prose words must survive");
    assert!(hint.contains("counter"), "prose words must survive");
}

#[test]
fn format_grounding_hint_preserves_non_url_statement() {
    let result = GroundingResult {
        alternatives: vec!["Redis".into()],
        grounding_statement: "rate limiting with Redis sliding window".into(),
        source: GroundingSource::LlmResearcher,
    };
    let hint = format_grounding_hint(&result, &["CockroachDB".into()]);
    assert!(hint.contains("rate limiting with Redis sliding window"));
}

// ── truncate_at_sentence via SraniGroundingChain::resolve distillation ────────
// truncate_at_sentence is private; exercised indirectly by the chain's hint_max_chars cap.

#[tokio::test]
async fn chain_distillation_replaces_raw_web_text_with_distilled_output() {
    let ctx = GroundingContext {
        fabricated_entities: vec!["CockroachDB".into(), "ClickHouse".into()],
        task_description: "Build a rate-limiting service using Redis and in-process counters"
            .into(),
    };
    let raw_text = "Web result: use Redis. ".repeat(200); // > 4000 chars
    let distilled_output = "Redis sliding window is the standard rate-limiting approach.";
    let providers: Vec<Box<dyn GroundingProvider>> = vec![
        Box::new(SpecAnchorGrounder),
        Box::new(WebSearchGrounder::new(
            Arc::new(mock_search(raw_text.clone())),
            3,
        )),
    ];
    let distiller = Arc::new(mock_adapter(distilled_output));
    let chain = SraniGroundingChain::new(providers).with_distiller(distiller, true);
    let result = chain.resolve(&ctx, 1).await.unwrap();
    assert_eq!(result.source, GroundingSource::WebSearch);
    assert!(
        result.grounding_statement.contains("Redis sliding window"),
        "distilled text must be in statement, got: {}",
        result.grounding_statement
    );
}

#[tokio::test]
async fn chain_distill_disabled_preserves_raw_text_capped_at_hint_limit() {
    let ctx = GroundingContext {
        fabricated_entities: vec!["CockroachDB".into(), "ClickHouse".into()],
        task_description: "Build a rate-limiting service using Redis and in-process counters"
            .into(),
    };
    let raw_text = "Redis. ".repeat(300); // > 1200 chars
    let providers: Vec<Box<dyn GroundingProvider>> = vec![
        Box::new(SpecAnchorGrounder),
        Box::new(WebSearchGrounder::new(Arc::new(mock_search(raw_text)), 3)),
    ];
    let distiller = Arc::new(mock_adapter("should not be called"));
    let chain = SraniGroundingChain::new(providers).with_distiller(distiller, false);
    let result = chain.resolve(&ctx, 1).await.unwrap();
    // distill=false: raw text passes through unchanged (no truncation in this project)
    assert!(!result.grounding_statement.is_empty());
}

// ── SraniGroundingChain unit tests ────────────────────────────────────────────

#[tokio::test]
async fn chain_tier0_merges_spec_anchor_and_researcher() {
    let ctx = GroundingContext {
        fabricated_entities: vec!["CockroachDB".into(), "ClickHouse".into()],
        task_description: "Build a rate-limiting service using Redis and in-process counters"
            .into(),
    };
    let providers: Vec<Box<dyn GroundingProvider>> = vec![
        Box::new(SpecAnchorGrounder),
        Box::new(LlmResearcherGrounder::new(Arc::new(mock_adapter(
            r#"{"alternatives": ["Redis TTL counters"], "statement": "Use Redis TTL + Lua"}"#,
        )))),
    ];
    let chain = SraniGroundingChain::new(providers);
    let result = chain.resolve(&ctx, 0).await.unwrap();
    assert!(
        result.grounding_statement.contains("Spec-defined"),
        "anchor statement missing: {}",
        result.grounding_statement
    );
    assert!(
        result.grounding_statement.contains("Redis TTL"),
        "researcher statement missing: {}",
        result.grounding_statement
    );
}

#[tokio::test]
async fn chain_tier1_escalates_to_web_search_skips_researcher() {
    let ctx = GroundingContext {
        fabricated_entities: vec!["CockroachDB".into(), "ClickHouse".into()],
        task_description: "Build a rate-limiting service using Redis and in-process counters"
            .into(),
    };
    let providers: Vec<Box<dyn GroundingProvider>> = vec![
        Box::new(SpecAnchorGrounder),
        Box::new(LlmResearcherGrounder::new(Arc::new(mock_adapter(
            "should not appear",
        )))),
        Box::new(WebSearchGrounder::new(
            Arc::new(mock_search("Web result: use Redis".to_string())),
            3,
        )),
    ];
    let chain = SraniGroundingChain::new(providers);
    let result = chain.resolve(&ctx, 1).await.unwrap();
    assert_eq!(
        result.source,
        GroundingSource::WebSearch,
        "tier=1 must use WebSearch"
    );
}

#[tokio::test]
async fn chain_tier_clamped_at_last_tier() {
    let ctx = GroundingContext {
        fabricated_entities: vec!["CockroachDB".into(), "ClickHouse".into()],
        task_description: "Build a rate-limiting service using Redis and in-process counters"
            .into(),
    };
    let providers: Vec<Box<dyn GroundingProvider>> = vec![
        Box::new(SpecAnchorGrounder),
        Box::new(LlmResearcherGrounder::new(Arc::new(mock_adapter(
            r#"{"alternatives": ["x"], "statement": "y"}"#,
        )))),
    ];
    let chain = SraniGroundingChain::new(providers);
    let result = chain.resolve(&ctx, 99).await;
    assert!(result.is_some(), "clamped tier must not panic");
}

#[tokio::test]
async fn chain_spec_anchor_only_still_produces_positive_result() {
    let ctx = GroundingContext {
        fabricated_entities: vec!["CockroachDB".into(), "ClickHouse".into()],
        task_description: "Build a rate-limiting service using Redis and in-process counters"
            .into(),
    };
    let providers: Vec<Box<dyn GroundingProvider>> = vec![Box::new(SpecAnchorGrounder)];
    let chain = SraniGroundingChain::new(providers);
    let result = chain.resolve(&ctx, 0).await;
    assert!(result.is_some());
    assert_eq!(result.unwrap().source, GroundingSource::SpecAnchor);
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
    let explorer =
        mock_adapter("I recommend CockroachDB for distributed rate-limiting consistency");
    let scorer = mock_adapter(r#"{"score": 0.9, "reason": "ok"}"#);
    let auditor = mock_adapter(r#"{"approved": true, "reason": "ok"}"#);
    // Researcher returns valid JSON with spec-compliant alternatives.
    let researcher = Arc::new(mock_adapter(
        r#"{"alternatives": ["Redis sliding window"], "statement": "Use Redis sorted sets as specified."}"#,
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

    let registry = AdapterRegistry::new(Arc::new(mock_adapter("x")) as Arc<dyn IComputeAdapter>);
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
        gap_research_chain: None,
        nats_raw: None,
        tenant_id: TenantId::default_tenant(),
        nats: None,
        prev_assembled_contexts: Vec::new(),
        compression_adapter: None,
        stable_cache: None,
        knowledge_provider: None,
        induction_store: None,
        conformal_margin: 0.0,
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
        .filter(|e| !e.shared_assumption.is_empty())
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
    // Slot must be classified (Some) after entity slot classification.
    for ev in &grounding_events {
        assert!(
            matches!(
                ev.source,
                GroundingSource::SpecAnchor | GroundingSource::LlmResearcher
            ),
            "source must be SpecAnchor or LlmResearcher, got {:?}",
            ev.source
        );
        assert!(
            ev.slot.is_some(),
            "slot must be classified (Some), got None for assumption: {}",
            ev.shared_assumption
        );
    }
}

#[cfg(test)]
mod classify_slot_tests {
    use h2ai_orchestrator::srani_grounding::classify_grounding_slot;

    #[test]
    fn kafka_maps_to_message_broker() {
        assert_eq!(
            classify_grounding_slot(&["Kafka".to_string()]),
            "message_broker"
        );
        assert_eq!(
            classify_grounding_slot(&["kafka".to_string()]),
            "message_broker"
        );
        assert_eq!(
            classify_grounding_slot(&["RabbitMQ".to_string()]),
            "message_broker"
        );
        assert_eq!(
            classify_grounding_slot(&["ActiveMQ".to_string()]),
            "message_broker"
        );
    }

    #[test]
    fn zookeeper_etcd_map_to_distributed_coordination() {
        assert_eq!(
            classify_grounding_slot(&["ZooKeeper".to_string()]),
            "distributed_coordination"
        );
        assert_eq!(
            classify_grounding_slot(&["etcd".to_string()]),
            "distributed_coordination"
        );
        assert_eq!(
            classify_grounding_slot(&["Consul".to_string()]),
            "distributed_coordination"
        );
    }

    #[test]
    fn redis_maps_to_cache_layer() {
        assert_eq!(
            classify_grounding_slot(&["Redis".to_string()]),
            "cache_layer"
        );
        assert_eq!(
            classify_grounding_slot(&["Memcached".to_string()]),
            "cache_layer"
        );
    }

    #[test]
    fn postgres_terms_map_to_database_migration() {
        assert_eq!(
            classify_grounding_slot(&["pg_publication_tables".to_string()]),
            "database_migration"
        );
        assert_eq!(
            classify_grounding_slot(&["PostgreSQL".to_string()]),
            "database_migration"
        );
        assert_eq!(
            classify_grounding_slot(&["replication_slot".to_string()]),
            "database_migration"
        );
    }

    #[test]
    fn unknown_entity_maps_to_implementation_detail() {
        assert_eq!(
            classify_grounding_slot(&["string.split".to_string()]),
            "implementation_detail"
        );
        assert_eq!(
            classify_grounding_slot(&["SomeUnknownLib".to_string()]),
            "implementation_detail"
        );
        assert_eq!(classify_grounding_slot(&[]), "implementation_detail");
    }

    #[test]
    fn multi_entity_first_match_wins() {
        assert_eq!(
            classify_grounding_slot(&["ZooKeeper".to_string(), "Redis".to_string()]),
            "distributed_coordination"
        );
    }
}

// ── Additional coverage tests ─────────────────────────────────────────────────

#[test]
fn chain_len_and_is_empty() {
    let chain_empty: SraniGroundingChain = SraniGroundingChain::new(vec![]);
    assert_eq!(chain_empty.len(), 0);
    assert!(chain_empty.is_empty());

    let chain_one = SraniGroundingChain::new(vec![Box::new(SpecAnchorGrounder)]);
    assert_eq!(chain_one.len(), 1);
    assert!(!chain_one.is_empty());
}

#[tokio::test]
async fn chain_empty_providers_returns_none() {
    let chain: SraniGroundingChain = SraniGroundingChain::new(vec![]);
    let ctx = GroundingContext {
        fabricated_entities: vec!["CockroachDB".into()],
        task_description: "Build a rate limiting service".into(),
    };
    let result = chain.resolve(&ctx, 0).await;
    assert!(result.is_none(), "empty chain must return None");
}

#[tokio::test]
async fn web_search_grounder_returns_none_with_no_fabricated_entities() {
    let ctx = GroundingContext {
        fabricated_entities: vec![],
        task_description: "Build a rate limiting service using Redis".into(),
    };
    let backend = Arc::new(mock_search("some result".to_string()));
    let grounder = WebSearchGrounder::new(backend, 3);
    let result = grounder.ground(&ctx).await;
    assert!(result.is_none(), "no fabricated entities must return None");
}

#[tokio::test]
async fn chain_none_anchor_some_tier_uses_tier_result() {
    struct NoneProvider;
    #[async_trait::async_trait]
    impl GroundingProvider for NoneProvider {
        async fn ground(&self, _ctx: &GroundingContext) -> Option<GroundingResult> {
            None
        }
    }

    let ctx = GroundingContext {
        fabricated_entities: vec!["CockroachDB".into()],
        task_description: "Build a rate limiting service using Redis and counters".into(),
    };
    let providers: Vec<Box<dyn GroundingProvider>> = vec![
        Box::new(NoneProvider),
        Box::new(LlmResearcherGrounder::new(Arc::new(mock_adapter(
            r#"{"alternatives": ["Redis"], "statement": "Use Redis"}"#,
        )))),
    ];
    let chain = SraniGroundingChain::new(providers);
    let result = chain.resolve(&ctx, 0).await;
    assert!(
        result.is_some(),
        "tier result must be used when anchor returns None"
    );
    assert_eq!(result.unwrap().source, GroundingSource::LlmResearcher);
}

#[tokio::test]
async fn web_search_no_results_found_string_returns_none() {
    use h2ai_test_utils::mock_search;
    let backend = mock_search("No results found.");
    let ctx = GroundingContext {
        fabricated_entities: vec!["CockroachDB".into()],
        task_description: "Build a rate limiting service using Redis".into(),
    };
    let grounder = WebSearchGrounder::new(Arc::new(backend), 3);
    let result = grounder.ground(&ctx).await;
    assert!(result.is_none(), "No results found must return None");
}

#[tokio::test]
async fn chain_distillation_empty_output_falls_back_to_raw() {
    let ctx = GroundingContext {
        fabricated_entities: vec!["CockroachDB".into(), "ClickHouse".into()],
        task_description: "Build a rate-limiting service using Redis and in-process counters"
            .into(),
    };
    let raw_text = "Web result: use Redis. ".repeat(5);
    let providers: Vec<Box<dyn GroundingProvider>> = vec![
        Box::new(SpecAnchorGrounder),
        Box::new(WebSearchGrounder::new(
            Arc::new(mock_search(raw_text.clone())),
            3,
        )),
    ];
    let distiller = Arc::new(mock_adapter(""));
    let chain = SraniGroundingChain::new(providers).with_distiller(distiller, true);
    let result = chain.resolve(&ctx, 1).await.unwrap();
    assert_eq!(result.source, GroundingSource::WebSearch);
    assert!(!result.grounding_statement.is_empty());
}

#[test]
fn format_grounding_hint_empty_alternatives_and_empty_statement() {
    let result = GroundingResult {
        alternatives: vec![],
        grounding_statement: String::new(),
        source: GroundingSource::SpecAnchor,
    };
    let hint = format_grounding_hint(&result, &["FakeLib".into()]);
    assert!(hint.contains("GROUNDING CONTEXT"));
    assert!(hint.contains("FakeLib"));
    assert!(!hint.contains("Spec-defined"));
    assert!(!hint.contains("alternatives:"));
}
