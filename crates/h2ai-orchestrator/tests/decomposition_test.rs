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
use h2ai_constraints::types::{ConstraintDoc, ConstraintPredicate, ConstraintSeverity};
use h2ai_context::embedding::EmbeddingModel;
use h2ai_orchestrator::decomposition::{
    compute_role_diversity, corpus_fallback, parse_decomposition_response, prune_by_orthogonality,
    run_decomposition_agent,
};
use h2ai_test_utils::{failing_adapter, mock_adapter, MockIComputeAdapter};
use h2ai_types::adapter::ComputeResponse;
use h2ai_types::config::{AdapterKind, ParetoWeights};
use h2ai_types::manifest::{CotStyle, ExplorerSlotConfig};
use std::sync::{Arc, Mutex};

#[test]
fn parse_valid_json_array() {
    let json = r#"[
      {
        "role_frame": "You are a security engineer.",
        "cot_style": "devil_s_advocate",
        "focus_mandate": "Responsible for CONSTRAINT-001.",
        "rejection_criteria": "The most likely attacker exploitation path."
      },
      {
        "role_frame": "You are a systems architect.",
        "cot_style": "first_principles",
        "focus_mandate": "Responsible for CONSTRAINT-002.",
        "rejection_criteria": "Irreversible technical debt."
      }
    ]"#;
    let slots = parse_decomposition_response(json).unwrap();
    assert_eq!(slots.len(), 2);
    assert_eq!(slots[0].role_frame, "You are a security engineer.");
    assert_eq!(slots[0].cot_style, CotStyle::DevilsAdvocate);
    assert_eq!(slots[0].focus_mandate, "Responsible for CONSTRAINT-001.");
    assert_eq!(
        slots[0].rejection_criteria,
        "The most likely attacker exploitation path."
    );
    assert_eq!(slots[1].cot_style, CotStyle::FirstPrinciples);
}

#[test]
fn parse_strips_surrounding_text() {
    let response = "Here is the committee:\n```json\n[\
      {\"role_frame\": \"You are a performance engineer.\", \"cot_style\": \"step_by_step\"}\
    ]\n```\nEnd.";
    let slots = parse_decomposition_response(response).unwrap();
    assert_eq!(slots.len(), 1);
    assert_eq!(slots[0].role_frame, "You are a performance engineer.");
}

#[test]
fn parse_drops_empty_role_frame_slots() {
    let json = r#"[
      {"role_frame": "", "cot_style": "none"},
      {"role_frame": "You are a security engineer.", "cot_style": "none"}
    ]"#;
    let slots = parse_decomposition_response(json).unwrap();
    assert_eq!(slots.len(), 1);
}

#[test]
fn parse_returns_error_on_no_json_array() {
    assert!(parse_decomposition_response("no array here").is_err());
}

#[test]
fn parse_returns_error_on_all_empty_role_frames() {
    let json = r#"[{"role_frame": "", "cot_style": "none"}]"#;
    assert!(parse_decomposition_response(json).is_err());
}

#[test]
fn parse_optional_fields_default_to_empty() {
    let json = r#"[{"role_frame": "You are a security engineer.", "cot_style": "none"}]"#;
    let slots = parse_decomposition_response(json).unwrap();
    assert!(slots[0].focus_mandate.is_empty());
    assert!(slots[0].rejection_criteria.is_empty());
}

/// Reasoning models (DeepSeek-R1, Qwen3) wrap their output in <think>...</think>.
/// The parser must strip these blocks before scanning for the JSON array.
#[test]
fn parse_strips_thinking_tags_from_reasoning_model() {
    let response = "<think>Let me think about the problem [latency, billing]...</think>\n[\
      {\"role_frame\": \"You are a latency engineer.\", \"cot_style\": \"first_principles\"}\
    ]";
    let slots = parse_decomposition_response(response).unwrap();
    assert_eq!(slots.len(), 1);
    assert_eq!(slots[0].role_frame, "You are a latency engineer.");
}

#[test]
fn parse_thinking_tags_with_nested_brackets_do_not_confuse_scanner() {
    let response = "<think>[placeholder] and [array] in thinking</think>\n[\
      {\"role_frame\": \"You are a systems architect.\", \"cot_style\": \"step_by_step\"}\
    ]";
    let slots = parse_decomposition_response(response).unwrap();
    assert_eq!(slots.len(), 1);
}

/// Regression: rfind(']') would find the last `]` in the entire string, not the matching
/// one. A `]` in trailing postamble text caused serde_json to receive garbage JSON.
#[test]
fn parse_trailing_bracket_in_postamble_does_not_confuse_scanner() {
    let response = "Here is the committee:\n[\
      {\"role_frame\": \"You are a performance engineer.\", \"cot_style\": \"step_by_step\"}\
    ]\nSee options [A] and [B] above.";
    let slots = parse_decomposition_response(response).unwrap();
    assert_eq!(slots.len(), 1);
    assert_eq!(slots[0].role_frame, "You are a performance engineer.");
}

/// Reasoning models (26B+) sometimes emit arrays for optional string fields.
/// The parser must coerce them to "; "-joined strings rather than failing.
#[test]
fn parse_coerces_array_focus_mandate_to_string() {
    let json = r#"[{
        "role_frame": "You are a performance engineer.",
        "cot_style": "first_principles",
        "focus_mandate": ["Ensure P99 < 100ms", "Track histogram per instance"],
        "rejection_criteria": "Latency regression under load"
    }]"#;
    let slots = parse_decomposition_response(json).unwrap();
    assert_eq!(slots.len(), 1);
    assert!(slots[0].focus_mandate.contains("Ensure P99 < 100ms"));
    assert!(slots[0]
        .focus_mandate
        .contains("Track histogram per instance"));
}

#[test]
fn parse_coerces_array_rejection_criteria_to_string() {
    let json = r#"[{
        "role_frame": "You are a security engineer.",
        "cot_style": "devil_s_advocate",
        "rejection_criteria": ["SQL injection via input", "Auth bypass via token reuse"]
    }]"#;
    let slots = parse_decomposition_response(json).unwrap();
    assert_eq!(slots.len(), 1);
    assert!(slots[0].rejection_criteria.contains("SQL injection"));
    assert!(slots[0].rejection_criteria.contains("Auth bypass"));
}

// ── Orthogonality pruner ─────────────────────────────────────────────────────

struct IdenticalModel;
impl EmbeddingModel for IdenticalModel {
    fn embed(&self, _: &str) -> Vec<f32> {
        vec![1.0, 0.0]
    }
}

struct OrthogonalModel;
impl EmbeddingModel for OrthogonalModel {
    fn embed(&self, text: &str) -> Vec<f32> {
        if text.contains("security") {
            vec![1.0, 0.0, 0.0, 0.0]
        } else if text.contains("performance") {
            vec![0.0, 1.0, 0.0, 0.0]
        } else if text.contains("correctness") {
            vec![0.0, 0.0, 1.0, 0.0]
        } else {
            vec![0.0, 0.0, 0.0, 1.0]
        }
    }
}

fn make_slot(role: &str) -> ExplorerSlotConfig {
    ExplorerSlotConfig {
        role_frame: role.into(),
        cot_style: CotStyle::None,
        focus_mandate: String::new(),
        rejection_criteria: String::new(),
        ..Default::default()
    }
}

#[test]
fn prune_drops_most_similar_slot_when_over_n_max() {
    let slots = vec![
        make_slot("security engineer"),
        make_slot("security engineer"),
        make_slot("security engineer"),
    ];
    let model = IdenticalModel;
    let pruned = prune_by_orthogonality(slots, 2, &model);
    assert_eq!(pruned.len(), 2);
}

#[test]
fn prune_keeps_all_when_at_or_below_n_max() {
    let slots = vec![
        make_slot("security engineer"),
        make_slot("performance engineer"),
    ];
    let model = OrthogonalModel;
    let pruned = prune_by_orthogonality(slots, 3, &model);
    assert_eq!(pruned.len(), 2);
}

#[test]
fn prune_drops_redundant_not_orthogonal() {
    let slots = vec![
        make_slot("security engineer first"),
        make_slot("performance engineer"),
        make_slot("security engineer second"),
    ];
    let model = OrthogonalModel;
    let pruned = prune_by_orthogonality(slots, 2, &model);
    assert_eq!(pruned.len(), 2);
    assert!(pruned.iter().any(|s| s.role_frame.contains("performance")));
}

// ── Corpus fallback ──────────────────────────────────────────────────────────

fn make_constraint(id: &str, domains: Vec<&str>) -> ConstraintDoc {
    ConstraintDoc {
        id: id.to_string(),
        source_file: format!("{id}.yaml"),
        description: String::new(),
        severity: ConstraintSeverity::Advisory,
        predicate: ConstraintPredicate::LengthRange {
            min_chars: None,
            max_chars: None,
        },
        remediation_hint: None,
        domains: domains.into_iter().map(str::to_string).collect(),
        mandatory_for_tags: vec![],
        related_to: vec![],
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
    }
}

#[test]
fn corpus_fallback_one_slot_per_domain() {
    let corpus = vec![
        make_constraint("C-001", vec!["security"]),
        make_constraint("C-002", vec!["security", "performance"]),
        make_constraint("C-003", vec!["correctness"]),
    ];
    let weights = ParetoWeights::new(0.33, 0.34, 0.33).unwrap();
    let slots = corpus_fallback(&corpus, &weights, 5);
    // 3 distinct domains: security, performance, correctness
    assert_eq!(slots.len(), 3);
    let roles: Vec<&str> = slots.iter().map(|s| s.role_frame.as_str()).collect();
    assert!(roles.iter().any(|r| r.contains("security")));
    assert!(roles.iter().any(|r| r.contains("performance")));
}

#[test]
fn corpus_fallback_respects_n_max() {
    let corpus = vec![
        make_constraint("C-001", vec!["security"]),
        make_constraint("C-002", vec!["performance"]),
        make_constraint("C-003", vec!["correctness"]),
        make_constraint("C-004", vec!["consistency"]),
    ];
    let weights = ParetoWeights::new(0.33, 0.34, 0.33).unwrap();
    let slots = corpus_fallback(&corpus, &weights, 2);
    assert_eq!(slots.len(), 2);
}

#[test]
fn corpus_fallback_empty_corpus_returns_default_slot() {
    let weights = ParetoWeights::new(0.33, 0.34, 0.33).unwrap();
    let slots = corpus_fallback(&[], &weights, 3);
    assert_eq!(slots.len(), 1);
    assert!(!slots[0].role_frame.is_empty());
}

#[test]
fn corpus_fallback_untagged_constraints_produce_default_slot() {
    let corpus = vec![make_constraint("C-001", vec![])];
    let weights = ParetoWeights::new(0.33, 0.34, 0.33).unwrap();
    let slots = corpus_fallback(&corpus, &weights, 3);
    assert_eq!(slots.len(), 1);
    assert!(!slots[0].role_frame.is_empty());
}

#[test]
fn corpus_fallback_compliance_domain_produces_analyst_slot() {
    let corpus = vec![make_constraint("C-010", vec!["compliance"])];
    let weights = ParetoWeights::new(0.33, 0.34, 0.33).unwrap();
    let slots = corpus_fallback(&corpus, &weights, 5);
    assert_eq!(slots.len(), 1);
    assert!(slots[0].role_frame.contains("regulatory compliance"));
}

#[test]
fn corpus_fallback_regulatory_and_audit_domains_produce_analyst_slot() {
    let corpus = vec![
        make_constraint("C-011", vec!["regulatory"]),
        make_constraint("C-012", vec!["audit"]),
    ];
    let weights = ParetoWeights::new(0.33, 0.34, 0.33).unwrap();
    let slots = corpus_fallback(&corpus, &weights, 5);
    assert_eq!(slots.len(), 2);
    assert!(slots
        .iter()
        .all(|s| s.role_frame.contains("regulatory compliance")));
}

#[test]
fn corpus_fallback_unknown_domain_produces_architect_slot() {
    let corpus = vec![make_constraint("C-020", vec!["accessibility"])];
    let weights = ParetoWeights::new(0.33, 0.34, 0.33).unwrap();
    let slots = corpus_fallback(&corpus, &weights, 5);
    assert_eq!(slots.len(), 1);
    assert!(slots[0].role_frame.contains("senior software architect"));
}

// ── run_decomposition_agent ──────────────────────────────────────────────────

#[tokio::test]
async fn run_decomposition_agent_uses_llm_slots_on_success() {
    let adapter = mock_adapter(
        r#"[
          {"role_frame": "You are a security engineer.", "cot_style": "devil_s_advocate"},
          {"role_frame": "You are a performance engineer.", "cot_style": "first_principles"}
        ]"#,
    );
    let weights = ParetoWeights::new(0.33, 0.34, 0.33).unwrap();
    let slots = run_decomposition_agent(
        "design a caching layer",
        &[],
        &weights,
        2,
        5,
        &adapter,
        None,
        2048,
        8192,
        "",
    )
    .await
    .unwrap();
    assert_eq!(slots.len(), 2);
    assert!(slots[0].role_frame.contains("security"));
}

#[tokio::test]
async fn run_decomposition_agent_prunes_to_n_max() {
    let adapter = mock_adapter(
        r#"[
          {"role_frame": "You are a security engineer.", "cot_style": "none"},
          {"role_frame": "You are a performance engineer.", "cot_style": "none"},
          {"role_frame": "You are a correctness engineer.", "cot_style": "none"}
        ]"#,
    );
    let weights = ParetoWeights::new(0.33, 0.34, 0.33).unwrap();
    let slots =
        run_decomposition_agent("task", &[], &weights, 2, 2, &adapter, None, 2048, 8192, "")
            .await
            .unwrap();
    assert_eq!(slots.len(), 2);
}

#[tokio::test]
async fn run_decomposition_agent_returns_err_on_adapter_failure() {
    let adapter = failing_adapter();
    let weights = ParetoWeights::new(0.33, 0.34, 0.33).unwrap();
    let result = run_decomposition_agent(
        "design a caching layer",
        &[],
        &weights,
        2,
        5,
        &adapter,
        None,
        2048,
        8192,
        "",
    )
    .await;
    assert!(result.is_err(), "adapter failure must propagate as Err");
}

#[tokio::test]
async fn run_decomposition_agent_returns_err_on_parse_failure() {
    let adapter = mock_adapter("not valid json");
    let weights = ParetoWeights::new(0.33, 0.34, 0.33).unwrap();
    let result = run_decomposition_agent(
        "design something",
        &[],
        &weights,
        2,
        5,
        &adapter,
        None,
        2048,
        8192,
        "",
    )
    .await;
    assert!(result.is_err(), "parse failure must propagate as Err");
}

#[tokio::test]
async fn decomposition_failure_propagates_as_err() {
    // Decomposition failure is fatal — no fallback. The task-handler publishes TaskFailed
    // and stops. This test verifies that a non-JSON response from the LLM propagates as Err.
    let adapter = mock_adapter("Here is my analysis of the problem... [no JSON array]");
    let corpus = vec![make_constraint("CONSTRAINT-003", vec!["rtb", "latency"])];
    let weights = ParetoWeights::new(0.5, 0.4, 0.1).unwrap();

    let result = run_decomposition_agent(
        "design a DSP onboarding flow",
        &corpus,
        &weights,
        2,
        5,
        &adapter,
        None,
        2048,
        8192,
        "",
    )
    .await;
    assert!(
        result.is_err(),
        "non-JSON response must propagate as Err — no fallback"
    );
}

// ── Request-capturing adapter ─────────────────────────────────────────────────

fn capturing_adapter(responses: Vec<&str>) -> (MockIComputeAdapter, Arc<Mutex<Vec<String>>>) {
    let responses: Vec<String> = responses.into_iter().map(str::to_string).collect();
    let captured: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let captured_clone = captured.clone();
    let call_count = Arc::new(Mutex::new(0usize));
    let mut m = MockIComputeAdapter::new();
    m.expect_execute().returning(move |req| {
        captured_clone.lock().unwrap().push(req.task.clone());
        let mut count = call_count.lock().unwrap();
        let response = responses
            .get(*count)
            .cloned()
            .unwrap_or_else(|| responses.last().cloned().unwrap_or_default());
        *count += 1;
        Ok(ComputeResponse {
            output: response,
            token_cost: 50,
            adapter_kind: AdapterKind::CloudGeneric {
                endpoint: String::new(),
                api_key_env: String::new(),
                model: None,
                provider: Default::default(),
            },
            tokens_used: None,
            reasoning_trace: None,
        })
    });
    m.expect_kind()
        .return_const(AdapterKind::CloudGeneric {
            endpoint: String::new(),
            api_key_env: String::new(),
            model: None,
            provider: Default::default(),
        })
        .times(0..);
    (m, captured)
}

/// STEP3 prompt must contain the corpus domain vocabulary so the LLM can emit
/// verbatim strings. This is the primary fix for C3 vocabulary mismatch.
#[tokio::test]
async fn step3_prompt_contains_corpus_domain_vocabulary() {
    let json_slots = r#"[
        {
            "role_frame": "You are a security engineer.",
            "cot_style": "devil_s_advocate",
            "focus_mandate": "auth constraints",
            "rejection_criteria": "Token forgery.",
            "constraint_domains": ["auth"],
            "search_enabled": false
        },
        {
            "role_frame": "You are a performance engineer.",
            "cot_style": "first_principles",
            "focus_mandate": "latency constraints",
            "rejection_criteria": "SLO breach under load.",
            "constraint_domains": ["latency"],
            "search_enabled": false
        }
    ]"#;

    let (adapter, captured) = capturing_adapter(vec![
        "Step 1 analysis: auth misses token expiry; latency misses P99 under concurrent load.",
        "Step 2 roles: security engineer for auth; performance engineer for latency.",
        json_slots,
    ]);

    let corpus = vec![
        make_constraint("C-001", vec!["auth"]),
        make_constraint("C-002", vec!["latency"]),
    ];
    let weights = ParetoWeights::new(0.33, 0.34, 0.33).unwrap();

    let slots = run_decomposition_agent(
        "design an authentication service with latency SLOs",
        &corpus,
        &weights,
        2,
        5,
        &adapter,
        None,
        2048,
        8192,
        "",
    )
    .await
    .unwrap();

    let tasks = captured.lock().unwrap();
    assert_eq!(tasks.len(), 3, "pipeline must make exactly 3 adapter calls");

    // STEP3 task (index 2) must contain the exact corpus vocabulary strings so the
    // LLM knows which domain tags to emit verbatim.
    let step3_task = &tasks[2];
    assert!(
        step3_task.contains("\"auth\""),
        "STEP3 prompt must list corpus domain 'auth' — got:\n{step3_task}"
    );
    assert!(
        step3_task.contains("\"latency\""),
        "STEP3 prompt must list corpus domain 'latency' — got:\n{step3_task}"
    );

    // Output slots should carry the exact vocabulary strings the LLM echoed back.
    let auth_slot = slots
        .iter()
        .find(|s| s.constraint_domains.contains(&"auth".to_string()));
    assert!(
        auth_slot.is_some(),
        "at least one slot must claim domain 'auth'"
    );
}

/// When corpus is empty, STEP3 prompt must still render without panicking,
/// and the domains placeholder must indicate no vocabulary is defined.
#[tokio::test]
async fn step3_prompt_handles_empty_corpus_gracefully() {
    let json_slots = r#"[
        {"role_frame": "You are a senior architect.", "cot_style": "step_by_step",
         "constraint_domains": [], "search_enabled": false}
    ]"#;

    let (adapter, captured) = capturing_adapter(vec![
        "Step 1: no constraints, general analysis.",
        "Step 2: one general architect role.",
        json_slots,
    ]);

    let weights = ParetoWeights::new(0.33, 0.34, 0.33).unwrap();
    let slots = run_decomposition_agent(
        "design a caching layer",
        &[],
        &weights,
        1,
        5,
        &adapter,
        None,
        2048,
        8192,
        "",
    )
    .await
    .unwrap();

    let tasks = captured.lock().unwrap();
    let step3_task = &tasks[2];
    // Empty corpus must not produce a placeholder that looks like a real domain list.
    assert!(
        step3_task.contains("no corpus domains defined"),
        "empty corpus must produce clear 'no corpus domains' message in STEP3 — got:\n{step3_task}"
    );
    assert_eq!(slots.len(), 1);
}

// ── 3-step pipeline chain tests ───────────────────────────────────────────────

fn sequential_mock_adapter(responses: Vec<&str>) -> (MockIComputeAdapter, Arc<Mutex<usize>>) {
    let responses: Vec<String> = responses.into_iter().map(str::to_string).collect();
    let call_count = Arc::new(Mutex::new(0usize));
    let count_clone = call_count.clone();
    let mut m = MockIComputeAdapter::new();
    m.expect_execute().returning(move |_req| {
        let mut count = count_clone.lock().unwrap();
        let response = responses
            .get(*count)
            .cloned()
            .unwrap_or_else(|| responses.last().cloned().unwrap_or_default());
        *count += 1;
        Ok(ComputeResponse {
            output: response,
            token_cost: 50,
            adapter_kind: AdapterKind::CloudGeneric {
                endpoint: String::new(),
                api_key_env: String::new(),
                model: None,
                provider: Default::default(),
            },
            tokens_used: None,
            reasoning_trace: None,
        })
    });
    m.expect_kind()
        .return_const(AdapterKind::CloudGeneric {
            endpoint: String::new(),
            api_key_env: String::new(),
            model: None,
            provider: Default::default(),
        })
        .times(0..);
    (m, call_count)
}

/// Step 1 and Step 2 return free-text; Step 3 returns JSON.
/// Verifies the pipeline produces slots from Step 3 output only.
#[tokio::test]
async fn pipeline_uses_step3_json_output_for_slot_construction() {
    let json_slots = r#"[
      {
        "role_frame": "You are a distributed systems engineer who has debugged budget race conditions.",
        "cot_style": "backward_chaining",
        "focus_mandate": "budget-pacing idempotency (CONSTRAINT-004)",
        "rejection_criteria": "Any path where Redis TTL expires before deduplication is confirmed."
      },
      {
        "role_frame": "You are a compliance architect who has led financial audit reviews.",
        "cot_style": "devil_s_advocate",
        "focus_mandate": "immutable audit log (CONSTRAINT-005)",
        "rejection_criteria": "Any design where Kafka publish follows application ack instead of preceding it."
      },
      {
        "role_frame": "You are an SRE who designs for simultaneous billing and audit failures.",
        "cot_style": "first_principles",
        "focus_mandate": "cross-domain cascade",
        "rejection_criteria": "Inconsistency between CockroachDB operational and ClickHouse audit under partial failure."
      }
    ]"#;

    let (adapter, call_count) = sequential_mock_adapter(vec![
        "Step 1 analysis: C-004 misses Redis TTL race; C-005 misses Kafka-before-ack ordering.",
        "Step 2 roles: Engineer 1 anchored to race condition; Engineer 2 anchored to audit ordering.",
        json_slots,
    ]);
    let corpus = vec![
        make_constraint("CONSTRAINT-004", vec!["budget-pacing"]),
        make_constraint("CONSTRAINT-005", vec!["compliance", "audit"]),
    ];
    let weights = ParetoWeights::new(0.33, 0.34, 0.33).unwrap();

    let slots = run_decomposition_agent(
        "Design a budget deduction service with immutable audit trail",
        &corpus,
        &weights,
        3,
        5,
        &adapter,
        None,
        2048,
        8192,
        "",
    )
    .await
    .unwrap();

    assert_eq!(slots.len(), 3);
    assert!(slots[0].role_frame.contains("race conditions"));
    assert_eq!(slots[0].cot_style, CotStyle::BackwardChaining);
    assert!(slots[1].role_frame.contains("audit reviews"));
    assert_eq!(slots[1].cot_style, CotStyle::DevilsAdvocate);
    assert!(slots[2].role_frame.contains("SRE"));
    assert_eq!(slots[2].cot_style, CotStyle::FirstPrinciples);
    // Verify call count: exactly 3 steps
    let calls = *call_count.lock().unwrap();
    assert_eq!(calls, 3, "pipeline must make exactly 3 adapter calls");
}

/// Step 1 failure propagates immediately — pipeline does not continue to Step 2.
#[tokio::test]
async fn pipeline_fails_fast_on_step1_adapter_error() {
    let adapter = failing_adapter();
    let weights = ParetoWeights::new(0.33, 0.34, 0.33).unwrap();
    let result = run_decomposition_agent(
        "design something",
        &[],
        &weights,
        2,
        5,
        &adapter,
        None,
        2048,
        8192,
        "",
    )
    .await;
    assert!(result.is_err());
}

/// Steps 1-2 succeed with free text; Step 3 returns non-JSON → parse error.
#[tokio::test]
async fn pipeline_fails_on_step3_json_parse_error() {
    let (adapter, call_count) = sequential_mock_adapter(vec![
        "Step 1: analysis output.",
        "Step 2: role design output.",
        "Step 3 produced no JSON array here.",
    ]);
    let weights = ParetoWeights::new(0.33, 0.34, 0.33).unwrap();
    let result =
        run_decomposition_agent("task", &[], &weights, 2, 5, &adapter, None, 2048, 8192, "").await;
    assert!(result.is_err(), "Step 3 non-JSON must propagate as Err");
    let calls = *call_count.lock().unwrap();
    assert_eq!(calls, 3, "all 3 steps must be attempted before parse error");
}

#[test]
fn parse_response_captures_constraint_domains() {
    let json = r#"[
        {
            "role_frame": "You are a security engineer.",
            "cot_style": "devil_s_advocate",
            "focus_mandate": "Ensure auth constraints are met.",
            "rejection_criteria": "Token forgery attack.",
            "constraint_domains": ["security", "auth"],
            "search_enabled": false
        }
    ]"#;
    let slots = parse_decomposition_response(json).unwrap();
    assert_eq!(slots[0].constraint_domains, vec!["security", "auth"]);
    assert!(!slots[0].search_enabled);
}

#[test]
fn parse_response_captures_search_enabled() {
    let json = r#"[
        {
            "role_frame": "You are a security researcher.",
            "cot_style": "step_by_step",
            "focus_mandate": "Check latest CVEs.",
            "rejection_criteria": "Unpatched known vuln.",
            "constraint_domains": ["security"],
            "search_enabled": true
        }
    ]"#;
    let slots = parse_decomposition_response(json).unwrap();
    assert!(slots[0].search_enabled);
    assert_eq!(slots[0].constraint_domains, vec!["security"]);
}

#[test]
fn parse_response_defaults_new_fields_when_absent() {
    let json = r#"[
        {
            "role_frame": "You are a performance engineer.",
            "cot_style": "first_principles",
            "focus_mandate": "Ensure latency SLOs.",
            "rejection_criteria": "Bottleneck under load."
        }
    ]"#;
    let slots = parse_decomposition_response(json).unwrap();
    assert!(slots[0].constraint_domains.is_empty());
    assert!(!slots[0].search_enabled);
}

#[test]
fn corpus_fallback_populates_constraint_domains() {
    let mut doc = ConstraintDoc::new_llm_judge("SEC-001", "No SQL injection.");
    doc.domains = vec!["security".into()];
    let slots = corpus_fallback(&[doc], &ParetoWeights::new(0.33, 0.33, 0.34).unwrap(), 4);
    assert!(!slots.is_empty());
    let security_slot = slots
        .iter()
        .find(|s| s.constraint_domains.contains(&"security".to_string()));
    assert!(
        security_slot.is_some(),
        "expected a slot with constraint_domains=['security']"
    );
}

#[test]
fn step1_includes_thinking_context_when_provided() {
    use h2ai_config::prompts::DECOMPOSITION_STEP1_TASK;
    let rendered = DECOMPOSITION_STEP1_TASK.render(&[
        (
            "thinking_context",
            "PRIOR THINKING CONTEXT:\nUse P95 histogram.\n\n",
        ),
        ("description", "design RTB timeout"),
        ("constraints", "CONSTRAINT-003"),
    ]);
    assert!(rendered.contains("Use P95 histogram"));
    assert!(rendered.contains("design RTB timeout"));
}

#[test]
fn step1_empty_thinking_context_produces_no_prefix() {
    use h2ai_config::prompts::DECOMPOSITION_STEP1_TASK;
    let rendered = DECOMPOSITION_STEP1_TASK.render(&[
        ("thinking_context", ""),
        ("description", "design RTB timeout"),
        ("constraints", "CONSTRAINT-003"),
    ]);
    assert!(!rendered.contains("PRIOR THINKING CONTEXT"));
    assert!(rendered.contains("design RTB timeout"));
}

// ── LlmJudge + Composite rubric extraction via run_decomposition_agent ────────

#[tokio::test]
async fn pipeline_extracts_llm_judge_rubric_into_step1_context() {
    let json_slot = r#"[{"role_frame": "You are a security engineer.", "cot_style": "step_by_step", "focus_mandate": "security constraint SEC-001", "rejection_criteria": "any security violation"}]"#;
    let (adapter, _call_count) = sequential_mock_adapter(vec![
        "Analysis: this constraint covers no-SQL-injection.",
        "Roles: one security engineer.",
        json_slot,
    ]);

    // new_llm_judge creates Composite { And, [LlmJudge { rubric }] }
    // → exercises both Composite and LlmJudge arms of extract_rubric
    let mut constraint = ConstraintDoc::new_llm_judge("SEC-001", "No SQL injection allowed.");
    constraint.domains = vec!["security".into()];
    constraint.remediation_hint = Some("Sanitize all user inputs.".into());

    let weights = ParetoWeights::new(0.33, 0.34, 0.33).unwrap();
    let slots = run_decomposition_agent(
        "Design a secure auth service",
        &[constraint],
        &weights,
        2,
        5,
        &adapter,
        None,
        2048,
        8192,
        "Think carefully.",
    )
    .await
    .unwrap();
    assert_eq!(slots.len(), 1);
}

// ── Untagged constraint produces "untagged" in step1 context ──────────────────

#[tokio::test]
async fn pipeline_handles_untagged_constraint_with_empty_domains() {
    let json_slot = r#"[{"role_frame": "You are an architect.", "cot_style": "step_by_step", "focus_mandate": "meet constraints", "rejection_criteria": "any violation"}]"#;
    let (adapter, _call_count) = sequential_mock_adapter(vec![
        "Analysis: general constraint.",
        "Roles: one architect.",
        json_slot,
    ]);

    // Constraint with empty domains → "untagged" in step1_analyze_task
    // Also empty corpus_domains → step3_assemble_json_task gets empty domains list
    let constraint = ConstraintDoc {
        id: "GEN-001".into(),
        source_file: "gen.yaml".into(),
        description: "General quality constraint.".into(),
        severity: ConstraintSeverity::Advisory,
        predicate: ConstraintPredicate::LengthRange {
            min_chars: None,
            max_chars: None,
        },
        remediation_hint: None,
        domains: vec![], // empty → "untagged"
        mandatory_for_tags: vec![],
        related_to: vec![],
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
    };

    let weights = ParetoWeights::new(0.33, 0.34, 0.33).unwrap();
    let slots = run_decomposition_agent(
        "Design a quality service",
        &[constraint],
        &weights,
        1,
        3,
        &adapter,
        None,
        2048,
        8192,
        "",
    )
    .await
    .unwrap();
    assert_eq!(slots.len(), 1);
}

// ── Line 47: value_to_string array with non-String item ───────────────────────

#[test]
fn parse_coerces_array_with_numeric_item_to_string() {
    // Array focus_mandate where one item is a number (not a String) → line 47 executes
    let json = r#"[{
        "role_frame": "You are a security engineer.",
        "cot_style": "none",
        "focus_mandate": ["Ensure reliability", 42]
    }]"#;
    let slots = parse_decomposition_response(json).unwrap();
    assert_eq!(slots.len(), 1);
    assert!(
        slots[0].focus_mandate.contains("42"),
        "numeric array item must be stringified"
    );
}

// ── Line 51: value_to_string with non-String, non-Array value ────────────────

#[test]
fn parse_coerces_numeric_role_frame_to_string() {
    // role_frame is a number → value_to_string hits the `other => other.to_string()` arm
    let json = r#"[{
        "role_frame": 99,
        "cot_style": "none"
    }]"#;
    let slots = parse_decomposition_response(json).unwrap();
    assert_eq!(slots.len(), 1);
    assert_eq!(slots[0].role_frame, "99");
}

// ── Line 106: NoJsonArray when ] comes before [ in response ───────────────────

#[test]
fn parse_returns_error_when_closing_bracket_before_opening() {
    // ] appears before [ → end <= start → DecompositionError::NoJsonArray
    let response = "] this text [ does not contain a valid array";
    let result = parse_decomposition_response(response);
    assert!(
        result.is_err(),
        "response with ] before [ must produce NoJsonArray error"
    );
}

// ── Line 153: prune_by_orthogonality with single slot (n==1) ─────────────────

#[test]
fn prune_single_slot_returns_it_unchanged() {
    // n == 1 → mean_sim[0] = 0.0 (the else branch at line 153)
    let slots = vec![make_slot("security engineer")];
    let model = IdenticalModel;
    let pruned = prune_by_orthogonality(slots, 1, &model);
    assert_eq!(pruned.len(), 1, "single slot must not be pruned");
}

// ── Lines 178-180: compute_role_diversity with a real EmbeddingModel ──────────

#[test]
fn compute_role_diversity_with_model_returns_non_one() {
    // The Some(m) branch in compute_role_diversity — previously untested
    let slots = vec![
        make_slot("security engineer"),
        make_slot("performance engineer"),
    ];
    let model = OrthogonalModel;
    let diversity = compute_role_diversity(&slots, Some(&model));
    assert!(
        diversity > 0.0,
        "orthogonal slots must produce positive diversity"
    );
}

#[test]
fn compute_role_diversity_none_model_returns_one() {
    let slots = vec![make_slot("any role"), make_slot("another role")];
    let diversity = compute_role_diversity(&slots, None);
    assert!((diversity - 1.0).abs() < 1e-9, "None model must return 1.0");
}
