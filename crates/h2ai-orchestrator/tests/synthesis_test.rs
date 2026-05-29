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
use chrono::Utc;
use h2ai_config::H2AIConfig;
use h2ai_orchestrator::synthesis::{
    CritiqueDocument, CritiqueVerdict, SynthesisError, SynthesisInput, SynthesisPhase,
};
use h2ai_test_utils::{mock_adapter, sequenced_adapter, MockIComputeAdapter};
use h2ai_types::adapter::AdapterError;
use h2ai_types::config::AdapterKind;
use h2ai_types::events::ProposalEvent;
use h2ai_types::identity::{ExplorerId, TaskId};
use h2ai_types::sizing::TauValue;
use std::sync::{Arc, Mutex};

fn make_proposal(output: &str) -> ProposalEvent {
    ProposalEvent {
        task_id: TaskId::new(),
        explorer_id: ExplorerId::new(),
        tau: TauValue::new(0.7).unwrap(),
        generation: 0,
        raw_output: output.to_string(),
        token_cost: 100,
        adapter_kind: AdapterKind::CloudGeneric {
            endpoint: "mock://test".into(),
            api_key_env: "NONE".into(),
            model: None,
            provider: Default::default(),
        },
        timestamp: Utc::now(),
    }
}

#[test]
fn critique_document_deserializes_from_json() {
    let json = r#"{
        "proposal_critiques": [
            {
                "proposal_id": "exp_001",
                "strengths": ["Good coverage"],
                "weaknesses": ["Misses latency"],
                "verdict": "partial"
            }
        ],
        "contradictions": [
            {
                "proposals": ["exp_001", "exp_002"],
                "conflict_description": "Redis vs stateless",
                "resolution": "stateless wins"
            }
        ],
        "synthesis_guidance": "Build on exp_001."
    }"#;
    let doc: CritiqueDocument = serde_json::from_str(json).unwrap();
    assert_eq!(doc.proposal_critiques.len(), 1);
    assert_eq!(doc.contradictions.len(), 1);
    assert!(matches!(
        doc.proposal_critiques[0].verdict,
        CritiqueVerdict::Partial
    ));
    assert_eq!(doc.synthesis_guidance, "Build on exp_001.");
}

#[tokio::test]
async fn synthesis_phase_succeeds_with_valid_critique_and_synthesis() {
    let valid_critique = r#"{
        "proposal_critiques": [
            {"proposal_id": "p1", "strengths": ["s1"], "weaknesses": ["w1"], "verdict": "partial"},
            {"proposal_id": "p2", "strengths": ["s2"], "weaknesses": ["w2"], "verdict": "strong"}
        ],
        "contradictions": [],
        "synthesis_guidance": "Use p2 as foundation."
    }"#;

    let adapter = sequenced_adapter(vec![
        valid_critique.to_string(),
        "Unified synthesis output combining both proposals.".to_string(),
    ]);

    let proposals = vec![
        make_proposal("proposal one text"),
        make_proposal("proposal two text"),
    ];
    let cfg = H2AIConfig::default();
    let input = SynthesisInput {
        task_description: "Implement auth system",
        constraint_list: "Must be stateless",
        proposals: &proposals,
        adapter: &adapter,
        cfg: &cfg,
    };

    let result = SynthesisPhase::run(input).await;
    assert!(result.is_ok(), "expected Ok, got: {:?}", result);
    let output = result.unwrap();
    assert_eq!(
        output.synthesis_text,
        "Unified synthesis output combining both proposals."
    );
    assert_eq!(output.critique_doc.proposal_critiques.len(), 2);
    assert_eq!(output.critique_tokens, 10);
    assert_eq!(output.synthesis_tokens, 10);
}

#[tokio::test]
async fn synthesis_phase_retries_critique_once_on_bad_json() {
    // First call: bad JSON. Second call (retry): valid JSON. Third call: synthesis.
    let valid_critique = r#"{
        "proposal_critiques": [
            {"proposal_id": "p1", "strengths": ["s1"], "weaknesses": [], "verdict": "strong"}
        ],
        "contradictions": [],
        "synthesis_guidance": "Use p1."
    }"#;

    let adapter = sequenced_adapter(vec![
        "not valid json at all".to_string(), // first attempt — bad JSON
        valid_critique.to_string(),          // retry — valid JSON
        "Synthesis text after retry.".to_string(), // synthesis call
    ]);

    let proposals = vec![make_proposal("text"), make_proposal("more text")];
    let cfg = H2AIConfig::default();
    let input = SynthesisInput {
        task_description: "task",
        constraint_list: "none",
        proposals: &proposals,
        adapter: &adapter,
        cfg: &cfg,
    };

    let result = SynthesisPhase::run(input).await;
    assert!(
        result.is_ok(),
        "expected retry to succeed, got: {:?}",
        result
    );
    assert_eq!(
        result.unwrap().synthesis_text,
        "Synthesis text after retry."
    );
}

#[tokio::test]
async fn synthesis_phase_returns_critique_failed_on_two_bad_json() {
    let adapter = mock_adapter("not valid json");
    let proposals = vec![make_proposal("text one"), make_proposal("text two")];
    let cfg = H2AIConfig::default();
    let input = SynthesisInput {
        task_description: "task",
        constraint_list: "none",
        proposals: &proposals,
        adapter: &adapter,
        cfg: &cfg,
    };

    let result = SynthesisPhase::run(input).await;
    assert!(
        matches!(result, Err(SynthesisError::CritiqueFailed(_))),
        "expected CritiqueFailed, got: {:?}",
        result
    );
}

#[test]
fn critique_verdict_weak_deserializes() {
    let json =
        r#"{"proposal_id": "p1", "strengths": [], "weaknesses": ["too vague"], "verdict": "weak"}"#;
    use h2ai_orchestrator::synthesis::ProposalCritique;
    let critique: ProposalCritique = serde_json::from_str(json).unwrap();
    assert!(matches!(critique.verdict, CritiqueVerdict::Weak));
}

#[tokio::test]
async fn synthesis_phase_returns_adapter_error_on_invalid_tau() {
    let adapter = mock_adapter("{}");
    let proposals = vec![make_proposal("text")];
    let cfg = H2AIConfig {
        synthesis_tau: -1.0, // invalid tau → TauValue::new fails
        ..H2AIConfig::default()
    };
    let input = SynthesisInput {
        task_description: "task",
        constraint_list: "none",
        proposals: &proposals,
        adapter: &adapter,
        cfg: &cfg,
    };
    let result = SynthesisPhase::run(input).await;
    assert!(
        matches!(result, Err(SynthesisError::AdapterError(_))),
        "invalid tau must produce AdapterError, got: {:?}",
        result
    );
}

#[tokio::test]
async fn synthesis_phase_returns_adapter_error_when_stage2_fails() {
    // Stage 1 (critique) succeeds with valid JSON; Stage 2 (synthesis adapter) fails.
    let valid_critique = r#"{
        "proposal_critiques": [{"proposal_id": "p1", "strengths": ["s"], "weaknesses": [], "verdict": "strong"}],
        "contradictions": [],
        "synthesis_guidance": "go."
    }"#;

    let call_count = Arc::new(Mutex::new(0usize));
    let call_count2 = call_count.clone();
    let first_output = valid_critique.to_string();
    let kind = AdapterKind::CloudGeneric {
        endpoint: "mock://failsecond".into(),
        api_key_env: "NONE".into(),
        model: None,
        provider: Default::default(),
    };
    let kind2 = kind.clone();
    let mut adapter = MockIComputeAdapter::new();
    adapter.expect_execute().returning(move |_| {
        let n = {
            let mut c = call_count2.lock().unwrap();
            let v = *c;
            *c += 1;
            v
        };
        if n == 0 {
            Ok(h2ai_types::adapter::ComputeResponse {
                output: first_output.clone(),
                token_cost: 10,
                adapter_kind: kind.clone(),
                tokens_used: None,
                reasoning_trace: None,
            })
        } else {
            Err(AdapterError::NetworkError("stage2 network failure".into()))
        }
    });
    adapter.expect_kind().return_const(kind2).times(0..);

    let proposals = vec![make_proposal("proposal text")];
    let cfg = H2AIConfig::default();
    let input = SynthesisInput {
        task_description: "task",
        constraint_list: "none",
        proposals: &proposals,
        adapter: &adapter,
        cfg: &cfg,
    };
    let result = SynthesisPhase::run(input).await;
    assert!(
        matches!(result, Err(SynthesisError::AdapterError(_))),
        "stage 2 adapter failure must produce AdapterError, got: {:?}",
        result
    );
}
