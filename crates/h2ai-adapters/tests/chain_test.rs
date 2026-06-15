use h2ai_adapters::chain::{execute_chain, tournament_merge};
use h2ai_test_utils::mock_adapter;
use h2ai_types::chain::{ChainedRequest, ChainStep};
use h2ai_types::adapter::AdapterError;
use h2ai_test_utils::failing_adapter;
use std::sync::Mutex;
use async_trait::async_trait;
use h2ai_types::adapter::{ComputeRequest, ComputeResponse, IComputeAdapter};
use h2ai_types::config::AdapterKind;

// ── CapturingAdapter records every system_context it receives ─────────────────

struct CapturingAdapter {
    contexts: Mutex<Vec<String>>,
    response: String,
    kind: AdapterKind,
}

impl std::fmt::Debug for CapturingAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CapturingAdapter").finish()
    }
}

#[async_trait]
impl IComputeAdapter for CapturingAdapter {
    async fn execute(&self, req: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
        self.contexts.lock().unwrap().push(req.system_context.clone());
        Ok(ComputeResponse {
            output: self.response.clone(),
            token_cost: 0,
            adapter_kind: self.kind.clone(),
            tokens_used: None,
            reasoning_trace: None,
        })
    }
    fn kind(&self) -> &AdapterKind { &self.kind }
}

fn capturing(response: &str) -> CapturingAdapter {
    CapturingAdapter {
        contexts: Mutex::new(vec![]),
        response: response.to_string(),
        kind: AdapterKind::CloudGeneric {
            endpoint: "http://test".into(),
            api_key_env: "TEST".into(),
            model: None,
            provider: Default::default(),
        },
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn single_step_returns_adapter_output() {
    let adapter = mock_adapter("hello world");
    let result = execute_chain(&adapter, ChainedRequest {
        initial_system_context: "sys".into(),
        steps: vec![ChainStep {
            template: "task".into(),
            tau: h2ai_types::sizing::TauValue::new(0.3).unwrap(),
            max_tokens: 256,
        }],
    })
    .await
    .unwrap();

    assert_eq!(result, "hello world");
}

#[tokio::test]
async fn empty_steps_returns_empty_string() {
    let adapter = mock_adapter("anything");
    let result = execute_chain(&adapter, ChainedRequest {
        initial_system_context: "sys".into(),
        steps: vec![],
    })
    .await
    .unwrap();

    assert_eq!(result, "");
}

#[tokio::test]
async fn two_steps_output_becomes_next_system_context() {
    // Step 0 outputs "step0-output"; step 1 must receive it as system_context.
    let adapter = capturing("step0-output");

    let _ = execute_chain(&adapter, ChainedRequest {
        initial_system_context: "initial-ctx".into(),
        steps: vec![
            ChainStep {
                template: "task0".into(),
                tau: h2ai_types::sizing::TauValue::new(0.3).unwrap(),
                max_tokens: 256,
            },
            ChainStep {
                template: "task1".into(),
                tau: h2ai_types::sizing::TauValue::new(0.3).unwrap(),
                max_tokens: 256,
            },
        ],
    })
    .await
    .unwrap();

    let ctxs = adapter.contexts.lock().unwrap();
    assert_eq!(ctxs.len(), 2);
    assert_eq!(ctxs[0], "initial-ctx", "step 0 must receive initial_system_context");
    assert_eq!(ctxs[1], "step0-output", "step 1 must receive step 0's output as system_context");
}

#[tokio::test]
async fn adapter_error_propagates() {
    let result = execute_chain(&failing_adapter(), ChainedRequest {
        initial_system_context: "sys".into(),
        steps: vec![ChainStep {
            template: "t".into(),
            tau: h2ai_types::sizing::TauValue::new(0.1).unwrap(),
            max_tokens: 64,
        }],
    })
    .await;

    assert!(matches!(result, Err(AdapterError::NetworkError(_))));
}

fn tau() -> h2ai_types::sizing::TauValue {
    h2ai_types::sizing::TauValue::new(0.3).unwrap()
}

#[tokio::test]
async fn tournament_merge_single_proposal_no_adapter_call() {
    // One proposal → returned as-is; adapter must not be called.
    let adapter = failing_adapter();  // would error if called
    let result = tournament_merge(
        &adapter,
        "sys",
        vec!["only".into()],
        "merge {proposal_b}",
        tau(),
        128,
    )
    .await
    .unwrap();
    assert_eq!(result, "only");
}

#[tokio::test]
async fn tournament_merge_empty_returns_empty_no_adapter_call() {
    let adapter = failing_adapter();
    let result = tournament_merge(&adapter, "sys", vec![], "merge {proposal_b}", tau(), 128)
        .await
        .unwrap();
    assert_eq!(result, "");
}

#[tokio::test]
async fn tournament_merge_two_proposals_one_adapter_call() {
    // Exactly one merge call. Template placeholder must be rendered.
    let adapter = capturing("merged-output");

    let result = tournament_merge(
        &adapter,
        "sys",
        vec!["proposal-a".into(), "proposal-b".into()],
        "combine: {proposal_b}",
        tau(),
        256,
    )
    .await
    .unwrap();

    assert_eq!(result, "merged-output");

    let ctxs = adapter.contexts.lock().unwrap();
    assert_eq!(ctxs.len(), 1, "exactly one merge call");
    // proposal-a must appear in the system context (as "Current Best")
    assert!(
        ctxs[0].contains("proposal-a"),
        "system context must contain proposal A"
    );
}

#[tokio::test]
async fn tournament_merge_four_proposals_two_rounds() {
    // Round 0: (A,B) → M1 and (C,D) → M2 in parallel.
    // Round 1: (M1,M2) → final.
    // Total calls = 3.
    let adapter = capturing("merged");

    tournament_merge(
        &adapter,
        "sys",
        vec!["a".into(), "b".into(), "c".into(), "d".into()],
        "m {proposal_b}",
        tau(),
        128,
    )
    .await
    .unwrap();

    let ctxs = adapter.contexts.lock().unwrap();
    assert_eq!(ctxs.len(), 3, "4 proposals → 3 merge calls across 2 rounds");
}

#[tokio::test]
async fn tournament_merge_three_proposals_bye_passes_through() {
    // Round 0: (A,B) → M1; C is a bye.
    // Round 1: (M1,C) → final.
    // Total calls = 2.
    let adapter = capturing("merged");

    tournament_merge(
        &adapter,
        "sys",
        vec!["a".into(), "b".into(), "c".into()],
        "m {proposal_b}",
        tau(),
        128,
    )
    .await
    .unwrap();

    let ctxs = adapter.contexts.lock().unwrap();
    assert_eq!(ctxs.len(), 2, "3 proposals → 2 merge calls");
}

#[tokio::test]
async fn tournament_merge_krum_winner_first_in_system_context() {
    // proposals[0] is the Krum winner. It must appear in system_context of round-0's
    // first pair (not as {proposal_b}).
    let adapter = capturing("ok");

    tournament_merge(
        &adapter,
        "base-sys",
        vec!["KRUM-LEADER".into(), "challenger".into()],
        "vs {proposal_b}",
        tau(),
        128,
    )
    .await
    .unwrap();

    let ctxs = adapter.contexts.lock().unwrap();
    // The one call's system_context must contain KRUM-LEADER (in "Current Best" section)
    assert!(
        ctxs[0].contains("KRUM-LEADER"),
        "Krum leader must be in system_context (position-primacy), got: {}",
        ctxs[0]
    );
}
