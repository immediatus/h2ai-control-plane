use async_trait::async_trait;
use h2ai_constraints::types::{ConstraintDoc, ConstraintPredicate, ConstraintSeverity};
use h2ai_context::embedding::EmbeddingModel;
use h2ai_orchestrator::decomposition::{
    corpus_fallback, parse_decomposition_response, prune_by_orthogonality, run_decomposition_agent,
};
use h2ai_types::adapter::{AdapterError, ComputeRequest, ComputeResponse, IComputeAdapter};
use h2ai_types::config::{AdapterKind, ParetoWeights};
use h2ai_types::manifest::{CotStyle, ExplorerSlotConfig};

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

// ── run_decomposition_agent ──────────────────────────────────────────────────

#[derive(Debug)]
struct MockDecompositionAdapter {
    response: String,
}

#[async_trait]
impl IComputeAdapter for MockDecompositionAdapter {
    async fn execute(&self, _req: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
        Ok(ComputeResponse {
            output: self.response.clone(),
            token_cost: 100,
            adapter_kind: AdapterKind::CloudGeneric {
                endpoint: String::new(),
                api_key_env: String::new(),
                model: None,
            },
            tokens_used: None,
            reasoning_trace: None,
        })
    }
    fn kind(&self) -> &AdapterKind {
        static KIND: std::sync::OnceLock<AdapterKind> = std::sync::OnceLock::new();
        KIND.get_or_init(|| AdapterKind::CloudGeneric {
            endpoint: String::new(),
            api_key_env: String::new(),
            model: None,
        })
    }
}

#[derive(Debug)]
struct FailingAdapter;

#[async_trait]
impl IComputeAdapter for FailingAdapter {
    async fn execute(&self, _req: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
        Err(AdapterError::Timeout)
    }
    fn kind(&self) -> &AdapterKind {
        static KIND: std::sync::OnceLock<AdapterKind> = std::sync::OnceLock::new();
        KIND.get_or_init(|| AdapterKind::CloudGeneric {
            endpoint: String::new(),
            api_key_env: String::new(),
            model: None,
        })
    }
}

#[tokio::test]
async fn run_decomposition_agent_uses_llm_slots_on_success() {
    let adapter = MockDecompositionAdapter {
        response: r#"[
          {"role_frame": "You are a security engineer.", "cot_style": "devil_s_advocate"},
          {"role_frame": "You are a performance engineer.", "cot_style": "first_principles"}
        ]"#
        .into(),
    };
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
    let adapter = MockDecompositionAdapter {
        response: r#"[
          {"role_frame": "You are a security engineer.", "cot_style": "none"},
          {"role_frame": "You are a performance engineer.", "cot_style": "none"},
          {"role_frame": "You are a correctness engineer.", "cot_style": "none"}
        ]"#
        .into(),
    };
    let weights = ParetoWeights::new(0.33, 0.34, 0.33).unwrap();
    let slots =
        run_decomposition_agent("task", &[], &weights, 2, 2, &adapter, None, 2048, 8192, "")
            .await
            .unwrap();
    assert_eq!(slots.len(), 2);
}

#[tokio::test]
async fn run_decomposition_agent_returns_err_on_adapter_failure() {
    let adapter = FailingAdapter;
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
    let adapter = MockDecompositionAdapter {
        response: "not valid json".into(),
    };
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
    let adapter = MockDecompositionAdapter {
        response: "Here is my analysis of the problem... [no JSON array]".into(),
    };
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

/// Captures all `task` strings from `execute` calls while returning sequential responses.
/// Used to assert on the content of prompts sent to each pipeline step.
#[derive(Debug)]
struct CapturingAdapter {
    responses: Vec<String>,
    captured_tasks: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    call_count: std::sync::Arc<std::sync::Mutex<usize>>,
}

impl CapturingAdapter {
    fn new(responses: Vec<&str>) -> (Self, std::sync::Arc<std::sync::Mutex<Vec<String>>>) {
        let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let adapter = Self {
            responses: responses.into_iter().map(str::to_string).collect(),
            captured_tasks: captured.clone(),
            call_count: std::sync::Arc::new(std::sync::Mutex::new(0)),
        };
        (adapter, captured)
    }
}

#[async_trait]
impl IComputeAdapter for CapturingAdapter {
    async fn execute(&self, req: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
        self.captured_tasks.lock().unwrap().push(req.task.clone());
        let mut count = self.call_count.lock().unwrap();
        let response = self
            .responses
            .get(*count)
            .cloned()
            .unwrap_or_else(|| self.responses.last().cloned().unwrap_or_default());
        *count += 1;
        Ok(ComputeResponse {
            output: response,
            token_cost: 50,
            adapter_kind: AdapterKind::CloudGeneric {
                endpoint: String::new(),
                api_key_env: String::new(),
                model: None,
            },
            tokens_used: None,
            reasoning_trace: None,
        })
    }
    fn kind(&self) -> &AdapterKind {
        static KIND: std::sync::OnceLock<AdapterKind> = std::sync::OnceLock::new();
        KIND.get_or_init(|| AdapterKind::CloudGeneric {
            endpoint: String::new(),
            api_key_env: String::new(),
            model: None,
        })
    }
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

    let (adapter, captured) = CapturingAdapter::new(vec![
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

    let (adapter, captured) = CapturingAdapter::new(vec![
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

/// Returns different responses in sequence (Step1 → Step2 → Step3).
/// The counter tracks which call we're on so each step gets appropriate content.
#[derive(Debug)]
struct SequentialMockAdapter {
    responses: Vec<String>,
    call_count: std::sync::Arc<std::sync::Mutex<usize>>,
}

impl SequentialMockAdapter {
    fn new(responses: Vec<&str>) -> Self {
        Self {
            responses: responses.into_iter().map(str::to_string).collect(),
            call_count: std::sync::Arc::new(std::sync::Mutex::new(0)),
        }
    }
}

#[async_trait]
impl IComputeAdapter for SequentialMockAdapter {
    async fn execute(&self, _req: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
        let mut count = self.call_count.lock().unwrap();
        let response = self
            .responses
            .get(*count)
            .cloned()
            .unwrap_or_else(|| self.responses.last().cloned().unwrap_or_default());
        *count += 1;
        Ok(ComputeResponse {
            output: response,
            token_cost: 50,
            adapter_kind: AdapterKind::CloudGeneric {
                endpoint: String::new(),
                api_key_env: String::new(),
                model: None,
            },
            tokens_used: None,
            reasoning_trace: None,
        })
    }
    fn kind(&self) -> &AdapterKind {
        static KIND: std::sync::OnceLock<AdapterKind> = std::sync::OnceLock::new();
        KIND.get_or_init(|| AdapterKind::CloudGeneric {
            endpoint: String::new(),
            api_key_env: String::new(),
            model: None,
        })
    }
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

    let adapter = SequentialMockAdapter::new(vec![
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
    let calls = *adapter.call_count.lock().unwrap();
    assert_eq!(calls, 3, "pipeline must make exactly 3 adapter calls");
}

/// Step 1 failure propagates immediately — pipeline does not continue to Step 2.
#[tokio::test]
async fn pipeline_fails_fast_on_step1_adapter_error() {
    let adapter = FailingAdapter;
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
    let adapter = SequentialMockAdapter::new(vec![
        "Step 1: analysis output.",
        "Step 2: role design output.",
        "Step 3 produced no JSON array here.",
    ]);
    let weights = ParetoWeights::new(0.33, 0.34, 0.33).unwrap();
    let result =
        run_decomposition_agent("task", &[], &weights, 2, 5, &adapter, None, 2048, 8192, "").await;
    assert!(result.is_err(), "Step 3 non-JSON must propagate as Err");
    let calls = *adapter.call_count.lock().unwrap();
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
