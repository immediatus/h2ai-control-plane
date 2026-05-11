use chrono::Utc;
use h2ai_config::H2AIConfig;
use h2ai_orchestrator::synthesis::{
    CritiqueDocument, CritiqueVerdict, SynthesisError, SynthesisInput, SynthesisPhase,
};
use h2ai_types::adapter::{AdapterError, ComputeRequest, ComputeResponse, IComputeAdapter};
use h2ai_types::config::AdapterKind;
use h2ai_types::events::ProposalEvent;
use h2ai_types::identity::{ExplorerId, TaskId};
use h2ai_types::sizing::TauValue;

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
        },
        timestamp: Utc::now(),
    }
}

// An adapter that returns the same output for every call
#[derive(Debug)]
struct FixedAdapter {
    output: String,
    cost: u64,
    kind: AdapterKind,
}

impl FixedAdapter {
    fn new(output: String, cost: u64) -> Self {
        Self {
            output,
            cost,
            kind: AdapterKind::CloudGeneric {
                endpoint: "mock://fixed".into(),
                api_key_env: "NONE".into(),
                model: None,
            },
        }
    }
}

#[async_trait::async_trait]
impl IComputeAdapter for FixedAdapter {
    async fn execute(&self, _req: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
        Ok(ComputeResponse {
            output: self.output.clone(),
            token_cost: self.cost,
            adapter_kind: self.kind.clone(),
            tokens_used: None,
        })
    }
    fn kind(&self) -> &AdapterKind {
        &self.kind
    }
}

// An adapter that returns different outputs on successive calls
use std::sync::{Arc, Mutex};

#[derive(Debug)]
struct SequencedAdapter {
    responses: Arc<Mutex<Vec<String>>>,
    kind: AdapterKind,
}

impl SequencedAdapter {
    fn new(responses: Vec<String>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses)),
            kind: AdapterKind::CloudGeneric {
                endpoint: "mock://sequenced".into(),
                api_key_env: "NONE".into(),
                model: None,
            },
        }
    }
}

#[async_trait::async_trait]
impl IComputeAdapter for SequencedAdapter {
    async fn execute(&self, _req: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
        let mut responses = self.responses.lock().unwrap();
        let output = if responses.is_empty() {
            "fallback".to_string()
        } else {
            responses.remove(0)
        };
        Ok(ComputeResponse {
            output,
            token_cost: 100,
            adapter_kind: self.kind.clone(),
            tokens_used: None,
        })
    }
    fn kind(&self) -> &AdapterKind {
        &self.kind
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

    let adapter = SequencedAdapter::new(vec![
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
    assert_eq!(output.critique_tokens, 100);
    assert_eq!(output.synthesis_tokens, 100);
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

    let adapter = SequencedAdapter::new(vec![
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
    let adapter = FixedAdapter::new("not valid json".to_string(), 10);
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
