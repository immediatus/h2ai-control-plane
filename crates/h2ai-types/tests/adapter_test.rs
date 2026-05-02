use h2ai_types::adapter::{AdapterError, ComputeRequest, ComputeResponse};
use h2ai_types::config::AdapterKind;
use h2ai_types::physics::TauValue;

#[test]
fn compute_request_serde_round_trip() {
    let req = ComputeRequest {
        system_context: "You must use stateless auth per ADR-004.".into(),
        task: "Write a JWT validation middleware.".into(),
        tau: TauValue::new(0.6).unwrap(),
        max_tokens: 2048,
    };
    let json = serde_json::to_string(&req).unwrap();
    let back: ComputeRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(req.tau, back.tau);
    assert_eq!(req.max_tokens, back.max_tokens);
    assert_eq!(req.system_context, back.system_context);
    assert_eq!(req.task, back.task);
}

#[test]
fn compute_response_serde_round_trip() {
    let resp = ComputeResponse {
        output: "fn validate_jwt(token: &str) -> Result<Claims, JwtError> { ... }".into(),
        token_cost: 312,
        adapter_kind: AdapterKind::CloudGeneric {
            endpoint: "https://api.example.com".into(),
            api_key_env: "CLOUD_API_KEY".into(),
        },
        tokens_used: None,
    };
    let json = serde_json::to_string(&resp).unwrap();
    let back: ComputeResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(resp.token_cost, back.token_cost);
    assert_eq!(resp.output, back.output);
    assert_eq!(resp.adapter_kind, back.adapter_kind);
}

#[test]
fn adapter_error_timeout_display() {
    let err = AdapterError::Timeout;
    assert!(err.to_string().contains("timed out"));
}

#[test]
fn adapter_error_oom_display() {
    let err = AdapterError::OomPanic("CUDA out of memory".into());
    assert!(err.to_string().contains("CUDA out of memory"));
}

#[test]
fn adapter_error_network_display() {
    let err = AdapterError::NetworkError("connection refused".into());
    assert!(err.to_string().contains("connection refused"));
}

// ── AdapterRegistry tests ─────────────────────────────────────────────────────

#[derive(Debug)]
struct LabelAdapter(String, h2ai_types::config::AdapterKind);

#[async_trait::async_trait]
impl h2ai_types::adapter::IComputeAdapter for LabelAdapter {
    async fn execute(
        &self,
        _req: h2ai_types::adapter::ComputeRequest,
    ) -> Result<h2ai_types::adapter::ComputeResponse, h2ai_types::adapter::AdapterError> {
        Ok(h2ai_types::adapter::ComputeResponse {
            output: self.0.clone(),
            token_cost: 0,
            adapter_kind: self.1.clone(),
            tokens_used: None,
        })
    }
    fn kind(&self) -> &h2ai_types::config::AdapterKind {
        &self.1
    }
}

fn label(name: &str) -> std::sync::Arc<dyn h2ai_types::adapter::IComputeAdapter> {
    std::sync::Arc::new(LabelAdapter(
        name.into(),
        h2ai_types::config::AdapterKind::CloudGeneric {
            endpoint: "mock://test".into(),
            api_key_env: "NONE".into(),
        },
    ))
}

#[test]
fn registry_scoring_falls_back_to_reasoning_when_not_set() {
    let reasoning = label("reasoning");
    let reg = h2ai_types::adapter::AdapterRegistry::new(reasoning.clone());
    let resolved = reg.resolve(&h2ai_types::adapter::TaskProfile::Scoring) as *const _;
    let expected = reasoning.as_ref() as *const _;
    // Raw pointer comparison is valid here: Arc::as_ref() and resolve() both produce
    // a reference into the same Arc allocation, so data pointers are identical.
    assert_eq!(
        resolved, expected,
        "scoring must fall back to reasoning when not configured"
    );
}

#[test]
fn registry_scoring_uses_dedicated_adapter_when_set() {
    let scoring = label("scoring");
    let reg =
        h2ai_types::adapter::AdapterRegistry::new(label("reasoning")).with_scoring(scoring.clone());
    let resolved = reg.resolve(&h2ai_types::adapter::TaskProfile::Scoring) as *const _;
    let expected = scoring.as_ref() as *const _;
    // Raw pointer comparison is valid here: Arc::as_ref() and resolve() both produce
    // a reference into the same Arc allocation, so data pointers are identical.
    assert_eq!(
        resolved, expected,
        "scoring must return the dedicated adapter"
    );
}

#[test]
fn registry_structural_falls_back_to_reasoning_when_not_set() {
    let reasoning = label("reasoning");
    let reg = h2ai_types::adapter::AdapterRegistry::new(reasoning.clone());
    let resolved = reg.resolve(&h2ai_types::adapter::TaskProfile::Structural) as *const _;
    let expected = reasoning.as_ref() as *const _;
    // Raw pointer comparison is valid here: Arc::as_ref() and resolve() both produce
    // a reference into the same Arc allocation, so data pointers are identical.
    assert_eq!(
        resolved, expected,
        "structural must fall back to reasoning when not configured"
    );
}

#[test]
fn registry_structural_uses_dedicated_adapter_when_set() {
    let structural = label("structural");
    let reg = h2ai_types::adapter::AdapterRegistry::new(label("reasoning"))
        .with_structural(structural.clone());
    let resolved = reg.resolve(&h2ai_types::adapter::TaskProfile::Structural) as *const _;
    let expected = structural.as_ref() as *const _;
    // Raw pointer comparison is valid here: Arc::as_ref() and resolve() both produce
    // a reference into the same Arc allocation, so data pointers are identical.
    assert_eq!(
        resolved, expected,
        "structural must return the dedicated adapter"
    );
}

#[test]
fn registry_reasoning_resolves_to_reasoning_adapter() {
    let reasoning = label("reasoning");
    let reg = h2ai_types::adapter::AdapterRegistry::new(reasoning.clone());
    let resolved = reg.resolve(&h2ai_types::adapter::TaskProfile::Reasoning) as *const _;
    let expected = reasoning.as_ref() as *const _;
    // Raw pointer comparison is valid here: Arc::as_ref() and resolve() both produce
    // a reference into the same Arc allocation, so data pointers are identical.
    assert_eq!(resolved, expected);
}

#[test]
fn registry_all_three_resolve_independently() {
    let r = label("r");
    let sc = label("sc");
    let st = label("st");
    let reg = h2ai_types::adapter::AdapterRegistry::new(r.clone())
        .with_scoring(sc.clone())
        .with_structural(st.clone());
    // Raw pointer comparison is valid here: Arc::as_ref() and resolve() both produce
    // a reference into the same Arc allocation, so data pointers are identical.
    assert_eq!(
        reg.resolve(&h2ai_types::adapter::TaskProfile::Reasoning) as *const _,
        r.as_ref() as *const _
    );
    assert_eq!(
        reg.resolve(&h2ai_types::adapter::TaskProfile::Scoring) as *const _,
        sc.as_ref() as *const _
    );
    assert_eq!(
        reg.resolve(&h2ai_types::adapter::TaskProfile::Structural) as *const _,
        st.as_ref() as *const _
    );
}
