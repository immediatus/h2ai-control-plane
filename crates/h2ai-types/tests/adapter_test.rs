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
    };
    let json = serde_json::to_string(&resp).unwrap();
    let back: ComputeResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(resp.token_cost, back.token_cost);
    assert_eq!(resp.output, back.output);
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
