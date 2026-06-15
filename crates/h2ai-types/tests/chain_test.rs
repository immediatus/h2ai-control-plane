use h2ai_types::chain::{ChainStep, ChainedRequest};
use h2ai_types::sizing::TauValue;

#[test]
fn chain_step_roundtrip() {
    let step = ChainStep {
        template: "fill this".into(),
        tau: TauValue::new(0.3).unwrap(),
        max_tokens: 512,
    };
    let json = serde_json::to_string(&step).unwrap();
    let restored: ChainStep = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.template, "fill this");
    assert_eq!(restored.max_tokens, 512);
}

#[test]
fn chained_request_roundtrip() {
    let req = ChainedRequest {
        initial_system_context: "sys".into(),
        steps: vec![
            ChainStep {
                template: "t1".into(),
                tau: TauValue::new(0.2).unwrap(),
                max_tokens: 256,
            },
            ChainStep {
                template: "t2".into(),
                tau: TauValue::new(0.5).unwrap(),
                max_tokens: 128,
            },
        ],
    };
    let json = serde_json::to_string(&req).unwrap();
    let restored: ChainedRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.steps.len(), 2);
    assert_eq!(restored.steps[0].template, "t1");
    assert_eq!(restored.steps[1].max_tokens, 128);
}
