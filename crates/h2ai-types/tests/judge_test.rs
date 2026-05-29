use h2ai_types::config::{AdapterFamily, AdapterKind};
use h2ai_types::judge::{JudgePersona, PanelDiversityKind};
use std::path::PathBuf;

#[test]
fn judge_persona_system_prompt_prefix_distinct() {
    let prefixes = [
        JudgePersona::Literal.system_prompt_prefix(),
        JudgePersona::Contextual.system_prompt_prefix(),
        JudgePersona::Skeptical.system_prompt_prefix(),
    ];
    for p in &prefixes {
        assert!(!p.is_empty());
    }
    assert_ne!(prefixes[0], prefixes[1]);
    assert_ne!(prefixes[1], prefixes[2]);
    assert_ne!(prefixes[0], prefixes[2]);
}

#[test]
fn judge_persona_roundtrips_serde() {
    let p = JudgePersona::Skeptical;
    let json = serde_json::to_string(&p).unwrap();
    let back: JudgePersona = serde_json::from_str(&json).unwrap();
    assert_eq!(p, back);
}

#[test]
fn panel_diversity_kind_roundtrips_serde() {
    let k = PanelDiversityKind::CrossFamily;
    let json = serde_json::to_string(&k).unwrap();
    let back: PanelDiversityKind = serde_json::from_str(&json).unwrap();
    assert_eq!(k, back);
}

#[test]
fn adapter_family_from_kind_all_variants() {
    assert_eq!(
        AdapterFamily::from_kind(&AdapterKind::Anthropic {
            api_key_env: "K".into(),
            model: "claude-3".into()
        }),
        AdapterFamily::Anthropic
    );
    assert_eq!(
        AdapterFamily::from_kind(&AdapterKind::OpenAI {
            api_key_env: "K".into(),
            model: "gpt-4".into()
        }),
        AdapterFamily::OpenAI
    );
    assert_eq!(
        AdapterFamily::from_kind(&AdapterKind::LocalLlamaCpp {
            model_path: PathBuf::from("/tmp/model.gguf"),
            n_threads: 4,
        }),
        AdapterFamily::Local
    );
    assert_eq!(
        AdapterFamily::from_kind(&AdapterKind::Ollama {
            endpoint: "http://localhost:11434".into(),
            model: "llama3".into(),
        }),
        AdapterFamily::Local
    );
    assert_eq!(
        AdapterFamily::from_kind(&AdapterKind::CloudGeneric {
            endpoint: "https://example.com".into(),
            api_key_env: "K".into(),
            model: None,
            provider: Default::default(),
        }),
        AdapterFamily::Cloud
    );
    assert_eq!(
        AdapterFamily::from_kind(&AdapterKind::A2a {
            endpoint: "https://example.com".into(),
            auth_scheme: "none".into(),
            auth_token_env: String::new(),
            timeout_minutes: 5,
            poll_interval_ms: 1000,
            max_poll_interval_ms: 30000,
            agent_card_cache_ttl_s: 300,
        }),
        AdapterFamily::Cloud
    );
}

#[test]
fn adapter_kind_family_method_matches_from_kind() {
    let kind = AdapterKind::Anthropic {
        api_key_env: "K".into(),
        model: "c".into(),
    };
    assert_eq!(kind.family(), AdapterFamily::from_kind(&kind));
}

#[test]
fn oracle_verdict_serde_roundtrip() {
    use h2ai_types::sizing::OracleVerdict;

    let verdict = OracleVerdict {
        details: serde_json::json!({"pass_count": 3, "total_count": 4}),
    };
    let json = serde_json::to_string(&verdict).unwrap();
    let back: OracleVerdict = serde_json::from_str(&json).unwrap();
    assert_eq!(back.details["pass_count"], 3);
    assert_eq!(back, verdict);
}
