use h2ai_adapters::factory::AdapterFactory;
use h2ai_types::config::AdapterKind;

#[test]
fn factory_builds_cloud_generic_adapter() {
    let kind = AdapterKind::CloudGeneric {
        endpoint: "https://api.example.com".into(),
        api_key_env: "MY_KEY".into(),
    };
    let adapter = AdapterFactory::build(&kind);
    assert!(adapter.is_ok());
    assert_eq!(adapter.unwrap().kind(), &kind);
}

#[test]
fn factory_builds_openai_adapter() {
    let kind = AdapterKind::OpenAI {
        api_key_env: "OPENAI_API_KEY".into(),
        model: "gpt-4o".into(),
    };
    let adapter = AdapterFactory::build(&kind);
    assert!(adapter.is_ok());
    assert_eq!(adapter.unwrap().kind(), &kind);
}

#[test]
fn factory_builds_anthropic_adapter() {
    let kind = AdapterKind::Anthropic {
        api_key_env: "ANTHROPIC_API_KEY".into(),
        model: "claude-3-5-sonnet-20241022".into(),
    };
    let adapter = AdapterFactory::build(&kind);
    assert!(adapter.is_ok());
    assert_eq!(adapter.unwrap().kind(), &kind);
}

#[test]
fn factory_builds_ollama_adapter() {
    let kind = AdapterKind::Ollama {
        endpoint: "http://localhost:11434".into(),
        model: "llama3.2".into(),
    };
    let adapter = AdapterFactory::build(&kind);
    assert!(adapter.is_ok());
    assert_eq!(adapter.unwrap().kind(), &kind);
}

#[test]
fn factory_returns_error_for_local_llamacpp() {
    let kind = AdapterKind::LocalLlamaCpp {
        model_path: std::path::PathBuf::from("/models/llama.gguf"),
        n_threads: 4,
    };
    let result = AdapterFactory::build(&kind);
    assert!(result.is_err(), "LocalLlamaCpp FFI is not yet wired");
    let err = result.unwrap_err();
    assert!(
        err.contains("LocalLlamaCpp"),
        "error should mention LocalLlamaCpp: {err}"
    );
}
