use h2ai_adapters::factory::AdapterFactory;
use h2ai_config::AdapterProfile;
use h2ai_types::config::AdapterKind;

#[test]
fn build_from_profiles_finds_named_profile() {
    let profiles = vec![AdapterProfile {
        name: "my-mock".into(),
        kind: AdapterKind::Ollama {
            endpoint: "http://localhost:11434".into(),
            model: "llama3".into(),
        },
    }];
    // Just check it resolves without error — we can't call execute() without a server.
    let result = AdapterFactory::build_from_profiles("my-mock", &profiles);
    assert!(result.is_ok());
}

#[test]
fn build_from_profiles_errors_on_missing_name() {
    let profiles: Vec<AdapterProfile> = vec![];
    let result = AdapterFactory::build_from_profiles("nonexistent", &profiles);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("nonexistent"));
}

#[test]
fn build_from_profiles_errors_on_llama_cpp() {
    use std::path::PathBuf;
    let profiles = vec![AdapterProfile {
        name: "local".into(),
        kind: AdapterKind::LocalLlamaCpp {
            model_path: PathBuf::from("/models/llama.gguf"),
            n_threads: 8,
        },
    }];
    let result = AdapterFactory::build_from_profiles("local", &profiles);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.contains("LocalLlamaCpp") || err.contains("Ollama"));
}
