use h2ai_config::{AdapterProfile, H2AIConfig};
use h2ai_types::config::AdapterKind;
use std::io::Write;

#[test]
fn h2ai_config_default_has_empty_profiles() {
    let cfg = H2AIConfig::default();
    assert!(cfg.adapter_profiles.is_empty());
}

#[test]
fn config_with_profiles_round_trips_json() {
    let mut cfg = H2AIConfig::default();
    cfg.adapter_profiles = vec![AdapterProfile {
        name: "my-ollama".into(),
        kind: AdapterKind::Ollama {
            endpoint: "http://localhost:11434".into(),
            model: "llama3".into(),
        },
    }];
    let json = serde_json::to_string(&cfg).unwrap();
    let back: H2AIConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(back.adapter_profiles.len(), 1);
    assert_eq!(back.adapter_profiles[0].name, "my-ollama");
}

#[test]
fn load_from_file_round_trips_adapter_profiles() {
    let mut file = tempfile::NamedTempFile::new().unwrap();
    let cfg = H2AIConfig {
        adapter_profiles: vec![AdapterProfile {
            name: "file-ollama".into(),
            kind: AdapterKind::Ollama {
                endpoint: "http://localhost:11434".into(),
                model: "mistral".into(),
            },
        }],
        ..H2AIConfig::default()
    };
    write!(file, "{}", serde_json::to_string(&cfg).unwrap()).unwrap();
    let loaded = H2AIConfig::load_from_file(file.path()).unwrap();
    assert_eq!(loaded.adapter_profiles.len(), 1);
    assert_eq!(loaded.adapter_profiles[0].name, "file-ollama");
}
