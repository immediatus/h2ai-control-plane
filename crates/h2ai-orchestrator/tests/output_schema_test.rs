use h2ai_orchestrator::output_schema::{
    schema_error_to_engine, validate_output, SchemaValidationResult,
};
use h2ai_types::config::OutputSchemaConfig;

fn schema(json: &str) -> OutputSchemaConfig {
    OutputSchemaConfig {
        schema_json: json.to_string(),
    }
}

#[test]
fn skipped_when_no_config() {
    let result = validate_output(r#"{"key": "value"}"#, None);
    assert_eq!(result, SchemaValidationResult::Skipped);
}

#[test]
fn valid_output_passes_schema() {
    let cfg = schema(
        r#"{"type": "object", "properties": {"score": {"type": "number"}}, "required": ["score"]}"#,
    );
    let result = validate_output(r#"{"score": 0.85}"#, Some(&cfg));
    assert_eq!(result, SchemaValidationResult::Valid);
}

#[test]
fn invalid_output_fails_schema() {
    let cfg = schema(r#"{"type": "object", "required": ["score"]}"#);
    let result = validate_output(r#"{"other": "field"}"#, Some(&cfg));
    assert!(matches!(result, SchemaValidationResult::Invalid(_)));
}

#[test]
fn non_json_output_fails_schema() {
    let cfg = schema(r#"{"type": "object"}"#);
    let result = validate_output("not json at all", Some(&cfg));
    assert!(matches!(result, SchemaValidationResult::Invalid(_)));
}

#[test]
fn invalid_schema_json_returns_invalid() {
    let cfg = schema("not valid json schema");
    let result = validate_output(r#"{"x": 1}"#, Some(&cfg));
    assert!(matches!(result, SchemaValidationResult::Invalid(_)));
}

#[test]
fn as_invalid_msg_returns_none_for_valid() {
    assert!(SchemaValidationResult::Valid.as_invalid_msg().is_none());
}

#[test]
fn as_invalid_msg_returns_none_for_skipped() {
    assert!(SchemaValidationResult::Skipped.as_invalid_msg().is_none());
}

#[test]
fn schema_error_to_engine_invalid_returns_some() {
    let result = SchemaValidationResult::Invalid("bad output".to_string());
    assert_eq!(
        schema_error_to_engine(&result),
        Some("bad output".to_string())
    );
}

#[test]
fn schema_error_to_engine_valid_returns_none() {
    assert!(schema_error_to_engine(&SchemaValidationResult::Valid).is_none());
}

#[test]
fn schema_error_to_engine_skipped_returns_none() {
    assert!(schema_error_to_engine(&SchemaValidationResult::Skipped).is_none());
}
