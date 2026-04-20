use h2ai_types::config::OutputSchemaConfig;

#[derive(Debug, Clone, PartialEq)]
pub enum SchemaValidationResult {
    Valid,
    Invalid(String),
    Skipped,
}

impl SchemaValidationResult {
    /// Returns the error message if the result is `Invalid`, otherwise `None`.
    pub fn as_invalid_msg(&self) -> Option<&str> {
        match self {
            Self::Invalid(msg) => Some(msg.as_str()),
            _ => None,
        }
    }
}

pub fn validate_output(
    output: &str,
    config: Option<&OutputSchemaConfig>,
) -> SchemaValidationResult {
    let cfg = match config {
        Some(c) => c,
        None => return SchemaValidationResult::Skipped,
    };

    let schema_value: serde_json::Value = match serde_json::from_str(&cfg.schema_json) {
        Ok(v) => v,
        Err(e) => return SchemaValidationResult::Invalid(format!("invalid schema JSON: {e}")),
    };

    let output_value: serde_json::Value = match serde_json::from_str(output) {
        Ok(v) => v,
        Err(_) => {
            return SchemaValidationResult::Invalid(
                "output is not valid JSON — schema validation requires JSON output".into(),
            )
        }
    };

    let validator = match jsonschema::validator_for(&schema_value) {
        Ok(v) => v,
        Err(e) => return SchemaValidationResult::Invalid(format!("schema compile error: {e}")),
    };

    if validator.is_valid(&output_value) {
        SchemaValidationResult::Valid
    } else {
        let errors: Vec<String> = validator
            .iter_errors(&output_value)
            .map(|e| e.to_string())
            .collect();
        SchemaValidationResult::Invalid(errors.join("; "))
    }
}

/// Convert a schema validation result into an error message for use in the TAO loop.
pub fn schema_error_to_engine(result: &SchemaValidationResult) -> Option<String> {
    match result {
        SchemaValidationResult::Invalid(msg) => Some(msg.clone()),
        _ => None,
    }
}
