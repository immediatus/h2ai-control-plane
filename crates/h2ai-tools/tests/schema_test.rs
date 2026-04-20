use h2ai_tools::registry::ToolRegistry;
use h2ai_tools::shell::ShellExecutor;
use h2ai_tools::ToolExecutor;

#[test]
fn shell_schema_has_correct_name() {
    let exec = ShellExecutor::default();
    let schema = exec.schema();
    assert_eq!(schema.name, "shell");
}

#[test]
fn shell_schema_parameters_is_object() {
    let exec = ShellExecutor::default();
    let schema = exec.schema();
    assert_eq!(schema.parameters["type"], "object");
    assert!(schema.parameters["properties"]["command"].is_object());
    let required = schema.parameters["required"].as_array().unwrap();
    assert!(required.iter().any(|v| v == "command"));
}

#[test]
fn registry_all_schemas_includes_shell() {
    let registry = ToolRegistry::default_with_shell();
    let schemas = registry.all_schemas();
    assert_eq!(schemas.len(), 1);
    // Use find rather than index: HashMap iteration order is undefined.
    assert!(schemas.iter().any(|s| s.name == "shell"));
}

#[test]
fn registry_all_schemas_empty_when_no_tools() {
    let registry = ToolRegistry::new();
    assert!(registry.all_schemas().is_empty());
}
