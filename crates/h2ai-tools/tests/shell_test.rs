use h2ai_tools::error::ToolError;
use h2ai_tools::shell::ShellExecutor;
use h2ai_tools::ToolExecutor;

// ── basic success ────────────────────────────────────────────────────────────

#[tokio::test]
async fn shell_returns_stdout_on_success() {
    let exec = ShellExecutor::default();
    let out = exec
        .execute_structured("echo", &["hello_world".to_owned()])
        .await
        .unwrap();
    assert!(out.contains("hello_world"), "got: {out}");
}

// ── JSON execute() surface ───────────────────────────────────────────────────

#[tokio::test]
async fn execute_json_parses_command_and_args() {
    let exec = ShellExecutor::default();
    let out = exec
        .execute(r#"{"command": "echo", "args": ["hi_from_json"]}"#)
        .await
        .unwrap();
    assert!(out.contains("hi_from_json"), "got: {out}");
}

#[tokio::test]
async fn execute_json_args_optional() {
    let exec = ShellExecutor::default();
    // `args` field absent — should default to empty vec
    let out = exec.execute(r#"{"command": "true"}"#).await.unwrap();
    // `true` exits 0 with no output
    assert!(out.is_empty() || out.trim().is_empty(), "got: {out}");
}

#[tokio::test]
async fn execute_malformed_json_returns_malformed_input() {
    let exec = ShellExecutor::default();
    let err = exec.execute("not json at all").await.unwrap_err();
    assert!(
        matches!(err, ToolError::MalformedInput(_)),
        "expected MalformedInput, got: {err:?}"
    );
}

#[tokio::test]
async fn execute_missing_command_field_returns_malformed_input() {
    let exec = ShellExecutor::default();
    let err = exec.execute(r#"{"args": ["foo"]}"#).await.unwrap_err();
    assert!(
        matches!(err, ToolError::MalformedInput(_)),
        "expected MalformedInput, got: {err:?}"
    );
}

// ── path traversal guard ─────────────────────────────────────────────────────

#[tokio::test]
async fn path_traversal_with_slash_returns_not_permitted() {
    let exec = ShellExecutor::default();
    let err = exec
        .execute_structured("/bin/echo", &["hello".to_owned()])
        .await
        .unwrap_err();
    assert!(
        matches!(err, ToolError::NotPermitted(_)),
        "expected NotPermitted, got: {err:?}"
    );
}

#[tokio::test]
async fn path_traversal_with_backslash_returns_not_permitted() {
    let exec = ShellExecutor::default();
    let err = exec.execute_structured("..\\cmd", &[]).await.unwrap_err();
    assert!(
        matches!(err, ToolError::NotPermitted(_)),
        "expected NotPermitted, got: {err:?}"
    );
}

// ── allowlist guard ──────────────────────────────────────────────────────────

#[tokio::test]
async fn allowlist_permits_listed_command() {
    let exec = ShellExecutor::new(vec!["echo".to_owned()], 5);
    let out = exec
        .execute_structured("echo", &["allowed".to_owned()])
        .await
        .unwrap();
    assert!(out.contains("allowed"), "got: {out}");
}

#[tokio::test]
async fn allowlist_blocks_unlisted_command() {
    let exec = ShellExecutor::new(vec!["echo".to_owned()], 5);
    let err = exec.execute_structured("ls", &[]).await.unwrap_err();
    assert!(
        matches!(err, ToolError::NotPermitted(_)),
        "expected NotPermitted, got: {err:?}"
    );
}

#[tokio::test]
async fn empty_allowlist_is_unrestricted() {
    // No allowlist → any command is permitted (subject to other guards).
    let exec = ShellExecutor::new(vec![], 5);
    let out = exec
        .execute_structured("echo", &["unrestricted".to_owned()])
        .await
        .unwrap();
    assert!(out.contains("unrestricted"), "got: {out}");
}

// ── nonzero exit / missing binary ────────────────────────────────────────────

#[tokio::test]
async fn shell_missing_binary_returns_io_error() {
    let exec = ShellExecutor::default();
    let result = exec
        .execute_structured("nonexistent_command_xyz_123", &[])
        .await;
    assert!(result.is_err(), "missing binary must return error");
}

#[tokio::test]
async fn shell_nonzero_exit_returns_shell_failed() {
    let exec = ShellExecutor::default();
    // `false` exits with code 1 and produces no stdout/stderr.
    let err = exec.execute_structured("false", &[]).await.unwrap_err();
    assert!(
        matches!(err, ToolError::ShellFailed { .. }),
        "expected ShellFailed, got: {err:?}"
    );
}

// ── timeout ──────────────────────────────────────────────────────────────────

#[tokio::test]
async fn shell_timeout_returns_timeout_error() {
    let exec = ShellExecutor::new(vec![], 1);
    let err = exec
        .execute_structured("sleep", &["10".to_owned()])
        .await
        .unwrap_err();
    assert!(
        matches!(err, ToolError::Timeout),
        "long command must timeout; got: {err:?}"
    );
}
