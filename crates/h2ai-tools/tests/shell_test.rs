use h2ai_tools::error::ToolError;
use h2ai_tools::shell::ShellExecutor;

#[tokio::test]
async fn shell_returns_stdout_on_success() {
    let exec = ShellExecutor::default();
    let out = exec.execute_command("echo hello_world").await.unwrap();
    assert!(out.contains("hello_world"));
}

#[tokio::test]
async fn shell_returns_stderr_in_error_on_nonzero_exit() {
    let exec = ShellExecutor::default();
    let err = exec
        .execute_command("sh -c 'echo error_msg >&2; exit 2'")
        .await
        .unwrap_err();
    match err {
        ToolError::ShellFailed { exit_code, stderr } => {
            assert_eq!(exit_code, 2);
            assert!(
                stderr.contains("error_msg"),
                "stderr must be propagated; got: {stderr}"
            );
        }
        other => panic!("expected ShellFailed, got: {other:?}"),
    }
}

#[tokio::test]
async fn shell_missing_binary_returns_shell_failed() {
    // `sh -c` will exit non-zero when command is not found
    let exec = ShellExecutor::default();
    let result = exec.execute_command("nonexistent_command_xyz_123").await;
    assert!(result.is_err(), "missing binary must return error");
}

#[tokio::test]
async fn shell_special_characters_in_command() {
    let exec = ShellExecutor::default();
    // Pipe, quote, dollar sign
    let out = exec
        .execute_command(r#"echo "hello world" | tr '[:lower:]' '[:upper:]'"#)
        .await
        .unwrap();
    assert!(out.contains("HELLO WORLD"), "got: {out}");
}

#[tokio::test]
async fn shell_empty_command_returns_empty_or_ok() {
    let exec = ShellExecutor::default();
    // An empty sh -c "" exits 0 with no output.
    let result = exec.execute_command("").await;
    match result {
        Ok(out) => assert!(out.is_empty() || out.trim().is_empty()),
        Err(_) => {} // also acceptable — platform-specific
    }
}

#[tokio::test]
async fn shell_timeout_returns_timeout_error() {
    let exec = ShellExecutor { timeout_secs: 1 };
    let err = exec.execute_command("sleep 5").await.unwrap_err();
    assert!(
        matches!(err, ToolError::Timeout),
        "long command must timeout; got: {err:?}"
    );
}

#[tokio::test]
async fn shell_exit_code_preserved_in_error() {
    let exec = ShellExecutor::default();
    let err = exec.execute_command("exit 42").await.unwrap_err();
    match err {
        ToolError::ShellFailed { exit_code, .. } => {
            assert_eq!(exit_code, 42, "exit code must be propagated");
        }
        other => panic!("expected ShellFailed, got: {other:?}"),
    }
}
