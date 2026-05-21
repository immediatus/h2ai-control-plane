//! Execution perimeter tests — proves the injection surface is closed and
//! `WaveMode` construction contracts are enforced.

use h2ai_config::H2AIConfig;
use h2ai_tools::error::ToolError;
use h2ai_tools::registry::ToolRegistry;
use h2ai_tools::shell::ShellExecutor;
use h2ai_tools::ToolExecutor;
use h2ai_types::agent::{AgentTool, WaveMode};

// ── Group 1: Structured input contract ───────────────────────────────────────

#[tokio::test]
async fn valid_json_executes_correctly() {
    let exec = ShellExecutor::default();
    let out = exec
        .execute(r#"{"command":"echo","args":["hello"]}"#)
        .await
        .unwrap();
    assert!(out.contains("hello"));
}

#[tokio::test]
async fn raw_shell_string_is_malformed_input() {
    let exec = ShellExecutor::default();
    let err = exec.execute("echo hello").await.unwrap_err();
    assert!(
        matches!(err, ToolError::MalformedInput(_)),
        "raw string must fail with MalformedInput; got: {err:?}"
    );
}

#[tokio::test]
async fn injection_payload_passed_as_literal_arg() {
    // The malicious payload must be inert — echo receives it as a literal string.
    let exec = ShellExecutor::default();
    let out = exec
        .execute(r#"{"command":"echo","args":["hello; rm -rf /"]}"#)
        .await
        .unwrap();
    assert!(
        out.contains("hello; rm -rf /"),
        "injection payload must be a literal arg; got: {out:?}"
    );
}

#[tokio::test]
async fn path_traversal_command_is_not_permitted() {
    let exec = ShellExecutor::default();
    let err = exec
        .execute(r#"{"command":"/usr/bin/rm","args":["-rf","/"]}"#)
        .await
        .unwrap_err();
    assert!(
        matches!(err, ToolError::NotPermitted(_)),
        "absolute path must be NotPermitted; got: {err:?}"
    );
}

#[tokio::test]
async fn backslash_path_traversal_is_not_permitted() {
    let exec = ShellExecutor::default();
    let err = exec
        .execute(r#"{"command":"..\\..\\windows\\system32\\cmd","args":[]}"#)
        .await
        .unwrap_err();
    assert!(
        matches!(err, ToolError::NotPermitted(_)),
        "backslash path must be NotPermitted; got: {err:?}"
    );
}

// ── Group 2: Allowlist enforcement ───────────────────────────────────────────

#[tokio::test]
async fn permitted_command_succeeds() {
    let exec = ShellExecutor::new(vec!["echo".into(), "ls".into()], 5);
    let out = exec
        .execute(r#"{"command":"echo","args":["allowed"]}"#)
        .await
        .unwrap();
    assert!(out.contains("allowed"));
}

#[tokio::test]
async fn blocked_command_returns_not_permitted() {
    let exec = ShellExecutor::new(vec!["git".into(), "ls".into()], 5);
    let err = exec
        .execute(r#"{"command":"rm","args":["-rf","/"]}"#)
        .await
        .unwrap_err();
    assert!(
        matches!(err, ToolError::NotPermitted(_)),
        "rm not in allowlist must be NotPermitted; got: {err:?}"
    );
}

#[tokio::test]
async fn empty_allowlist_permits_all_commands() {
    let exec = ShellExecutor::new(vec![], 5);
    let out = exec
        .execute(r#"{"command":"echo","args":["open"]}"#)
        .await
        .unwrap();
    assert!(out.contains("open"));
}

// ── Group 3: WaveMode construction contracts ─────────────────────────────────

fn test_cfg(normal: &[&str], hardened: &[&str]) -> H2AIConfig {
    H2AIConfig {
        shell_allowlist: normal
            .iter()
            .map(std::string::ToString::to_string)
            .collect(),
        shell_hardened_allowlist: hardened
            .iter()
            .map(std::string::ToString::to_string)
            .collect(),
        ..Default::default()
    }
}

#[tokio::test]
async fn normal_mode_uses_full_allowlist() {
    let cfg = test_cfg(&["echo", "git"], &["echo"]);
    let registry = ToolRegistry::for_wave(&cfg, WaveMode::Normal);
    let out = registry
        .execute(
            AgentTool::Shell,
            r#"{"command":"git","args":["--version"]}"#,
        )
        .await
        .unwrap();
    assert!(
        out.contains("git"),
        "git must be available in Normal mode; got: {out:?}"
    );
}

#[tokio::test]
async fn hardened_mode_uses_restricted_allowlist() {
    let cfg = test_cfg(&["echo", "git"], &["echo"]);
    let registry = ToolRegistry::for_wave(&cfg, WaveMode::Hardened);
    let err = registry
        .execute(
            AgentTool::Shell,
            r#"{"command":"git","args":["--version"]}"#,
        )
        .await
        .unwrap_err();
    assert!(
        matches!(err, ToolError::NotPermitted(_)),
        "git must be blocked in Hardened mode; got: {err:?}"
    );
}

#[tokio::test]
async fn command_in_both_modes_succeeds_in_both() {
    let cfg = test_cfg(&["echo", "git"], &["echo"]);
    let normal_registry = ToolRegistry::for_wave(&cfg, WaveMode::Normal);
    let hardened_registry = ToolRegistry::for_wave(&cfg, WaveMode::Hardened);

    let normal_out = normal_registry
        .execute(AgentTool::Shell, r#"{"command":"echo","args":["shared"]}"#)
        .await
        .unwrap();
    let hardened_out = hardened_registry
        .execute(AgentTool::Shell, r#"{"command":"echo","args":["shared"]}"#)
        .await
        .unwrap();

    assert!(normal_out.contains("shared"));
    assert!(hardened_out.contains("shared"));
}

// ── Group 4: Process group reaper (Unix only) ─────────────────────────────────

#[cfg(unix)]
#[tokio::test]
async fn timeout_kills_entire_process_group() {
    let exec = ShellExecutor::new(vec![], 1); // 1-second timeout

    // sh backgrounds a sleep with a unique duration (99991 s), then waits.
    // The unusual duration serves as a process-unique sentinel for pgrep.
    // Without PGID kill, the backgrounded sleep would survive the timeout as an orphan.
    let sentinel = "99991"; // duration unique enough to avoid pgrep false positives
    let input = format!(r#"{{"command":"sh","args":["-c","sleep {sentinel} & wait"]}}"#);
    let err = exec.execute(&input).await.unwrap_err();

    assert!(
        matches!(err, ToolError::Timeout),
        "expected Timeout; got: {err:?}"
    );

    // Give the OS a moment to reap.
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    // Verify no sentinel process survives.
    let ps_out = std::process::Command::new("pgrep")
        .args(["-f", &format!("sleep {sentinel}")])
        .output()
        .unwrap();
    assert!(
        ps_out.stdout.is_empty(),
        "orphaned sleep process must not survive PGID kill; pgrep output: {:?}",
        String::from_utf8_lossy(&ps_out.stdout)
    );
}
