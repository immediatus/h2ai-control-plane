use crate::error::ToolError;
use async_trait::async_trait;
use std::collections::HashSet;
use std::time::Duration;
use tokio::process::Command;
use tokio::time::timeout;

const MAX_OUTPUT_BYTES: usize = 1024 * 1024; // 1 MiB

pub struct ShellExecutor {
    timeout_secs: u64,
    /// Empty = unrestricted (backward-compat sentinel). Non-empty = allowlist enforced.
    allowlist: HashSet<String>,
}

impl Default for ShellExecutor {
    fn default() -> Self {
        Self::new(vec![], 5)
    }
}

impl ShellExecutor {
    pub fn new(allowlist: Vec<String>, timeout_secs: u64) -> Self {
        Self {
            timeout_secs,
            allowlist: allowlist.into_iter().collect(),
        }
    }

    /// Guard 1 – JSON parse.
    /// Guard 2 – path traversal (rejects any `command` containing `/` or `\`).
    /// Guard 3 – allowlist O(1) lookup.
    /// Then: Command::new(command).args(args) — no shell interpreter.
    pub async fn execute_structured(
        &self,
        command: &str,
        args: &[String],
    ) -> Result<String, ToolError> {
        // Guard 2: path traversal
        if command.contains('/') || command.contains('\\') {
            return Err(ToolError::NotPermitted(command.to_owned()));
        }

        // Guard 3: allowlist
        if !self.allowlist.is_empty() && !self.allowlist.contains(command) {
            return Err(ToolError::NotPermitted(command.to_owned()));
        }

        self.spawn_and_wait(command, args).await
    }

    #[cfg(unix)]
    async fn spawn_and_wait(&self, command: &str, args: &[String]) -> Result<String, ToolError> {
        let child = Command::new(command)
            .args(args)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .process_group(0) // child becomes PGID leader; all descendants inherit PGID
            .kill_on_drop(true)
            .spawn()
            .map_err(ToolError::Io)?;

        // Capture PGID before await — child.id() returns None after process completes.
        let pgid = child.id().map(|pid| pid as i32);

        match timeout(
            Duration::from_secs(self.timeout_secs),
            child.wait_with_output(),
        )
        .await
        {
            Ok(Ok(output)) => {
                if output.status.success() {
                    let mut bytes = output.stdout;
                    bytes.truncate(MAX_OUTPUT_BYTES);
                    Ok(String::from_utf8_lossy(&bytes).into_owned())
                } else {
                    let mut bytes = output.stderr;
                    bytes.truncate(MAX_OUTPUT_BYTES);
                    Err(ToolError::ShellFailed {
                        exit_code: output.status.code().unwrap_or(-1),
                        stderr: String::from_utf8_lossy(&bytes).into_owned(),
                    })
                }
            }
            Ok(Err(e)) => Err(ToolError::Io(e)),
            Err(_elapsed) => {
                // Kill entire process group — catches all descendants, not just the child.
                if let Some(pgid) = pgid {
                    // SAFETY: pgid is a valid positive i32 from child.id().
                    let rc = unsafe { libc::kill(-pgid, libc::SIGKILL) };
                    if rc != 0 {
                        let errno = unsafe { *libc::__errno_location() };
                        // ESRCH (3) = group already dead; anything else is unexpected.
                        if errno != libc::ESRCH {
                            tracing::warn!(pgid, errno, "kill(-pgid, SIGKILL) failed unexpectedly");
                        }
                    }
                }
                Err(ToolError::Timeout)
            }
        }
    }

    #[cfg(not(unix))]
    async fn spawn_and_wait(&self, command: &str, args: &[String]) -> Result<String, ToolError> {
        let child = Command::new(command)
            .args(args)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(ToolError::Io)?;

        match timeout(
            Duration::from_secs(self.timeout_secs),
            child.wait_with_output(),
        )
        .await
        {
            Ok(Ok(output)) => {
                if output.status.success() {
                    let mut bytes = output.stdout;
                    bytes.truncate(MAX_OUTPUT_BYTES);
                    Ok(String::from_utf8_lossy(&bytes).into_owned())
                } else {
                    let mut bytes = output.stderr;
                    bytes.truncate(MAX_OUTPUT_BYTES);
                    Err(ToolError::ShellFailed {
                        exit_code: output.status.code().unwrap_or(-1),
                        stderr: String::from_utf8_lossy(&bytes).into_owned(),
                    })
                }
            }
            Ok(Err(e)) => Err(ToolError::Io(e)),
            Err(_elapsed) => {
                tracing::warn!(
                    "process group termination unavailable on non-Unix platforms; \
                     orphaned subprocesses may survive timeout"
                );
                Err(ToolError::Timeout)
            }
        }
    }
}

#[async_trait]
impl crate::ToolExecutor for ShellExecutor {
    fn schema(&self) -> crate::ToolSchema {
        crate::ToolSchema {
            name: "shell",
            description: "Execute a binary directly (no shell interpreter). \
                          Returns stdout on success, stderr on failure.",
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Binary name only — no path separators, no shell syntax"
                    },
                    "args": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Arguments passed as literals to the binary. No shell expansion occurs."
                    }
                },
                "required": ["command"]
            }),
        }
    }

    /// Expects JSON input: `{"command": "git", "args": ["log", "--oneline"]}`.
    /// Guard 1: JSON parse → MalformedInput.
    /// Guard 2: path traversal in `command` → NotPermitted.
    /// Guard 3: allowlist lookup → NotPermitted.
    async fn execute(&self, input: &str) -> Result<String, ToolError> {
        // Guard 1: JSON parse
        let v: serde_json::Value =
            serde_json::from_str(input).map_err(|e| ToolError::MalformedInput(e.to_string()))?;

        let command = v["command"]
            .as_str()
            .ok_or_else(|| ToolError::MalformedInput("missing 'command' field".into()))?;

        let args: Vec<String> = match v["args"].as_array() {
            None => vec![],
            Some(arr) => arr
                .iter()
                .enumerate()
                .map(|(i, a)| {
                    a.as_str().map(String::from).ok_or_else(|| {
                        ToolError::MalformedInput(format!("args[{i}] is not a string"))
                    })
                })
                .collect::<Result<_, _>>()?,
        };

        self.execute_structured(command, &args).await
    }
}
