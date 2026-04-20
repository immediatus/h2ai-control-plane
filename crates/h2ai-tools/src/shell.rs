use crate::error::ToolError;
use async_trait::async_trait;
use std::time::Duration;
use tokio::process::Command;
use tokio::time::timeout;

const MAX_OUTPUT_BYTES: usize = 1024 * 1024; // 1 MiB

pub struct ShellExecutor {
    pub timeout_secs: u64,
}

impl Default for ShellExecutor {
    fn default() -> Self {
        Self { timeout_secs: 5 }
    }
}

impl ShellExecutor {
    pub async fn execute_command(&self, command: &str) -> Result<String, ToolError> {
        // kill_on_drop(true) ensures the child process is killed when the Child
        // handle is dropped — which happens when the timeout future is cancelled.
        let child = Command::new("sh")
            .arg("-c")
            .arg(command)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(ToolError::Io)?;

        match timeout(Duration::from_secs(self.timeout_secs), child.wait_with_output()).await {
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
            Err(_) => Err(ToolError::Timeout),
        }
    }
}

#[async_trait]
impl crate::ToolExecutor for ShellExecutor {
    async fn execute(&self, input: &str) -> Result<String, ToolError> {
        self.execute_command(input).await
    }
}
