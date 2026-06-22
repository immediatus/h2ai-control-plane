use crate::error::ToolError;
use crate::{ToolExecutor, ToolSchema};
use async_trait::async_trait;
use serde_json::json;

const PERMITTED_OPS: &[&str] = &["read_file", "list_directory"];

#[async_trait]
pub trait McpBackend: Send + Sync {
    async fn call(&self, op: &str, path: &str) -> Result<String, ToolError>;
}

// ── Live: stdio subprocess (MCP JSON-RPC 2.0) ────────────────────────────────

pub struct StdioMcpBackend {
    command: String,
    args: Vec<String>,
    timeout_secs: u64,
}

impl StdioMcpBackend {
    pub fn new(command: impl Into<String>, args: Vec<String>, timeout_secs: u64) -> Self {
        Self {
            command: command.into(),
            args,
            timeout_secs,
        }
    }
}

#[async_trait]
impl McpBackend for StdioMcpBackend {
    async fn call(&self, op: &str, path: &str) -> Result<String, ToolError> {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
        use tokio::process::Command;
        use tokio::time::{timeout, Duration};

        let request = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": op,
                "arguments": { "path": path }
            }
        });
        let request_line = serde_json::to_string(&request)
            .map_err(|e| ToolError::MalformedInput(e.to_string()))?
            + "\n";

        let mut child = Command::new(&self.command)
            .args(&self.args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|e| ToolError::InitializationFailed(e.to_string()))?;

        let mut stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();

        // Capture PID before the async block consumes `child` to avoid race where
        // the process exits before we can read the ID for SIGKILL.
        #[cfg(unix)]
        let child_pid = child.id();

        let result = timeout(Duration::from_secs(self.timeout_secs), async {
            // BrokenPipe on stdin write is non-fatal: the subprocess may have
            // already written its response to stdout without reading input.
            if let Err(e) = stdin.write_all(request_line.as_bytes()).await {
                if e.kind() != std::io::ErrorKind::BrokenPipe {
                    return Err(ToolError::Io(e));
                }
            }
            drop(stdin);

            let mut reader = BufReader::new(stdout);
            let mut line = String::new();
            reader.read_line(&mut line).await.map_err(ToolError::Io)?;

            let resp: serde_json::Value = serde_json::from_str(line.trim())
                .map_err(|e| ToolError::MalformedInput(e.to_string()))?;

            if let Some(err) = resp.get("error") {
                return Err(ToolError::ExecutionFailed(err.to_string()));
            }

            let content = resp
                .pointer("/result/content/0/text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            Ok(content)
        })
        .await;

        // Kill process group on Unix regardless of outcome (same pattern as ShellExecutor).
        #[cfg(unix)]
        if let Some(pid) = child_pid {
            #[allow(clippy::cast_possible_wrap)]
            unsafe {
                libc::kill(-(pid as i32), libc::SIGKILL)
            };
        }
        #[cfg(not(unix))]
        tracing::warn!("MCP process group kill not supported on this platform; child may linger");
        let _ = child.wait().await;

        result.map_err(|_| ToolError::Timeout)?
    }
}

// ── Executor ─────────────────────────────────────────────────────────────────

pub struct McpExecutor {
    backend: Box<dyn McpBackend>,
}

impl McpExecutor {
    #[must_use]
    pub fn new(backend: Box<dyn McpBackend>) -> Self {
        Self { backend }
    }
}

#[async_trait]
impl ToolExecutor for McpExecutor {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "file_system",
            description: "Read files or list directories in the workspace (read-only).",
            parameters: json!({
                "type": "object",
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": ["read_file", "list_directory"],
                        "description": "Operation: read_file returns file contents; list_directory returns entry names."
                    },
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file or directory."
                    }
                },
                "required": ["op", "path"]
            }),
        }
    }

    async fn execute(&self, input: &str) -> Result<String, ToolError> {
        let v: serde_json::Value =
            serde_json::from_str(input).map_err(|e| ToolError::MalformedInput(e.to_string()))?;

        let op = v["op"]
            .as_str()
            .ok_or_else(|| ToolError::MalformedInput("missing 'op' field".into()))?;
        let path = v["path"]
            .as_str()
            .ok_or_else(|| ToolError::MalformedInput("missing 'path' field".into()))?;

        if !PERMITTED_OPS.contains(&op) {
            return Err(ToolError::NotPermitted(format!("op not allowed: {op}")));
        }

        self.backend.call(op, path).await
    }
}
