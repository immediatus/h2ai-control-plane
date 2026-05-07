//! H2AI Oracle Sidecar
//!
//! Subscribes to `h2ai.oracle.pending`, runs the specified test suite against
//! the winning merged output via the ShellExecutor perimeter, and publishes
//! `OracleResultEvent` to `h2ai.oracle.results`.
//!
//! Environment variables:
//!   NATS_URL     — NATS server URL (default: nats://localhost:4222)
//!   ORACLE_TIMEOUT_SECS — override per-spec timeout (optional)

use futures::StreamExt;
use h2ai_eval::parse_result;
use h2ai_tools::shell::ShellExecutor;
use h2ai_types::events::{OraclePendingEvent, OracleResultEvent};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{info, warn};

const PENDING_SUBJECT: &str = "h2ai.oracle.pending";
const RESULTS_SUBJECT: &str = "h2ai.oracle.results";

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("h2ai_eval=info".parse().unwrap()),
        )
        .init();

    let nats_url = std::env::var("NATS_URL").unwrap_or_else(|_| "nats://localhost:4222".to_owned());

    info!("connecting to NATS at {nats_url}");
    let nc = async_nats::connect(&nats_url)
        .await
        .expect("NATS connect failed");

    let mut sub = nc
        .subscribe(PENDING_SUBJECT)
        .await
        .expect("subscribe failed");

    info!("oracle sidecar ready — listening on {PENDING_SUBJECT}");

    while let Some(msg) = sub.next().await {
        let nc = nc.clone();
        tokio::spawn(async move {
            let pending: OraclePendingEvent = match serde_json::from_slice(&msg.payload) {
                Ok(p) => p,
                Err(e) => {
                    warn!(error = %e, "failed to parse OraclePendingEvent — dropping");
                    return;
                }
            };

            let task_id = pending.task_id.clone();
            info!(task_id = %task_id, "running oracle for task");

            let result = evaluate(&pending).await;

            match serde_json::to_vec(&result) {
                Ok(payload) => {
                    if let Err(e) = nc.publish(RESULTS_SUBJECT, payload.into()).await {
                        warn!(task_id = %task_id, error = %e, "failed to publish OracleResultEvent");
                    } else {
                        info!(
                            task_id = %task_id,
                            passed = result.passed,
                            score = result.score,
                            residual = result.residual,
                            duration_ms = result.duration_ms,
                            "oracle result published"
                        );
                    }
                }
                Err(e) => {
                    warn!(task_id = %task_id, error = %e, "failed to serialize OracleResultEvent");
                }
            }
        });
    }
}

/// Run the oracle test suite against the winning output.
///
/// Writes the output to a temp file, invokes the test runner via ShellExecutor
/// (which enforces process-group timeout and no shell interpreter), then parses
/// pass/fail counts from the combined stdout+stderr.
async fn evaluate(pending: &OraclePendingEvent) -> OracleResultEvent {
    let spec = &pending.oracle_spec;
    let timeout_secs = (spec.timeout_ms / 1000).max(1);

    let start_ms = now_ms();

    // Write winning output to a temp file
    let (output_file, output_path) =
        match write_temp_output(&pending.winning_output, &spec.language) {
            Ok(p) => p,
            Err(e) => {
                warn!(error = %e, "failed to create temp file for oracle output");
                return failure_result(pending, start_ms);
            }
        };

    // Build executor — open allowlist since sidecar only runs operator-defined test suites
    let executor = ShellExecutor::new(vec![], timeout_secs);

    // Build command and args based on language
    let (command, args) = build_command(&spec.language, &spec.test_suite, &output_path);

    let result = executor.execute_structured(&command, &args).await;

    // Temp file cleanup — ignore errors (OS will GC)
    drop(output_file);

    let duration_ms = now_ms() - start_ms;
    let timestamp_ms = now_ms();

    let (passed_bool, pass_count, fail_count, total) = match result {
        Ok(stdout) => parse_result(&stdout, true),
        Err(h2ai_tools::error::ToolError::ShellFailed { stderr, .. }) => {
            parse_result(&stderr, false)
        }
        Err(h2ai_tools::error::ToolError::Timeout) => {
            warn!(task_id = %pending.task_id, "oracle timed out after {timeout_secs}s");
            (false, 0, 1, 1)
        }
        Err(e) => {
            warn!(task_id = %pending.task_id, error = %e, "oracle execution error");
            (false, 0, 1, 1)
        }
    };

    let _ = (pass_count, fail_count); // counts informational
    let score = if total > 0 {
        pass_count as f64 / total as f64
    } else if passed_bool {
        1.0
    } else {
        0.0
    };
    let residual = (pending.q_confidence - passed_bool as u8 as f64).abs();

    OracleResultEvent {
        task_id: pending.task_id.clone(),
        q_confidence: pending.q_confidence,
        n_used: pending.n_used,
        passed: passed_bool,
        score,
        residual,
        domain: pending.domain.clone(),
        oracle_type: spec.oracle_type.clone(),
        duration_ms,
        timestamp_ms,
    }
}

/// Build the (command, args) pair for the test runner.
///
/// Language dispatch:
/// - "python" → `python3 -m pytest <test_suite> --tb=short -q`
/// - "javascript" → `node <test_suite>`
/// - other → `<language> <test_suite> <output_file>`
///
/// The output file path is always appended as the last arg so test suites
/// can locate the output via positional arg or the `H2AI_OUTPUT_FILE` convention.
/// Note: ShellExecutor does not support env var injection — suites must read
/// the output path from argv[-1].
fn build_command(language: &str, test_suite: &str, output_path: &str) -> (String, Vec<String>) {
    match language {
        "python" => (
            "python3".to_owned(),
            vec![
                "-m".to_owned(),
                "pytest".to_owned(),
                test_suite.to_owned(),
                "--tb=short".to_owned(),
                "-q".to_owned(),
                format!("--output-file={output_path}"),
            ],
        ),
        "javascript" => (
            "node".to_owned(),
            vec![test_suite.to_owned(), output_path.to_owned()],
        ),
        lang => (
            lang.to_owned(),
            vec![test_suite.to_owned(), output_path.to_owned()],
        ),
    }
}

/// Write the winning output to a language-appropriate temp file.
/// Returns the `NamedTempFile` (keep alive for its lifetime) and the path string.
fn write_temp_output(
    output: &str,
    language: &str,
) -> Result<(tempfile::NamedTempFile, String), std::io::Error> {
    use std::io::Write;

    let suffix = match language {
        "python" => ".py",
        "javascript" => ".js",
        "rust" => ".rs",
        _ => ".txt",
    };

    let mut f = tempfile::Builder::new()
        .prefix("h2ai_oracle_")
        .suffix(suffix)
        .tempfile()?;
    f.write_all(output.as_bytes())?;
    f.flush()?;

    let path = f.path().to_string_lossy().into_owned();
    Ok((f, path))
}

fn failure_result(pending: &OraclePendingEvent, start_ms: u64) -> OracleResultEvent {
    OracleResultEvent {
        task_id: pending.task_id.clone(),
        q_confidence: pending.q_confidence,
        n_used: pending.n_used,
        passed: false,
        score: 0.0,
        residual: pending.q_confidence, // |q - 0.0|
        domain: pending.domain.clone(),
        oracle_type: pending.oracle_spec.oracle_type.clone(),
        duration_ms: now_ms() - start_ms,
        timestamp_ms: now_ms(),
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
