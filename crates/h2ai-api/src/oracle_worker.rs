use futures::StreamExt;
use h2ai_types::events::{OraclePendingEvent, OracleResultEvent};

pub struct OracleWorker {
    pub nats_raw: async_nats::Client,
}

impl OracleWorker {
    pub fn new(nats_raw: async_nats::Client) -> Self {
        Self { nats_raw }
    }

    pub async fn run(self) {
        let allowlist = vec![
            "cargo".to_string(),
            "python".to_string(),
            "pytest".to_string(),
        ];

        let mut sub = match self.nats_raw.subscribe("h2ai.oracle.pending").await {
            Ok(s) => s,
            Err(e) => {
                tracing::error!(error = %e, "OracleWorker: failed to subscribe");
                return;
            }
        };
        tracing::info!("OracleWorker: subscribed to h2ai.oracle.pending");

        while let Some(msg) = sub.next().await {
            if let Ok(ev) = serde_json::from_slice::<OraclePendingEvent>(&msg.payload) {
                // Build executor per-message to honour oracle_spec.timeout_ms.
                let timeout_secs = ev.oracle_spec.timeout_ms.div_ceil(1000).max(1);
                let executor =
                    h2ai_tools::shell::ShellExecutor::new(allowlist.clone(), timeout_secs);
                let result = Self::execute_oracle(&executor, &ev).await;
                match serde_json::to_vec(&result) {
                    Ok(payload) => {
                        let _ = self
                            .nats_raw
                            .publish("h2ai.oracle.results", payload.into())
                            .await;
                        tracing::debug!(
                            task_id = %ev.task_id,
                            passed = result.passed,
                            score = result.score,
                            "oracle result published"
                        );
                    }
                    Err(e) => {
                        tracing::error!(error = %e, task_id = %ev.task_id, "OracleWorker: failed to serialize result");
                    }
                }
            } else {
                tracing::warn!("OracleWorker: failed to parse OraclePendingEvent");
            }
        }
    }

    async fn execute_oracle(
        executor: &h2ai_tools::shell::ShellExecutor,
        ev: &OraclePendingEvent,
    ) -> OracleResultEvent {
        let start = std::time::Instant::now();
        let timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let (passed, score) =
            match build_test_command(&ev.oracle_spec.language, &ev.oracle_spec.test_suite) {
                None => {
                    tracing::warn!(
                        language = %ev.oracle_spec.language,
                        "OracleWorker: unsupported language"
                    );
                    (false, 0.0)
                }
                Some((cmd, args)) => match executor.execute_structured(&cmd, &args).await {
                    Ok(stdout) => {
                        let (pass_n, total_n) = parse_test_counts(&stdout);
                        let passed = total_n > 0 && pass_n == total_n;
                        // Zero total means output was unparseable; treat as failure, not perfect score.
                        let score = if total_n > 0 {
                            pass_n as f64 / total_n as f64
                        } else {
                            0.0
                        };
                        (passed, score)
                    }
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            task_id = %ev.task_id,
                            "oracle execution failed"
                        );
                        (false, 0.0)
                    }
                },
            };

        let passed_f64 = if passed { 1.0_f64 } else { 0.0_f64 };
        OracleResultEvent {
            task_id: ev.task_id.clone(),
            q_confidence: ev.q_confidence,
            n_used: ev.n_used,
            passed,
            score,
            residual: (ev.q_confidence - passed_f64).abs(),
            domain: ev.domain.clone(),
            oracle_type: ev.oracle_spec.oracle_type.clone(),
            duration_ms: start.elapsed().as_millis() as u64,
            timestamp_ms,
        }
    }
}

/// Parse `(passed, total)` from test runner stdout.
///
/// Scans lines from the end, looking for word pairs where the second word is
/// `"passed"` or `"failed"`. Returns `(0, 0)` when no match is found.
pub fn parse_test_counts(stdout: &str) -> (u64, u64) {
    for line in stdout.lines().rev() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let words: Vec<&str> = line.split_whitespace().collect();
        let mut pass_n: Option<u64> = None;
        let mut fail_n: Option<u64> = None;

        for w in words.windows(2) {
            let keyword = w[1].trim_end_matches(|c: char| !c.is_alphabetic());
            if keyword == "passed" {
                pass_n = w[0].parse().ok();
            }
            if keyword == "failed" {
                fail_n = w[0].parse().ok();
            }
        }

        if pass_n.is_some() || fail_n.is_some() {
            let p = pass_n.unwrap_or(0);
            let f = fail_n.unwrap_or(0);
            return (p, p + f);
        }
    }

    (0, 0)
}

/// Build the test command for the given language and test suite.
///
/// Returns `(command, args)` or `None` for unsupported languages.
pub fn build_test_command(language: &str, test_suite: &str) -> Option<(String, Vec<String>)> {
    match language {
        "rust" => Some((
            "cargo".to_string(),
            vec![
                "nextest".to_string(),
                "run".to_string(),
                test_suite.to_string(),
            ],
        )),
        "python" => Some((
            "python".to_string(),
            vec![
                "-m".to_string(),
                "pytest".to_string(),
                test_suite.to_string(),
                "-v".to_string(),
            ],
        )),
        "pytest" => Some((
            "pytest".to_string(),
            vec![test_suite.to_string(), "-v".to_string()],
        )),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_counts_cargo_nextest() {
        let out = "Summary [2.3s] 42 tests run: 40 passed, 2 failed";
        let (p, t) = parse_test_counts(out);
        assert_eq!(p, 40);
        assert_eq!(t, 42);
    }

    #[test]
    fn parse_counts_pytest() {
        let out = "===== 35 passed, 5 failed in 3.2s =====";
        let (p, t) = parse_test_counts(out);
        assert_eq!(p, 35);
        assert_eq!(t, 40);
    }

    #[test]
    fn parse_counts_all_passed() {
        let out = "42 passed in 1.5s";
        let (p, t) = parse_test_counts(out);
        assert_eq!(p, 42);
        assert_eq!(t, 42);
    }

    #[test]
    fn parse_counts_unparseable() {
        let (p, t) = parse_test_counts("no test output");
        assert_eq!(p, 0);
        assert_eq!(t, 0);
    }

    #[test]
    fn build_command_rust() {
        let (cmd, args) = build_test_command("rust", "my::tests").unwrap();
        assert_eq!(cmd, "cargo");
        assert!(args.contains(&"nextest".to_string()));
        assert!(args.contains(&"my::tests".to_string()));
    }

    #[test]
    fn build_command_python() {
        let (cmd, args) = build_test_command("python", "tests/").unwrap();
        assert_eq!(cmd, "python");
        assert!(args.contains(&"-m".to_string()));
    }

    #[test]
    fn build_command_unknown_returns_none() {
        assert!(build_test_command("java", "test/").is_none());
    }

    #[test]
    fn parse_counts_zero_total_returns_zero_zero() {
        // Unparseable stdout must return (0,0), not (0, undefined).
        // execute_oracle maps total_n==0 to score=0.0 and passed=false.
        let (p, t) = parse_test_counts("error: could not find tests");
        assert_eq!(p, 0);
        assert_eq!(t, 0);
    }
}
