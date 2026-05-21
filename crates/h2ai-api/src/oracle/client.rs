use h2ai_types::identity::TaskId;
use h2ai_types::sizing::{OracleDomain, OracleSpec};
use serde::Serialize;
use std::time::Duration;

#[derive(Debug, Serialize)]
struct OracleRequest<'a> {
    task_id: &'a str,
    output: &'a str,
    domain: &'a OracleDomain,
}

/// Response from the external oracle service.
#[derive(Debug, Clone)]
pub struct OracleResponse {
    pub passed: bool,
    pub score: f64,
    pub details: serde_json::Value,
}

impl OracleResponse {
    fn error(reason: &str) -> Self {
        Self {
            passed: false,
            score: 0.0,
            details: serde_json::json!({ "error": reason }),
        }
    }
}

/// HTTP client for the external oracle service.
///
/// Sends `POST runner_uri` with `{ task_id, output, domain }`.
/// Returns `OracleResponse { passed, score, details }`.
/// Never panics — all errors produce `passed=false, score=0.0`.
pub struct OracleClient {
    http: reqwest::Client,
}

impl OracleClient {
    #[must_use]
    pub fn new() -> Self {
        Self {
            http: reqwest::Client::new(),
        }
    }

    pub async fn evaluate(
        &self,
        spec: &OracleSpec,
        task_id: &TaskId,
        output: &str,
    ) -> OracleResponse {
        if spec.runner_uri.is_empty() {
            return OracleResponse::error("runner_uri is empty");
        }

        let task_id_str = task_id.to_string();
        let body = OracleRequest {
            task_id: &task_id_str,
            output,
            domain: &spec.domain,
        };

        let timeout = Duration::from_millis(spec.timeout_ms);
        let result =
            tokio::time::timeout(timeout, self.http.post(&spec.runner_uri).json(&body).send())
                .await;

        match result {
            Err(_elapsed) => {
                tracing::warn!(
                    runner = %spec.runner_uri,
                    timeout_ms = spec.timeout_ms,
                    "oracle: HTTP call timed out"
                );
                OracleResponse::error("timeout")
            }
            Ok(Err(e)) => {
                tracing::warn!(error = %e, runner = %spec.runner_uri, "oracle: HTTP call failed");
                OracleResponse::error(&e.to_string())
            }
            Ok(Ok(resp)) => {
                if !resp.status().is_success() {
                    let status = resp.status().as_u16();
                    tracing::warn!(status, runner = %spec.runner_uri, "oracle: non-2xx response");
                    return OracleResponse::error(&format!("HTTP {status}"));
                }
                match resp.json::<serde_json::Value>().await {
                    Err(e) => {
                        tracing::warn!(error = %e, "oracle: failed to parse response JSON");
                        OracleResponse::error(&e.to_string())
                    }
                    Ok(json) => {
                        let passed = json
                            .get("passed")
                            .and_then(serde_json::Value::as_bool)
                            .unwrap_or(false);
                        let score = json
                            .get("score")
                            .and_then(serde_json::Value::as_f64)
                            .unwrap_or(0.0);
                        let details = json
                            .get("details")
                            .cloned()
                            .unwrap_or(serde_json::Value::Null);
                        OracleResponse {
                            passed,
                            score,
                            details,
                        }
                    }
                }
            }
        }
    }
}

impl Default for OracleClient {
    fn default() -> Self {
        Self::new()
    }
}
