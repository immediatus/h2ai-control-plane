use crate::jaccard::{jaccard, tokenize};
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::physics::TauValue;

/// Near-zero temperature for deterministic similarity scoring via SLM.
pub const SIMILARITY_TAU: f64 = 0.05;

const SIMILARITY_MAX_TOKENS: u64 = 32;

const SYSTEM_PROMPT: &str = "You are a semantic similarity scorer. \
    Given two texts, output ONLY valid JSON with a single key \"score\" \
    containing a float in [0.0, 1.0]. 1.0 = identical meaning, 0.0 = unrelated.";

#[derive(Debug)]
#[allow(dead_code)]
pub(crate) enum SimilarityError {
    Adapter(String),
    ParseFailure(String),
    ScoreOutOfRange(f64),
}

/// Semantic similarity between two text strings in [0, 1].
///
/// - **With adapter**: dispatches a JSON scoring prompt to an SLM via
///   `IComputeAdapter`. Falls back to token Jaccard on any error (adapter
///   failure, malformed JSON, out-of-range score).
/// - **Without adapter** (`None`): token-level Jaccard — zero extra cost,
///   identical to existing behaviour; existing call-sites pay nothing until
///   they supply an adapter.
pub async fn semantic_jaccard(a: &str, b: &str, adapter: Option<&dyn IComputeAdapter>) -> f64 {
    let Some(adapter) = adapter else {
        return jaccard(&tokenize(a), &tokenize(b));
    };
    match query_similarity(a, b, adapter).await {
        Ok(score) => score,
        Err(_) => jaccard(&tokenize(a), &tokenize(b)),
    }
}

async fn query_similarity(
    a: &str,
    b: &str,
    adapter: &dyn IComputeAdapter,
) -> Result<f64, SimilarityError> {
    let task = format!(
        "Text A: {a}\nText B: {b}\n\nOutput ONLY valid JSON: {{\"score\": <float 0.0-1.0>}}"
    );
    let req = ComputeRequest {
        system_context: SYSTEM_PROMPT.to_string(),
        task,
        // SAFETY: 0.05 ∈ [0, 1] — compile-time-known constant.
        tau: TauValue::new(SIMILARITY_TAU).expect("SIMILARITY_TAU in valid range"),
        max_tokens: SIMILARITY_MAX_TOKENS,
    };
    let response = adapter
        .execute(req)
        .await
        .map_err(|e| SimilarityError::Adapter(e.to_string()))?;
    parse_score(&response.output)
}

/// Parse `{"score": <float>}` from raw SLM output.
///
/// Tolerates leading/trailing text so that a model preamble like
/// "Sure! {\"score\": 0.9}" still extracts correctly.
pub(crate) fn parse_score(raw: &str) -> Result<f64, SimilarityError> {
    let start = raw
        .find('{')
        .ok_or_else(|| SimilarityError::ParseFailure(raw.to_string()))?;
    let end = raw[start..]
        .find('}')
        .map(|i| start + i + 1)
        .ok_or_else(|| SimilarityError::ParseFailure(raw.to_string()))?;
    let v: serde_json::Value = serde_json::from_str(&raw[start..end])
        .map_err(|e| SimilarityError::ParseFailure(e.to_string()))?;
    let score = v["score"]
        .as_f64()
        .ok_or_else(|| SimilarityError::ParseFailure("missing numeric 'score' field".into()))?;
    if !(0.0..=1.0).contains(&score) {
        return Err(SimilarityError::ScoreOutOfRange(score));
    }
    Ok(score)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_score unit tests ────────────────────────────────────────────────

    #[test]
    fn parse_score_exact_json() {
        assert!((parse_score(r#"{"score": 0.85}"#).unwrap() - 0.85).abs() < 1e-9);
    }

    #[test]
    fn parse_score_integer_value() {
        assert!((parse_score(r#"{"score": 1}"#).unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn parse_score_with_preamble() {
        let raw = r#"Sure! {"score": 0.7} done."#;
        assert!((parse_score(raw).unwrap() - 0.7).abs() < 1e-9);
    }

    #[test]
    fn parse_score_no_brace_returns_err() {
        assert!(parse_score("no json here").is_err());
    }

    #[test]
    fn parse_score_missing_score_field_returns_err() {
        assert!(parse_score(r#"{"value": 0.5}"#).is_err());
    }

    #[test]
    fn parse_score_out_of_range_returns_err() {
        assert!(parse_score(r#"{"score": 1.5}"#).is_err());
        assert!(parse_score(r#"{"score": -0.1}"#).is_err());
    }

    #[test]
    fn parse_score_zero_and_one_boundary() {
        assert!((parse_score(r#"{"score": 0.0}"#).unwrap() - 0.0).abs() < 1e-9);
        assert!((parse_score(r#"{"score": 1.0}"#).unwrap() - 1.0).abs() < 1e-9);
    }

    // ── semantic_jaccard integration tests (using MockAdapter) ───────────────

    #[tokio::test]
    async fn semantic_jaccard_none_identical_text_is_one() {
        let text = "stateless jwt auth token ADR-001";
        let sim = semantic_jaccard(text, text, None).await;
        assert!((sim - 1.0).abs() < 1e-9, "identical text with None must be 1.0");
    }

    #[tokio::test]
    async fn semantic_jaccard_none_disjoint_text_is_zero() {
        let sim = semantic_jaccard("jwt stateless auth", "redis cache store", None).await;
        assert_eq!(sim, 0.0);
    }

    #[tokio::test]
    async fn semantic_jaccard_adapter_returns_score() {
        use h2ai_adapters::MockAdapter;
        let adapter = MockAdapter::new(r#"{"score": 0.9}"#.to_string());
        let sim = semantic_jaccard("jwt auth", "bearer token auth", Some(&adapter)).await;
        assert!((sim - 0.9).abs() < 1e-9, "must use adapter score when parseable");
    }

    #[tokio::test]
    async fn semantic_jaccard_adapter_bad_response_falls_back_to_jaccard() {
        use h2ai_adapters::MockAdapter;
        let adapter = MockAdapter::new("I cannot compute that.".to_string());
        let text = "stateless jwt auth";
        let sim = semantic_jaccard(text, text, Some(&adapter)).await;
        let fallback = jaccard(&tokenize(text), &tokenize(text));
        assert!((sim - fallback).abs() < 1e-9, "bad adapter response must fall back to token Jaccard");
    }

    #[tokio::test]
    async fn semantic_jaccard_adapter_out_of_range_falls_back_to_jaccard() {
        use h2ai_adapters::MockAdapter;
        let adapter = MockAdapter::new(r#"{"score": 2.5}"#.to_string());
        let text = "jwt auth stateless";
        let sim = semantic_jaccard(text, text, Some(&adapter)).await;
        let fallback = jaccard(&tokenize(text), &tokenize(text));
        assert!((sim - fallback).abs() < 1e-9, "out-of-range score must fall back to token Jaccard");
    }
}
