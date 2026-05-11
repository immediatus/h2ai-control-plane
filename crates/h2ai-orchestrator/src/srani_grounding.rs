use async_trait::async_trait;
use h2ai_config::prompts::{
    SRANI_DISTILL_SYSTEM, SRANI_DISTILL_TASK, SRANI_RESEARCHER_SYSTEM, SRANI_RESEARCHER_TASK,
};
use h2ai_tools::web_search::WebSearchBackend;
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::sizing::TauValue;
use std::collections::HashSet;
use std::sync::Arc;

use crate::specification_grounding::extract_arch_nouns;

pub use h2ai_types::events::GroundingSource;

// ─── Data types ───────────────────────────────────────────────────────────────

pub struct GroundingContext {
    pub fabricated_entities: Vec<String>,
    pub task_description: String,
}

pub struct GroundingResult {
    pub alternatives: Vec<String>,
    pub grounding_statement: String,
    pub source: GroundingSource,
}

// ─── Trait ────────────────────────────────────────────────────────────────────

#[async_trait]
pub trait GroundingProvider: Send + Sync {
    async fn ground(&self, ctx: &GroundingContext) -> Option<GroundingResult>;
}

// ─── SpecAnchorGrounder ───────────────────────────────────────────────────────

pub struct SpecAnchorGrounder;

#[async_trait]
impl GroundingProvider for SpecAnchorGrounder {
    async fn ground(&self, ctx: &GroundingContext) -> Option<GroundingResult> {
        let fabricated_set: HashSet<&String> = ctx.fabricated_entities.iter().collect();
        let spec_nouns = extract_arch_nouns(&ctx.task_description);
        let mut alternatives: Vec<String> = spec_nouns
            .into_iter()
            .filter(|n| !fabricated_set.contains(n))
            .collect();
        alternatives.sort();

        let grounding_statement = if alternatives.is_empty() {
            String::new()
        } else {
            format!("Spec-defined components: {}", alternatives.join(", "))
        };

        Some(GroundingResult {
            alternatives,
            grounding_statement,
            source: GroundingSource::SpecAnchor,
        })
    }
}

// ─── LlmResearcherGrounder ───────────────────────────────────────────────────

pub struct LlmResearcherGrounder {
    adapter: Arc<dyn IComputeAdapter>,
}

impl LlmResearcherGrounder {
    pub fn new(adapter: Arc<dyn IComputeAdapter>) -> Self {
        Self { adapter }
    }
}

#[async_trait]
impl GroundingProvider for LlmResearcherGrounder {
    async fn ground(&self, ctx: &GroundingContext) -> Option<GroundingResult> {
        let fabricated = ctx.fabricated_entities.join(", ");
        let research_req = ComputeRequest {
            system_context: SRANI_RESEARCHER_SYSTEM.as_str().into(),
            task: SRANI_RESEARCHER_TASK.render(&[
                ("fabricated", &fabricated),
                ("task_description", &ctx.task_description),
            ]),
            tau: TauValue::new(0.3).unwrap(),
            max_tokens: 512,
        };
        let response = self.adapter.execute(research_req).await.ok()?.output;
        let v: serde_json::Value = serde_json::from_str(&response).ok()?;
        let alternatives: Vec<String> = v["alternatives"]
            .as_array()?
            .iter()
            .filter_map(|a| a.as_str().map(String::from))
            .collect();
        let statement = v["statement"].as_str().unwrap_or("").to_string();

        Some(GroundingResult {
            alternatives,
            grounding_statement: statement,
            source: GroundingSource::LlmResearcher,
        })
    }
}

// ─── WebSearchGrounder ───────────────────────────────────────────────────────

pub struct WebSearchGrounder {
    backend: Arc<dyn WebSearchBackend>,
    max_results: usize,
}

impl WebSearchGrounder {
    pub fn new(backend: Arc<dyn WebSearchBackend>, max_results: usize) -> Self {
        Self {
            backend,
            max_results,
        }
    }

    /// Generate 2–3 targeted search queries that will surface real technical
    /// content rather than generic text. Rules:
    ///
    /// 1. **Domain + implementation**: pulls SO Q&A about how the domain is
    ///    actually built — e.g. "rate limiting sliding window Redis implementation".
    /// 2. **Entity grounding**: asks whether the hallucinated entity belongs in
    ///    this context — e.g. "CockroachDB rate limiting use case".
    /// 3. **Alternatives / comparison**: surfaces what engineers actually use —
    ///    e.g. "rate limiting Redis token bucket alternatives comparison".
    pub fn build_queries(ctx: &GroundingContext) -> Vec<String> {
        // Pull meaningful domain words: skip short stop-words, take up to 5.
        let domain_words: Vec<&str> = ctx
            .task_description
            .split_whitespace()
            .filter(|w| {
                w.len() > 3
                    && !matches!(
                        *w,
                        "with"
                            | "using"
                            | "that"
                            | "from"
                            | "into"
                            | "over"
                            | "Build"
                            | "build"
                            | "Make"
                            | "make"
                            | "Create"
                            | "create"
                            | "Implement"
                            | "implement"
                            | "Design"
                            | "design"
                    )
            })
            .take(5)
            .collect();
        let domain = domain_words.join(" ");

        let mut queries = Vec::with_capacity(3);

        // Q1 — core implementation: what do engineers actually use for this?
        if !domain.is_empty() {
            queries.push(format!("{domain} implementation"));
        }

        // Q2 — entity grounding: is the hallucinated component real for this use case?
        if let Some(entity) = ctx.fabricated_entities.first() {
            let short_domain: String = domain_words
                .iter()
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join(" ");
            queries.push(format!("{entity} {short_domain}"));
        }

        // Q3 — alternatives: what does the industry actually recommend?
        if !domain.is_empty() {
            queries.push(format!("{domain} best practices alternatives"));
        }

        queries
    }
}

#[async_trait]
impl GroundingProvider for WebSearchGrounder {
    async fn ground(&self, ctx: &GroundingContext) -> Option<GroundingResult> {
        if ctx.fabricated_entities.is_empty() {
            return None;
        }

        let queries = Self::build_queries(ctx);
        let mut sections: Vec<String> = Vec::new();

        for (i, query) in queries.iter().enumerate() {
            match self.backend.search(query, self.max_results).await {
                Ok(text) if !text.is_empty() && text != "No results found." => {
                    sections.push(format!("=== Query {}: {} ===\n{}", i + 1, query, text));
                }
                _ => {}
            }
        }

        if sections.is_empty() {
            return None;
        }

        Some(GroundingResult {
            alternatives: vec![],
            grounding_statement: sections.join("\n\n"),
            source: GroundingSource::WebSearch,
        })
    }
}

// ─── SraniGroundingChain ─────────────────────────────────────────────────────

pub struct SraniGroundingChain {
    providers: Vec<Box<dyn GroundingProvider>>,
    /// Optional LLM adapter that distills raw web-search text into concise facts.
    distiller: Option<Arc<dyn IComputeAdapter>>,
    /// Max chars of raw text fed to the distiller (or hint if no distiller).
    raw_max_chars: usize,
    /// Max chars of the final grounding statement injected into the hint.
    hint_max_chars: usize,
    /// Whether to run the distillation step when `distiller` is Some.
    distill_enabled: bool,
}

impl SraniGroundingChain {
    pub fn new(providers: Vec<Box<dyn GroundingProvider>>) -> Self {
        Self {
            providers,
            distiller: None,
            raw_max_chars: 4000,
            hint_max_chars: 1200,
            distill_enabled: true,
        }
    }

    pub fn with_distiller(
        mut self,
        distiller: Arc<dyn IComputeAdapter>,
        raw_max_chars: usize,
        hint_max_chars: usize,
        distill_enabled: bool,
    ) -> Self {
        self.distiller = Some(distiller);
        self.raw_max_chars = raw_max_chars;
        self.hint_max_chars = hint_max_chars;
        self.distill_enabled = distill_enabled;
        self
    }

    pub fn len(&self) -> usize {
        self.providers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }

    /// Run providers[0] (anchor) always, plus providers[tier+1] clamped to len-1.
    /// If the tier provider returns a WebSearch result and a distiller is configured,
    /// run the distillation pass before returning.
    pub async fn resolve(&self, ctx: &GroundingContext, tier: usize) -> Option<GroundingResult> {
        if self.providers.is_empty() {
            return None;
        }
        let anchor = self.providers[0].ground(ctx).await;

        let tier_idx = if self.providers.len() > 1 {
            (tier + 1).min(self.providers.len() - 1)
        } else {
            0
        };
        let tier_result = if tier_idx > 0 {
            self.providers[tier_idx].ground(ctx).await
        } else {
            None
        };

        let mut merged = merge_grounding(anchor, tier_result)?;

        // Distillation: when the merged result carries web-search content, compact it
        // with the LLM so only the most relevant facts reach the explorer hint.
        if merged.source == GroundingSource::WebSearch && !merged.grounding_statement.is_empty() {
            // Always cap raw content before any further processing.
            let raw = truncate_at_sentence(&merged.grounding_statement, self.raw_max_chars);

            let distilled = if self.distill_enabled {
                if let Some(ref adapter) = self.distiller {
                    distill_with_llm(adapter, &raw, ctx).await.unwrap_or(raw)
                } else {
                    raw
                }
            } else {
                raw
            };

            merged.grounding_statement = truncate_at_sentence(&distilled, self.hint_max_chars);
        }

        Some(merged)
    }
}

/// Truncates `s` at the last `. ` within `max_chars`, or at the char boundary if none found.
fn truncate_at_sentence(s: &str, max_chars: usize) -> String {
    if s.len() <= max_chars {
        return s.to_string();
    }
    let budget = &s[..max_chars];
    budget
        .rfind(". ")
        .map(|i| s[..i + 1].to_string())
        .unwrap_or_else(|| budget.to_string())
}

/// Calls the distiller LLM to extract key technical facts from raw search text.
async fn distill_with_llm(
    adapter: &Arc<dyn IComputeAdapter>,
    raw: &str,
    ctx: &GroundingContext,
) -> Option<String> {
    let req = ComputeRequest {
        system_context: SRANI_DISTILL_SYSTEM.as_str().into(),
        task: SRANI_DISTILL_TASK.render(&[
            ("task_description", &ctx.task_description),
            ("raw_results", raw),
        ]),
        tau: TauValue::new(0.2).unwrap(),
        max_tokens: 256,
    };
    let result = adapter.execute(req).await.ok()?;
    let text = result.output.trim().to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

fn merge_grounding(
    anchor: Option<GroundingResult>,
    tier: Option<GroundingResult>,
) -> Option<GroundingResult> {
    match (anchor, tier) {
        (None, None) => None,
        (Some(a), None) => Some(a),
        (None, Some(t)) => Some(t),
        (Some(a), Some(t)) => {
            let mut alternatives = a.alternatives.clone();
            for alt in &t.alternatives {
                if !alternatives.contains(alt) {
                    alternatives.push(alt.clone());
                }
            }
            let statement = match (
                a.grounding_statement.is_empty(),
                t.grounding_statement.is_empty(),
            ) {
                (true, _) => t.grounding_statement.clone(),
                (_, true) => a.grounding_statement.clone(),
                _ => format!("{}\n{}", a.grounding_statement, t.grounding_statement),
            };
            Some(GroundingResult {
                alternatives,
                grounding_statement: statement,
                source: t.source,
            })
        }
    }
}

// ─── Hint formatter ───────────────────────────────────────────────────────────

/// Removes bare URLs (http/https tokens) from text so they don't reach the LLM hint.
fn strip_urls(s: &str) -> String {
    s.split_whitespace()
        .filter(|w| !w.starts_with("http://") && !w.starts_with("https://"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Formats the grounding hint block injected into the explorer's retry context.
/// The `grounding_statement` is assumed to have already been distilled/truncated
/// by `SraniGroundingChain::resolve`; this function only strips residual URLs.
pub fn format_grounding_hint(result: &GroundingResult, fabricated: &[String]) -> String {
    let entities = fabricated.join(", ");
    let spec_line = if result.alternatives.is_empty() {
        String::new()
    } else {
        format!(
            "Spec-defined components: {}\n",
            result.alternatives.join(", ")
        )
    };
    let avoid_line = format!("Avoid (not in spec): {entities}\n");
    let alts_line = if result.grounding_statement.is_empty() {
        String::new()
    } else {
        let clean = strip_urls(&result.grounding_statement);
        format!("Spec-compliant alternatives: {clean}\n")
    };
    format!(
        "\n\n--- GROUNDING CONTEXT ---\n\
         {spec_line}{avoid_line}{alts_line}\
         Design using the spec-defined components listed above.\n\
         ---"
    )
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use h2ai_adapters::mock::MockAdapter;
    use h2ai_tools::error::ToolError;
    use h2ai_tools::web_search::MockSearchBackend;

    fn ctx_rate_limiting() -> GroundingContext {
        GroundingContext {
            fabricated_entities: vec!["CockroachDB".into(), "ClickHouse".into()],
            task_description: "Build a rate-limiting service using Redis and in-process counters"
                .into(),
        }
    }

    fn ctx_empty_task() -> GroundingContext {
        GroundingContext {
            fabricated_entities: vec!["CockroachDB".into()],
            task_description: "do something simple".into(),
        }
    }

    // ── SpecAnchorGrounder (4) ────────────────────────────────────────────────

    #[tokio::test]
    async fn spec_anchor_extracts_spec_entities_as_alternatives() {
        let grounder = SpecAnchorGrounder;
        let result = grounder.ground(&ctx_rate_limiting()).await.unwrap();
        assert!(
            result.alternatives.contains(&"Redis".to_string()),
            "expected Redis in alternatives, got {:?}",
            result.alternatives
        );
        assert!(
            !result.alternatives.contains(&"CockroachDB".to_string()),
            "CockroachDB should not be promoted — it is fabricated"
        );
    }

    #[tokio::test]
    async fn spec_anchor_excludes_fabricated_from_alternatives() {
        let ctx = GroundingContext {
            fabricated_entities: vec!["Redis".into(), "CockroachDB".into()],
            task_description: "Build a rate-limiting service using Redis and in-process counters"
                .into(),
        };
        let grounder = SpecAnchorGrounder;
        let result = grounder.ground(&ctx).await.unwrap();
        assert!(
            !result.alternatives.contains(&"Redis".to_string()),
            "Redis is fabricated — must not appear in alternatives"
        );
    }

    #[tokio::test]
    async fn spec_anchor_empty_spec_still_produces_result() {
        let grounder = SpecAnchorGrounder;
        let result = grounder.ground(&ctx_empty_task()).await;
        assert!(result.is_some());
        assert!(result.unwrap().alternatives.is_empty());
    }

    #[tokio::test]
    async fn spec_anchor_source_tag_is_correct() {
        let grounder = SpecAnchorGrounder;
        let result = grounder.ground(&ctx_rate_limiting()).await.unwrap();
        assert_eq!(result.source, GroundingSource::SpecAnchor);
    }

    // ── LlmResearcherGrounder (3) ─────────────────────────────────────────────

    #[tokio::test]
    async fn llm_researcher_happy_path() {
        let adapter = Arc::new(MockAdapter::new(
            r#"{"alternatives": ["Redis TTL counters", "sliding window"], "statement": "Use Redis TTL + Lua for rate limiting"}"#.into(),
        ));
        let grounder = LlmResearcherGrounder::new(adapter);
        let result = grounder.ground(&ctx_rate_limiting()).await.unwrap();
        assert!(!result.alternatives.is_empty());
        assert_eq!(result.source, GroundingSource::LlmResearcher);
    }

    #[tokio::test]
    async fn llm_researcher_invalid_json_returns_none() {
        let adapter = Arc::new(MockAdapter::new("not json at all !!!".into()));
        let grounder = LlmResearcherGrounder::new(adapter);
        let result = grounder.ground(&ctx_rate_limiting()).await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn llm_researcher_adapter_error_returns_none() {
        let adapter = Arc::new(MockAdapter::new(r#"{"statement": "use Redis"}"#.into()));
        let grounder = LlmResearcherGrounder::new(adapter);
        let result = grounder.ground(&ctx_rate_limiting()).await;
        assert!(
            result.is_none(),
            "missing alternatives field must return None"
        );
    }

    // ── WebSearchGrounder (3) ─────────────────────────────────────────────────

    #[tokio::test]
    async fn web_search_produces_web_search_source() {
        let backend = Arc::new(MockSearchBackend::new(
            "Redis sliding-window counter is the standard approach for rate limiting".to_string(),
        ));
        let grounder = WebSearchGrounder::new(backend, 3);
        let result = grounder.ground(&ctx_rate_limiting()).await.unwrap();
        assert_eq!(result.source, GroundingSource::WebSearch);
        assert!(!result.grounding_statement.is_empty());
    }

    #[tokio::test]
    async fn web_search_empty_results_returns_none() {
        let backend = Arc::new(MockSearchBackend::new("".to_string()));
        let grounder = WebSearchGrounder::new(backend, 3);
        let result = grounder.ground(&ctx_rate_limiting()).await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn web_search_error_returns_none() {
        struct FailingBackend;
        #[async_trait]
        impl WebSearchBackend for FailingBackend {
            async fn search(&self, _q: &str, _n: usize) -> Result<String, ToolError> {
                Err(ToolError::MalformedInput("network error".into()))
            }
        }
        let grounder = WebSearchGrounder::new(Arc::new(FailingBackend), 3);
        let result = grounder.ground(&ctx_rate_limiting()).await;
        assert!(result.is_none());
    }

    // ── truncate_at_sentence ──────────────────────────────────────────────────

    #[test]
    fn truncate_at_sentence_within_budget_returns_unchanged() {
        let s = "Short sentence. Another one.";
        assert_eq!(truncate_at_sentence(s, 200), s);
    }

    #[test]
    fn truncate_at_sentence_over_budget_breaks_at_sentence_boundary() {
        let s = "First sentence. Second sentence. Third sentence that is very long indeed.";
        let result = truncate_at_sentence(s, 30);
        assert!(
            result.ends_with('.'),
            "must end at a sentence boundary, got: {result:?}"
        );
        assert!(result.len() <= 30, "must not exceed budget");
        assert!(
            result.contains("First"),
            "must contain first sentence, got: {result:?}"
        );
    }

    #[test]
    fn truncate_at_sentence_no_sentence_boundary_truncates_at_char_limit() {
        let s = "OneWordNoBreakAtAll"; // no ". "
        let result = truncate_at_sentence(s, 10);
        assert_eq!(result.len(), 10);
    }

    // ── strip_urls ────────────────────────────────────────────────────────────

    #[test]
    fn strip_urls_removes_http_tokens() {
        let s = "Redis https://redis.io/docs sliding window http://example.com counter";
        let result = strip_urls(s);
        assert!(!result.contains("https://"), "https:// must be removed");
        assert!(!result.contains("http://"), "http:// must be removed");
        assert!(result.contains("Redis"), "prose words must survive");
        assert!(result.contains("counter"), "prose words must survive");
    }

    #[test]
    fn strip_urls_preserves_non_url_text() {
        let s = "rate limiting with Redis sliding window";
        assert_eq!(strip_urls(s), s);
    }

    // ── SraniGroundingChain (4) ───────────────────────────────────────────────

    #[tokio::test]
    async fn chain_tier0_merges_spec_anchor_and_researcher() {
        let providers: Vec<Box<dyn GroundingProvider>> = vec![
            Box::new(SpecAnchorGrounder),
            Box::new(LlmResearcherGrounder::new(Arc::new(MockAdapter::new(
                r#"{"alternatives": ["Redis TTL counters"], "statement": "Use Redis TTL + Lua"}"#
                    .into(),
            )))),
        ];
        let chain = SraniGroundingChain::new(providers);
        let result = chain.resolve(&ctx_rate_limiting(), 0).await.unwrap();
        assert!(
            result.grounding_statement.contains("Spec-defined"),
            "anchor statement missing: {}",
            result.grounding_statement
        );
        assert!(
            result.grounding_statement.contains("Redis TTL"),
            "researcher statement missing: {}",
            result.grounding_statement
        );
    }

    #[tokio::test]
    async fn chain_tier1_escalates_to_web_search_skips_researcher() {
        let providers: Vec<Box<dyn GroundingProvider>> = vec![
            Box::new(SpecAnchorGrounder),
            Box::new(LlmResearcherGrounder::new(Arc::new(MockAdapter::new(
                "should not appear".into(),
            )))),
            Box::new(WebSearchGrounder::new(
                Arc::new(MockSearchBackend::new("Web result: use Redis".to_string())),
                3,
            )),
        ];
        let chain = SraniGroundingChain::new(providers);
        let result = chain.resolve(&ctx_rate_limiting(), 1).await.unwrap();
        assert_eq!(
            result.source,
            GroundingSource::WebSearch,
            "tier=1 must use WebSearch"
        );
    }

    #[tokio::test]
    async fn chain_tier_clamped_at_last_tier() {
        let providers: Vec<Box<dyn GroundingProvider>> = vec![
            Box::new(SpecAnchorGrounder),
            Box::new(LlmResearcherGrounder::new(Arc::new(MockAdapter::new(
                r#"{"alternatives": ["x"], "statement": "y"}"#.into(),
            )))),
        ];
        let chain = SraniGroundingChain::new(providers);
        let result = chain.resolve(&ctx_rate_limiting(), 99).await;
        assert!(result.is_some(), "clamped tier must not panic");
    }

    #[tokio::test]
    async fn chain_spec_anchor_only_still_produces_positive_result() {
        let providers: Vec<Box<dyn GroundingProvider>> = vec![Box::new(SpecAnchorGrounder)];
        let chain = SraniGroundingChain::new(providers);
        let result = chain.resolve(&ctx_rate_limiting(), 0).await;
        assert!(result.is_some());
        assert_eq!(result.unwrap().source, GroundingSource::SpecAnchor);
    }

    // ── distillation integration ──────────────────────────────────────────────

    #[tokio::test]
    async fn chain_distillation_replaces_raw_web_text_with_distilled_output() {
        let raw_text = "Web result: use Redis. ".repeat(200); // > 4000 chars
        let distilled_output = "Redis sliding window is the standard rate-limiting approach.";
        let providers: Vec<Box<dyn GroundingProvider>> = vec![
            Box::new(SpecAnchorGrounder),
            Box::new(WebSearchGrounder::new(
                Arc::new(MockSearchBackend::new(raw_text.clone())),
                3,
            )),
        ];
        let distiller = Arc::new(MockAdapter::new(distilled_output.into()));
        let chain = SraniGroundingChain::new(providers).with_distiller(distiller, 4000, 1200, true);
        let result = chain.resolve(&ctx_rate_limiting(), 1).await.unwrap();
        assert_eq!(result.source, GroundingSource::WebSearch);
        // Distilled output should appear; raw repetitive text should not.
        assert!(
            result.grounding_statement.contains("Redis sliding window"),
            "distilled text must be in statement, got: {}",
            result.grounding_statement
        );
        assert!(
            result.grounding_statement.len() <= 1200,
            "hint must respect hint_max_chars, len={}",
            result.grounding_statement.len()
        );
    }

    #[tokio::test]
    async fn chain_distill_disabled_preserves_raw_text_capped_at_hint_limit() {
        let raw_text = "Redis. ".repeat(300); // > 1200 chars
        let providers: Vec<Box<dyn GroundingProvider>> = vec![
            Box::new(SpecAnchorGrounder),
            Box::new(WebSearchGrounder::new(
                Arc::new(MockSearchBackend::new(raw_text)),
                3,
            )),
        ];
        let distiller = Arc::new(MockAdapter::new("should not be called".into()));
        let chain = SraniGroundingChain::new(providers).with_distiller(
            distiller, 4000, 1200, false, // disabled
        );
        let result = chain.resolve(&ctx_rate_limiting(), 1).await.unwrap();
        assert!(
            result.grounding_statement.len() <= 1200,
            "hint must still be capped even with distill=false, len={}",
            result.grounding_statement.len()
        );
    }
}
