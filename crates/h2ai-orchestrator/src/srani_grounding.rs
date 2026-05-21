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
    ///    this context — e.g. "`CockroachDB` rate limiting use case".
    /// 3. **Alternatives / comparison**: surfaces what engineers actually use —
    ///    e.g. "rate limiting Redis token bucket alternatives comparison".
    #[must_use]
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
                .copied()
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
    #[must_use]
    pub fn new(providers: Vec<Box<dyn GroundingProvider>>) -> Self {
        Self {
            providers,
            distiller: None,
            raw_max_chars: 4000,
            hint_max_chars: 1200,
            distill_enabled: true,
        }
    }

    #[must_use]
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

    #[must_use]
    pub fn len(&self) -> usize {
        self.providers.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }

    /// Run providers[0] (anchor) always, plus providers[tier+1] clamped to len-1.
    /// If the tier provider returns a `WebSearch` result and a distiller is configured,
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
        .map_or_else(|| budget.to_string(), |i| s[..=i].to_string())
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
                (_, true) => a.grounding_statement,
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
///
/// The `grounding_statement` is assumed to have already been distilled/truncated
/// by `SraniGroundingChain::resolve`; this function only strips residual URLs.
#[must_use]
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
