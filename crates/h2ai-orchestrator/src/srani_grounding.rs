use async_trait::async_trait;
use h2ai_config::prompts::{
    I1_SYNTHESIS_VALIDATOR_TASK, SRANI_DISTILL_SYSTEM, SRANI_DISTILL_TASK, SRANI_RESEARCHER_SYSTEM,
    SRANI_RESEARCHER_TASK,
};
use h2ai_tools::web_search::WebSearchBackend;
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::gap_i1::{DomainSynthesis, KnowledgeGapRecord};
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
    /// Whether to run the distillation step when `distiller` is Some.
    distill_enabled: bool,
    /// Minimum character count to trigger distillation. Compact results skip it.
    compress_threshold: usize,
}

impl SraniGroundingChain {
    #[must_use]
    pub fn new(providers: Vec<Box<dyn GroundingProvider>>) -> Self {
        Self {
            providers,
            distiller: None,
            distill_enabled: true,
            compress_threshold: 800,
        }
    }

    #[must_use]
    pub fn with_distiller(
        mut self,
        distiller: Arc<dyn IComputeAdapter>,
        distill_enabled: bool,
    ) -> Self {
        self.distiller = Some(distiller);
        self.distill_enabled = distill_enabled;
        self
    }

    /// Override the minimum character threshold below which distillation is skipped.
    #[must_use]
    pub fn with_compress_threshold(mut self, threshold: usize) -> Self {
        self.compress_threshold = threshold;
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

        // Distillation: when the merged result carries web-search content, compress it
        // with the LLM so only the most relevant facts reach the explorer hint.
        // No character truncation — LLM compression is the only size-reduction mechanism.
        if merged.source == GroundingSource::WebSearch
            && merged.grounding_statement.len() >= self.compress_threshold
            && self.distill_enabled
        {
            if let Some(ref adapter) = self.distiller {
                if let Some(distilled) =
                    distill_with_llm(adapter, &merged.grounding_statement, ctx).await
                {
                    merged.grounding_statement = distilled;
                }
            }
        }

        Some(merged)
    }
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

// ─── helpers ───────────────────────────────────────────────────────────

/// Build web search queries targeted at a specific constraint knowledge gap.
pub fn gap_queries_from_record(record: &KnowledgeGapRecord, check_text: &str) -> Vec<String> {
    vec![
        format!(
            "correct implementation {} instead of {}",
            check_text
                .split_whitespace()
                .take(6)
                .collect::<Vec<_>>()
                .join(" "),
            record
                .incorrect_concept
                .split_whitespace()
                .take(4)
                .collect::<Vec<_>>()
                .join(" ")
        ),
        format!(
            "{} failure race condition known bug documentation",
            record
                .incorrect_concept
                .split_whitespace()
                .take(5)
                .collect::<Vec<_>>()
                .join(" ")
        ),
        record.gap_query.clone(),
    ]
}

/// Returns true if a `DomainSynthesis` meets the minimum confidence threshold.
pub fn synthesis_meets_threshold(synth: &DomainSynthesis, min_confidence: f64) -> bool {
    synth.confidence >= min_confidence
}

/// Research a constraint knowledge gap using web search + LLM synthesis validation.
///
/// Builds a `GroundingContext` from the gap record, runs the full `SraniGroundingChain`
/// distillation pipeline (DDG search → truncate → LLM distill → truncate),
/// then asks the LLM (via `I1_SYNTHESIS_VALIDATOR_TASK`) to score the synthesis.
/// Returns `Some(DomainSynthesis)` only if `synthesis_meets_threshold` passes.
pub async fn run_gap_researcher(
    record: &KnowledgeGapRecord,
    check_text: &str,
    adapter: &Arc<dyn IComputeAdapter>,
    chain: Option<&SraniGroundingChain>,
    min_confidence: f64,
    timeout_secs: u64,
) -> Option<DomainSynthesis> {
    let queries = gap_queries_from_record(record, check_text);

    let ctx = GroundingContext {
        fabricated_entities: queries,
        task_description: check_text.to_string(),
    };

    let (correct_pattern, source) = if let Some(ch) = chain {
        // Run full distillation pipeline: DDG search → truncate → LLM distill → truncate
        let grounding = tokio::time::timeout(
            std::time::Duration::from_secs(timeout_secs),
            ch.resolve(&ctx, 0),
        )
        .await
        .ok()??;
        let stmt = if grounding.grounding_statement.is_empty() {
            record.gap_query.clone()
        } else {
            grounding.grounding_statement.clone()
        };
        (stmt, grounding.alternatives.first().cloned())
    } else {
        // LLM-only path: use gap_query directly as the evidence seed.
        (record.gap_query.clone(), None)
    };

    let validation_task = I1_SYNTHESIS_VALIDATOR_TASK
        .replace("{check_text}", check_text)
        .replace("{incorrect_pattern}", &record.incorrect_concept)
        .replace("{correct_pattern}", &correct_pattern)
        .replace("{mechanistic_reason}", &record.gap_query);

    let req = ComputeRequest {
        system_context: "You are a domain synthesis validator. Respond only with valid JSON."
            .into(),
        task: validation_task,
        tau: TauValue::new(0.2).unwrap(),
        max_tokens: 256,
    };

    // Validation is a small 256-token call — cap it at 60s independently of web-search timeout.
    let response = tokio::time::timeout(std::time::Duration::from_secs(60), adapter.execute(req))
        .await
        .ok()?
        .ok()?
        .output;
    let v: serde_json::Value = serde_json::from_str(&response).ok()?;
    let score = v["score"].as_f64().unwrap_or(0.0);
    let reason = v["reason"].as_str().unwrap_or("").to_string();

    let synth = DomainSynthesis {
        check_id: (record.constraint_id.clone(), record.check_idx),
        incorrect_pattern: record.incorrect_concept.clone(),
        correct_pattern,
        mechanistic_reason: reason,
        source,
        confidence: score,
    };

    if synthesis_meets_threshold(&synth, min_confidence) {
        Some(synth)
    } else {
        None
    }
}

// ─── Slot classifier ─────────────────────────────────────────────────────────

/// Classifies SRANI-detected fabricated entities into a repair context slot name.
/// First entity matching a known technology domain wins.
/// Falls back to "implementation_detail" for unknown entities or empty input.
///
/// NOTE: "nats" is intentionally excluded from the message_broker pattern.
/// The codebase uses NATS (nats.io) heavily as its own message bus infrastructure
/// (NatsClient, nats:// URLs, nats_dispatch fields). Matching on the substring
/// "nats" would produce false positives for any NATS-related variable or URL.
#[must_use]
pub fn classify_grounding_slot(entities: &[String]) -> String {
    for entity in entities {
        let e = entity.to_lowercase();
        if e.contains("kafka") || e.contains("rabbitmq") || e.contains("activemq") {
            return "message_broker".to_string();
        }
        if e.contains("zookeeper")
            || e.contains("etcd")
            || e.contains("consul")
            || e.contains("chubby")
        {
            return "distributed_coordination".to_string();
        }
        if e.contains("redis") || e.contains("memcached") || e.contains("dragonfly") {
            return "cache_layer".to_string();
        }
        if e.starts_with("pg_") || e.contains("postgres") || e.contains("replication_slot") {
            return "database_migration".to_string();
        }
    }
    "implementation_detail".to_string()
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

#[cfg(test)]
mod gap_i1_tests {
    use super::*;
    use h2ai_types::gap_i1::{DomainSynthesis, KnowledgeGapRecord};

    #[test]
    fn gap_queries_from_record_are_non_empty() {
        let record = KnowledgeGapRecord {
            constraint_id: "CONSTRAINT-008".to_string(),
            check_idx: 1,
            incorrect_concept: "SETNX as standalone idempotency primitive".to_string(),
            gap_query: "Redis Lua EVAL atomic quota update without SETNX".to_string(),
            pass_rate_across_waves: 0.0,
        };
        let queries = gap_queries_from_record(&record, "Does design use Lua EVAL for CAS?");
        assert_eq!(queries.len(), 3);
        assert!(queries.iter().all(|q| !q.is_empty()));
    }

    #[test]
    fn domain_synthesis_below_min_confidence_is_rejected() {
        let synth = DomainSynthesis {
            check_id: ("C".to_string(), 0),
            incorrect_pattern: "wrong".to_string(),
            correct_pattern: "right".to_string(),
            mechanistic_reason: "because".to_string(),
            source: None,
            confidence: 0.5,
        };
        assert!(!synthesis_meets_threshold(&synth, 0.7));
    }

    #[test]
    fn domain_synthesis_above_min_confidence_is_accepted() {
        let synth = DomainSynthesis {
            check_id: ("C".to_string(), 0),
            incorrect_pattern: "wrong".to_string(),
            correct_pattern: "right".to_string(),
            mechanistic_reason: "because".to_string(),
            source: None,
            confidence: 0.85,
        };
        assert!(synthesis_meets_threshold(&synth, 0.7));
    }

    #[test]
    fn synthesis_meets_threshold_works_with_optional_grounding_chain() {
        // Pure function test — just verify None doesn't break compilation
        // (run_gap_researcher is async so just test the helper functions here)
        // run_gap_researcher now accepts Option<&SraniGroundingChain> instead of
        // Option<&WebSearchGrounder> — this test validates the helper functions.
        let record = KnowledgeGapRecord {
            constraint_id: "C".to_string(),
            check_idx: 0,
            incorrect_concept: "SETNX as lock".to_string(),
            gap_query: "Redis Lua EVAL atomic CAS".to_string(),
            pass_rate_across_waves: 0.0,
        };
        let queries = gap_queries_from_record(&record, "Does design use Lua EVAL?");
        assert_eq!(queries.len(), 3);
        // Verify that passing None for the chain (type-checked as Option<&SraniGroundingChain>)
        // is expressible in user code.
        let chain: Option<&SraniGroundingChain> = None;
        assert!(chain.is_none());
    }
}
