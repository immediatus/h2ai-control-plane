use async_trait::async_trait;
use h2ai_config::prompts;
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::sizing::TauValue;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::LazyLock;

use crate::gap_checkers::{Gap, GapCheckContext, GapChecker, GapKind, GapSeverity, GapSource};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FindingKind {
    Entity,
    Claim,
}

impl FindingKind {
    pub fn label(&self) -> &'static str {
        match self {
            FindingKind::Entity => "entity",
            FindingKind::Claim => "claim",
        }
    }
}

#[derive(Debug, Clone)]
pub struct GroundingFinding {
    pub text: String,
    pub kind: FindingKind,
    pub reason: String,
    pub confidence: f64,
}

#[async_trait]
pub trait GroundingJudge: Send + Sync {
    async fn judge(&self, output: &str, spec: &str) -> Vec<GroundingFinding>;
}

// ── Regex patterns ────────────────────────────────────────────────────────────

/// CamelCase compound: at least two capitalised words concatenated.
static CAMEL_RE: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b").unwrap());

/// Branded tech suffix: word ending in DB, MQ, KV, Store, Cache, Lock, Hub, Bus, Queue, Service.
static SUFFIX_RE: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(r"\b\w+(?:DB|MQ|KV|Store|Cache|Lock|Hub|Bus|Queue|Service)\b").unwrap()
});

/// Curated infrastructure lexicon — single-word terms that don't match the patterns above.
static LEXICON: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "etcd",
        "Consul",
        "Vault",
        "Envoy",
        "Istio",
        "Cassandra",
        "DynamoDB",
        "BigTable",
        "Spanner",
        "Zookeeper",
        "FoundationDB",
        "TiKV",
        "CockroachDB",
        "YugabyteDB",
        "ScyllaDB",
        "Citus",
        "TimescaleDB",
        "InfluxDB",
        "Prometheus",
        "Grafana",
        "Jaeger",
        "OpenTelemetry",
        "Kafka",
        "Redis",
        "Pulsar",
        "RabbitMQ",
        "NATS",
        "Elasticsearch",
        "Solr",
        "Clickhouse",
        "Druid",
        "Pinot",
        "Flink",
        "Spark",
        "Trino",
        "Presto",
        "Kubernetes",
        "Nomad",
        "Terraform",
        "Ansible",
        "Helm",
    ]
    .into_iter()
    .collect()
});

// ── HeuristicGroundingJudge ───────────────────────────────────────────────────

pub struct HeuristicGroundingJudge;

#[async_trait]
impl GroundingJudge for HeuristicGroundingJudge {
    async fn judge(&self, output: &str, spec: &str) -> Vec<GroundingFinding> {
        check_ungrounded_entities(output, spec)
    }
}

// ── Pure detection functions ──────────────────────────────────────────────────

/// Extracts architectural nouns (CamelCase compounds, branded tech suffixes, lexicon words).
pub fn extract_arch_nouns(text: &str) -> HashSet<String> {
    let mut result = HashSet::new();
    for m in CAMEL_RE.find_iter(text) {
        result.insert(m.as_str().to_string());
    }
    for m in SUFFIX_RE.find_iter(text) {
        result.insert(m.as_str().to_string());
    }
    for word in text.split(|c: char| !c.is_alphanumeric()) {
        if LEXICON.contains(word) {
            result.insert(word.to_string());
        }
    }
    result
}

/// Pure detection function: arch nouns in output absent from spec → GroundingFinding (Entity, 0.8).
pub fn check_ungrounded_entities(output: &str, spec: &str) -> Vec<GroundingFinding> {
    let output_nouns = extract_arch_nouns(output);
    if output_nouns.is_empty() {
        return vec![];
    }
    let spec_nouns = extract_arch_nouns(spec);
    let mut ungrounded: Vec<String> = output_nouns.difference(&spec_nouns).cloned().collect();
    ungrounded.sort();
    ungrounded
        .into_iter()
        .map(|text| GroundingFinding {
            reason: format!("entity '{}' absent from task specification", text),
            text,
            kind: FindingKind::Entity,
            confidence: 0.8,
        })
        .collect()
}

// ── LlmGroundingJudge ────────────────────────────────────────────────────────

pub struct LlmGroundingJudge {
    adapter: Arc<dyn IComputeAdapter>,
    max_tokens: u64,
    tau: TauValue,
}

impl LlmGroundingJudge {
    pub fn new(adapter: Arc<dyn IComputeAdapter>, max_tokens: u64, tau: f64) -> Self {
        Self {
            adapter,
            max_tokens,
            tau: TauValue::new(tau)
                .expect("grounding tau is validated in H2AIConfig::validate(); this expect is unreachable with valid config"),
        }
    }
}

#[async_trait]
impl GroundingJudge for LlmGroundingJudge {
    async fn judge(&self, output: &str, spec: &str) -> Vec<GroundingFinding> {
        let req = ComputeRequest {
            system_context: prompts::GROUNDING_JUDGE_SYSTEM.to_string(),
            task: prompts::GROUNDING_JUDGE_TASK
                .replace("{spec}", spec)
                .replace("{output}", output),
            tau: self.tau,
            max_tokens: self.max_tokens,
        };
        let raw = match self.adapter.execute(req).await {
            Ok(r) => r.output,
            Err(_) => return vec![],
        };
        parse_grounding_response(&raw)
    }
}

pub fn parse_grounding_response(raw: &str) -> Vec<GroundingFinding> {
    let start = raw.find('{').unwrap_or(raw.len());
    let end = raw.rfind('}').map(|i| i + 1).unwrap_or(raw.len());
    if start >= end {
        return vec![];
    }
    let trimmed = &raw[start..end];
    let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) else {
        return vec![];
    };
    let Some(arr) = v.get("findings").and_then(|f| f.as_array()) else {
        return vec![];
    };
    arr.iter()
        .filter_map(|item| {
            let text = item.get("text")?.as_str()?.to_string();
            let kind_str = item.get("kind")?.as_str()?;
            let kind = match kind_str {
                "entity" => FindingKind::Entity,
                "claim" => FindingKind::Claim,
                _ => return None,
            };
            let reason = item
                .get("reason")
                .and_then(|r| r.as_str())
                .unwrap_or("")
                .to_string();
            let confidence = item.get("confidence")?.as_f64()?;
            if confidence < 0.5 {
                return None;
            }
            Some(GroundingFinding {
                text,
                kind,
                reason,
                confidence,
            })
        })
        .collect()
}

// ── CompositeGroundingJudge ──────────────────────────────────────────────────

pub struct CompositeGroundingJudge {
    judges: Vec<Arc<dyn GroundingJudge>>,
}

impl CompositeGroundingJudge {
    pub fn new(judges: Vec<Arc<dyn GroundingJudge>>) -> Self {
        Self { judges }
    }
}

#[async_trait]
impl GroundingJudge for CompositeGroundingJudge {
    async fn judge(&self, output: &str, spec: &str) -> Vec<GroundingFinding> {
        let futures = self.judges.iter().map(|j| j.judge(output, spec));
        let results = futures::future::join_all(futures).await;
        let mut seen: HashSet<String> = HashSet::new();
        results
            .into_iter()
            .flatten()
            .filter(|f| seen.insert(f.text.clone()))
            .collect()
    }
}

// ── GroundingChecker ─────────────────────────────────────────────────────────

pub struct GroundingChecker {
    judge: Arc<dyn GroundingJudge>,
    effective_spec: String,
    min_confidence: f64,
}

impl GroundingChecker {
    pub fn new(
        judge: Arc<dyn GroundingJudge>,
        effective_spec: String,
        min_confidence: f64,
    ) -> Self {
        Self {
            judge,
            effective_spec,
            min_confidence,
        }
    }
}

pub fn confidence_to_severity(confidence: f64) -> GapSeverity {
    if confidence >= 0.9 {
        GapSeverity::High
    } else if confidence >= 0.7 {
        GapSeverity::Medium
    } else {
        GapSeverity::Low
    }
}

#[async_trait]
impl GapChecker for GroundingChecker {
    // _ctx unused now; future extension: pass ctx.verified_provision_list to the judge
    // to exclude already-verified provisions from entity flagging.
    async fn check(&self, document: &str, _ctx: &GapCheckContext) -> Vec<Gap> {
        self.judge
            .judge(document, &self.effective_spec)
            .await
            .into_iter()
            .filter(|f| f.confidence >= self.min_confidence)
            .map(|f| Gap {
                id: format!("grounding:{}", f.text.to_lowercase().replace(' ', "_")),
                kind: GapKind::UngroundedContent,
                severity: confidence_to_severity(f.confidence),
                description: format!("[{}] {}: {}", f.kind.label(), f.text, f.reason),
                affected_provisions: vec![],
                depends_on: None,
                source: GapSource::GroundingCheck,
            })
            .collect()
    }
}
