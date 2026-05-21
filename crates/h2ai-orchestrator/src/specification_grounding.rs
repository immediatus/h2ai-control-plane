use regex::Regex;
use std::collections::HashSet;
use std::sync::LazyLock;

/// Result of checking N proposals against a task specification for correlated fabrication.
#[derive(Debug, Clone)]
pub struct GroundingCheckResult {
    /// Correlated Fabrication Index: max pairwise overlap of ungrounded entity sets.
    /// 0.0 = no shared fabrication, 1.0 = all proposals share the same fabricated entity.
    pub cfi: f64,
    /// Entities present in ≥2 proposals but absent from the specification.
    pub shared_ungrounded: Vec<String>,
    /// Per-proposal ungrounded entity sets (indexed by proposal position).
    pub per_proposal_ungrounded: Vec<HashSet<String>>,
    /// Number of proposals checked.
    pub proposal_count: usize,
}

// ── Regex patterns ────────────────────────────────────────────────────────────

/// CamelCase compound: at least two capitalised words concatenated.
/// Matches: `ZooKeeper`, `ElasticSearch`, OpenTelemetry
static CAMEL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b").unwrap());

/// Branded tech suffix: word ending in DB, MQ, KV, Store, Cache, Lock, Hub, Bus, Queue, Service.
/// Matches: `CockroachDB`, `RedisLock`, `MessageQueue`
static SUFFIX_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b\w+(?:DB|MQ|KV|Store|Cache|Lock|Hub|Bus|Queue|Service)\b").unwrap()
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

/// Extract architectural nouns from `text` using regex + curated lexicon.
///
/// A noun qualifies if it:
/// - Is a CamelCase compound (≥2 capitalised words), OR
/// - Ends with a branded tech suffix (DB, MQ, KV, Store, Cache, Lock, Hub, Bus, Queue, Service), OR
/// - Appears in the curated infrastructure lexicon.
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

/// Check all `proposals` against `spec` for correlated ungrounded architectural entities.
///
/// Returns `None` when fewer than 2 proposals are provided.
///
/// CFI formula:
/// ```text
/// Ungrounded(O_i) = V_arch(O_i) \ V_arch(spec)
/// CFI_pair(i,j)   = |U_i ∩ U_j| / (max(|U_i|, |U_j|) + ε)
/// CFI             = max over all pairs (i,j), i < j
/// ```
#[must_use]
pub fn check_specification_grounding(
    spec: &str,
    proposals: &[&str],
) -> Option<GroundingCheckResult> {
    if proposals.len() < 2 {
        return None;
    }

    let spec_nouns = extract_arch_nouns(spec);

    let per_proposal_ungrounded: Vec<HashSet<String>> = proposals
        .iter()
        .map(|p| {
            extract_arch_nouns(p)
                .into_iter()
                .filter(|n| !spec_nouns.contains(n))
                .collect()
        })
        .collect();

    let n = proposals.len();
    let mut cfi = 0.0_f64;
    let mut shared_map: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

    for i in 0..n {
        for j in (i + 1)..n {
            let ui = &per_proposal_ungrounded[i];
            let uj = &per_proposal_ungrounded[j];
            let intersection_count = ui.intersection(uj).count() as f64;
            let max_len = ui.len().max(uj.len());
            let pair_cfi = if max_len == 0 {
                0.0
            } else {
                intersection_count / max_len as f64
            };
            if pair_cfi > cfi {
                cfi = pair_cfi;
            }
            for entity in ui.intersection(uj) {
                *shared_map.entry(entity.clone()).or_insert(0) += 1;
            }
        }
    }

    let mut shared_ungrounded: Vec<String> = shared_map.into_keys().collect();
    shared_ungrounded.sort();

    Some(GroundingCheckResult {
        cfi,
        shared_ungrounded,
        per_proposal_ungrounded,
        proposal_count: proposals.len(),
    })
}
