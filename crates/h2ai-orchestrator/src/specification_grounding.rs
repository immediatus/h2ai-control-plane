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
/// Matches: ZooKeeper, ElasticSearch, OpenTelemetry
static CAMEL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b").unwrap());

/// Branded tech suffix: word ending in DB, MQ, KV, Store, Cache, Lock, Hub, Bus, Queue, Service.
/// Matches: CockroachDB, RedisLock, MessageQueue
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── extract_arch_nouns ──────────────────────────────────────────────

    #[test]
    fn camelcase_compound_extracted() {
        let nouns =
            extract_arch_nouns("Use ZooKeeper for coordination and ElasticSearch for indexing");
        assert!(nouns.contains("ZooKeeper"), "got: {nouns:?}");
        assert!(nouns.contains("ElasticSearch"), "got: {nouns:?}");
    }

    #[test]
    fn branded_suffix_extracted() {
        let nouns =
            extract_arch_nouns("Store state in CockroachDB and use RedisLock for mutual exclusion");
        assert!(nouns.contains("CockroachDB"), "got: {nouns:?}");
        assert!(nouns.contains("RedisLock"), "got: {nouns:?}");
    }

    #[test]
    fn lexicon_terms_extracted() {
        let nouns = extract_arch_nouns("Use etcd for service discovery alongside Consul and Vault");
        assert!(nouns.contains("etcd"), "got: {nouns:?}");
        assert!(nouns.contains("Consul"), "got: {nouns:?}");
        assert!(nouns.contains("Vault"), "got: {nouns:?}");
    }

    #[test]
    fn plain_english_not_extracted() {
        let nouns = extract_arch_nouns("Use a cache and a queue for better performance");
        assert!(!nouns.contains("cache"), "got: {nouns:?}");
        assert!(!nouns.contains("queue"), "got: {nouns:?}");
    }

    // ── check_specification_grounding ──────────────────────────────────

    #[test]
    fn fewer_than_two_proposals_returns_none() {
        let result = check_specification_grounding("use Redis", &["use Redis and ZooKeeper"]);
        assert!(result.is_none());
    }

    #[test]
    fn redis_in_spec_is_not_ungrounded() {
        let spec = "Use Redis for caching and Kafka for messaging";
        let p1 = "Use Redis for caching and Kafka for messaging with retry logic";
        let p2 = "Use Redis and Kafka with exponential backoff";
        let result = check_specification_grounding(spec, &[p1, p2]).unwrap();
        assert_eq!(result.cfi, 0.0, "Redis is grounded; CFI must be 0");
        assert!(result.shared_ungrounded.is_empty());
    }

    #[test]
    fn cockroachdb_not_in_spec_gives_cfi_one() {
        let spec = "Use Redis for caching and Kafka for event log";
        let p1 = "Use Redis and Kafka. CockroachDB advisory locks prevent double-spend.";
        let p2 = "Use Redis and Kafka. CockroachDB provides distributed locking.";
        let result = check_specification_grounding(spec, &[p1, p2]).unwrap();
        assert!(
            (result.cfi - 1.0).abs() < 1e-9,
            "expected CFI=1.0, got {}",
            result.cfi
        );
        assert!(
            result.shared_ungrounded.iter().any(|e| e == "CockroachDB"),
            "CockroachDB must be in shared_ungrounded; got {:?}",
            result.shared_ungrounded
        );
    }

    #[test]
    fn partial_overlap_gives_cfi_between_zero_and_one() {
        // Ungrounded(p1) = {CockroachDB, YugabyteDB}, Ungrounded(p2) = {CockroachDB}
        // CFI = 1 / max(2,1) = 0.5
        let spec = "Use Redis for state";
        let p1 = "Use Redis. CockroachDB and YugabyteDB provide ACID guarantees.";
        let p2 = "Use Redis. CockroachDB provides distributed transactions.";
        let result = check_specification_grounding(spec, &[p1, p2]).unwrap();
        assert!(
            result.cfi > 0.0 && result.cfi < 1.0,
            "expected 0 < CFI < 1, got {}",
            result.cfi
        );
        assert!(
            result.shared_ungrounded.iter().any(|e| e == "CockroachDB"),
            "CockroachDB must be in shared_ungrounded"
        );
    }

    #[test]
    fn no_shared_ungrounded_gives_cfi_zero() {
        let spec = "Use Redis and Kafka";
        let p1 = "Use Redis and Kafka with ZooKeeper for coordination";
        let p2 = "Use Redis and Kafka with Consul for service discovery";
        let result = check_specification_grounding(spec, &[p1, p2]).unwrap();
        assert_eq!(result.cfi, 0.0, "no shared ungrounded; CFI must be 0");
    }

    #[test]
    fn proposal_count_matches_input_length() {
        let spec = "Use Kafka";
        let p1 = "Kafka and CockroachDB";
        let p2 = "Kafka and CockroachDB";
        let p3 = "Kafka and CockroachDB";
        let result = check_specification_grounding(spec, &[p1, p2, p3]).unwrap();
        assert_eq!(result.proposal_count, 3);
    }
}
