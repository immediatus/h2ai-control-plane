use crate::factory::ProviderKind;
use crate::provider::KnowledgeProvider;
use crate::types::{KnowledgeNode, KnowledgeQuery, KnowledgeResult, NodeSource};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

fn word_overlap_score(query: &str, text: &str) -> f32 {
    fn words(s: &str) -> std::collections::HashSet<String> {
        s.split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 2)
            .map(|w| w.to_lowercase())
            .collect()
    }
    let qw = words(query);
    if qw.is_empty() {
        return 0.0;
    }
    let tw = words(text);
    let overlap = qw.intersection(&tw).count();
    (overlap as f32 / qw.len() as f32 * 0.6).min(0.6)
}

pub struct SkillProvider {
    nodes: Arc<RwLock<Vec<KnowledgeNode>>>,
}

impl SkillProvider {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            nodes: Arc::new(RwLock::new(vec![])),
        })
    }

    pub fn push_all(&self, new_nodes: Vec<KnowledgeNode>) {
        self.nodes.write().expect("SkillProvider nodes lock poisoned").extend(new_nodes);
    }

    pub fn len(&self) -> usize {
        self.nodes.read().expect("SkillProvider nodes lock poisoned").len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.read().expect("SkillProvider nodes lock poisoned").is_empty()
    }
}

#[async_trait]
impl KnowledgeProvider for SkillProvider {
    async fn query(&self, query: &KnowledgeQuery<'_>) -> KnowledgeResult {
        let nodes = self.nodes.read().expect("SkillProvider nodes lock poisoned");
        let mut results: Vec<(KnowledgeNode, f32)> = nodes
            .iter()
            .filter(|n| query.depths.contains(&n.depth))
            .filter_map(|n| {
                let domain_match = query.tags.iter().any(|t| n.domains.contains(t))
                    || n.domains.iter().any(|d| query.text.contains(d.as_str()));
                if !domain_match {
                    return None;
                }
                let raw =
                    (0.4 + word_overlap_score(query.text, &n.synthesis)).min(1.0) * n.importance;
                if raw < 0.1 {
                    return None;
                }
                Some((n.clone(), raw))
            })
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        KnowledgeResult {
            nodes: results,
            global_included: false,
            surfaced_tensions: vec![],
            ppr_expanded: false,
        }
    }
    async fn global_summary(&self) -> Option<KnowledgeNode> {
        None
    }
    fn is_ready(&self) -> bool {
        true
    }
    fn kind(&self) -> &ProviderKind {
        &ProviderKind::Skill
    }
}

/// Post-filter merged knowledge nodes to those whose `domains` intersect `tags`.
/// When `tags` is empty, no filtering is applied — no signal → no change.
/// Untagged nodes (`domains.is_empty()`) are always retained to prevent starvation.
/// Falls back to the unfiltered set if the intersection would be empty.
pub(crate) fn scope_by_domains(
    nodes: Vec<(KnowledgeNode, f32)>,
    tags: &[String],
) -> Vec<(KnowledgeNode, f32)> {
    if tags.is_empty() {
        return nodes;
    }
    let tag_set: std::collections::HashSet<&str> =
        tags.iter().map(String::as_str).collect();
    let filtered: Vec<_> = nodes
        .iter()
        .filter(|(n, _)| {
            n.domains.is_empty()
                || n.domains.iter().any(|d| tag_set.contains(d.as_str()))
        })
        .cloned()
        .collect();
    if filtered.is_empty() {
        nodes
    } else {
        filtered
    }
}

pub struct CompositeProvider {
    providers: Vec<Arc<dyn KnowledgeProvider>>,
    /// Maps node_id → accumulated penalty [0.0, 0.9]. Applied as score multiplier (1 - penalty).
    violation_map: Arc<std::sync::RwLock<std::collections::HashMap<String, f32>>>,
    /// Populated lazily in query(). Maps node_id → is_synthetic.
    /// Lets record_violations skip Synthetic nodes without receiving source info explicitly.
    source_cache: Arc<std::sync::RwLock<std::collections::HashMap<String, bool>>>,
    /// When true, post-filter query results by domain intersection with query.tags.
    domain_scoping: bool,
}

impl CompositeProvider {
    pub fn new(providers: Vec<Arc<dyn KnowledgeProvider>>, domain_scoping: bool) -> Arc<Self> {
        Arc::new(Self {
            providers,
            violation_map: Arc::new(std::sync::RwLock::new(std::collections::HashMap::new())),
            source_cache: Arc::new(std::sync::RwLock::new(std::collections::HashMap::new())),
            domain_scoping,
        })
    }

    /// Record violation co-occurrence for retrieved node IDs.
    /// Skips Synthetic nodes — they document failures, not guidance.
    /// Delta is accumulated and capped at 0.9 per node.
    pub fn record_violations(&self, node_ids: &[String], delta: f32) {
        let cache = self.source_cache.read().expect("source_cache lock poisoned");
        let mut map = self.violation_map.write().expect("violation_map lock poisoned");
        for id in node_ids {
            if cache.get(id.as_str()).copied().unwrap_or(false) {
                continue; // Synthetic node — exempt
            }
            let penalty = map.entry(id.clone()).or_insert(0.0);
            *penalty = (*penalty + delta).min(0.9);
        }
    }

    /// Returns the accumulated violation penalty for a node (0.0 = none, 0.9 = max).
    /// Used in tests to verify penalty state directly.
    pub fn violation_penalty_for(&self, node_id: &str) -> f32 {
        self.violation_map
            .read()
            .expect("violation_map lock poisoned")
            .get(node_id)
            .copied()
            .unwrap_or(0.0)
    }

    fn penalised_score(&self, node_id: &str, score: f32) -> f32 {
        let penalty = self.violation_map
            .read()
            .expect("violation_map lock poisoned")
            .get(node_id)
            .copied()
            .unwrap_or(0.0);
        score * (1.0 - penalty)
    }
}

#[async_trait]
impl KnowledgeProvider for CompositeProvider {
    async fn query(&self, query: &KnowledgeQuery<'_>) -> KnowledgeResult {
        let results =
            futures::future::join_all(self.providers.iter().map(|p| p.query(query))).await;

        let mut merged: HashMap<String, (KnowledgeNode, f32)> = HashMap::new();
        let mut global_included = false;
        let mut ppr_expanded = false;
        let mut surfaced_tensions = Vec::new();

        for result in results {
            global_included |= result.global_included;
            ppr_expanded |= result.ppr_expanded;
            surfaced_tensions.extend(result.surfaced_tensions);
            for (node, score) in result.nodes {
                let penalized = self.penalised_score(&node.id, score);
                merged
                    .entry(node.id.clone())
                    .and_modify(|e| {
                        if penalized > e.1 {
                            e.1 = penalized;
                        }
                    })
                    .or_insert((node, penalized));
            }
        }

        let mut nodes: Vec<(KnowledgeNode, f32)> = merged.into_values().collect();
        nodes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // GAP-F4 Phase 1b: filter by domain intersection BEFORE top_k truncation so
        // in-domain nodes ranked below top_k are promoted rather than discarded.
        if self.domain_scoping {
            nodes = scope_by_domains(nodes, query.tags);
        }

        nodes.truncate(query.top_k);

        {
            let mut cache = self.source_cache.write().expect("source_cache lock poisoned");
            for (node, _) in &nodes {
                cache.entry(node.id.clone())
                    .or_insert(matches!(node.source, NodeSource::Synthetic));
            }
        }

        KnowledgeResult {
            nodes,
            global_included,
            surfaced_tensions,
            ppr_expanded,
        }
    }
    async fn global_summary(&self) -> Option<KnowledgeNode> {
        None
    }
    fn is_ready(&self) -> bool {
        self.providers.iter().all(|p| p.is_ready())
    }
    fn kind(&self) -> &ProviderKind {
        &ProviderKind::Composite
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{NodeDepth, NodeSource, RetrievalMode, SearchScope};

    fn skill_node(id: &str, domains: &[&str], synthesis: &str) -> KnowledgeNode {
        KnowledgeNode {
            id: id.to_string(),
            depth: NodeDepth::Leaf,
            source: NodeSource::Synthetic,
            domains: domains.iter().map(|s| s.to_string()).collect(),
            synthesis: synthesis.to_string(),
            failure_modes: vec![],
            invariants: vec![],
            importance: 0.8,
            entry_points: vec![],
            tensions: vec![],
            cross_references: vec![],
            related: vec![],
        }
    }

    fn make_query<'a>(text: &'a str, tags: &'a [String]) -> KnowledgeQuery<'a> {
        static LEAF: &[NodeDepth] = &[NodeDepth::Leaf];
        KnowledgeQuery {
            text,
            tags,
            explicit_ids: &[],
            top_k: 10,
            depths: LEAF,
            mode: RetrievalMode::CollapsedTree,
            scope: SearchScope::Auto,
            expand_hops: 0,
        }
    }

    #[tokio::test]
    async fn empty_provider_returns_empty_result() {
        let provider = SkillProvider::new();
        let tags: Vec<String> = vec!["auth".into()];
        let result = provider.query(&make_query("auth token", &tags)).await;
        assert!(result.nodes.is_empty());
    }

    #[tokio::test]
    async fn domain_tag_match_returns_node_with_nonzero_score() {
        let provider = SkillProvider::new();
        provider.push_all(vec![skill_node("s1", &["auth"], "auth token validation failed")]);
        let tags: Vec<String> = vec!["auth".into()];
        let result = provider.query(&make_query("auth", &tags)).await;
        assert_eq!(result.nodes.len(), 1);
        assert!(result.nodes[0].1 > 0.1, "score must be above threshold");
    }

    #[tokio::test]
    async fn no_domain_overlap_excludes_node() {
        let provider = SkillProvider::new();
        provider.push_all(vec![skill_node("s1", &["auth"], "auth token failed")]);
        let tags: Vec<String> = vec!["billing".into()];
        let result = provider.query(&make_query("billing invoice", &tags)).await;
        assert!(result.nodes.is_empty(), "no domain overlap → empty result");
    }

    #[tokio::test]
    async fn text_match_boosts_score() {
        let provider = SkillProvider::new();
        provider.push_all(vec![
            skill_node("s1", &["auth"], "auth topology retry occurred"),
            skill_node("s2", &["auth"], "auth basic setup"),
        ]);
        let tags: Vec<String> = vec!["auth".into()];
        let result = provider.query(&make_query("topology retry", &tags)).await;
        let s1_score = result
            .nodes
            .iter()
            .find(|(n, _)| n.id == "s1")
            .map(|(_, s)| *s)
            .unwrap_or(0.0);
        let s2_score = result
            .nodes
            .iter()
            .find(|(n, _)| n.id == "s2")
            .map(|(_, s)| *s)
            .unwrap_or(0.0);
        assert!(
            s1_score > s2_score,
            "synthesis match must boost score: s1={s1_score} s2={s2_score}"
        );
    }

    #[tokio::test]
    async fn below_threshold_score_excluded() {
        // importance=0.1, domain match=0.4, zero word overlap → 0.4*0.1=0.04 < 0.1 threshold
        let provider = SkillProvider::new();
        let mut node = skill_node("s1", &["auth"], "completely unrelated xyz123 qwerty zxcv");
        node.importance = 0.1;
        provider.push_all(vec![node]);
        let tags: Vec<String> = vec!["auth".into()];
        // query text shares no words with "completely unrelated xyz123 qwerty zxcv"
        let result = provider
            .query(&make_query("topology retry validation", &tags))
            .await;
        assert!(result.nodes.is_empty(), "score below 0.1 must be excluded");
    }

    struct StaticProvider(Vec<(KnowledgeNode, f32)>);

    #[async_trait]
    impl KnowledgeProvider for StaticProvider {
        async fn query(&self, _query: &KnowledgeQuery<'_>) -> KnowledgeResult {
            KnowledgeResult {
                nodes: self.0.clone(),
                global_included: false,
                surfaced_tensions: vec![],
                ppr_expanded: false,
            }
        }
        async fn global_summary(&self) -> Option<KnowledgeNode> {
            None
        }
        fn is_ready(&self) -> bool {
            true
        }
        fn kind(&self) -> &ProviderKind {
            &ProviderKind::Skill
        }
    }

    #[tokio::test]
    async fn composite_merges_results_from_both_providers() {
        let n1 = skill_node("n1", &["auth"], "auth");
        let n2 = skill_node("n2", &["billing"], "billing");
        let p1: Arc<dyn KnowledgeProvider> = Arc::new(StaticProvider(vec![(n1, 0.8)]));
        let p2: Arc<dyn KnowledgeProvider> = Arc::new(StaticProvider(vec![(n2, 0.6)]));
        let composite = CompositeProvider::new(vec![p1, p2], false);
        let tags: Vec<String> = vec![];
        let result = composite.query(&make_query("test", &tags)).await;
        assert_eq!(result.nodes.len(), 2);
        assert_eq!(result.nodes[0].1, 0.8, "highest score first");
    }

    #[tokio::test]
    async fn composite_dedup_higher_score_wins() {
        let n = skill_node("n1", &["auth"], "auth");
        let p1: Arc<dyn KnowledgeProvider> = Arc::new(StaticProvider(vec![(n.clone(), 0.3)]));
        let p2: Arc<dyn KnowledgeProvider> = Arc::new(StaticProvider(vec![(n.clone(), 0.7)]));
        let composite = CompositeProvider::new(vec![p1, p2], false);
        let tags: Vec<String> = vec![];
        let result = composite.query(&make_query("auth", &tags)).await;
        assert_eq!(result.nodes.len(), 1, "dedup: same id → single entry");
        assert_eq!(result.nodes[0].1, 0.7, "higher score wins");
    }

    #[tokio::test]
    async fn composite_respects_top_k() {
        let nodes: Vec<(KnowledgeNode, f32)> = (0..12)
            .map(|i| (skill_node(&format!("n{i}"), &["auth"], "auth"), 0.5))
            .collect();
        let p: Arc<dyn KnowledgeProvider> = Arc::new(StaticProvider(nodes));
        let composite = CompositeProvider::new(vec![p], false);
        static LEAF: &[NodeDepth] = &[NodeDepth::Leaf];
        let tags: Vec<String> = vec![];
        let query = KnowledgeQuery {
            text: "auth",
            tags: &tags,
            explicit_ids: &[],
            top_k: 5,
            depths: LEAF,
            mode: RetrievalMode::CollapsedTree,
            scope: SearchScope::Auto,
            expand_hops: 0,
        };
        let result = composite.query(&query).await;
        assert_eq!(result.nodes.len(), 5, "top_k=5 must truncate to 5");
    }

    #[tokio::test]
    async fn composite_is_ready_only_when_all_providers_ready() {
        struct NotReady;
        #[async_trait]
        impl KnowledgeProvider for NotReady {
            async fn query(&self, _: &KnowledgeQuery<'_>) -> KnowledgeResult {
                KnowledgeResult {
                    nodes: vec![],
                    global_included: false,
                    surfaced_tensions: vec![],
                    ppr_expanded: false,
                }
            }
            async fn global_summary(&self) -> Option<KnowledgeNode> {
                None
            }
            fn is_ready(&self) -> bool {
                false
            }
            fn kind(&self) -> &ProviderKind {
                &ProviderKind::Skill
            }
        }
        let composite = CompositeProvider::new(vec![Arc::new(NotReady)], false);
        assert!(!composite.is_ready());
    }

    #[tokio::test]
    async fn topic_node_returned_when_depths_include_topic() {
        let provider = SkillProvider::new();
        let node = KnowledgeNode {
            id: "t1".to_string(),
            depth: NodeDepth::Topic,
            source: NodeSource::Synthetic,
            domains: vec!["auth".to_string()],
            synthesis: "auth topic node".to_string(),
            failure_modes: vec![],
            invariants: vec![],
            importance: 0.8,
            entry_points: vec![],
            tensions: vec![],
            cross_references: vec![],
            related: vec![],
        };
        provider.push_all(vec![node]);
        static TOPIC: &[NodeDepth] = &[NodeDepth::Topic];
        let tags: Vec<String> = vec!["auth".into()];
        let query = KnowledgeQuery {
            text: "auth",
            tags: &tags,
            explicit_ids: &[],
            top_k: 10,
            depths: TOPIC,
            mode: RetrievalMode::CollapsedTree,
            scope: SearchScope::Auto,
            expand_hops: 0,
        };
        let result = provider.query(&query).await;
        assert_eq!(result.nodes.len(), 1, "Topic node must be returned when depths includes Topic");
    }

    #[tokio::test]
    async fn leaf_only_depths_excludes_topic_node() {
        let provider = SkillProvider::new();
        let node = KnowledgeNode {
            id: "t1".to_string(),
            depth: NodeDepth::Topic,
            source: NodeSource::Synthetic,
            domains: vec!["auth".to_string()],
            synthesis: "auth topic node".to_string(),
            failure_modes: vec![],
            invariants: vec![],
            importance: 0.8,
            entry_points: vec![],
            tensions: vec![],
            cross_references: vec![],
            related: vec![],
        };
        provider.push_all(vec![node]);
        let tags: Vec<String> = vec!["auth".into()];
        let result = provider.query(&make_query("auth", &tags)).await; // make_query uses LEAF depths
        assert!(result.nodes.is_empty(), "Topic node must NOT be returned when depths is [Leaf]");
    }

    #[tokio::test]
    async fn violation_penalty_reduces_score_for_wiki_node() {
        let wiki_node = KnowledgeNode {
            id: "wiki-1".to_string(),
            depth: NodeDepth::Leaf,
            source: NodeSource::WikiYaml { path: "auth.yaml".to_string() },
            domains: vec!["auth".to_string()],
            synthesis: "auth wiki guidance".to_string(),
            failure_modes: vec![],
            invariants: vec![],
            importance: 1.0,
            entry_points: vec![],
            tensions: vec![],
            cross_references: vec![],
            related: vec![],
        };
        let p: Arc<dyn KnowledgeProvider> = Arc::new(StaticProvider(vec![(wiki_node, 0.8)]));
        let composite = CompositeProvider::new(vec![p], false);

        // First query — populates source_cache
        let tags: Vec<String> = vec!["auth".into()];
        let s0 = composite.query(&make_query("auth guidance", &tags)).await
            .nodes.first().map(|(_, s)| *s).unwrap_or(0.0);

        // Apply one violation delta
        composite.record_violations(&["wiki-1".to_string()], 0.1);

        // Second query — score should be reduced by factor (1 - 0.1) = 0.9
        let s1 = composite.query(&make_query("auth guidance", &tags)).await
            .nodes.first().map(|(_, s)| *s).unwrap_or(0.0);

        assert!(s1 < s0, "penalised score must be lower: s0={s0} s1={s1}");
        let expected = s0 * 0.9;
        assert!(
            (s1 - expected).abs() < 1e-5,
            "penalised score must be s0 * 0.9: expected={expected} got={s1}"
        );
    }

    #[tokio::test]
    async fn violation_penalty_exempt_for_synthetic_nodes() {
        let synth_node = KnowledgeNode {
            id: "skill-1".to_string(),
            depth: NodeDepth::Leaf,
            source: NodeSource::Synthetic,
            domains: vec!["auth".to_string()],
            synthesis: "auth skill node".to_string(),
            failure_modes: vec![],
            invariants: vec![],
            importance: 1.0,
            entry_points: vec![],
            tensions: vec![],
            cross_references: vec![],
            related: vec![],
        };
        let p: Arc<dyn KnowledgeProvider> = Arc::new(StaticProvider(vec![(synth_node, 0.8)]));
        let composite = CompositeProvider::new(vec![p], false);

        let tags: Vec<String> = vec!["auth".into()];
        // Populate source_cache via query
        let s0 = composite.query(&make_query("auth", &tags)).await
            .nodes.first().map(|(_, s)| *s).unwrap_or(0.0);

        // Attempt to penalise the synthetic node — must be no-op
        composite.record_violations(&["skill-1".to_string()], 0.1);

        let s1 = composite.query(&make_query("auth", &tags)).await
            .nodes.first().map(|(_, s)| *s).unwrap_or(0.0);

        assert!(
            (s1 - s0).abs() < 1e-5,
            "Synthetic node score must not be reduced: s0={s0} s1={s1}"
        );
    }

    // --- CompositeProvider domain_scoping integration test ---

    #[tokio::test]
    async fn composite_domain_scoping_filters_out_of_domain_nodes() {
        // Provider returns one in-domain node (billing) and one out-of-domain node (auth).
        let billing_node = skill_node("billing-1", &["billing"], "billing invoice processing");
        let auth_node = skill_node("auth-1", &["auth"], "auth token validation");
        let p: Arc<dyn KnowledgeProvider> =
            Arc::new(StaticProvider(vec![(billing_node, 0.8), (auth_node, 0.7)]));

        // domain_scoping: true — only billing-tagged nodes should survive the post-filter.
        let composite = CompositeProvider::new(vec![p], true);
        let tags: Vec<String> = vec!["billing".into()];
        let result = composite.query(&make_query("invoice", &tags)).await;

        assert_eq!(
            result.nodes.len(),
            1,
            "domain_scoping=true must remove the out-of-domain auth node"
        );
        assert_eq!(
            result.nodes[0].0.id, "billing-1",
            "only the billing-domain node must survive"
        );
    }

    #[tokio::test]
    async fn composite_domain_scoping_false_returns_all_nodes() {
        // Sanity check: with domain_scoping disabled, both nodes are returned.
        let billing_node = skill_node("billing-2", &["billing"], "billing invoice processing");
        let auth_node = skill_node("auth-2", &["auth"], "auth token validation");
        let p: Arc<dyn KnowledgeProvider> =
            Arc::new(StaticProvider(vec![(billing_node, 0.8), (auth_node, 0.7)]));

        let composite = CompositeProvider::new(vec![p], false);
        let tags: Vec<String> = vec!["billing".into()];
        let result = composite.query(&make_query("invoice", &tags)).await;

        assert_eq!(
            result.nodes.len(),
            2,
            "domain_scoping=false must not filter: both nodes must be returned"
        );
    }

    #[tokio::test]
    async fn violation_penalty_capped_at_0_9() {
        let wiki_node = KnowledgeNode {
            id: "wiki-cap".to_string(),
            depth: NodeDepth::Leaf,
            source: NodeSource::WikiYaml { path: "x.yaml".to_string() },
            domains: vec!["auth".to_string()],
            synthesis: "auth".to_string(),
            failure_modes: vec![],
            invariants: vec![],
            importance: 1.0,
            entry_points: vec![],
            tensions: vec![],
            cross_references: vec![],
            related: vec![],
        };
        let p: Arc<dyn KnowledgeProvider> = Arc::new(StaticProvider(vec![(wiki_node, 1.0)]));
        let composite = CompositeProvider::new(vec![p], false);

        let tags: Vec<String> = vec!["auth".into()];
        let s0 = composite.query(&make_query("auth", &tags)).await
            .nodes.first().map(|(_, s)| *s).unwrap_or(0.0);

        // Apply 20 violations — penalty must cap at 0.9
        for _ in 0..20 {
            composite.record_violations(&["wiki-cap".to_string()], 0.1);
        }

        let s_final = composite.query(&make_query("auth", &tags)).await
            .nodes.first().map(|(_, s)| *s).unwrap_or(0.0);

        let min_allowed = s0 * 0.1;
        assert!(
            s_final >= min_allowed - 1e-5,
            "score must not drop below 10% of original after cap: min={min_allowed} got={s_final}"
        );
        assert!(
            composite.violation_penalty_for("wiki-cap") <= 0.9 + 1e-5,
            "penalty must be capped at 0.9"
        );
    }
}

#[cfg(test)]
mod scope_by_domains_tests {
    use super::*;

    fn node(id: &str, domains: &[&str]) -> (KnowledgeNode, f32) {
        (KnowledgeNode {
            id: id.to_string(),
            depth: crate::types::NodeDepth::Leaf,
            synthesis: id.to_string(),
            invariants: vec![],
            failure_modes: vec![],
            domains: domains.iter().map(|s| s.to_string()).collect(),
            entry_points: vec![],
            tensions: vec![],
            cross_references: vec![],
            related: vec![],
            source: crate::types::NodeSource::Synthetic,
            importance: 0.5,
        }, 0.8)
    }

    #[test]
    fn empty_tags_no_filtering() {
        let nodes = vec![node("a", &["auth"]), node("b", &["billing"])];
        let result = scope_by_domains(nodes.clone(), &[]);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn filters_off_domain_node() {
        let nodes = vec![node("auth-node", &["auth"]), node("billing-node", &["billing"])];
        let result = scope_by_domains(nodes, &["billing".to_string()]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0.id, "billing-node");
    }

    #[test]
    fn retains_untagged_nodes() {
        // Nodes with empty domains are always retained (no starvation guarantee).
        let nodes = vec![node("untagged", &[]), node("auth-node", &["auth"])];
        let result = scope_by_domains(nodes, &["billing".to_string()]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0.id, "untagged");
    }

    #[test]
    fn falls_back_to_unfiltered_when_filter_empties_result() {
        let nodes = vec![node("auth-node", &["auth"])];
        let result = scope_by_domains(nodes.clone(), &["billing".to_string()]);
        // No billing nodes → fallback → return all
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0.id, "auth-node");
    }

    #[test]
    fn multi_domain_node_retained_on_any_match() {
        let nodes = vec![node("multi", &["auth", "billing"])];
        let result = scope_by_domains(nodes, &["billing".to_string()]);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn scoping_disabled_no_change() {
        // When domain_scoping flag is false, CompositeProvider must return identical
        // node sets regardless of query.tags — this is the no-regression guarantee.
        let nodes = vec![node("auth-node", &["auth"]), node("billing-node", &["billing"])];
        // scope_by_domains is the pure function; flag=false means it is never called.
        // Verify the pure function itself passes through when tags is empty (the guard path).
        let result = scope_by_domains(nodes.clone(), &[]);
        assert_eq!(result.len(), 2, "empty tags → no filtering → both nodes returned");
    }

    #[test]
    fn skill_nodes_filtered_by_same_rule() {
        // Synthetic-source nodes (cross-task skill nodes) are not exempt from domain
        // filtering — they use the same domains field as wiki nodes.
        let mut skill_node = node("skill:t1:billing:topic", &["billing"]);
        skill_node.0.source = crate::types::NodeSource::Synthetic;
        let mut wiki_node = node("wiki:auth", &["auth"]);
        wiki_node.0.source = crate::types::NodeSource::WikiYaml { path: "auth.yaml".into() };
        let nodes = vec![skill_node, wiki_node];
        let result = scope_by_domains(nodes, &["billing".to_string()]);
        assert_eq!(result.len(), 1, "only billing-domain skill node must survive");
        assert_eq!(result[0].0.id, "skill:t1:billing:topic");
    }
}
