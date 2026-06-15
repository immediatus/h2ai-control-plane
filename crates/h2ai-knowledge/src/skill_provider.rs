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
        self.nodes
            .write()
            .expect("SkillProvider nodes lock poisoned")
            .extend(new_nodes);
    }

    pub fn len(&self) -> usize {
        self.nodes
            .read()
            .expect("SkillProvider nodes lock poisoned")
            .len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes
            .read()
            .expect("SkillProvider nodes lock poisoned")
            .is_empty()
    }
}

#[async_trait]
impl KnowledgeProvider for SkillProvider {
    async fn query(&self, query: &KnowledgeQuery<'_>) -> KnowledgeResult {
        let nodes = self
            .nodes
            .read()
            .expect("SkillProvider nodes lock poisoned");
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
pub fn scope_by_domains(
    nodes: Vec<(KnowledgeNode, f32)>,
    tags: &[String],
) -> Vec<(KnowledgeNode, f32)> {
    if tags.is_empty() {
        return nodes;
    }
    let tag_set: std::collections::HashSet<&str> = tags.iter().map(String::as_str).collect();
    let filtered: Vec<_> = nodes
        .iter()
        .filter(|(n, _)| {
            n.domains.is_empty() || n.domains.iter().any(|d| tag_set.contains(d.as_str()))
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
        let cache = self
            .source_cache
            .read()
            .expect("source_cache lock poisoned");
        let mut map = self
            .violation_map
            .write()
            .expect("violation_map lock poisoned");
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
        let penalty = self
            .violation_map
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
            let mut cache = self
                .source_cache
                .write()
                .expect("source_cache lock poisoned");
            for (node, _) in &nodes {
                cache
                    .entry(node.id.clone())
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
