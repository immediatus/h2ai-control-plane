use crate::bm25plus::Bm25PlusRetriever;
use crate::factory::{ProviderKind, ScoringConfig};
use crate::graph::ConstraintGraph;
use crate::source::KnowledgeSource;
use crate::types::{
    KnowledgeNode, KnowledgeQuery, KnowledgeResult, NodeDepth, NodeSource, RetrievalMode,
    SurfacedTension,
};
use async_trait::async_trait;
use h2ai_constraints::resolver::ConstraintResolver;
use h2ai_constraints::source::{FsConstraintIndex, FsConstraintStore};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[async_trait]
pub trait KnowledgeProvider: Send + Sync {
    async fn query(&self, query: &KnowledgeQuery<'_>) -> KnowledgeResult;
    async fn global_summary(&self) -> Option<KnowledgeNode>;
    fn is_ready(&self) -> bool;
    fn kind(&self) -> &ProviderKind;
}

#[allow(dead_code)]
pub struct Bm25WikiProvider {
    global_node: Option<KnowledgeNode>,
    topic_nodes: Vec<KnowledgeNode>,
    leaf_items: Vec<KnowledgeNode>,
    topic_index: Bm25PlusRetriever,
    leaf_indices: HashMap<String, Bm25PlusRetriever>,
    collapsed_index: Bm25PlusRetriever,
    /// PPR graph built from leaf nodes only; topic IDs are not valid PPR seeds.
    graph: ConstraintGraph,
    scoring: ScoringConfig,
}

impl Bm25WikiProvider {
    pub fn leaf_count(&self) -> usize {
        self.leaf_items.len()
    }

    pub fn topic_count(&self) -> usize {
        self.topic_nodes.len()
    }

    pub async fn build(source: Arc<dyn KnowledgeSource>, scoring: ScoringConfig) -> Self {
        // Pass 1: load
        let (global_node, topic_nodes, leaf_items) =
            load_all(source, scoring.global_synthesis_max_chars).await;

        // Pass 2: BM25+ topic index
        let topic_index = {
            let docs: Vec<(String, String)> = topic_nodes
                .iter()
                .map(|n| {
                    let text = format!(
                        "{} {} {}",
                        n.synthesis,
                        n.invariants.join(" "),
                        n.failure_modes.join(" ")
                    );
                    (n.id.clone(), text)
                })
                .collect();
            Bm25PlusRetriever::build(docs.iter().map(|(id, t)| (id.as_str(), t.as_str())))
        };

        // Per-domain leaf indices
        let mut domain_leaves: HashMap<String, Vec<(String, String)>> = HashMap::new();
        for item in &leaf_items {
            for domain in &item.domains {
                domain_leaves
                    .entry(domain.clone())
                    .or_default()
                    .push((item.id.clone(), item.synthesis.clone()));
            }
        }
        let leaf_indices: HashMap<String, Bm25PlusRetriever> = domain_leaves
            .into_iter()
            .map(|(domain, pairs)| {
                let retriever = Bm25PlusRetriever::build(
                    pairs.iter().map(|(id, text)| (id.as_str(), text.as_str())),
                );
                (domain, retriever)
            })
            .collect();

        // Collapsed index (all nodes)
        let collapsed_index = {
            let topic_pairs: Vec<(String, String)> = topic_nodes
                .iter()
                .map(|n| (n.id.clone(), n.synthesis.clone()))
                .collect();
            let leaf_pairs: Vec<(String, String)> = leaf_items
                .iter()
                .map(|n| (n.id.clone(), n.synthesis.clone()))
                .collect();
            let all: Vec<(&str, &str)> = topic_pairs
                .iter()
                .chain(leaf_pairs.iter())
                .map(|(id, t)| (id.as_str(), t.as_str()))
                .collect();
            Bm25PlusRetriever::build(all.into_iter())
        };

        // Pass 3: PPR graph
        let graph = ConstraintGraph::build(&leaf_items);

        Self {
            global_node,
            topic_nodes,
            leaf_items,
            topic_index,
            leaf_indices,
            collapsed_index,
            graph,
            scoring,
        }
    }
}

async fn load_all(
    source: Arc<dyn KnowledgeSource>,
    global_synthesis_max_chars: usize,
) -> (
    Option<KnowledgeNode>,
    Vec<KnowledgeNode>,
    Vec<KnowledgeNode>,
) {
    let (global, wiki_topics, items) = tokio::join!(
        source.global_node(),
        source.wiki_nodes(),
        source.all_items(),
    );

    let leaf_nodes: Vec<KnowledgeNode> = items.into_iter().map(source_item_to_leaf).collect();

    let topic_nodes = if wiki_topics.is_empty() {
        synthesize_topic_nodes(&leaf_nodes)
    } else {
        wiki_topics
    };

    let global_node = global.or_else(|| {
        let synthesis = topic_nodes
            .iter()
            .map(|t| t.synthesis.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        if synthesis.is_empty() {
            return None;
        }
        let trimmed: String = synthesis.chars().take(global_synthesis_max_chars).collect();
        let domains: Vec<String> = topic_nodes
            .iter()
            .flat_map(|t| t.domains.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        Some(KnowledgeNode {
            id: "global:overview".to_string(),
            depth: NodeDepth::Global,
            synthesis: trimmed,
            invariants: vec![],
            failure_modes: vec![],
            domains,
            entry_points: vec![],
            tensions: vec![],
            cross_references: vec![],
            related: vec![],
            source: NodeSource::Synthetic,
            importance: 1.0,
        })
    });

    (global_node, topic_nodes, leaf_nodes)
}

fn source_item_to_leaf(item: crate::source::SourceItem) -> KnowledgeNode {
    KnowledgeNode {
        id: item.id.clone(),
        depth: NodeDepth::Leaf,
        synthesis: item.summary,
        invariants: vec![],
        failure_modes: vec![],
        domains: item.domains,
        entry_points: vec![],
        tensions: vec![],
        cross_references: item.cross_refs,
        related: item.related,
        source: NodeSource::YamlConstraint { id: item.id },
        importance: 0.7,
    }
}

fn synthesize_topic_nodes(leaves: &[KnowledgeNode]) -> Vec<KnowledgeNode> {
    let mut by_domain: HashMap<String, Vec<&KnowledgeNode>> = HashMap::new();
    for leaf in leaves {
        for domain in &leaf.domains {
            by_domain.entry(domain.clone()).or_default().push(leaf);
        }
    }
    let mut domains: Vec<_> = by_domain.into_iter().collect();
    domains.sort_by(|a, b| a.0.cmp(&b.0));
    domains
        .into_iter()
        .map(|(domain, nodes)| {
            let synthesis = nodes
                .iter()
                .map(|n| n.synthesis.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            KnowledgeNode {
                id: format!("topic:{domain}"),
                depth: NodeDepth::Topic,
                synthesis,
                invariants: vec![],
                failure_modes: vec![],
                domains: vec![domain],
                entry_points: nodes.iter().map(|n| n.id.clone()).collect(),
                tensions: vec![],
                cross_references: vec![],
                related: nodes.iter().map(|n| n.id.clone()).collect(),
                source: NodeSource::Synthetic,
                importance: 0.8,
            }
        })
        .collect()
}

fn boost_leaf_score(base: f32, node: &KnowledgeNode, query_text: &str, id_boost: f32) -> f32 {
    let mut score = base;
    if query_text.contains(&node.id) {
        score += id_boost;
    }
    score
}

#[async_trait]
impl KnowledgeProvider for Bm25WikiProvider {
    async fn query(&self, query: &KnowledgeQuery<'_>) -> KnowledgeResult {
        // 1. Explicit ID bypass: fetch leaf nodes by exact ID + topic nodes by entry_point match.
        // This guarantees constraint-specific wiki articles are always returned regardless of
        // BM25 vocabulary overlap (e.g., "CONSTRAINT-005" → kafka-audit topic node).
        if !query.explicit_ids.is_empty() {
            let mut nodes: Vec<(KnowledgeNode, f32)> = Vec::new();
            for id in query.explicit_ids {
                if let Some(n) = self.leaf_items.iter().find(|n| &n.id == id) {
                    nodes.push((n.clone(), 1.0));
                }
            }
            for topic in &self.topic_nodes {
                if topic
                    .entry_points
                    .iter()
                    .any(|ep| query.explicit_ids.contains(ep))
                {
                    nodes.push((topic.clone(), 1.0));
                }
            }
            return KnowledgeResult {
                global_included: false,
                surfaced_tensions: vec![],
                ppr_expanded: false,
                nodes,
            };
        }

        let mut results: Vec<(KnowledgeNode, f32)> = Vec::new();
        let mut global_included = false;

        // 2. Global node — always include when depth requested
        if query.depths.contains(&NodeDepth::Global) {
            if let Some(ref g) = self.global_node {
                results.push((g.clone(), 1.0));
                global_included = true;
            }
        }

        // 3. Retrieve topic and leaf nodes by mode
        match query.mode {
            RetrievalMode::CollapsedTree => {
                if query.depths.contains(&NodeDepth::Topic) {
                    let hits = self.topic_index.query(query.text, query.top_k);
                    for hit in hits {
                        if let Some(node) = self.topic_nodes.iter().find(|n| n.id == hit.id) {
                            results.push((node.clone(), hit.score));
                        }
                    }
                }
                if query.depths.contains(&NodeDepth::Leaf) {
                    let hits = self.collapsed_index.query(query.text, query.top_k);
                    for hit in hits {
                        if let Some(node) = self.leaf_items.iter().find(|n| n.id == hit.id) {
                            let score = boost_leaf_score(
                                hit.score * self.scoring.leaf_score_multiplier,
                                node,
                                query.text,
                                self.scoring.id_in_query_boost,
                            );
                            results.push((node.clone(), score));
                        }
                    }
                }
            }
            RetrievalMode::TreeTraversal => {
                // Route to topic clusters first
                let matched_topics: Vec<(KnowledgeNode, f32)> =
                    if query.depths.contains(&NodeDepth::Topic)
                        || query.depths.contains(&NodeDepth::Leaf)
                    {
                        let hits = self
                            .topic_index
                            .query(query.text, self.scoring.topic_cluster_top_k);
                        hits.into_iter()
                            .filter_map(|h| {
                                self.topic_nodes
                                    .iter()
                                    .find(|n| n.id == h.id)
                                    .map(|n| (n.clone(), h.score))
                            })
                            .collect()
                    } else {
                        vec![]
                    };

                // Add matched topic nodes to results
                if query.depths.contains(&NodeDepth::Topic) {
                    for (topic, score) in &matched_topics {
                        results.push((topic.clone(), *score));
                    }
                }

                // Retrieve leaves from matched topic domains
                if query.depths.contains(&NodeDepth::Leaf) {
                    let target_domains: HashSet<String> = matched_topics
                        .iter()
                        .flat_map(|(t, _)| t.domains.clone())
                        .collect();

                    let domains_to_search: Vec<String> = if target_domains.is_empty() {
                        self.leaf_indices.keys().cloned().collect()
                    } else {
                        target_domains.into_iter().collect()
                    };

                    let mut seen_leaf_ids: HashSet<String> = HashSet::new();
                    for domain in &domains_to_search {
                        if let Some(idx) = self.leaf_indices.get(domain) {
                            let per_domain_k =
                                (query.top_k / domains_to_search.len().max(1)).max(2);
                            let hits = idx.query(query.text, per_domain_k);
                            for hit in hits {
                                if seen_leaf_ids.insert(hit.id.clone()) {
                                    if let Some(node) =
                                        self.leaf_items.iter().find(|n| n.id == hit.id)
                                    {
                                        let mut score = boost_leaf_score(
                                            hit.score * self.scoring.leaf_score_multiplier,
                                            node,
                                            query.text,
                                            self.scoring.id_in_query_boost,
                                        );
                                        // Entry point boost
                                        if matched_topics
                                            .iter()
                                            .any(|(t, _)| t.entry_points.contains(&node.id))
                                        {
                                            score += self.scoring.entry_point_boost;
                                        }
                                        results.push((node.clone(), score));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // 4. PPR expansion on leaf results
        let mut ppr_expanded = false;
        if query.expand_hops > 0 && query.depths.contains(&NodeDepth::Leaf) {
            let leaf_hit_ids: Vec<String> = results
                .iter()
                .filter(|(n, _)| n.depth == NodeDepth::Leaf)
                .map(|(n, _)| n.id.clone())
                .collect();
            if !leaf_hit_ids.is_empty() {
                let seed_refs: Vec<&str> = leaf_hit_ids.iter().map(|s| s.as_str()).collect();
                let ppr_hits = self.graph.ppr(
                    &seed_refs,
                    self.scoring.ppr_alpha,
                    query.top_k,
                    self.scoring.ppr_max_iter,
                );
                let existing_ids: HashSet<String> =
                    results.iter().map(|(n, _)| n.id.clone()).collect();
                for (id, ppr_score) in ppr_hits {
                    if !existing_ids.contains(&id) {
                        if let Some(node) = self.leaf_items.iter().find(|n| n.id == id) {
                            results.push((
                                node.clone(),
                                ppr_score * self.scoring.ppr_score_multiplier,
                            ));
                            ppr_expanded = true;
                        }
                    }
                }
            }
        }

        // 5. Surface tensions when ≥2 matched topic clusters have cross-domain tensions
        let matched_topic_domains: HashSet<String> = results
            .iter()
            .filter(|(n, _)| n.depth == NodeDepth::Topic)
            .flat_map(|(n, _)| n.domains.clone())
            .collect();

        let mut surfaced_tensions = Vec::new();
        for (node, _) in results.iter().filter(|(n, _)| n.depth == NodeDepth::Topic) {
            for tension in &node.tensions {
                if matched_topic_domains.contains(&tension.domain) {
                    let domain_a = node.domains.first().cloned().unwrap_or_default();
                    let already = surfaced_tensions.iter().any(|t: &SurfacedTension| {
                        (t.domain_a == domain_a && t.domain_b == tension.domain)
                            || (t.domain_a == tension.domain && t.domain_b == domain_a)
                    });
                    if !already {
                        surfaced_tensions.push(SurfacedTension {
                            domain_a,
                            domain_b: tension.domain.clone(),
                            reason: tension.reason.clone(),
                        });
                    }
                }
            }
        }

        // 6. Dedup by ID (higher score wins) and sort descending
        let mut deduped: HashMap<String, (KnowledgeNode, f32)> = HashMap::new();
        for (node, score) in results {
            let entry = deduped
                .entry(node.id.clone())
                .or_insert((node.clone(), score));
            if score > entry.1 {
                *entry = (node, score);
            }
        }
        let mut final_results: Vec<(KnowledgeNode, f32)> = deduped.into_values().collect();
        final_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let cap = query.top_k + if global_included { 1 } else { 0 };
        final_results.truncate(cap);

        KnowledgeResult {
            nodes: final_results,
            global_included,
            surfaced_tensions,
            ppr_expanded,
        }
    }

    async fn global_summary(&self) -> Option<KnowledgeNode> {
        self.global_node.clone()
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn kind(&self) -> &ProviderKind {
        &ProviderKind::Bm25Wiki
    }
}

/// Zero-change fallback that delegates to the existing ConstraintResolver.
/// Used when `knowledge` is absent from config.
pub struct PassthroughProvider {
    resolver: ConstraintResolver,
}

impl PassthroughProvider {
    pub fn new(resolver: ConstraintResolver) -> Self {
        Self { resolver }
    }

    pub fn new_from_path(path: &std::path::Path) -> Self {
        let (index, store) = FsConstraintStore::load(path).unwrap_or_else(|_| {
            let store = FsConstraintStore::from_docs(vec![]);
            let index = FsConstraintIndex::from_docs(&[]);
            (index, store)
        });
        Self {
            resolver: ConstraintResolver::new(Arc::new(index), Arc::new(store)),
        }
    }
}

#[async_trait]
impl KnowledgeProvider for PassthroughProvider {
    async fn query(&self, query: &KnowledgeQuery<'_>) -> KnowledgeResult {
        let resolved = self
            .resolver
            .resolve(query.explicit_ids, query.tags, query.text)
            .await;
        let nodes = resolved
            .into_iter()
            .map(|doc| {
                let node = KnowledgeNode {
                    id: doc.id.clone(),
                    depth: NodeDepth::Leaf,
                    synthesis: doc.description.clone(),
                    invariants: vec![],
                    failure_modes: vec![],
                    domains: doc.domains.clone(),
                    entry_points: vec![],
                    tensions: vec![],
                    cross_references: vec![],
                    related: doc.related_to.clone(),
                    source: NodeSource::Synthetic,
                    importance: 0.7,
                };
                (node, 1.0f32)
            })
            .collect();
        KnowledgeResult {
            nodes,
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
        &ProviderKind::Passthrough
    }
}
