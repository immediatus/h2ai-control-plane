use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeDepth {
    Global,
    Topic,
    Leaf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeSource {
    YamlConstraint { id: String },
    WikiYaml { path: String },
    Synthetic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensionRef {
    pub domain: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossRef {
    pub id: String,
    pub domain: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeNode {
    pub id: String,
    pub depth: NodeDepth,
    pub synthesis: String,
    pub invariants: Vec<String>,
    pub failure_modes: Vec<String>,
    pub domains: Vec<String>,
    pub entry_points: Vec<String>,
    pub tensions: Vec<TensionRef>,
    pub cross_references: Vec<CrossRef>,
    pub related: Vec<String>,
    pub source: NodeSource,
    pub importance: f32,
}

#[derive(Debug, Clone, Default)]
pub enum RetrievalMode {
    #[default]
    TreeTraversal,
    CollapsedTree,
}

#[derive(Debug, Clone, Default)]
pub enum SearchScope {
    #[default]
    Auto,
    Local,
    Global,
}

#[derive(Debug)]
pub struct KnowledgeQuery<'a> {
    pub text: &'a str,
    pub tags: &'a [String],
    pub explicit_ids: &'a [String],
    pub top_k: usize,
    pub depths: &'a [NodeDepth],
    pub mode: RetrievalMode,
    pub scope: SearchScope,
    pub expand_hops: u8,
}

impl<'a> KnowledgeQuery<'a> {
    pub fn all_depths(text: &'a str) -> Self {
        static ALL: &[NodeDepth] = &[NodeDepth::Global, NodeDepth::Topic, NodeDepth::Leaf];
        Self {
            text,
            tags: &[],
            explicit_ids: &[],
            top_k: 10,
            depths: ALL,
            mode: RetrievalMode::TreeTraversal,
            scope: SearchScope::Auto,
            expand_hops: 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SurfacedTension {
    pub domain_a: String,
    pub domain_b: String,
    pub reason: String,
}

#[derive(Debug)]
pub struct KnowledgeResult {
    pub nodes: Vec<(KnowledgeNode, f32)>,
    pub global_included: bool,
    pub surfaced_tensions: Vec<SurfacedTension>,
    pub ppr_expanded: bool,
}
