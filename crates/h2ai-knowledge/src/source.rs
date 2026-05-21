use std::path::PathBuf;

use async_trait::async_trait;
use serde::Deserialize;

use crate::types::{CrossRef, KnowledgeNode, NodeDepth, NodeSource, TensionRef};

// ---------------------------------------------------------------------------
// Public output type
// ---------------------------------------------------------------------------

/// A flat, scored-ready item built from a single constraint document.
#[derive(Debug, Clone)]
pub struct SourceItem {
    pub id: String,
    pub summary: String,
    pub domains: Vec<String>,
    pub tags: Vec<String>,
    pub related: Vec<String>,
    pub cross_refs: Vec<CrossRef>,
    pub source_ref: String,
}

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

#[async_trait]
pub trait KnowledgeSource: Send + Sync {
    /// All leaf-level constraint items.
    async fn all_items(&self) -> Vec<SourceItem>;

    /// Topic-depth wiki nodes (one per non-overview wiki YAML file).
    async fn wiki_nodes(&self) -> Vec<KnowledgeNode>;

    /// The single global-depth overview node, if present.
    async fn global_node(&self) -> Option<KnowledgeNode>;
}

// ---------------------------------------------------------------------------
// YamlDirSource
// ---------------------------------------------------------------------------

pub struct YamlDirSource {
    corpus_dir: PathBuf,
}

impl YamlDirSource {
    pub fn new(corpus_dir: impl Into<PathBuf>) -> Self {
        Self {
            corpus_dir: corpus_dir.into(),
        }
    }
}

#[async_trait]
impl KnowledgeSource for YamlDirSource {
    async fn all_items(&self) -> Vec<SourceItem> {
        match h2ai_constraints::loader::load_corpus(&self.corpus_dir) {
            Ok(docs) => docs
                .into_iter()
                .map(|doc| {
                    let cross_refs = doc
                        .related_to
                        .iter()
                        .map(|id| CrossRef {
                            id: id.clone(),
                            domain: String::new(),
                            reason: String::new(),
                        })
                        .collect();
                    SourceItem {
                        summary: format!("{}: {}", doc.id, doc.description),
                        id: doc.id,
                        domains: doc.domains,
                        tags: doc.mandatory_for_tags,
                        related: doc.related_to,
                        cross_refs,
                        source_ref: doc.source_file,
                    }
                })
                .collect(),
            Err(e) => {
                tracing::warn!(
                    corpus_dir = %self.corpus_dir.display(),
                    error = %e,
                    "failed to load constraint corpus; returning empty"
                );
                vec![]
            }
        }
    }

    async fn wiki_nodes(&self) -> Vec<KnowledgeNode> {
        let wiki_dir = self.corpus_dir.join("wiki");
        if !wiki_dir.exists() {
            return vec![];
        }

        let mut entries = match std::fs::read_dir(&wiki_dir) {
            Ok(e) => e.filter_map(std::result::Result::ok).collect::<Vec<_>>(),
            Err(e) => {
                tracing::warn!(
                    wiki_dir = %wiki_dir.display(),
                    error = %e,
                    "failed to read wiki directory"
                );
                return vec![];
            }
        };
        entries.sort_by_key(std::fs::DirEntry::file_name);

        let mut nodes = Vec::new();
        for entry in entries {
            let path = entry.path();
            let ext = path.extension().and_then(|e| e.to_str());
            if ext != Some("yaml") && ext != Some("yml") {
                continue;
            }
            // Skip _overview.yaml/.yml — handled by global_node()
            let fname = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if fname == "_overview.yaml" || fname == "_overview.yml" {
                continue;
            }

            let content = match std::fs::read_to_string(&path) {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!(path = %path.display(), error = %e, "failed to read wiki node file");
                    continue;
                }
            };

            let yaml: WikiNodeYaml = match serde_yaml::from_str(&content) {
                Ok(y) => y,
                Err(e) => {
                    tracing::warn!(path = %path.display(), error = %e, "failed to parse wiki node YAML");
                    continue;
                }
            };

            // Derive ID: use yaml.id if non-empty, else derive from stem
            let id = if yaml.id.is_empty() {
                let stem = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown");
                format!("topic:{stem}")
            } else {
                yaml.id.clone()
            };

            let tensions = yaml
                .tensions
                .iter()
                .map(|t| TensionRef {
                    domain: t.domain.clone(),
                    reason: t.reason.clone(),
                })
                .collect();

            let cross_references = yaml
                .cross_references
                .iter()
                .map(|c| CrossRef {
                    id: c.id.clone(),
                    domain: c.domain.clone(),
                    reason: c.reason.clone(),
                })
                .collect();

            nodes.push(KnowledgeNode {
                id,
                depth: NodeDepth::Topic,
                synthesis: yaml.synthesis,
                invariants: yaml.invariants,
                failure_modes: yaml.failure_modes_covered,
                domains: yaml.domains,
                entry_points: yaml.entry_points,
                tensions,
                cross_references,
                related: yaml.related,
                source: NodeSource::WikiYaml {
                    path: path.to_string_lossy().to_string(),
                },
                importance: 0.8,
            });
        }
        nodes
    }

    async fn global_node(&self) -> Option<KnowledgeNode> {
        let overview_path = self.corpus_dir.join("wiki/_overview.yaml");
        if !overview_path.exists() {
            return None;
        }

        let content = match std::fs::read_to_string(&overview_path) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(path = %overview_path.display(), error = %e, "failed to read _overview.yaml");
                return None;
            }
        };

        let yaml: OverviewYaml = match serde_yaml::from_str(&content) {
            Ok(y) => y,
            Err(e) => {
                tracing::warn!(path = %overview_path.display(), error = %e, "failed to parse _overview.yaml");
                return None;
            }
        };

        Some(KnowledgeNode {
            id: "global:overview".to_string(),
            depth: NodeDepth::Global,
            synthesis: yaml.synthesis,
            invariants: yaml.key_invariants,
            failure_modes: vec![],
            domains: yaml.domains_covered,
            entry_points: vec![],
            tensions: vec![],
            cross_references: vec![],
            related: vec![],
            source: NodeSource::WikiYaml {
                path: "_overview.yaml".to_string(),
            },
            importance: 1.0,
        })
    }
}

// ---------------------------------------------------------------------------
// Private deserialization structs
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct WikiNodeYaml {
    #[serde(default)]
    id: String,
    #[serde(default)]
    synthesis: String,
    #[serde(default)]
    invariants: Vec<String>,
    #[serde(default)]
    failure_modes_covered: Vec<String>,
    #[serde(default)]
    domains: Vec<String>,
    #[serde(default)]
    entry_points: Vec<String>,
    #[serde(default)]
    tensions: Vec<TensionRefYaml>,
    #[serde(default)]
    cross_references: Vec<CrossRefYaml>,
    #[serde(default)]
    related: Vec<String>,
}

#[derive(Deserialize)]
struct TensionRefYaml {
    domain: String,
    #[serde(default)]
    reason: String,
}

#[derive(Deserialize)]
struct CrossRefYaml {
    id: String,
    #[serde(default)]
    domain: String,
    #[serde(default)]
    reason: String,
}

#[derive(Deserialize)]
struct OverviewYaml {
    #[serde(default)]
    synthesis: String,
    #[serde(default)]
    key_invariants: Vec<String>,
    #[serde(default)]
    domains_covered: Vec<String>,
}
