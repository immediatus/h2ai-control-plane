use crate::provider::{Bm25WikiProvider, KnowledgeProvider, PassthroughProvider};
use crate::source::YamlDirSource;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderKind {
    Bm25Wiki,
    Passthrough,
    Skill,
    Composite,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceKind {
    YamlDir { path: PathBuf },
}

/// Tunable scoring parameters for `BM25Wiki` retrieval.
/// All fields have defaults so omitting `[knowledge.scoring]` in TOML is a zero-behaviour-change upgrade.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoringConfig {
    /// Multiplier applied to BM25+ scores for direct leaf hits before boosts.
    #[serde(default = "ScoringConfig::default_leaf_multiplier")]
    pub leaf_score_multiplier: f32,
    /// Score boost added when the constraint ID appears literally in the query text.
    #[serde(default = "ScoringConfig::default_id_boost")]
    pub id_in_query_boost: f32,
    /// Score boost added when a leaf node is listed as an `entry_point` of a matched topic cluster.
    #[serde(default = "ScoringConfig::default_entry_point_boost")]
    pub entry_point_boost: f32,
    /// Multiplier applied to raw PPR probability mass when scoring PPR-expanded nodes.
    #[serde(default = "ScoringConfig::default_ppr_multiplier")]
    pub ppr_score_multiplier: f32,
    /// PPR teleportation probability (restart probability). Standard value: 0.15.
    #[serde(default = "ScoringConfig::default_ppr_alpha")]
    pub ppr_alpha: f32,
    /// Number of PPR power-iteration steps. 20 converges for graphs up to ~1k nodes.
    #[serde(default = "ScoringConfig::default_ppr_max_iter")]
    pub ppr_max_iter: usize,
    /// Maximum number of topic clusters to match in `TreeTraversal` mode.
    #[serde(default = "ScoringConfig::default_topic_cluster_top_k")]
    pub topic_cluster_top_k: usize,
    /// Maximum characters to retain in the synthesized global overview node.
    #[serde(default = "ScoringConfig::default_global_synthesis_max_chars")]
    pub global_synthesis_max_chars: usize,
}

impl ScoringConfig {
    const fn default_leaf_multiplier() -> f32 {
        0.7
    }
    const fn default_id_boost() -> f32 {
        0.15
    }
    const fn default_entry_point_boost() -> f32 {
        0.10
    }
    const fn default_ppr_multiplier() -> f32 {
        0.3
    }
    const fn default_ppr_alpha() -> f32 {
        0.15
    }
    const fn default_ppr_max_iter() -> usize {
        20
    }
    const fn default_topic_cluster_top_k() -> usize {
        3
    }
    const fn default_global_synthesis_max_chars() -> usize {
        600
    }
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            leaf_score_multiplier: Self::default_leaf_multiplier(),
            id_in_query_boost: Self::default_id_boost(),
            entry_point_boost: Self::default_entry_point_boost(),
            ppr_score_multiplier: Self::default_ppr_multiplier(),
            ppr_alpha: Self::default_ppr_alpha(),
            ppr_max_iter: Self::default_ppr_max_iter(),
            topic_cluster_top_k: Self::default_topic_cluster_top_k(),
            global_synthesis_max_chars: Self::default_global_synthesis_max_chars(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KnowledgeConfig {
    pub provider: ProviderKind,
    pub source: SourceKind,
    #[serde(default)]
    pub scoring: ScoringConfig,
}

pub struct KnowledgeProviderFactory;

impl KnowledgeProviderFactory {
    pub async fn build_provider(cfg: &KnowledgeConfig) -> Arc<dyn KnowledgeProvider> {
        match &cfg.provider {
            ProviderKind::Bm25Wiki => {
                let source = Self::build_source(cfg);
                Arc::new(Bm25WikiProvider::build(source, cfg.scoring.clone()).await)
            }
            ProviderKind::Passthrough => match &cfg.source {
                SourceKind::YamlDir { path } => Arc::new(PassthroughProvider::new_from_path(path)),
            },
            ProviderKind::Skill => crate::skill_provider::SkillProvider::new(),
            ProviderKind::Composite => crate::skill_provider::CompositeProvider::new(vec![], false),
        }
    }

    /// Build a `Bm25WikiProvider` from a constraint corpus directory.
    ///
    /// Used when no explicit `[knowledge]` config is present but a `wiki/` subdirectory
    /// exists under the constraint corpus path — so the explorer's knowledge-gathering
    /// cycle can surface domain articles without requiring manually-written hints.
    pub async fn build_from_constraint_corpus(
        corpus_path: &std::path::Path,
    ) -> Arc<dyn KnowledgeProvider> {
        let cfg = KnowledgeConfig {
            provider: ProviderKind::Bm25Wiki,
            source: SourceKind::YamlDir {
                path: corpus_path.to_path_buf(),
            },
            scoring: ScoringConfig::default(),
        };
        Self::build_provider(&cfg).await
    }

    #[must_use]
    pub fn build_source(cfg: &KnowledgeConfig) -> Arc<dyn crate::source::KnowledgeSource> {
        match &cfg.source {
            SourceKind::YamlDir { path } => Arc::new(YamlDirSource::new(path.clone())),
        }
    }
}
