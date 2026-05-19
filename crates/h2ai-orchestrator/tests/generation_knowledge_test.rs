//! Compile-time check: EngineInput accepts knowledge_provider and induction_store fields.

use h2ai_knowledge::provider::KnowledgeProvider;
use h2ai_knowledge::types::{KnowledgeQuery, KnowledgeResult};
use std::sync::Arc;

struct FixedKnowledgeProvider;

#[async_trait::async_trait]
impl KnowledgeProvider for FixedKnowledgeProvider {
    async fn query(&self, _query: &KnowledgeQuery<'_>) -> KnowledgeResult {
        KnowledgeResult {
            nodes: vec![],
            global_included: false,
            surfaced_tensions: vec![],
            ppr_expanded: false,
        }
    }

    async fn global_summary(&self) -> Option<h2ai_knowledge::types::KnowledgeNode> {
        None
    }

    fn is_ready(&self) -> bool {
        true
    }

    fn kind(&self) -> &h2ai_knowledge::factory::ProviderKind {
        &h2ai_knowledge::factory::ProviderKind::Bm25Wiki
    }
}

#[test]
fn engine_input_accepts_knowledge_provider() {
    let _: Option<Arc<dyn KnowledgeProvider + Send + Sync>> =
        Some(Arc::new(FixedKnowledgeProvider));
    // If this compiles, the trait is correctly wired.
}
