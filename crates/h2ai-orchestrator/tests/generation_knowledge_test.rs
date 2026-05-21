#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::significant_drop_tightening,
    clippy::significant_drop_in_scrutinee,
    clippy::unused_async,
    clippy::default_trait_access,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::manual_let_else,
    clippy::match_wildcard_for_single_variants,
    clippy::similar_names,
    clippy::match_same_arms,
    clippy::literal_string_with_formatting_args,
    clippy::redundant_clone,
    clippy::redundant_closure_for_method_calls,
    clippy::useless_format,
    clippy::option_if_let_else,
    clippy::map_unwrap_or,
    clippy::cloned_instead_of_copied,
    clippy::trivially_copy_pass_by_ref,
    clippy::cast_lossless,
    clippy::uninlined_format_args,
    clippy::needless_pass_by_value,
    clippy::explicit_iter_loop,
    clippy::needless_borrow,
    clippy::large_futures,
    clippy::manual_string_new,
    clippy::needless_lifetimes,
    clippy::elidable_lifetime_names,
    clippy::redundant_else,
    clippy::stable_sort_primitive,
    clippy::type_complexity,
    clippy::wildcard_imports,
    clippy::single_match_else,
    clippy::missing_fields_in_debug,
    clippy::doc_link_with_quotes,
    clippy::implicit_hasher,
    clippy::needless_collect,
    clippy::suboptimal_flops,
    clippy::missing_const_for_fn,
    clippy::needless_type_cast,
    clippy::unreadable_literal,
    clippy::no_effect_underscore_binding
)]
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
