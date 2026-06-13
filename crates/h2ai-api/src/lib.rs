#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_lossless,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::significant_drop_tightening,
    clippy::significant_drop_in_scrutinee,
    clippy::unused_async,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::option_if_let_else,
    clippy::manual_let_else,
    clippy::match_wildcard_for_single_variants,
    clippy::similar_names,
    clippy::match_same_arms,
    clippy::needless_pass_by_value,
    clippy::doc_markdown,
    clippy::doc_link_with_quotes,
    clippy::implicit_hasher,
    clippy::literal_string_with_formatting_args,
    clippy::missing_fields_in_debug,
    clippy::redundant_else,
    clippy::map_unwrap_or,
    clippy::type_complexity,
    clippy::float_cmp,
    clippy::large_futures,
    clippy::suboptimal_flops,
    clippy::needless_type_cast,
    clippy::elidable_lifetime_names
)]
pub mod bootstrap;
pub mod debug_record;
pub mod error;
pub mod metrics;
pub mod opro;
pub mod oracle;
pub mod oracle_worker;
pub mod recovery;
pub mod rho_ema;
pub mod routes;
pub mod shadow_auditor;
pub mod state;
pub mod task_pipeline;
pub mod tenant_registry;
