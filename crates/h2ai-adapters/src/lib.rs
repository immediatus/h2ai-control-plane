//! Concrete [`h2ai_types::adapter::IComputeAdapter`] implementations.
//!
//! - [`a2a::A2aExplorerAdapter`] — generic HTTP adapter with configurable auth scheme
//!   (`Bearer`, `ApiKey`, `None`) and output format (`Text`, `Json`); includes
//!   exponential backoff and proposal extraction helpers.
//! - [`cloud::CloudGenericAdapter`] — OpenAI-compatible HTTP endpoint (no model field)
//! - [`openai::OpenAIAdapter`] — `OpenAI` Chat Completions with model selection
//! - [`anthropic::AnthropicAdapter`] — Anthropic Messages API
//! - [`ollama::OllamaAdapter`] — Ollama native `/api/chat`
//! - [`factory`] — builds any adapter from `AdapterKind` config

pub mod a2a;
pub mod anthropic;
pub mod chain;
pub mod cloud;
pub mod factory;
pub mod ollama;
pub mod openai;
