//! Concrete [`h2ai_types::adapter::IComputeAdapter`] implementations.
//!
//! - [`cloud::CloudGenericAdapter`] — OpenAI-compatible HTTP endpoint (no model field)
//! - [`openai::OpenAIAdapter`] — `OpenAI` Chat Completions with model selection
//! - [`anthropic::AnthropicAdapter`] — Anthropic Messages API
//! - [`ollama::OllamaAdapter`] — Ollama native `/api/chat`
//! - [`factory`] — builds any adapter from `AdapterKind` config

pub mod a2a;
pub mod anthropic;
pub mod cloud;
pub mod factory;
pub mod ollama;
pub mod openai;
