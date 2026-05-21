use crate::a2a::A2aExplorerAdapter;
use crate::anthropic::AnthropicAdapter;
use crate::cloud::CloudGenericAdapter;
use crate::ollama::OllamaAdapter;
use crate::openai::OpenAIAdapter;
use h2ai_config::AdapterProfile;
use h2ai_types::adapter::IComputeAdapter;
use h2ai_types::config::AdapterKind;
use std::sync::Arc;

pub struct AdapterFactory;

impl AdapterFactory {
    /// Build a compute adapter from an `AdapterKind`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the adapter cannot be constructed (e.g. unsupported variant,
    /// or `A2aExplorerAdapter::new` fails).
    pub fn build(kind: &AdapterKind) -> Result<Arc<dyn IComputeAdapter>, String> {
        Self::build_with_thinking(kind, true)
    }

    /// Build a compute adapter from an `AdapterKind`, controlling the thinking mode flag.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the adapter cannot be constructed (e.g. unsupported variant,
    /// or `A2aExplorerAdapter::new` fails).
    pub fn build_with_thinking(
        kind: &AdapterKind,
        enable_thinking: bool,
    ) -> Result<Arc<dyn IComputeAdapter>, String> {
        match kind {
            AdapterKind::CloudGeneric {
                endpoint,
                api_key_env,
                model,
            } => Ok(Arc::new(CloudGenericAdapter::with_thinking(
                endpoint.clone(),
                api_key_env.clone(),
                model.clone(),
                enable_thinking,
            ))),
            AdapterKind::OpenAI { api_key_env, model } => {
                Ok(Arc::new(OpenAIAdapter::with_thinking(
                    "https://api.openai.com/v1".into(),
                    api_key_env.clone(),
                    model.clone(),
                    enable_thinking,
                )))
            }
            AdapterKind::Anthropic { api_key_env, model } => Ok(Arc::new(AnthropicAdapter::new(
                "https://api.anthropic.com".into(),
                api_key_env.clone(),
                model.clone(),
            ))),
            AdapterKind::Ollama { endpoint, model } => Ok(Arc::new(OllamaAdapter::new(
                endpoint.clone(),
                model.clone(),
            ))),
            AdapterKind::LocalLlamaCpp { .. } => {
                Err("LocalLlamaCpp FFI adapter is not yet wired. \
                 Use AdapterKind::Ollama with a local Ollama server for local inference."
                    .into())
            }
            AdapterKind::A2a {
                endpoint,
                auth_scheme,
                auth_token_env,
                timeout_minutes,
                poll_interval_ms,
                max_poll_interval_ms,
                agent_card_cache_ttl_s,
            } => A2aExplorerAdapter::new(
                endpoint.clone(),
                auth_scheme.clone(),
                auth_token_env.clone(),
                *timeout_minutes,
                *poll_interval_ms,
                *max_poll_interval_ms,
                *agent_card_cache_ttl_s,
            )
            .map(|a| Arc::new(a) as Arc<dyn IComputeAdapter>),
        }
    }

    /// Build an adapter by looking up `name` in `profiles`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if no profile with that name exists or if `build()` fails.
    pub fn build_from_profiles(
        name: &str,
        profiles: &[AdapterProfile],
    ) -> Result<Arc<dyn IComputeAdapter>, String> {
        let profile = profiles
            .iter()
            .find(|p| p.name == name)
            .ok_or_else(|| format!("adapter profile '{name}' not found"))?;
        Self::build(&profile.kind)
    }
}
