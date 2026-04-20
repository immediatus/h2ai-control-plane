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
    pub fn build(kind: &AdapterKind) -> Result<Arc<dyn IComputeAdapter>, String> {
        match kind {
            AdapterKind::CloudGeneric {
                endpoint,
                api_key_env,
            } => Ok(Arc::new(CloudGenericAdapter::new(
                endpoint.clone(),
                api_key_env.clone(),
            ))),
            AdapterKind::OpenAI { api_key_env, model } => Ok(Arc::new(OpenAIAdapter::new(
                "https://api.openai.com/v1".into(),
                api_key_env.clone(),
                model.clone(),
            ))),
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
        }
    }

    /// Build an adapter by looking up `name` in `profiles`.
    ///
    /// Returns `Err` if no profile with that name exists or if `build()` fails
    /// for the matched profile's kind.
    pub fn build_from_profiles(
        name: &str,
        profiles: &[AdapterProfile],
    ) -> Result<Arc<dyn IComputeAdapter>, String> {
        let profile = profiles
            .iter()
            .find(|p| p.name == name)
            .ok_or_else(|| format!("adapter profile '{}' not found", name))?;
        Self::build(&profile.kind)
    }
}
