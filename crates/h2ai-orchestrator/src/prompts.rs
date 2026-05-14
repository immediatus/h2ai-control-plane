use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

struct PromptCacheEntry {
    text: String,
    fetched_at: Instant,
}

struct PromptCache {
    entries: HashMap<String, PromptCacheEntry>, // key = "{adapter_name}/{prompt_key}"
}

impl PromptCache {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    fn get(&self, key: &str, ttl: Duration) -> Option<&str> {
        self.entries.get(key).and_then(|e| {
            if e.fetched_at.elapsed() < ttl {
                Some(e.text.as_str())
            } else {
                None
            }
        })
    }

    fn insert(&mut self, key: String, text: String) {
        self.entries.insert(
            key,
            PromptCacheEntry {
                text,
                fetched_at: Instant::now(),
            },
        );
    }
}

static PROMPT_CACHE: OnceLock<Mutex<PromptCache>> = OnceLock::new();

fn cache() -> &'static Mutex<PromptCache> {
    PROMPT_CACHE.get_or_init(|| Mutex::new(PromptCache::new()))
}

/// Resolve a prompt for an adapter, checking NATS for an active variant first.
/// Falls back to `default_text` if no variant is active or NATS is unavailable.
/// Uses a 30-second in-memory cache to avoid NATS roundtrips on every call.
pub async fn resolve_prompt(
    adapter_name: &str,
    prompt_key: &str,
    default_text: &str,
    nats: Option<&h2ai_state::NatsClient>,
) -> String {
    let cache_key = format!("{}/{}", adapter_name, prompt_key);
    let ttl = Duration::from_secs(30);

    // Cache hit
    {
        let cache = cache().lock().unwrap();
        if let Some(text) = cache.get(&cache_key, ttl) {
            return text.to_string();
        }
    }

    // Cache miss — try NATS
    if let Some(nats_client) = nats {
        if let Ok(Some(variant_id)) = nats_client
            .get_active_variant_ptr(adapter_name, prompt_key)
            .await
        {
            if let Ok(Some(variant)) = nats_client
                .get_prompt_variant(adapter_name, prompt_key, &variant_id)
                .await
            {
                let text = variant.text.clone();
                cache().lock().unwrap().insert(cache_key, text.clone());
                return text;
            }
        }
    }

    // Fallback to default (from config)
    default_text.to_string()
}
