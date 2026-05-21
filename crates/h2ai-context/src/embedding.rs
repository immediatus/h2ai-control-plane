#[cfg(feature = "fastembed-embed")]
use std::sync::Mutex;

/// A text embedding model that maps strings to dense float vectors.
///
/// Implement this trait to plug in any embedding model (sentence-transformers,
/// ONNX Runtime, remote API, etc.). The returned vector must be L2-normalised
/// for `semantic_jaccard` to give cosine similarity correctly; if your model
/// returns un-normalised vectors, normalise them in your `embed` implementation.
///
/// # Fallback behaviour
///
/// All functions that accept `Option<&dyn EmbeddingModel>` fall back to
/// exact-string equality when `None` is passed — 1.0 for identical strings,
/// 0.0 for all others.
pub trait EmbeddingModel: Send + Sync {
    /// Embed `text` into a dense float vector.
    ///
    /// The vector dimension must be consistent across all calls on the same
    /// model instance. Returning an empty vec is treated as "no embedding
    /// available" and triggers the exact-equality fallback.
    fn embed(&self, text: &str) -> Vec<f32>;
}

/// Compute cosine similarity between two embedding vectors.
///
/// Uses the dot-product formula `Σ aᵢbᵢ / (‖a‖ · ‖b‖)`.  For L2-normalised
/// unit vectors (as returned by [`EmbeddingModel::embed`]) this reduces to the
/// plain dot product `Σ aᵢbᵢ`.  Returns 0.0 for empty, zero-norm, or
/// mismatched-dimension inputs.  Result is clamped to [-1, 1] to guard against
/// floating-point rounding; [`semantic_jaccard`] further clamps to [0, 1].
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-9 || norm_b < 1e-9 {
        return 0.0;
    }
    f64::from((dot / (norm_a * norm_b)).clamp(-1.0, 1.0))
}

/// Semantic similarity between two text strings in [0, 1].
///
/// - **With model**: cosine similarity between embedding vectors. Catches
///   semantic equivalents that token overlap misses ("redis" ≈ "key-value store").
/// - **Without model** (`None`): exact-string equality — 1.0 for identical strings,
///   0.0 for all others. Token Jaccard was removed because it is meaningless for
///   LLM outputs; irrelevant tasks surface as verification failures downstream.
///
/// The return value is in [0, 1] in both modes.
#[must_use]
pub fn semantic_jaccard(a: &str, b: &str, model: Option<&dyn EmbeddingModel>) -> f64 {
    model.map_or_else(
        || if a == b { 1.0 } else { 0.0 },
        |m| {
            let va = m.embed(a);
            let vb = m.embed(b);
            if va.is_empty() || vb.is_empty() {
                // Model returned no vector — fall back to exact equality.
                if a == b {
                    1.0
                } else {
                    0.0
                }
            } else {
                // Cosine similarity ∈ [-1, 1]; clamp to [0, 1] — negative cosine
                // means opposite directions which we treat as zero similarity.
                cosine_similarity(&va, &vb).max(0.0)
            }
        },
    )
}

/// Concrete [`EmbeddingModel`] backed by fastembed-rs and ONNX Runtime.
///
/// Wraps `fastembed::TextEmbedding` behind a `Mutex` so it can be shared across async tasks.
/// Model weights are downloaded on first construction and cached to `~/.cache/fastembed/`.
/// A single warmup embed is performed at construction time so that the first calibration
/// request pays no ONNX cold-start cost.  All returned vectors are L2-normalised, meaning
/// cosine similarity equals the plain dot product.
///
/// Requires the `fastembed-embed` Cargo feature and a compatible ONNX Runtime on the host.
/// Build with: `cargo build -p h2ai-context --features fastembed-embed`.
#[cfg(feature = "fastembed-embed")]
pub struct FastEmbedModel {
    inner: Mutex<fastembed::TextEmbedding>,
}

#[cfg(feature = "fastembed-embed")]
impl FastEmbedModel {
    /// Construct using the default model (`all-MiniLM-L6-v2`, 22 MB download).
    ///
    /// Suitable for development and deployments that do not specify an embedding model
    /// in `H2AIConfig`.  Prefer [`new_with`][Self::new_with] in production so the
    /// operator-configured model name is honoured.
    ///
    /// # Errors
    ///
    /// Returns an error if the fastembed model cannot be initialised (e.g. download failure,
    /// ONNX runtime error).
    pub fn new() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Self::new_with_fastembed_model(fastembed::EmbeddingModel::AllMiniLML6V2)
    }

    /// Construct using the model name from `H2AIConfig::embedding_model_name`.
    ///
    /// Translates the config-level [`EmbeddingModelName`][h2ai_config::EmbeddingModelName]
    /// variant to the corresponding `fastembed::EmbeddingModel`, then delegates to the
    /// same initialisation path as [`new`][Self::new].  Use this constructor in `main.rs`
    /// so operators can switch models without recompiling.
    ///
    /// # Errors
    ///
    /// Returns an error if the fastembed model cannot be initialised (e.g. download failure,
    /// ONNX runtime error).
    pub fn new_with(
        name: &h2ai_config::EmbeddingModelName,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        use h2ai_config::EmbeddingModelName;
        let fastembed_model = match name {
            EmbeddingModelName::AllMiniLmL6V2 => fastembed::EmbeddingModel::AllMiniLML6V2,
            EmbeddingModelName::BgeSmallEnV1_5 => fastembed::EmbeddingModel::BGESmallENV15,
        };
        Self::new_with_fastembed_model(fastembed_model)
    }

    fn new_with_fastembed_model(
        model: fastembed::EmbeddingModel,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let opts = fastembed::InitOptions {
            model_name: model,
            show_download_progress: true,
            ..Default::default()
        };
        let inner = fastembed::TextEmbedding::try_new(opts)?;
        let instance = Self {
            inner: Mutex::new(inner),
        };
        // Warmup: force ONNX model load now so the first calibration request pays no cold-start cost.
        let _ = instance.embed("warmup");
        Ok(instance)
    }
}

#[cfg(feature = "fastembed-embed")]
impl EmbeddingModel for FastEmbedModel {
    fn embed(&self, text: &str) -> Vec<f32> {
        let model = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match model.embed(vec![text], None) {
            Ok(mut embeddings) => {
                if embeddings.is_empty() {
                    return vec![];
                }
                l2_normalize(embeddings.remove(0))
            }
            Err(_) => vec![],
        }
    }
}

#[cfg(feature = "fastembed-embed")]
fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}
