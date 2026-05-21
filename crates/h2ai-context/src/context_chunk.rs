use h2ai_types::memory::MemoryTier;

/// A context chunk tagged with its memory consolidation tier and creation timestamp.
///
/// The tier determines both how quickly the chunk's relevance decays (`decay_weight`)
/// and the minimum ensemble size required to reliably use it (`n_it_optimal`).
/// Procedural chunks (stable rules) need only 2 agents; working-memory chunks
/// (ephemeral observations) need up to 9.
#[derive(Debug, Clone)]
pub struct ContextChunk<'a> {
    pub content: &'a str,
    pub tier: MemoryTier,
    /// Unix timestamp (seconds) when this chunk was recorded.
    pub timestamp_secs: u64,
}

impl<'a> ContextChunk<'a> {
    #[must_use]
    pub const fn new(content: &'a str, tier: MemoryTier, timestamp_secs: u64) -> Self {
        Self {
            content,
            tier,
            timestamp_secs,
        }
    }

    /// Ebbinghaus decay weight at `now_secs`: `e^(-(now - t) / halflife)`.
    ///
    /// Returns 1.0 for chunks with `timestamp_secs >= now_secs` (future timestamps
    /// are treated as age zero via saturating subtraction — never penalised).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn decay_weight(&self, now_secs: u64) -> f64 {
        let halflife = self.tier.decay_halflife_secs() as f64;
        let age = now_secs.saturating_sub(self.timestamp_secs) as f64;
        (-age / halflife).exp()
    }

    /// Minimum ensemble size for reliable use of this chunk's tier.
    #[must_use]
    pub fn n_it_optimal(&self) -> usize {
        self.tier.n_it_optimal()
    }
}

/// Recommended ensemble size for a mixed set of chunks: worst-case (most uncertain) tier.
///
/// Use this as the `n_optimal` hint when an agent receives chunks of varying tiers in
/// a single context window — the ensemble must be large enough to handle the least
/// reliable chunk present.
#[must_use]
pub fn recommended_ensemble_size(chunks: &[ContextChunk<'_>]) -> usize {
    chunks
        .iter()
        .map(ContextChunk::n_it_optimal)
        .max()
        .unwrap_or(1)
}

/// Build an ordered context string from tiered chunks for LLM injection.
///
/// Ordering strategy:
/// 1. Chunks sorted by tier stability descending (Procedural first, Working last) so
///    that reliable constraints anchor the prompt head — mitigates lost-in-middle risk.
/// 2. Within each tier, sorted by decay weight descending (most recent first).
/// 3. Each chunk is prefixed with a `## [Tier] Memory` section header so the LLM
///    can reason about the reliability of each block.
///
/// The `manifest` is prepended as the first section. Chunks below `min_weight` are
/// excluded — pass `0.0` to include everything.
#[must_use]
pub fn build_tiered_context(
    manifest: &str,
    chunks: &[ContextChunk<'_>],
    now_secs: u64,
    min_weight: f64,
) -> String {
    let mut ordered: Vec<(&ContextChunk<'_>, f64)> = chunks
        .iter()
        .map(|c| (c, c.decay_weight(now_secs)))
        .filter(|(_, w)| *w >= min_weight)
        .collect();

    // Primary: higher tier first (Procedural=3 > Working=0). Secondary: higher weight first.
    ordered.sort_by(|(a, wa), (b, wb)| {
        b.tier
            .cmp(&a.tier)
            .then(wb.partial_cmp(wa).unwrap_or(std::cmp::Ordering::Equal))
    });

    let mut parts = vec![format!("## Task Manifest\n{manifest}")];
    for (chunk, _) in &ordered {
        let header = match chunk.tier {
            MemoryTier::Procedural => "## Procedural Memory",
            MemoryTier::Semantic => "## Semantic Memory",
            MemoryTier::Episodic => "## Episodic Memory",
            MemoryTier::Working => "## Working Memory",
        };
        parts.push(format!("{header}\n{}", chunk.content));
    }
    parts.join("\n\n")
}
