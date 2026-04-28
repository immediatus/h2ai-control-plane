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
    pub fn new(content: &'a str, tier: MemoryTier, timestamp_secs: u64) -> Self {
        Self { content, tier, timestamp_secs }
    }

    /// Ebbinghaus decay weight at `now_secs`: `e^(-(now - t) / halflife)`.
    ///
    /// Returns 1.0 for chunks with `timestamp_secs >= now_secs` (future timestamps
    /// are treated as age zero via saturating subtraction — never penalised).
    pub fn decay_weight(&self, now_secs: u64) -> f64 {
        let halflife = self.tier.decay_halflife_secs() as f64;
        let age = now_secs.saturating_sub(self.timestamp_secs) as f64;
        (-age / halflife).exp()
    }

    /// Minimum ensemble size for reliable use of this chunk's tier.
    pub fn n_it_optimal(&self) -> usize {
        self.tier.n_it_optimal()
    }
}

/// Recommended ensemble size for a mixed set of chunks: worst-case (most uncertain) tier.
///
/// Use this as the `n_optimal` hint when an agent receives chunks of varying tiers in
/// a single context window — the ensemble must be large enough to handle the least
/// reliable chunk present.
pub fn recommended_ensemble_size(chunks: &[ContextChunk<'_>]) -> usize {
    chunks.iter().map(|c| c.n_it_optimal()).max().unwrap_or(1)
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
        b.tier.cmp(&a.tier).then(wb.partial_cmp(wa).unwrap_or(std::cmp::Ordering::Equal))
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

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: u64 = 1_000_000;

    fn chunk(content: &'static str, tier: MemoryTier, age_secs: u64) -> ContextChunk<'static> {
        ContextChunk::new(content, tier, NOW.saturating_sub(age_secs))
    }

    // ── decay_weight ──────────────────────────────────────────────────────────

    #[test]
    fn decay_weight_at_age_zero_is_one() {
        let c = chunk("x", MemoryTier::Working, 0);
        assert!((c.decay_weight(NOW) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn decay_weight_at_one_time_constant_is_inv_e() {
        // exp(-t/τ) at t=τ → 1/e ≈ 0.368 (not 0.5 — halflife here means time constant)
        let halflife = MemoryTier::Working.decay_halflife_secs();
        let c = chunk("x", MemoryTier::Working, halflife);
        let expected = std::f64::consts::E.recip();
        assert!((c.decay_weight(NOW) - expected).abs() < 1e-9);
    }

    #[test]
    fn decay_weight_future_timestamp_clamped_to_one() {
        let c = ContextChunk::new("x", MemoryTier::Episodic, NOW + 3600);
        assert!((c.decay_weight(NOW) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn procedural_decays_slower_than_working_at_same_age() {
        let age = 3_600u64; // 1 hour
        let w = chunk("x", MemoryTier::Working, age).decay_weight(NOW);
        let p = chunk("x", MemoryTier::Procedural, age).decay_weight(NOW);
        assert!(p > w, "procedural decays slower: p={p:.4} w={w:.4}");
    }

    // ── n_it_optimal on ContextChunk ─────────────────────────────────────────

    #[test]
    fn n_it_optimal_delegates_to_tier() {
        assert_eq!(chunk("x", MemoryTier::Procedural, 0).n_it_optimal(), 2);
        assert_eq!(chunk("x", MemoryTier::Working, 0).n_it_optimal(), 9);
    }

    // ── recommended_ensemble_size ────────────────────────────────────────────

    #[test]
    fn recommended_ensemble_size_empty_is_one() {
        assert_eq!(recommended_ensemble_size(&[]), 1);
    }

    #[test]
    fn recommended_ensemble_size_takes_worst_case_tier() {
        let chunks = vec![
            chunk("rule", MemoryTier::Procedural, 0),
            chunk("obs", MemoryTier::Working, 0),
        ];
        assert_eq!(recommended_ensemble_size(&chunks), 9);
    }

    // ── build_tiered_context ─────────────────────────────────────────────────

    #[test]
    fn build_tiered_context_manifest_always_first() {
        let chunks = vec![chunk("working obs", MemoryTier::Working, 0)];
        let ctx = build_tiered_context("manifest text", &chunks, NOW, 0.0);
        assert!(ctx.starts_with("## Task Manifest"), "manifest must open the context");
    }

    #[test]
    fn build_tiered_context_procedural_before_working() {
        let chunks = vec![
            chunk("working obs", MemoryTier::Working, 0),
            chunk("stable rule", MemoryTier::Procedural, 0),
        ];
        let ctx = build_tiered_context("m", &chunks, NOW, 0.0);
        let proc_pos = ctx.find("## Procedural Memory").unwrap();
        let work_pos = ctx.find("## Working Memory").unwrap();
        assert!(proc_pos < work_pos, "Procedural must precede Working in context");
    }

    #[test]
    fn build_tiered_context_min_weight_filters_stale_chunks() {
        let halflife = MemoryTier::Working.decay_halflife_secs();
        let chunks = vec![
            chunk("fresh", MemoryTier::Working, 0),
            chunk("stale", MemoryTier::Working, halflife * 10), // weight ≈ 0.001
        ];
        let ctx = build_tiered_context("m", &chunks, NOW, 0.01);
        assert!(ctx.contains("fresh"));
        assert!(!ctx.contains("stale"), "chunk below min_weight must be excluded");
    }

    #[test]
    fn build_tiered_context_within_tier_recent_first() {
        let chunks = vec![
            chunk("old proc", MemoryTier::Procedural, 86_400),
            chunk("new proc", MemoryTier::Procedural, 0),
        ];
        let ctx = build_tiered_context("m", &chunks, NOW, 0.0);
        let pos_new = ctx.find("new proc").unwrap();
        let pos_old = ctx.find("old proc").unwrap();
        assert!(pos_new < pos_old, "more recent chunk must appear first within tier");
    }
}
