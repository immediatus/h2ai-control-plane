use h2ai_context::context_chunk::{build_tiered_context, recommended_ensemble_size, ContextChunk};
use h2ai_types::memory::MemoryTier;

const NOW: u64 = 1_000_000;

const fn chunk(content: &'static str, tier: MemoryTier, age_secs: u64) -> ContextChunk<'static> {
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

// ── n_it_optimal for Semantic and Episodic tiers ─────────────────────────

#[test]
fn n_it_optimal_semantic_tier() {
    assert!(chunk("x", MemoryTier::Semantic, 0).n_it_optimal() > 0);
}

#[test]
fn n_it_optimal_episodic_tier() {
    assert!(chunk("x", MemoryTier::Episodic, 0).n_it_optimal() > 0);
}

// ── build_tiered_context ─────────────────────────────────────────────────

#[test]
fn build_tiered_context_manifest_always_first() {
    let chunks = vec![chunk("working obs", MemoryTier::Working, 0)];
    let ctx = build_tiered_context("manifest text", &chunks, NOW, 0.0);
    assert!(
        ctx.starts_with("## Task Manifest"),
        "manifest must open the context"
    );
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
    assert!(
        proc_pos < work_pos,
        "Procedural must precede Working in context"
    );
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
    assert!(
        !ctx.contains("stale"),
        "chunk below min_weight must be excluded"
    );
}

#[test]
fn build_tiered_context_semantic_header_present() {
    let chunks = vec![chunk("semantic fact", MemoryTier::Semantic, 0)];
    let ctx = build_tiered_context("m", &chunks, NOW, 0.0);
    assert!(
        ctx.contains("## Semantic Memory"),
        "Semantic tier must produce its header"
    );
}

#[test]
fn build_tiered_context_episodic_header_present() {
    let chunks = vec![chunk("episodic event", MemoryTier::Episodic, 0)];
    let ctx = build_tiered_context("m", &chunks, NOW, 0.0);
    assert!(
        ctx.contains("## Episodic Memory"),
        "Episodic tier must produce its header"
    );
}

#[test]
fn build_tiered_context_empty_chunks_returns_manifest_only() {
    let ctx = build_tiered_context("only manifest", &[], NOW, 0.0);
    assert_eq!(ctx, "## Task Manifest\nonly manifest");
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
    assert!(
        pos_new < pos_old,
        "more recent chunk must appear first within tier"
    );
}
