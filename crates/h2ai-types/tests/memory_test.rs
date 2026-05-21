use h2ai_types::memory::MemoryTier;

// ── rho() values match spec ───────────────────────────────────────────────────

#[test]
#[allow(clippy::float_cmp)]
fn memory_tier_rho_values() {
    assert_eq!(MemoryTier::Working.rho(), 0.08);
    assert_eq!(MemoryTier::Episodic.rho(), 0.20);
    assert_eq!(MemoryTier::Semantic.rho(), 0.35);
    assert_eq!(MemoryTier::Procedural.rho(), 0.50);
}

// ── decay_halflife_secs values match spec ─────────────────────────────────────

#[test]
fn memory_tier_decay_halflife_working() {
    assert_eq!(MemoryTier::Working.decay_halflife_secs(), 3_600);
}

#[test]
fn memory_tier_decay_halflife_episodic() {
    assert_eq!(MemoryTier::Episodic.decay_halflife_secs(), 86_400);
}

#[test]
fn memory_tier_decay_halflife_semantic() {
    assert_eq!(MemoryTier::Semantic.decay_halflife_secs(), 604_800);
}

#[test]
fn memory_tier_decay_halflife_procedural() {
    assert_eq!(MemoryTier::Procedural.decay_halflife_secs(), 2_592_000);
}

// ── n_it_optimal returns positive sizes and follows rho ordering ──────────────

#[test]
fn memory_tier_n_it_optimal_ordering() {
    // Higher rho → fewer agents needed
    assert!(
        MemoryTier::Working.n_it_optimal() >= MemoryTier::Episodic.n_it_optimal(),
        "Working needs >= Episodic agents"
    );
    assert!(
        MemoryTier::Episodic.n_it_optimal() >= MemoryTier::Semantic.n_it_optimal(),
        "Episodic needs >= Semantic agents"
    );
    assert!(
        MemoryTier::Semantic.n_it_optimal() >= MemoryTier::Procedural.n_it_optimal(),
        "Semantic needs >= Procedural agents"
    );
}

#[test]
fn memory_tier_n_it_optimal_positive() {
    for tier in [
        MemoryTier::Working,
        MemoryTier::Episodic,
        MemoryTier::Semantic,
        MemoryTier::Procedural,
    ] {
        assert!(
            tier.n_it_optimal() >= 1,
            "n_it_optimal must be at least 1 for {tier:?}"
        );
    }
}

// ── Ord: tier ordering matches discriminant values ────────────────────────────

#[test]
fn memory_tier_ordering() {
    assert!(MemoryTier::Working < MemoryTier::Episodic);
    assert!(MemoryTier::Episodic < MemoryTier::Semantic);
    assert!(MemoryTier::Semantic < MemoryTier::Procedural);
}

// ── Clone / Copy / PartialEq / Eq ────────────────────────────────────────────

#[test]
fn memory_tier_clone_and_eq() {
    let a = MemoryTier::Episodic;
    let b = a;
    assert_eq!(a, b);
    assert_ne!(MemoryTier::Working, MemoryTier::Procedural);
}

#[test]
fn memory_tier_debug() {
    assert!(format!("{:?}", MemoryTier::Working).contains("Working"));
    assert!(format!("{:?}", MemoryTier::Procedural).contains("Procedural"));
}

// ── Serde round-trip ──────────────────────────────────────────────────────────

#[test]
fn memory_tier_serde_round_trip() {
    for tier in [
        MemoryTier::Working,
        MemoryTier::Episodic,
        MemoryTier::Semantic,
        MemoryTier::Procedural,
    ] {
        let json = serde_json::to_string(&tier).unwrap();
        let back: MemoryTier = serde_json::from_str(&json).unwrap();
        assert_eq!(tier, back);
    }
}
