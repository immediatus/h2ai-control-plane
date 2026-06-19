use h2ai_types::memory::{MemoryTier, RetryHintPattern, TenantMemoryStore};

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

// ── GAP-G1: RetryHintPattern and TenantMemoryStore ───────────────────────────

#[test]
fn retry_hint_pattern_success_rate_beta_prior() {
    // Beta(2,8) prior: (success + 2) / (attempt + 10)
    // With 0 successes, 0 attempts: (0+2)/(0+10) = 0.2
    let p = RetryHintPattern {
        trigger_tags: vec!["billing".to_string()],
        exit_reason_kind: "ZeroSurvival".to_string(),
        hint_text: "use append-only".to_string(),
        success_count: 0,
        attempt_count: 0,
    };
    let rate = p.success_rate();
    assert!(
        (rate - 0.2).abs() < 1e-9,
        "prior-only rate must be 0.2, got {rate}"
    );
}

#[test]
fn retry_hint_pattern_success_rate_with_observations() {
    // 2 successes, 5 attempts: (2+2)/(5+10) = 4/15 ≈ 0.2667
    let p = RetryHintPattern {
        trigger_tags: vec!["billing".to_string()],
        exit_reason_kind: "ZeroSurvival".to_string(),
        hint_text: "use append-only".to_string(),
        success_count: 2,
        attempt_count: 5,
    };
    let rate = p.success_rate();
    let expected = 4.0 / 15.0;
    assert!(
        (rate - expected).abs() < 1e-9,
        "rate must be {expected}, got {rate}"
    );
}

#[test]
fn retry_hint_pattern_merge_counts_adds_g_counters() {
    let mut a = RetryHintPattern {
        trigger_tags: vec!["billing".to_string()],
        exit_reason_kind: "ZeroSurvival".to_string(),
        hint_text: "use append-only".to_string(),
        success_count: 3,
        attempt_count: 7,
    };
    let b = RetryHintPattern {
        trigger_tags: vec!["billing".to_string()],
        exit_reason_kind: "ZeroSurvival".to_string(),
        hint_text: "use append-only".to_string(),
        success_count: 1,
        attempt_count: 2,
    };
    a.merge_counts(&b);
    assert_eq!(a.success_count, 4);
    assert_eq!(a.attempt_count, 9);
}

#[test]
fn retry_hint_pattern_merge_is_commutative() {
    let base = RetryHintPattern {
        trigger_tags: vec![],
        exit_reason_kind: "ZeroSurvival".to_string(),
        hint_text: "hint".to_string(),
        success_count: 3,
        attempt_count: 7,
    };
    let delta = RetryHintPattern {
        trigger_tags: vec![],
        exit_reason_kind: "ZeroSurvival".to_string(),
        hint_text: "hint".to_string(),
        success_count: 1,
        attempt_count: 2,
    };
    // a.merge(b) vs b.merge(a) — both should have same counts
    let mut ab = base.clone();
    ab.merge_counts(&delta);
    let mut ba = delta.clone();
    ba.merge_counts(&base);
    assert_eq!(ab.success_count, ba.success_count);
    assert_eq!(ab.attempt_count, ba.attempt_count);
}

#[test]
fn tenant_memory_store_roundtrips_json() {
    use chrono::Utc;
    let store = TenantMemoryStore {
        tenant_id: "t1".to_string(),
        generated_at: Utc::now(),
        task_count_seen: 5,
        retry_hint_patterns: vec![RetryHintPattern {
            trigger_tags: vec!["billing".to_string()],
            exit_reason_kind: "ZeroSurvival".to_string(),
            hint_text: "use append-only".to_string(),
            success_count: 2,
            attempt_count: 5,
        }],
    };
    let json = serde_json::to_string(&store).unwrap();
    let back: TenantMemoryStore = serde_json::from_str(&json).unwrap();
    assert_eq!(back.tenant_id, "t1");
    assert_eq!(back.retry_hint_patterns.len(), 1);
    assert_eq!(back.retry_hint_patterns[0].success_count, 2);
}

// ── TagPatternBucket ─────────────────────────────────────────────────────────

#[test]
fn tag_pattern_bucket_roundtrip() {
    use h2ai_types::memory::{RetryHintPattern, TagPatternBucket};

    let bucket = TagPatternBucket {
        patterns: vec![RetryHintPattern {
            trigger_tags: vec!["http".to_string(), "timeout".to_string()],
            exit_reason_kind: "ZeroSurvival".to_string(),
            hint_text: "use idempotent retry".to_string(),
            success_count: 3,
            attempt_count: 5,
        }],
    };
    let json = serde_json::to_vec(&bucket).expect("serialize");
    let back: TagPatternBucket = serde_json::from_slice(&json).expect("deserialize");
    assert_eq!(back.patterns.len(), 1);
    assert_eq!(back.patterns[0].hint_text, "use idempotent retry");
}

#[test]
fn tag_pattern_bucket_default_is_empty() {
    use h2ai_types::memory::TagPatternBucket;
    let b = TagPatternBucket::default();
    assert!(b.patterns.is_empty());
}
