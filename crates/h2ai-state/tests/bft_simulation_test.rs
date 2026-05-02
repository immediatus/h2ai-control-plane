//! Simulation tests proving BFT properties.
//!
//! These tests are deterministic (no external randomness dependency) — they
//! enumerate a grid of configurations and verify invariants hold in all cases.
//!
//! ## What is proved
//!
//! 1. `krum_never_selects_byzantine_in_any_simulated_scenario` — for every
//!    (n, f, byzantine_text) configuration satisfying n ≥ 2f+3, Krum always
//!    returns an honest proposal.
//!
//! 2. `condorcet_vulnerable_when_byzantine_form_majority` — with f ≥ n/2
//!    Byzantine proposals presenting identical adversarial text, ConsensusMedian
//!    can select the Byzantine output.

use chrono::Utc;
use h2ai_context::jaccard::{jaccard, tokenize};
use h2ai_state::bft::ConsensusMedian;
use h2ai_state::krum::{krum_index, krum_score_subset, quorum_satisfied};
use h2ai_types::config::AdapterKind;
use h2ai_types::events::ProposalEvent;
use h2ai_types::identity::{ExplorerId, TaskId};
use h2ai_types::physics::TauValue;
use std::collections::HashSet;

fn build_token_sets(proposals: &[ProposalEvent]) -> Vec<HashSet<String>> {
    proposals.iter().map(|p| tokenize(&p.raw_output)).collect()
}

fn jaccard_distance_matrix(token_sets: &[HashSet<String>]) -> Vec<Vec<f64>> {
    let n = token_sets.len();
    let mut d = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dist = 1.0 - jaccard(&token_sets[i], &token_sets[j]);
            d[i][j] = dist;
            d[j][i] = dist;
        }
    }
    d
}

fn krum_select(proposals: &[ProposalEvent], f: usize) -> Option<&ProposalEvent> {
    if proposals.is_empty() {
        return None;
    }
    if f == 0 {
        return proposals.first();
    }
    if !quorum_satisfied(proposals.len(), f) {
        return None;
    }
    let distances = jaccard_distance_matrix(&build_token_sets(proposals));
    let k = proposals.len() - f - 2;
    krum_index(&distances, k).map(|i| &proposals[i])
}

fn multi_krum_select(proposals: &[ProposalEvent], f: usize, m: usize) -> Vec<&ProposalEvent> {
    if proposals.is_empty() || m == 0 {
        return vec![];
    }
    if f == 0 {
        return proposals.iter().take(m).collect();
    }
    if !quorum_satisfied(proposals.len(), f) {
        return vec![];
    }
    let distances = jaccard_distance_matrix(&build_token_sets(proposals));
    let mut remaining: Vec<usize> = (0..proposals.len()).collect();
    let mut selected = Vec::with_capacity(m);
    while selected.len() < m && remaining.len() > f + 2 {
        let k = remaining.len() - f - 2;
        let best_pos = (0..remaining.len())
            .min_by(|&a, &b| {
                let sa = krum_score_subset(remaining[a], &remaining, &distances, k);
                let sb = krum_score_subset(remaining[b], &remaining, &distances, k);
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("remaining is non-empty");
        selected.push(&proposals[remaining[best_pos]]);
        remaining.remove(best_pos);
    }
    selected
}

fn prop(text: &str) -> ProposalEvent {
    ProposalEvent {
        task_id: TaskId::new(),
        explorer_id: ExplorerId::new(),
        tau: TauValue::new(0.5).unwrap(),
        generation: 0,
        raw_output: text.into(),
        token_cost: text.len() as u64,
        adapter_kind: AdapterKind::CloudGeneric {
            endpoint: "mock".into(),
            api_key_env: "NONE".into(),
        },
        timestamp: Utc::now(),
    }
}

/// Honest proposal vocabulary: varies across simulated explorers but all share
/// the "stateless jwt auth" core that honest LLM outputs would share.
const HONEST_TEXTS: &[&str] = &[
    "stateless jwt auth token rotation ADR-001 compliant",
    "stateless jwt authentication bearer token ADR-001",
    "jwt stateless auth mechanism token rotation ADR-001",
    "stateless authentication jwt bearer rotation ADR-001",
    "jwt bearer stateless auth token ADR-001 compliant",
    "stateless token authentication jwt refresh ADR-001",
    "stateless jwt credential verification ADR-001",
    "jwt stateless token credential auth ADR-001",
    "stateless jwt auth signing rotation ADR-001",
    "jwt stateless bearer token auth ADR-001",
];

/// Byzantine proposal vocabulary: semantically unrelated to the honest texts.
const BYZANTINE_TEXTS: &[&str] = &[
    "blockchain hash proof-of-work cryptocurrency mining",
    "redis session store sliding window expiry cookie domain",
    "rainbow table hash collision preimage attack reversal",
    "elasticsearch document index shard replication cluster",
];

/// Build a scenario with `n_honest` honest proposals followed by `f` Byzantine ones.
fn build_scenario(n_honest: usize, f: usize, byz_idx: usize) -> Vec<ProposalEvent> {
    let byz_text = BYZANTINE_TEXTS[byz_idx % BYZANTINE_TEXTS.len()];
    let mut proposals: Vec<ProposalEvent> = HONEST_TEXTS
        .iter()
        .take(n_honest)
        .map(|t| prop(t))
        .collect();
    for _ in 0..f {
        proposals.push(prop(byz_text));
    }
    proposals
}

fn is_byzantine(text: &str) -> bool {
    BYZANTINE_TEXTS.contains(&text)
}

/// **Simulation 1**: Krum never selects a Byzantine proposal.
///
/// Exhaustively tests: f ∈ {1, 2, 3}, n = 2f+3 .. 2f+6, byz_texts × all 4.
///
/// This proves the Krum theorem holds in our implementation over the metric
/// (𝒫(Tokens), d_J) for the realistic vocabulary used.
#[test]
fn krum_never_selects_byzantine_in_any_simulated_scenario() {
    let mut scenarios_tested = 0u32;

    for f in 1usize..=3 {
        let n_min = 2 * f + 3;
        for n in n_min..=(n_min + 3) {
            let n_honest = n - f;
            if n_honest > HONEST_TEXTS.len() {
                continue;
            }
            for byz_idx in 0..BYZANTINE_TEXTS.len() {
                let proposals = build_scenario(n_honest, f, byz_idx);
                assert_eq!(proposals.len(), n);

                let selected = krum_select(&proposals, f).unwrap_or_else(|| {
                    panic!("Krum must return Some for n={n}, f={f} (quorum satisfied)")
                });

                assert!(
                    !is_byzantine(&selected.raw_output),
                    "SAFETY VIOLATION: Krum selected Byzantine proposal for n={n}, f={f}, byz_idx={byz_idx}.\n\
                     Selected: {}\n\
                     Byzantine: {}",
                    selected.raw_output,
                    BYZANTINE_TEXTS[byz_idx % BYZANTINE_TEXTS.len()]
                );
                scenarios_tested += 1;
            }
        }
    }

    // Sanity: ensure we actually ran scenarios (not vacuously true)
    assert!(
        scenarios_tested >= 36,
        "expected ≥ 36 scenarios, ran {scenarios_tested}"
    );
}

/// **Simulation 2**: ConsensusMedian is vulnerable when Byzantine proposals
/// form a majority-of-cluster.
///
/// Configuration: n=4 total, f=2 Byzantine. Byzantine proposals are identical
/// (Sybil/flooding attack). The Byzantine cluster has mean Jaccard ≥ honest
/// cluster mean Jaccard, so ConsensusMedian selects a Byzantine proposal.
///
/// This demonstrates WHY Krum is needed: ConsensusMedian has no quorum
/// protection. The attack works when f ≥ n/2 because the Byzantine cluster
/// can match or beat the honest cluster in pairwise similarity.
#[tokio::test]
async fn condorcet_vulnerable_when_byzantine_form_majority() {
    // 2 honest proposals, 2 Byzantine (identical — Sybil flooding attack)
    let h1 = prop("stateless jwt auth token rotation ADR-001 compliant");
    let h2 = prop("stateless jwt authentication bearer token ADR-001");
    let byz_text = "redis session store sliding window expiry cookie domain";
    let b1 = prop(byz_text);
    let b2 = prop(byz_text); // identical clone — maximises Byzantine cluster similarity

    // n=4, f=2: Krum CANNOT help here (n=4 < 2*2+3=7), but ConsensusMedian
    // should also fail — that's the point of this test.
    let proposals = vec![h1, h2, b1, b2];

    // ConsensusMedian scores — Byzantine proposals WIN, not just tie:
    // honest h1: mean_J = (J(h1,h2) + J(h1,b1) + J(h1,b2)) / 3 ≈ (0.22 + 0 + 0) / 3 ≈ 0.07
    // honest h2: same ≈ 0.07
    // byzantine b1: mean_J = (J(b1,h1) + J(b1,h2) + J(b1,b2)) / 3 ≈ (0 + 0 + 1.0) / 3 ≈ 0.33
    // byzantine b2: same ≈ 0.33
    //
    // Byzantine proposals score ~0.33 vs honest ~0.07. The Sybil clone gives each
    // Byzantine proposal perfect self-similarity (J=1.0) that the isolated honest
    // proposals cannot match. Rust's max_by returns the last maximum element (b2).
    let selected = ConsensusMedian::resolve(&proposals, None).await.unwrap();

    // The test ASSERTS the vulnerability — Condorcet fails under this attack.
    // If ConsensusMedian is ever fixed to handle this, update the assertion.
    assert_eq!(
        selected.raw_output, byz_text,
        "ConsensusMedian SHOULD select the Byzantine proposal under Sybil flood attack \
         (n=4, f=2). This test documents the known vulnerability. \
         If this assertion fails, ConsensusMedian has been hardened — remove or update this test."
    );
}

/// **Simulation 3**: Krum's quorum requirement is not vacuous — removing one
/// honest proposal from the minimal quorum configuration breaks the guarantee.
///
/// With n = 2f+2 (one below the quorum threshold), `krum_select` returns None.
/// This confirms the precondition is enforced, not just documented.
#[test]
fn krum_returns_none_below_quorum() {
    for f in 1usize..=3 {
        let n = 2 * f + 2; // one below quorum
        let proposals: Vec<_> = HONEST_TEXTS.iter().take(n).map(|t| prop(t)).collect();
        assert_eq!(proposals.len(), n);
        assert!(
            krum_select(&proposals, f).is_none(),
            "Krum must return None for n={n} < {} (f={f})",
            2 * f + 3
        );
    }
}

/// **Simulation 4**: Multi-Krum filters all Byzantine proposals when m = n − f − 2.
///
/// Configuration: n=7, f=2, m=3 (the maximum Multi-Krum can safely select).
/// Multi-Krum iterates while `remaining.len() > f+2`; for n=7, f=2 this yields
/// exactly `n − f − 2 = 3` survivors.  After selection, none of them are Byzantine.
///
/// Note: `m = n − f` would require selecting from a sub-pool below the quorum
/// threshold, at which point Krum's safety guarantee no longer holds (Byzantine
/// identical-clone / Sybil proposals would score 0 distance to each other and
/// beat honest ones).  The safe bound is `m ≤ n − f − 2`.
#[test]
fn multi_krum_selects_byzantine_free_survivors_up_to_safe_maximum() {
    let f = 2usize;
    let n_honest = 5;

    // Use *distinct* Byzantine texts to avoid the Sybil-clone pathology where
    // identical Byzantine proposals score 0 Krum distance to each other and
    // beat honest proposals in later rounds.
    let mut proposals: Vec<ProposalEvent> = HONEST_TEXTS
        .iter()
        .take(n_honest)
        .map(|t| prop(t))
        .collect();
    for byz_idx in 0..f {
        proposals.push(prop(BYZANTINE_TEXTS[byz_idx % BYZANTINE_TEXTS.len()]));
    }
    let n = proposals.len(); // 7
    assert_eq!(n, 7);

    // Safe maximum m = n − f − 2 = 3: Multi-Krum iterates while
    // `remaining.len() > f+2`, so it can produce at most n − f − 2 survivors.
    let m_safe = n - f - 2; // 3
    let survivors = multi_krum_select(&proposals, f, m_safe);
    assert_eq!(
        survivors.len(),
        m_safe,
        "must return exactly {m_safe} survivors"
    );

    for s in &survivors {
        assert!(
            !is_byzantine(&s.raw_output),
            "Multi-Krum survivor must not be Byzantine; got: {}",
            s.raw_output
        );
    }
}
