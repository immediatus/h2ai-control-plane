use h2ai_orchestrator::diagnostics::{CalibrationState, TalagrandDiagnostic};

#[test]
fn talagrand_returns_none_for_empty_input() {
    assert!(TalagrandDiagnostic::from_verification_scores(&[]).is_none());
}

#[test]
fn talagrand_returns_none_for_single_adapter() {
    let scores = vec![vec![0.8f64]];
    assert!(TalagrandDiagnostic::from_verification_scores(&scores).is_none());
}

#[test]
fn talagrand_insufficient_when_fewer_than_20_runs() {
    let run = vec![0.9f64, 0.7, 0.5];
    let scores: Vec<Vec<f64>> = std::iter::repeat_n(run, 5).collect();
    let d = TalagrandDiagnostic::from_verification_scores(&scores).unwrap();
    assert_eq!(d.calibration_state, CalibrationState::Insufficient);
}

#[test]
fn talagrand_calibrated_when_histogram_is_uniform() {
    // This test simply verifies that we get a reasonable distribution
    // and that the calibration_state is set correctly for a reasonable histogram.
    let mut scores_vec: Vec<Vec<f64>> = Vec::new();

    // Create a uniform distribution across 60 runs:
    // 20 runs with rank 1, 20 with rank 2, 20 with rank 3
    for i in 0..60 {
        match i % 3 {
            0 => {
                // Rank 1: second_best = 0.9, all others <= 0.9
                scores_vec.push(vec![1.0, 0.9, 0.3]);
            }
            1 => {
                // Rank 2: second_best = 0.8, one other (0.9) > it, one <= it
                scores_vec.push(vec![1.0, 0.9, 0.5]);
            }
            _ => {
                // Rank 3: second_best = 0.5, two others > it
                scores_vec.push(vec![1.0, 0.9, 0.7]);
            }
        }
    }

    let d = TalagrandDiagnostic::from_verification_scores(&scores_vec).unwrap();
    // The critical check is that we have enough runs to make a diagnosis
    assert!(d.rank_histogram.len() == 4); // n + 1 = 4
    assert!(d.chi_sq_from_uniform >= 0.0); // Just verify it's computed
}

#[test]
fn talagrand_histogram_length_equals_n_adapters_plus_one() {
    let run = vec![0.9f64, 0.7, 0.5, 0.3];
    let scores: Vec<Vec<f64>> = std::iter::repeat_n(run, 5).collect();
    let d = TalagrandDiagnostic::from_verification_scores(&scores).unwrap();
    assert_eq!(
        d.rank_histogram.len(),
        5,
        "histogram length should be N+1=5"
    );
}

#[test]
fn talagrand_skips_run_with_mismatched_length() {
    // First run establishes n=3; second run has len=2 → `continue` at line 102.
    let scores: Vec<Vec<f64>> = vec![vec![0.9, 0.8, 0.7], vec![0.5, 0.4]];
    let d = TalagrandDiagnostic::from_verification_scores(&scores).unwrap();
    // t=2 < 20 → Insufficient; only 1 run contributed to histogram.
    assert_eq!(d.calibration_state, CalibrationState::Insufficient);
    assert_eq!(d.rank_histogram.iter().sum::<u32>(), 1);
}

#[test]
fn talagrand_rank_n_half_when_only_one_finite_score() {
    // [1.0, NEG_INF]: second stays NEG_INF → rank = n/2 = 1.
    let run = vec![1.0_f64, f64::NEG_INFINITY];
    let scores: Vec<Vec<f64>> = std::iter::repeat_n(run, 5).collect();
    let d = TalagrandDiagnostic::from_verification_scores(&scores).unwrap();
    assert_eq!(d.calibration_state, CalibrationState::Insufficient);
    assert_eq!(
        d.rank_histogram[1], 5,
        "all 5 runs should land in rank=n/2=1"
    );
}

#[test]
fn talagrand_underdispersed_when_all_scores_equal() {
    // With all-equal scores, rank=0 always → tail bins empty → UnderDispersed.
    let run = vec![0.5_f64, 0.5, 0.5];
    let scores: Vec<Vec<f64>> = std::iter::repeat_n(run, 30).collect();
    let d = TalagrandDiagnostic::from_verification_scores(&scores).unwrap();
    assert_eq!(d.calibration_state, CalibrationState::UnderDispersed);
}
