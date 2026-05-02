use h2ai_state::weiszfeld::weiszfeld_select;

#[test]
fn weiszfeld_selects_honest_with_one_byzantine() {
    // 4 honest embeddings near [1,0,...,0], 1 Byzantine at [-1,0,...,0]
    let dim = 8;
    let honest: Vec<Vec<f32>> = (0..4)
        .map(|i| {
            let mut v = vec![0.0f32; dim];
            v[0] = 1.0 - 0.05 * i as f32; // slightly varied but near [1,0,...]
            v[1] = 0.05 * i as f32;
            v
        })
        .collect();
    let byzantine = vec![-1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    let mut embeddings = honest.clone();
    embeddings.push(byzantine);

    let selected = weiszfeld_select(&embeddings, 20);
    assert!(
        selected < 4,
        "Should select an honest proposal (index 0-3), got {}",
        selected
    );
}

#[test]
fn weiszfeld_single_returns_zero() {
    let embs = vec![vec![1.0f32, 0.0, 0.0]];
    assert_eq!(weiszfeld_select(&embs, 20), 0);
}

#[test]
fn weiszfeld_empty_returns_zero() {
    let embs: Vec<Vec<f32>> = vec![];
    assert_eq!(weiszfeld_select(&embs, 20), 0);
}
