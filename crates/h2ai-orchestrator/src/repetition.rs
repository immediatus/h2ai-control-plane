use h2ai_context::jaccard::{jaccard, tokenize};

/// Jaccard word-token similarity between two strings. Range [0.0, 1.0].
/// Two strings with identical token sets return 1.0 regardless of order.
/// Both empty strings return 0.0 (no tokens in common or distinct).
pub fn similarity(a: &str, b: &str) -> f64 {
    let ta = tokenize(a);
    let tb = tokenize(b);
    jaccard(&ta, &tb)
}
