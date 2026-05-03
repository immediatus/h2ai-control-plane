use std::collections::HashSet;

fn tokenize(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .filter(|t| t.len() > 1)
        .collect()
}

fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count() as f64;
    let union = a.union(b).count() as f64;
    intersection / union
}

/// Jaccard word-token similarity between two strings. Range [0.0, 1.0].
/// Two strings with identical token sets return 1.0 regardless of order.
/// Both empty strings return 0.0 (no tokens in common or distinct).
pub fn similarity(a: &str, b: &str) -> f64 {
    jaccard(&tokenize(a), &tokenize(b))
}
