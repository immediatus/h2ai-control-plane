pub struct CompactionConfig {
    pub max_tokens: usize,
    pub preserve_keywords: Vec<String>,
}

fn estimate_tokens(s: &str) -> usize {
    s.len().saturating_add(3) / 4
}

/// Compact `context` to fit within `config.max_tokens` (±N-token heuristic slack).
///
/// Strategy (Lost-in-Middle mitigation):
/// 1. If already within budget, return with missing preserve_keywords appended (no truncation).
/// 2. Otherwise keep a head and tail slice of the body, drop the middle, and
///    append any missing preserve_keywords at the end (checked against the
///    compacted body so keywords dropped from the middle are re-injected).
pub fn compact(context: &str, config: &CompactionConfig) -> String {
    // Fast path: check if we're already within budget with any suffix needed.
    let initial_suffix = build_keyword_suffix(context, &config.preserve_keywords);
    let suffix_tokens = estimate_tokens(&initial_suffix);
    let body_tokens = estimate_tokens(context);
    let total = body_tokens + if initial_suffix.is_empty() { 0 } else { suffix_tokens + 1 };

    if total <= config.max_tokens {
        return if initial_suffix.is_empty() {
            context.to_owned()
        } else {
            format!("{context}\n{initial_suffix}")
        };
    }

    // Need to truncate. Reserve tokens for suffix (all keywords) + marker.
    // We use all keywords as suffix since truncation may drop any of them.
    let all_keywords_suffix = config.preserve_keywords.join(" ");
    let suffix_reserve = if all_keywords_suffix.is_empty() {
        0
    } else {
        estimate_tokens(&all_keywords_suffix) + 1 // +1 for the newline
    };

    let marker = "\n[...compacted...]\n";
    let marker_tokens = estimate_tokens(marker);
    let body_budget = config
        .max_tokens
        .saturating_sub(suffix_reserve)
        .saturating_sub(marker_tokens);

    let char_budget = body_budget * 4;
    // When budget is zero (e.g. keywords alone exceed max_tokens), emit only the suffix.
    if char_budget == 0 {
        return if all_keywords_suffix.is_empty() {
            String::new()
        } else {
            all_keywords_suffix
        };
    }
    let half = char_budget / 2;

    let chars: Vec<char> = context.chars().collect();
    let head_end = half.min(chars.len());
    let tail_start = chars.len().saturating_sub(half);

    let body = if tail_start > head_end {
        let head: String = chars[..head_end].iter().collect();
        let tail: String = chars[tail_start..].iter().collect();
        format!("{head}{marker}{tail}")
    } else {
        chars[..head_end].iter().collect()
    };

    // After truncation, check which keywords are now missing from the body.
    let post_suffix = build_keyword_suffix(&body, &config.preserve_keywords);

    if post_suffix.is_empty() {
        body
    } else {
        format!("{body}\n{post_suffix}")
    }
}

fn build_keyword_suffix(context: &str, keywords: &[String]) -> String {
    let mut seen = std::collections::HashSet::new();
    let missing: Vec<&str> = keywords
        .iter()
        .filter(|kw| seen.insert(kw.as_str()) && !context.contains(kw.as_str()))
        .map(|s| s.as_str())
        .collect();
    if missing.is_empty() {
        String::new()
    } else {
        missing.join(" ")
    }
}
