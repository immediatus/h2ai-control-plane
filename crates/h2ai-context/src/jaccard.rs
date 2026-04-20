use std::collections::HashSet;

/// Common English function words that carry no domain signal.
/// Filtering them improves Jaccard accuracy for LLM output similarity.
const STOPWORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "shall", "can", "it", "its", "this",
    "that", "these", "those", "i", "we", "you", "he", "she", "they", "what", "which", "who", "not",
    "as", "if", "so", "also", "all", "any", "each", "some", "such", "than", "then", "there",
    "their", "them", "our", "your", "my", "s", "use", "uses", "used", "using", "provide",
    "provides", "provided", "approach",
];

pub fn tokenize(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .filter(|t| t.len() > 1 && !STOPWORDS.contains(&t.as_str()))
        .collect()
}

pub fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count() as f64;
    let union = a.union(b).count() as f64;
    intersection / union
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stopwords_filtered_from_tokens() {
        let tokens = tokenize("the proposed solution uses stateless JWT auth");
        assert!(!tokens.contains("the"), "stopword 'the' must be filtered");
        assert!(!tokens.contains("uses"), "stopword 'uses' must be filtered");
        assert!(
            tokens.contains("proposed"),
            "content word 'proposed' must remain"
        );
        assert!(tokens.contains("stateless"));
        assert!(tokens.contains("jwt"));
        assert!(tokens.contains("auth"));
    }

    #[test]
    fn single_char_tokens_filtered() {
        let tokens = tokenize("a b c stateless");
        assert!(!tokens.contains("a"), "single-char 'a' must be filtered");
        assert!(!tokens.contains("b"), "single-char 'b' must be filtered");
        assert!(tokens.contains("stateless"));
    }

    #[test]
    fn synonyms_have_higher_jaccard_after_stopword_removal() {
        let a = "the JWT is a stateless auth mechanism";
        let b = "the JSON Web Token provides stateless authentication";
        let j = jaccard(&tokenize(a), &tokenize(b));
        assert!(j > 0.0, "non-zero jaccard for partially overlapping text");
        assert!(j < 1.0, "different texts must not be identical");
    }

    #[test]
    fn identical_text_gives_jaccard_one() {
        let text = "stateless JWT authentication ADR-001 compliant";
        assert_eq!(jaccard(&tokenize(text), &tokenize(text)), 1.0);
    }

    #[test]
    fn completely_disjoint_text_gives_jaccard_zero() {
        let a = "stateless jwt authentication";
        let b = "redis session store expiry";
        assert_eq!(jaccard(&tokenize(a), &tokenize(b)), 0.0);
    }
}
