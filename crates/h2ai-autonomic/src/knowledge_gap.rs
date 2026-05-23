use h2ai_types::gap_i1::KnowledgeGapRecord;

pub fn detect_cold_checks(
    check_rates: &[((String, usize), f64)],
    threshold: f64,
) -> Vec<KnowledgeGapRecord> {
    check_rates
        .iter()
        .filter(|(_, rate)| *rate <= threshold)
        .map(|((constraint_id, check_idx), rate)| KnowledgeGapRecord {
            constraint_id: constraint_id.clone(),
            check_idx: *check_idx,
            incorrect_concept: String::new(), // filled by LLM extractor before researcher dispatch
            gap_query: String::new(),         // filled by LLM extractor before researcher dispatch
            pass_rate_across_waves: *rate,
        })
        .collect()
}

pub fn build_gap_queries(check_text: &str, incorrect_concept: &str) -> Vec<String> {
    vec![
        // Query 1: canonical implementation pattern
        format!(
            "correct implementation {} replace {}",
            extract_domain_keywords(check_text),
            extract_core_noun(incorrect_concept)
        ),
        // Query 2: known failure mode documentation
        format!(
            "{} failure mode race condition bug known issue",
            extract_core_noun(incorrect_concept)
        ),
        // Query 3: migration path from incorrect to correct
        format!(
            "migrate from {} to atomic alternative {}",
            extract_core_noun(incorrect_concept),
            extract_domain_keywords(check_text)
        ),
    ]
}

fn extract_core_noun(concept: &str) -> String {
    concept
        .split_whitespace()
        .take(6)
        .collect::<Vec<_>>()
        .join(" ")
}

fn extract_domain_keywords(check_text: &str) -> String {
    let words: Vec<&str> = check_text
        .split_whitespace()
        .filter(|w| {
            let clean = w.trim_matches(|c: char| !c.is_alphanumeric());
            clean.len() > 3
                && (clean
                    .chars()
                    .next()
                    .map(|c| c.is_uppercase())
                    .unwrap_or(false)
                    || clean
                        .chars()
                        .all(|c| c.is_uppercase() || c == '_' || c.is_ascii_digit()))
        })
        .take(5)
        .collect();
    if words.is_empty() {
        check_text
            .split_whitespace()
            .take(5)
            .collect::<Vec<_>>()
            .join(" ")
    } else {
        words.join(" ")
    }
}
