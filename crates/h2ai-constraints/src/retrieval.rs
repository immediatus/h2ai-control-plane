use crate::types::{ConstraintDoc, ConstraintMeta};
use std::collections::HashMap;

/// BM25 (Okapi BM25) sparse retrieval index for constraint corpus.
///
/// Provides semantic keyword retrieval over large corpora without external dependencies.
/// BM25 is the industry standard for first-stage retrieval (Elasticsearch, Lucene, Solr
/// all use it as their default ranking function).
///
/// Advantages over raw vocabulary matching:
/// - **TF saturation**: a term appearing 10× is not 10× more relevant than appearing 1×
/// - **IDF weighting**: rare terms (specific to few constraints) matter more than common ones
/// - **Length normalization**: long rubrics don't unfairly outscore short rubrics
///
/// At corpus scale (1M+ constraints), a dense ANN layer (fastembed + HNSW) should sit above
/// this as the first-stage filter, with BM25 as the re-ranker over ANN candidates. At small
/// to medium corpus size (<100K constraints), BM25 alone provides sufficient retrieval quality.
///
/// Standard BM25 parameters: k1=1.5 (TF saturation), b=0.75 (length normalization).
#[derive(Debug, Clone)]
pub struct ConstraintRetriever {
    entries: Vec<RetrieverEntry>,
    /// Corpus-wide inverse document frequency for each observed term.
    idf: HashMap<String, f32>,
    avg_doc_len: f32,
    /// TF saturation coefficient. Controls how much repeated terms matter. Standard: 1.5.
    k1: f32,
    /// Length normalization factor. 0 = no normalization, 1 = full normalization. Standard: 0.75.
    b: f32,
}

#[derive(Debug, Clone)]
struct RetrieverEntry {
    id: String,
    term_freqs: HashMap<String, u32>,
    doc_len: u32,
}

/// A single retrieval result with BM25 relevance score.
#[derive(Debug, Clone, PartialEq)]
pub struct ConstraintCandidate {
    pub id: String,
    /// BM25 score in [0, ∞). Higher is more relevant. Not bounded — use for ranking only.
    pub score: f32,
}

impl ConstraintRetriever {
    /// Build a BM25 index from a slice of `ConstraintDoc`.
    ///
    /// The indexed text is: `"{id} {description} {rubric_text}"` — all text a task
    /// description could plausibly match against.
    #[must_use]
    pub fn from_docs(docs: &[ConstraintDoc]) -> Self {
        Self::build(docs.iter().map(|d| {
            let rubric = extract_rubric_text(&d.predicate);
            let text = format!("{} {} {}", d.id, d.description, rubric);
            (d.id.as_str(), text)
        }))
    }

    /// Build from an iterator of `(id, text)` pairs. Used directly in tests.
    #[allow(clippy::cast_precision_loss)]
    pub fn build<'a>(docs: impl Iterator<Item = (&'a str, String)>) -> Self {
        let entries: Vec<RetrieverEntry> = docs
            .map(|(id, text)| {
                let term_freqs = tokenize(&text);
                let doc_len = term_freqs.values().sum();
                RetrieverEntry {
                    id: id.to_string(),
                    term_freqs,
                    doc_len,
                }
            })
            .collect();

        let n = entries.len() as f32;
        let avg_doc_len = if entries.is_empty() {
            0.0
        } else {
            entries.iter().map(|e| e.doc_len as f32).sum::<f32>() / n
        };

        // BM25 IDF: log((N - df + 0.5) / (df + 0.5) + 1) — Robertson-Walker variant
        // Adding +1 inside the log prevents negative IDF for very common terms.
        let mut df: HashMap<String, u32> = HashMap::new();
        for entry in &entries {
            for term in entry.term_freqs.keys() {
                *df.entry(term.clone()).or_default() += 1;
            }
        }
        let idf: HashMap<String, f32> = df
            .into_iter()
            .map(|(term, count)| {
                let score = ((n - count as f32 + 0.5) / (count as f32 + 0.5)).ln_1p();
                (term, score.max(0.0))
            })
            .collect();

        Self {
            entries,
            idf,
            avg_doc_len,
            k1: 1.5,
            b: 0.75,
        }
    }

    /// Query the index, returning up to `top_k` candidates sorted by descending BM25 score.
    ///
    /// Returns an empty vec if the index is empty or `top_k` is 0.
    /// Only candidates with score > 0 are returned.
    #[must_use]
    pub fn query(&self, text: &str, top_k: usize) -> Vec<ConstraintCandidate> {
        if self.entries.is_empty() || top_k == 0 {
            return vec![];
        }
        let query_terms = tokenize(text);
        if query_terms.is_empty() {
            return vec![];
        }
        let mut scores: Vec<ConstraintCandidate> = self
            .entries
            .iter()
            .map(|entry| ConstraintCandidate {
                id: entry.id.clone(),
                score: self.bm25_score(entry, &query_terms),
            })
            .filter(|c| c.score > 0.0)
            .collect();
        scores.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scores.truncate(top_k);
        scores
    }

    #[allow(clippy::cast_precision_loss)]
    fn bm25_score(&self, entry: &RetrieverEntry, query_terms: &HashMap<String, u32>) -> f32 {
        let dl = entry.doc_len as f32;
        let avgdl = self.avg_doc_len.max(1.0);
        let mut score = 0.0f32;
        for term in query_terms.keys() {
            let Some(&idf) = self.idf.get(term) else {
                continue;
            };
            let tf = *entry.term_freqs.get(term).unwrap_or(&0) as f32;
            if tf == 0.0 {
                continue;
            }
            let numerator = tf * (self.k1 + 1.0);
            let denominator = self.k1.mul_add(1.0 - self.b + self.b * dl / avgdl, tf);
            score += idf * numerator / denominator;
        }
        score
    }

    /// Number of indexed documents.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Extract the rubric string from a predicate for indexing purposes.
fn extract_rubric_text(pred: &crate::types::ConstraintPredicate) -> String {
    use crate::types::ConstraintPredicate;
    match pred {
        ConstraintPredicate::LlmJudge { rubric } => rubric.clone(),
        ConstraintPredicate::VocabularyPresence { terms, .. }
        | ConstraintPredicate::NegativeKeyword { terms } => terms.join(" "),
        ConstraintPredicate::RegexMatch { pattern, .. } => pattern.clone(),
        ConstraintPredicate::Composite { children, .. } => children
            .iter()
            .map(extract_rubric_text)
            .collect::<Vec<_>>()
            .join(" "),
        _ => String::new(),
    }
}

/// Lightweight tokenizer: lowercase, split on non-alphanumeric, filter stop-words and short tokens.
/// Produces term → frequency counts.
fn tokenize(text: &str) -> HashMap<String, u32> {
    let mut counts: HashMap<String, u32> = HashMap::new();
    for word in text.split(|c: char| !c.is_alphanumeric()) {
        let token = word.to_lowercase();
        if token.len() >= 3 && !is_stopword(&token) {
            *counts.entry(token).or_default() += 1;
        }
    }
    counts
}

fn is_stopword(token: &str) -> bool {
    // Common English and technical stop-words that add no signal for constraint retrieval.
    matches!(
        token,
        "the"
            | "and"
            | "not"
            | "are"
            | "this"
            | "that"
            | "with"
            | "for"
            | "all"
            | "any"
            | "must"
            | "will"
            | "may"
            | "can"
            | "use"
            | "its"
            | "per"
            | "two"
            | "one"
            | "from"
            | "into"
            | "both"
            | "such"
            | "also"
            | "when"
            | "then"
            | "each"
            | "have"
            | "been"
            | "was"
            | "has"
            | "had"
            | "does"
            | "did"
            | "but"
            | "than"
            | "more"
            | "only"
            | "after"
            | "which"
            | "these"
            | "those"
            | "their"
            | "they"
            | "would"
            | "should"
            | "could"
            | "never"
            | "without"
            | "between"
            | "above"
            | "below"
            | "during"
            | "within"
            | "where"
            | "what"
            | "how"
            | "who"
            | "ever"
            | "even"
            | "being"
            | "just"
            | "here"
            | "some"
            | "them"
            | "used"
            | "using"
            | "since"
            | "thus"
            | "hence"
            | "therefore"
    )
}

/// Resolve applicable `ConstraintMeta` for a task using both tag-based and BM25 semantic lookup.
///
/// Two-stage resolution:
/// 1. Tag intersection (O(tags), exact): mandatory constraints for the task context
/// 2. BM25 semantic search (O(corpus)): surface additional relevant constraints the
///    task description implies but didn't explicitly tag
///
/// The union of both stages is returned, deduplicated.
#[must_use]
pub fn resolve_with_retrieval<S: std::hash::BuildHasher>(
    query_text: &str,
    top_k: usize,
    retriever: &ConstraintRetriever,
    all_metas: &HashMap<String, ConstraintMeta, S>,
) -> Vec<ConstraintMeta> {
    let candidates = retriever.query(query_text, top_k);
    candidates
        .into_iter()
        .filter_map(|c| all_metas.get(&c.id).cloned())
        .collect()
}
