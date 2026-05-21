use std::collections::HashMap;

/// BM25+ (Lv & Zhai) sparse retrieval scorer.
///
/// BM25+ adds a DELTA floor to the per-term contribution, preventing long documents
/// from scoring 0 when they contain a query term. Standard BM25 length-normalizes
/// aggressively: a single match in a 300-token synthesis node can round to 0.
/// The DELTA=1.0 addend ensures every document that shares at least one query term
/// receives a strictly positive score contribution from that term.
///
/// Parameters follow standard BM25 conventions (Okapi BM25 defaults):
/// - K1 = 1.5 — TF saturation coefficient
/// - B  = 0.75 — length normalization factor
/// - δ  = 1.0  — BM25+ lower-bound addend (Lv & Zhai 2011)
#[derive(Debug, Clone)]
pub struct Bm25PlusRetriever {
    entries: Vec<BM25Entry>,
    idf: HashMap<String, f32>,
    avg_doc_len: f32,
}

#[derive(Debug, Clone)]
struct BM25Entry {
    id: String,
    term_freqs: HashMap<String, u32>,
    doc_len: u32,
}

/// A single retrieval result with BM25+ score.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    pub id: String,
    /// BM25+ score in (0, ∞). Use for ranking only — not bounded.
    pub score: f32,
}

const K1: f32 = 1.5;
const B: f32 = 0.75;
const DELTA: f32 = 1.0;

impl Bm25PlusRetriever {
    /// Build a BM25+ index from an iterator of `(id, text)` pairs.
    #[allow(clippy::cast_precision_loss)]
    pub fn build<'a>(docs: impl Iterator<Item = (&'a str, &'a str)>) -> Self {
        let entries: Vec<BM25Entry> = docs
            .map(|(id, text)| {
                let term_freqs = tokenize(text);
                let doc_len = term_freqs.values().sum();
                BM25Entry {
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

        // IDF: Robertson-Walker variant — ln((N - df + 0.5) / (df + 0.5) + 1)
        // The +1 inside the log prevents negative IDF for terms present in all documents.
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
        }
    }

    /// Query the index, returning up to `top_k` candidates sorted by descending BM25+ score.
    ///
    /// Only candidates with score > 0 are returned. Returns empty if corpus is empty,
    /// `top_k` is 0, or no query terms match any document.
    #[must_use]
    pub fn query(&self, text: &str, top_k: usize) -> Vec<Candidate> {
        if self.entries.is_empty() || top_k == 0 {
            return vec![];
        }
        let query_terms = tokenize(text);
        if query_terms.is_empty() {
            return vec![];
        }

        let mut scores: Vec<Candidate> = self
            .entries
            .iter()
            .map(|entry| Candidate {
                id: entry.id.clone(),
                score: self.bm25plus_score(entry, &query_terms),
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
    fn bm25plus_score(&self, entry: &BM25Entry, query_terms: &HashMap<String, u32>) -> f32 {
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
            // BM25+ tf normalization
            let tf_norm = tf * (K1 + 1.0) / K1.mul_add(1.0 - B + B * dl / avgdl, tf);
            // BM25+ score: IDF × (δ + tf_norm)  — the δ floor is the key upgrade
            score += idf * (DELTA + tf_norm);
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

/// Tokenizer: lowercase, split on non-alphanumeric, min 3 chars, stopword filter.
/// Produces term → frequency counts.
pub(crate) fn tokenize(text: &str) -> HashMap<String, u32> {
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
    matches!(
        token,
        "the"
            | "and"
            | "for"
            | "are"
            | "this"
            | "that"
            | "with"
            | "from"
            | "not"
            | "but"
            | "has"
            | "have"
            | "was"
            | "were"
            | "will"
            | "can"
            | "all"
            | "any"
            | "its"
            | "use"
            | "used"
            | "must"
    )
}
