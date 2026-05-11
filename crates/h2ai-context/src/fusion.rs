use std::collections::HashMap;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::Value;
use tantivy::schema::{Schema, STORED, TEXT};
use tantivy::{Index, TantivyDocument};

/// Reciprocal Rank Fusion constant k=60 from Cormack, Clarke & Buettcher (SIGIR 2009).
/// Value 60 was chosen as optimal across TREC datasets; it prevents rank-1 documents
/// from dominating. Not operator-configurable — changing it would invalidate the
/// published optimality claim.
pub const RRF_K: f64 = 60.0;

pub fn rrf_fuse(ranked_lists: &[Vec<(usize, f64)>], k: f64) -> Vec<(usize, f64)> {
    let mut scores: HashMap<usize, f64> = HashMap::new();
    for list in ranked_lists {
        for (rank, &(doc_idx, _)) in list.iter().enumerate() {
            *scores.entry(doc_idx).or_insert(0.0) += 1.0 / (k + (rank + 1) as f64);
        }
    }
    let mut fused: Vec<(usize, f64)> = scores.into_iter().collect();
    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    fused
}

/// Rank documents by BM25 score against `query` using a per-call RAM index.
///
/// Returns `(original_doc_index, bm25_score)` pairs sorted by score descending.
/// Documents with no term overlap with the query receive score 0.0 and appear last.
/// On query-parse failure or empty corpus, returns all docs unranked (score 0.0).
pub fn bm25_search(query: &str, docs: &[&str]) -> Vec<(usize, f64)> {
    if docs.is_empty() {
        return vec![];
    }

    let mut schema_builder = Schema::builder();
    let body_field = schema_builder.add_text_field("body", TEXT);
    let id_field = schema_builder.add_u64_field("id", STORED);
    let schema = schema_builder.build();

    let index = Index::create_in_ram(schema);
    let mut writer = index.writer(15_000_000).expect("tantivy writer");

    for (i, &doc_text) in docs.iter().enumerate() {
        let mut doc = TantivyDocument::default();
        doc.add_text(body_field, doc_text);
        doc.add_u64(id_field, i as u64);
        writer.add_document(doc).expect("add document");
    }
    writer.commit().expect("commit");

    let reader = index.reader().expect("reader");
    let searcher = reader.searcher();
    let query_parser = QueryParser::for_index(&index, vec![body_field]);

    let ranked = match query_parser.parse_query(query) {
        Ok(q) => searcher
            .search(&q, &TopDocs::with_limit(docs.len()))
            .unwrap_or_default()
            .into_iter()
            .map(|(score, addr)| {
                let doc: TantivyDocument = searcher.doc(addr).expect("doc retrieve");
                let orig_id = doc
                    .get_first(id_field)
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                (orig_id, score as f64)
            })
            .collect::<Vec<_>>(),
        Err(_) => vec![],
    };

    // Append docs not returned by BM25 (zero score) so result always has docs.len() entries.
    let mut seen: std::collections::HashSet<usize> = ranked.iter().map(|(i, _)| *i).collect();
    let mut result = ranked;
    for i in 0..docs.len() {
        if !seen.contains(&i) {
            result.push((i, 0.0));
            seen.insert(i);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_fuse_single_list_preserves_order() {
        let list = vec![(0usize, 0.9), (1, 0.7), (2, 0.3)];
        let fused = rrf_fuse(&[list], RRF_K);
        assert_eq!(fused[0].0, 0);
        assert_eq!(fused[1].0, 1);
        assert_eq!(fused[2].0, 2);
    }

    #[test]
    fn rrf_fuse_two_agreeing_lists_amplifies_top_doc() {
        let list_a = vec![(0usize, 0.9), (1, 0.5)];
        let list_b = vec![(0usize, 0.8), (1, 0.4)];
        let fused = rrf_fuse(&[list_a, list_b], RRF_K);
        assert_eq!(fused[0].0, 0, "doc ranked 1st in both lists must win");
        assert!(fused[0].1 > fused[1].1);
    }

    #[test]
    fn rrf_fuse_disagreeing_lists_give_equal_scores() {
        let list_a = vec![(0usize, 0.9), (1, 0.5)];
        let list_b = vec![(1usize, 0.9), (0, 0.5)];
        let fused = rrf_fuse(&[list_a, list_b], RRF_K);
        let score_0 = fused.iter().find(|(i, _)| *i == 0).unwrap().1;
        let score_1 = fused.iter().find(|(i, _)| *i == 1).unwrap().1;
        assert!(
            (score_0 - score_1).abs() < 1e-9,
            "mirrored ranks → equal RRF score"
        );
    }

    #[test]
    fn rrf_fuse_empty_input_returns_empty() {
        let fused: Vec<(usize, f64)> = rrf_fuse(&[], RRF_K);
        assert!(fused.is_empty());
    }

    #[test]
    fn bm25_search_relevant_doc_ranks_first() {
        let docs = [
            "jwt authentication stateless token bearer",
            "redis cache store eviction",
            "tcp socket connection timeout",
        ];
        let result = bm25_search("jwt authentication", &docs);
        assert_eq!(result[0].0, 0, "jwt doc must rank first for jwt query");
    }

    #[test]
    fn bm25_search_empty_docs_returns_empty() {
        let result = bm25_search("query", &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn bm25_search_returns_all_docs() {
        let docs = ["alpha", "beta", "gamma"];
        let result = bm25_search("alpha beta", &docs);
        assert_eq!(result.len(), 3);
    }
}
