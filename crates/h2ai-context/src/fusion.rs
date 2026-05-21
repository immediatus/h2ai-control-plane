use std::collections::HashMap;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::Value;
use tantivy::schema::{Schema, STORED, TEXT};
use tantivy::{Index, TantivyDocument};

/// Reciprocal Rank Fusion constant k=60 from Cormack, Clarke & Buettcher (SIGIR 2009).
///
/// Value 60 was chosen as optimal across TREC datasets; it prevents rank-1 documents
/// from dominating. Not operator-configurable — changing it would invalidate the
/// published optimality claim.
pub const RRF_K: f64 = 60.0;

#[must_use]
#[allow(clippy::cast_precision_loss)]
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
///
/// # Panics
///
/// Panics if the Tantivy in-RAM index writer or reader cannot be created (should not
/// happen in practice as the RAM index is unbounded and the writer budget is generous).
#[must_use]
#[allow(clippy::cast_precision_loss)]
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

    let ranked = query_parser.parse_query(query).map_or_else(
        |_| vec![],
        |q| {
            searcher
                .search(&q, &TopDocs::with_limit(docs.len()))
                .unwrap_or_default()
                .into_iter()
                .map(|(score, addr)| {
                    let doc: TantivyDocument = searcher.doc(addr).expect("doc retrieve");
                    #[allow(clippy::cast_possible_truncation)]
                    let orig_id = doc
                        .get_first(id_field)
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as usize;
                    (orig_id, f64::from(score))
                })
                .collect::<Vec<_>>()
        },
    );

    // Append docs not returned by BM25 (zero score) so result always has docs.len() entries.
    let mut seen: std::collections::HashSet<usize> = ranked.iter().map(|(i, _)| *i).collect();
    let mut result = ranked;
    for i in 0..docs.len() {
        if seen.insert(i) {
            result.push((i, 0.0));
        }
    }
    result
}
