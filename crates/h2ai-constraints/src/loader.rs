use std::collections::HashSet;
use std::path::Path;

use crate::types::ConstraintDoc;

/// Load a constraint corpus from a directory.
///
/// Only `.yaml` / `.yml` files are loaded — YAML is the sole supported format.
/// Iteration is sorted by filename for deterministic corpus ordering.
pub fn load_corpus(dir: impl AsRef<Path>) -> Result<Vec<ConstraintDoc>, std::io::Error> {
    let dir = dir.as_ref();
    if !dir.exists() {
        return Ok(vec![]);
    }

    let mut entries: Vec<_> = std::fs::read_dir(dir)?.filter_map(|e| e.ok()).collect();
    entries.sort_by_key(|e| e.file_name());

    let mut corpus = Vec::new();
    let mut seen_ids: HashSet<String> = HashSet::new();

    for entry in &entries {
        let path = entry.path();
        let ext = path.extension().and_then(|e| e.to_str());
        if ext == Some("yaml") || ext == Some("yml") {
            let content = std::fs::read_to_string(&path)?;
            if let Some(doc) = crate::yaml::parse_yaml_constraint(&path, &content) {
                if seen_ids.insert(doc.id.clone()) {
                    corpus.push(doc);
                }
            }
        }
    }

    Ok(corpus)
}
