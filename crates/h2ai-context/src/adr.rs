use crate::jaccard::tokenize;
use std::collections::HashSet;
use std::path::Path;

pub struct AdrConstraints {
    pub source: String,
    pub keywords: HashSet<String>,
}

/// Load all `.md` files from `dir` as ADR constraints, ignoring missing directories.
pub fn load_corpus(dir: impl AsRef<Path>) -> Result<Vec<AdrConstraints>, std::io::Error> {
    let dir = dir.as_ref();
    if !dir.exists() {
        return Ok(vec![]);
    }
    let mut corpus = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("md") {
            let content = std::fs::read_to_string(&path)?;
            let source = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_owned();
            corpus.push(parse_adr(&source, &content));
        }
    }
    Ok(corpus)
}

pub fn parse_adr(source: &str, content: &str) -> AdrConstraints {
    let keywords = extract_constraints_section(content)
        .map(|s| tokenize(&s))
        .unwrap_or_default();
    AdrConstraints {
        source: source.to_owned(),
        keywords,
    }
}

fn extract_constraints_section(content: &str) -> Option<String> {
    let lower = content.to_lowercase();
    let start = lower.find("## constraints")?;
    let after = &content[start + "## constraints".len()..];
    let end = after.find("\n## ").unwrap_or(after.len());
    Some(after[..end].to_owned())
}
