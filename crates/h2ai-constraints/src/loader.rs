use crate::types::{ConstraintDoc, ConstraintPredicate, ConstraintSeverity, VocabularyMode};
use std::collections::HashSet;
use std::path::Path;

pub fn load_corpus(dir: impl AsRef<Path>) -> Result<Vec<ConstraintDoc>, std::io::Error> {
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
            let id = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_owned();
            corpus.push(parse_constraint_doc(&id, &content));
        }
    }
    Ok(corpus)
}

pub fn parse_constraint_doc(id: &str, content: &str) -> ConstraintDoc {
    let (domains, mandatory_for_tags, content) = parse_frontmatter(content);
    let content = content.to_string();
    let lower = content.to_lowercase();
    let (section_text, severity) = find_constraint_section(&content, &lower);
    let terms = tokenize_section(section_text);
    let predicate = ConstraintPredicate::VocabularyPresence {
        mode: VocabularyMode::AllOf,
        terms,
    };
    ConstraintDoc {
        id: id.to_owned(),
        source_file: id.to_owned(),
        description: String::new(),
        severity,
        predicate,
        remediation_hint: None,
        domains,
        mandatory_for_tags,
    }
}

/// Parse a simple YAML-style frontmatter block from the start of a markdown file.
///
/// Returns `(domains, mandatory_for_tags, body)` where `body` is the content after
/// the closing `---`. Files without a `---` opener return empty vecs and the full content.
fn parse_frontmatter(content: &str) -> (Vec<String>, Vec<String>, &str) {
    let content = content.trim_start();
    if !content.starts_with("---") {
        return (vec![], vec![], content);
    }
    // Find the closing --- on its own line
    let after_open = match content.find('\n') {
        Some(i) => &content[i + 1..],
        None => return (vec![], vec![], content),
    };
    let close = match after_open.find("\n---") {
        Some(i) => i,
        None => return (vec![], vec![], content),
    };
    let frontmatter = &after_open[..close];
    let rest = &after_open[close + 4..]; // skip "\n---"
    let body = rest.trim_start_matches('\n');

    let domains = extract_yaml_list(frontmatter, "domains");
    let mandatory_for_tags = extract_yaml_list(frontmatter, "mandatory_for_tags");
    (domains, mandatory_for_tags, body)
}

/// Extract a YAML list from a frontmatter block:
///   key:\n    - val1\n    - val2
fn extract_yaml_list(block: &str, key: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut in_list = false;
    for line in block.lines() {
        let trimmed = line.trim();
        if trimmed == format!("{key}:").as_str() {
            in_list = true;
            continue;
        }
        if in_list {
            if let Some(val) = trimmed.strip_prefix("- ") {
                result.push(val.trim().to_string());
            } else if !trimmed.is_empty() && !trimmed.starts_with('#') {
                break; // hit next key
            }
        }
    }
    result
}

fn find_constraint_section<'a>(content: &'a str, lower: &str) -> (&'a str, ConstraintSeverity) {
    // Priority order: Hard > Soft > Advisory > plain Constraints
    let candidates: &[(&str, ConstraintSeverity)] = &[
        (
            "## hard constraints",
            ConstraintSeverity::Hard { threshold: 0.8 },
        ),
        (
            "## soft constraints",
            ConstraintSeverity::Soft { weight: 1.0 },
        ),
        ("## advisory", ConstraintSeverity::Advisory),
        (
            "## constraints",
            ConstraintSeverity::Hard { threshold: 0.8 },
        ),
    ];
    for (heading, severity) in candidates {
        if let Some(start) = lower.find(heading) {
            let after = &content[start + heading.len()..];
            let end = after.find("\n## ").unwrap_or(after.len());
            return (&after[..end], severity.clone());
        }
    }
    ("", ConstraintSeverity::Hard { threshold: 0.8 })
}

fn tokenize_section(section: &str) -> Vec<String> {
    section
        .split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|w| w.len() >= 3)
        .collect::<HashSet<_>>()
        .into_iter()
        .collect()
}
