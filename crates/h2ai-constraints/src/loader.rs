use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::source::{ConstraintError, ConstraintSource};
use crate::spec::SemanticSpec;
use crate::types::ConstraintDoc;

/// Filesystem YAML source — scans a directory for .yaml/.yml constraint files.
/// Files are loaded in sorted order for deterministic corpus ordering.
pub struct YamlDirSource {
    pub dir: PathBuf,
}

impl YamlDirSource {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }
}

impl ConstraintSource for YamlDirSource {
    fn load_all(&self) -> Result<Vec<SemanticSpec>, ConstraintError> {
        let dir = &self.dir;
        if !dir.exists() {
            return Ok(vec![]);
        }

        let mut entries: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| ConstraintError::Unavailable(e.to_string()))?
            .filter_map(std::result::Result::ok)
            .collect();
        entries.sort_by_key(std::fs::DirEntry::file_name);

        let mut specs = Vec::new();
        let mut seen_ids: HashSet<String> = HashSet::new();

        for entry in &entries {
            let path = entry.path();
            let ext = path.extension().and_then(|e| e.to_str());
            if ext != Some("yaml") && ext != Some("yml") {
                continue;
            }
            let content = std::fs::read_to_string(&path)
                .map_err(|e| ConstraintError::Unavailable(e.to_string()))?;

            match serde_yaml::from_str::<crate::yaml::ConstraintYaml>(&content) {
                Ok(yaml) => {
                    // Fix #2: check parsed struct, not raw string — avoids false-positive
                    // warnings when "predicates:" appears in rubric or description text.
                    if !yaml.predicates.is_empty() {
                        tracing::warn!(
                            path = %path.display(),
                            id = %yaml.id,
                            "constraint uses deprecated 'predicates:' array. Migrate to 'semantic:' section."
                        );
                    }
                    match yaml.into_semantic_spec() {
                        Ok(spec) => {
                            if seen_ids.insert(spec.id.clone()) {
                                specs.push(spec);
                            }
                        }
                        Err(msg) => {
                            tracing::warn!(path = %path.display(), error = %msg, "skipping constraint");
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        path = %path.display(),
                        error = %e,
                        "failed to parse YAML constraint; skipping"
                    );
                }
            }
        }

        Ok(specs)
    }
}

// ── Legacy function kept for internal callers ──────────────────────────────

/// Load a constraint corpus from a directory.
/// Used by `RuntimeConstraintStore::load()` and existing test fixtures.
///
/// # Errors
/// Returns `std::io::Error` if reading the directory or any YAML file fails.
pub fn load_corpus(dir: impl AsRef<Path>) -> Result<Vec<ConstraintDoc>, std::io::Error> {
    let dir = dir.as_ref();
    if !dir.exists() {
        return Ok(vec![]);
    }

    let mut entries: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(std::result::Result::ok)
        .collect();
    entries.sort_by_key(std::fs::DirEntry::file_name);

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
