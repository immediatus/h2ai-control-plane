use async_trait::async_trait;
use h2ai_types::memory::RetryHintPattern;

pub mod algorithmic;
pub mod nats_scheduler;

/// Context passed to the induction scheduler when triggered.
#[derive(Debug, Clone)]
pub struct InductionContext {
    pub tenant_id: String,
    pub task_class_tags: Vec<String>,
    pub violated_constraint_ids: Vec<String>,
}

/// Result returned by the induction scheduler.
#[derive(Debug, Clone)]
pub struct InductionResult {
    pub patterns: Vec<RetryHintPattern>,
    pub trigger_tags: Vec<String>,
}

impl InductionResult {
    /// Belief-compatibility gate (Phase 1): at least one tag in common between
    /// this result's trigger_tags and the provided current_tags.
    pub fn is_compatible_with(&self, current_tags: &[String]) -> bool {
        self.trigger_tags.iter().any(|t| current_tags.contains(t))
    }
}

/// Trait for scheduling retroactive induction over prior task history.
/// Inject as `Arc<dyn InductionScheduler>` for testability.
#[async_trait]
pub trait InductionScheduler: Send + Sync {
    /// Read-only: load relevant `RetryHintPattern` records before a task starts.
    /// Two-round SAD in `NatsInductionScheduler`; direct filter in `AlgorithmicInductionWorker`.
    /// Default returns `vec![]` so existing mock impls compile without change.
    async fn load_priming_hints(&self, _ctx: &InductionContext) -> Vec<RetryHintPattern> {
        vec![]
    }

    /// Read-write: retroactive induction triggered after zero-survival.
    /// Increments G-counter `attempt_count` for matched patterns.
    async fn run_retroactive(&self, ctx: &InductionContext) -> Option<InductionResult>;

    /// Increment G-counter `success_count` for hint texts that led to a successful resolution.
    /// Called from `engine.rs` when a wave resolves after induction hints were applied.
    /// Default is a no-op so existing mock impls compile without change.
    async fn record_success(&self, _hint_texts: &[String], _ctx: &InductionContext) {}
}

// ── Shingling pure functions ──────────────────────────────────────────────────

/// Normalize a string for trigram shingling.
///
/// Pipeline: split CamelCase → lowercase → strip version tokens (vX.Y or X.Y.Z)
/// → strip hex/numerics → strip punctuation → collapse whitespace.
pub fn normalize_for_shingling(s: &str) -> String {
    // Split CamelCase: insert space before each uppercase letter preceded by lowercase
    let mut camel_split = String::with_capacity(s.len() + 8);
    let chars: Vec<char> = s.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        if c.is_uppercase() && i > 0 && chars[i - 1].is_lowercase() {
            camel_split.push(' ');
        }
        camel_split.push(c);
    }

    // Lowercase
    let lower = camel_split.to_lowercase();

    // Strip version tokens: vX or vX.Y.Z patterns and bare numbers
    let words: Vec<&str> = lower.split_whitespace().collect();
    let filtered: Vec<&str> = words
        .iter()
        .copied()
        .filter(|w| {
            // Drop tokens that are: purely numeric, version-like (v25, v25.6), or hex
            let stripped = w.trim_start_matches('v');
            let is_version = stripped
                .chars()
                .next()
                .map(|c| c.is_ascii_digit())
                .unwrap_or(false)
                && stripped.chars().all(|c| c.is_ascii_digit() || c == '.');
            let is_numeric = w.chars().all(|c| c.is_ascii_digit());
            let is_hex = w.len() >= 6 && w.chars().all(|c| c.is_ascii_hexdigit());
            !is_version && !is_numeric && !is_hex
        })
        .collect();

    // Strip punctuation (keep letters, digits, spaces)
    let joined = filtered.join(" ");
    let cleaned: String = joined
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == ' ' {
                c
            } else {
                ' '
            }
        })
        .collect();

    // Collapse whitespace
    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Compute sorted, deduplicated character trigram shingles from a string.
pub fn trigram_shingles(s: &str) -> Vec<[u8; 3]> {
    let bytes = s.as_bytes();
    if bytes.len() < 3 {
        return vec![];
    }
    let mut shingles: Vec<[u8; 3]> = bytes.windows(3).map(|w| [w[0], w[1], w[2]]).collect();
    shingles.sort_unstable();
    shingles.dedup();
    shingles
}

/// Jaccard similarity between two sorted trigram shingle sets.
pub fn jaccard_shingles(a: &[[u8; 3]], b: &[[u8; 3]]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let mut intersection = 0usize;
    let mut ai = 0;
    let mut bi = 0;
    while ai < a.len() && bi < b.len() {
        match a[ai].cmp(&b[bi]) {
            std::cmp::Ordering::Equal => {
                intersection += 1;
                ai += 1;
                bi += 1;
            }
            std::cmp::Ordering::Less => ai += 1,
            std::cmp::Ordering::Greater => bi += 1,
        }
    }
    let union = a.len() + b.len() - intersection;
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Cluster strings by trigram Jaccard similarity using union-find.
///
/// Returns a Vec of cluster labels (same label = same cluster).
/// Strings with Jaccard >= threshold are merged into the same cluster.
/// A length-ratio gate rejects pairs where min_len/max_len < 0.5.
pub fn cluster_by_similarity(strings: &[String], threshold: f64) -> Vec<usize> {
    let n = strings.len();
    let mut parent: Vec<usize> = (0..n).collect();

    let normalized: Vec<_> = strings.iter().map(|s| normalize_for_shingling(s)).collect();
    let shingles: Vec<_> = normalized.iter().map(|s| trigram_shingles(s)).collect();

    fn find(parent: &mut Vec<usize>, x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }

    for i in 0..n {
        for j in (i + 1)..n {
            let len_i = normalized[i].len();
            let len_j = normalized[j].len();
            if len_i == 0 || len_j == 0 {
                continue;
            }
            let min_len = len_i.min(len_j) as f64;
            let max_len = len_i.max(len_j) as f64;
            if min_len / max_len < 0.5 {
                continue; // length-ratio gate
            }
            let j_sim = jaccard_shingles(&shingles[i], &shingles[j]);
            if j_sim >= threshold {
                let ri = find(&mut parent, i);
                let rj = find(&mut parent, j);
                if ri != rj {
                    parent[ri] = rj;
                }
            }
        }
    }

    // Canonicalize labels
    (0..n).map(|i| find(&mut parent, i)).collect()
}
