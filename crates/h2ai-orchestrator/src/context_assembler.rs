use h2ai_types::adapter::IComputeAdapter;
use regex::Regex;
use std::sync::OnceLock;

static CONSTRAINT_RE: OnceLock<Regex> = OnceLock::new();

#[inline]
fn constraint_regex() -> &'static Regex {
    CONSTRAINT_RE.get_or_init(|| Regex::new(r"C-\d{3}").unwrap())
}

/// Output of a single `ContextAssembler::build()` call.
#[derive(Debug, Clone)]
pub struct AssembledContext {
    pub text: String,
    pub token_estimate: usize,
    pub compression: CompressionKind,
    /// 1.0 = no compression, 0.3 = 70 % of original tokens removed.
    pub compression_ratio: f32,
    pub prev_wave_delta: bool,
    /// true if compression was stopped early by the quality guard.
    pub quality_clamped: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompressionKind {
    /// Compression disabled (no budget configured).
    None,
    /// Only rule-based transforms ran.
    RuleBased,
    /// Rule-based + section importance trimming; no LLM call.
    ImportanceScored,
    /// Rule-based + LLM summarization of low-importance sections.
    LlmSummarized,
}

/// Logical section of a context with importance metadata.
#[derive(Debug, Clone)]
pub struct Section {
    pub tag: SectionTag,
    pub text: String,
    /// 0.0–1.0; higher = more important; sections with preserve=true are never compressed.
    pub importance: f32,
    /// If true this section is passed through verbatim regardless of budget.
    pub preserve: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SectionTag {
    LeaderPrefix,
    Grounding,
    ActiveCtx,
    RetryContext,
    RoleFrame,
    Mandate,
    RejectionCriteria,
    Tombstone,
    /// Knowledge corpus overview. preserve=true, importance=1.0.
    GlobalKnowledge,
    /// Domain cluster synthesis. preserve=false, importance=0.8.
    TopicKnowledge,
    /// Cross-domain constraint tensions surfaced by Synthesizer knowledge query.
    /// preserve=false, importance=0.85. Only injected for Synthesizer slots.
    ConstraintTension,
    /// Compliance checklist from constraint binary_checks. preserve=true, importance=1.0.
    ComplianceChecklist,
}

/// Input to `ContextAssembler::build()`.
pub struct ContextAssemblerInput<'a> {
    pub active_ctx: &'a str,
    pub retry_context: Option<&'a str>,
    pub leader_prefix: Option<&'a str>,
    pub grounding: Option<&'a str>,
    pub tombstone: Option<&'a str>,
    pub role_frame: Option<&'a str>,
    pub mandate: Option<&'a str>,
    pub rejection_criteria: Option<&'a str>,
    /// Previous wave's assembled output for delta encoding. None on wave 0.
    pub prev_wave_blob: Option<&'a AssembledContext>,
    /// Token budget. None = compression disabled.
    pub budget: Option<usize>,
    /// Stop compression when ratio drops below this. None = 0.4 default.
    pub quality_guard_ratio: Option<f32>,
    /// Adapter for LLM summarization pass. None = skip LLM pass.
    pub compression_adapter: Option<&'a dyn IComputeAdapter>,
    /// Cross-task cache. None = no caching.
    pub stable_cache: Option<&'a stable_cache::StableContextCache>,
    /// Global knowledge node text. preserve=true, importance=1.0. None = omit.
    pub global_knowledge: Option<&'a str>,
    /// Topic cluster synthesis text. preserve=false, importance=0.8. None = omit.
    pub topic_knowledge: Option<&'a str>,
    /// Constraint tension text for Synthesizer slots. None = omit. preserve=false, importance=0.85.
    pub constraint_tensions: Option<&'a str>,
    /// Numbered compliance checklist derived from constraint binary_checks.
    /// preserve=true, importance=1.0. None = omit.
    pub compliance_checklist: Option<&'a str>,
}

pub struct ContextAssembler;

impl ContextAssembler {
    pub async fn build(input: ContextAssemblerInput<'_>) -> AssembledContext {
        // Step 0: if no budget configured, return raw assembled text.
        let Some(budget) = input.budget else {
            let text = assemble_raw(&input);
            let token_estimate = estimate_tokens(&text);
            return AssembledContext {
                text,
                token_estimate,
                compression: CompressionKind::None,
                compression_ratio: 1.0,
                prev_wave_delta: false,
                quality_clamped: false,
            };
        };

        let quality_threshold = input.quality_guard_ratio.unwrap_or(0.4);

        // Step 1: build and score sections
        let sections = build_sections(&input);
        let prev_text = input.prev_wave_blob.map(|p| p.text.as_str());
        let sections = score_sections(sections, prev_text);

        let original_tokens: usize = sections.iter().map(|s| estimate_tokens(&s.text)).sum();

        // Step 2: rule-based pass
        let wave = input.prev_wave_blob.map_or(0, |_| 1u32);
        let mut sections = sections;
        let delta_applied = rule_pass(
            &mut sections,
            RulePassInput {
                prev_wave_blob: input.prev_wave_blob,
                wave,
            },
        );
        let tokens_after_rule: usize = sections.iter().map(|s| estimate_tokens(&s.text)).sum();

        if tokens_after_rule <= budget {
            let text = sections_to_text(&sections);
            let ratio = if original_tokens > 0 {
                tokens_after_rule as f32 / original_tokens as f32
            } else {
                1.0
            };
            return AssembledContext {
                text,
                token_estimate: tokens_after_rule,
                compression: CompressionKind::RuleBased,
                compression_ratio: ratio,
                prev_wave_delta: delta_applied,
                quality_clamped: false,
            };
        }

        // Step 3: importance-scored trimming
        let sections = importance_trim(sections, budget);
        let tokens_after_trim: usize = sections.iter().map(|s| estimate_tokens(&s.text)).sum();
        let quality_clamped_trim =
            quality_guard(original_tokens, tokens_after_trim, quality_threshold);

        if tokens_after_trim <= budget || input.compression_adapter.is_none() {
            let text = sections_to_text(&sections);
            let ratio = if original_tokens > 0 {
                tokens_after_trim as f32 / original_tokens as f32
            } else {
                1.0
            };
            return AssembledContext {
                text,
                token_estimate: tokens_after_trim,
                compression: CompressionKind::ImportanceScored,
                compression_ratio: ratio,
                prev_wave_delta: delta_applied,
                quality_clamped: quality_clamped_trim,
            };
        }

        // Step 4: LLM summarization — target RetryContext first, then ActiveCtx
        let mut sections = sections;
        let target_tags = [SectionTag::RetryContext, SectionTag::ActiveCtx];
        let adapter = input.compression_adapter.unwrap();

        for tag in &target_tags {
            let tokens_now: usize = sections.iter().map(|s| estimate_tokens(&s.text)).sum();
            if tokens_now <= budget {
                break;
            }
            // Pre-compute other_tokens before mutable borrow of sections
            let other_tokens: usize = sections
                .iter()
                .filter(|s| s.tag != *tag)
                .map(|s| estimate_tokens(&s.text))
                .sum();
            let section_budget = budget.saturating_sub(other_tokens).max(64);
            if let Some(sec) = sections.iter_mut().find(|s| s.tag == *tag && !s.preserve) {
                let req = h2ai_types::adapter::ComputeRequest {
                    system_context: "You are a precision context compressor for an AI orchestration \
                        system. Preserve: constraint IDs (C-NNN), decisions, requirements, rejection \
                        criteria. Remove: restatements, filler, redundant examples.".to_string(),
                    task: format!(
                        "Compress the following to under {} tokens. \
                         Output ONLY the compressed text, no preamble:\n\n{}",
                        section_budget, sec.text
                    ),
                    tau: h2ai_types::sizing::TauValue::new(0.1).unwrap(),
                    max_tokens: (section_budget as u64).max(64),
                };
                if let Ok(resp) = adapter.execute(req).await {
                    sec.text = resp.output;
                }
                // LLM failure is non-fatal: keep trimmed text
            }
        }

        let tokens_after_llm: usize = sections.iter().map(|s| estimate_tokens(&s.text)).sum();
        let quality_clamped = quality_guard(original_tokens, tokens_after_llm, quality_threshold);
        let text = sections_to_text(&sections);
        let ratio = if original_tokens > 0 {
            tokens_after_llm as f32 / original_tokens as f32
        } else {
            1.0
        };
        AssembledContext {
            text,
            token_estimate: tokens_after_llm,
            compression: CompressionKind::LlmSummarized,
            compression_ratio: ratio,
            prev_wave_delta: delta_applied,
            quality_clamped,
        }
    }
}

/// Join non-empty sections into a single text string, separated by double newlines.
pub(crate) fn sections_to_text(sections: &[Section]) -> String {
    sections
        .iter()
        .filter(|s| !s.text.is_empty())
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Concatenate input pieces into a raw context string (no compression).
#[must_use]
pub fn assemble_raw(input: &ContextAssemblerInput<'_>) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(lp) = input.leader_prefix {
        if !lp.is_empty() {
            parts.push(lp.to_string());
        }
    }
    if let Some(g) = input.grounding {
        if !g.is_empty() {
            parts.push(format!("[STATE-OF-THE-ART]: {g}"));
        }
    }
    if let Some(rf) = input.role_frame {
        if !rf.is_empty() {
            parts.push(rf.to_string());
        }
    }
    if let Some(m) = input.mandate {
        if !m.is_empty() {
            parts.push(format!("[MANDATE]: {m}"));
        }
    }
    if let Some(rc) = input.rejection_criteria {
        if !rc.is_empty() {
            parts.push(format!(
                "[AFTER WRITING YOUR PROPOSAL, IDENTIFY THE BIGGEST RISK]: {rc}"
            ));
        }
    }
    parts.push(input.active_ctx.to_string());
    if let Some(ret) = input.retry_context {
        if !ret.is_empty() {
            parts.push(ret.to_string());
        }
    }
    if let Some(t) = input.tombstone {
        if !t.is_empty() {
            parts.push(t.to_string());
        }
    }
    if let Some(gk) = input.global_knowledge {
        if !gk.is_empty() {
            parts.push(format!("[KNOWLEDGE]:\n{gk}"));
        }
    }
    if let Some(tk) = input.topic_knowledge {
        if !tk.is_empty() {
            parts.push(format!("[DOMAIN KNOWLEDGE]:\n{tk}"));
        }
    }
    if let Some(ct) = input.constraint_tensions {
        if !ct.is_empty() {
            parts.push(format!("[CONSTRAINT TENSIONS]:\n{ct}"));
        }
    }
    if let Some(cl) = input.compliance_checklist {
        if !cl.is_empty() {
            parts.push(format!("[COMPLIANCE CHECKLIST]:\n{cl}"));
        }
    }
    parts.join("\n\n")
}

/// Content-type-aware token estimate. YAML/code is denser (~3 chars/token);
/// prose is sparser (~4.5 chars/token).
pub(crate) fn estimate_tokens(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    let chars = text.chars().count();
    let code_lines = text
        .lines()
        .filter(|l| {
            let t = l.trim_start();
            t.starts_with('-') || t.starts_with('{') || t.starts_with('[') || t.starts_with('#')
        })
        .count();
    let total_lines = text.lines().count().max(1);
    let code_ratio = code_lines as f64 / total_lines as f64;
    let chars_per_token = 1.5f64.mul_add(1.0 - code_ratio, 3.0);
    (chars as f64 / chars_per_token).ceil() as usize
}

/// Decompose a `ContextAssemblerInput` into an ordered list of Sections.
/// Empty strings are silently skipped — callers should not pass `Some("")`.
#[must_use]
pub fn build_sections(input: &ContextAssemblerInput<'_>) -> Vec<Section> {
    let mut sections = Vec::new();
    if let Some(lp) = input.leader_prefix {
        if !lp.is_empty() {
            sections.push(Section {
                tag: SectionTag::LeaderPrefix,
                text: lp.to_string(),
                importance: 1.0,
                preserve: true,
            });
        }
    }
    if let Some(g) = input.grounding {
        if !g.is_empty() {
            sections.push(Section {
                tag: SectionTag::Grounding,
                text: g.to_string(),
                importance: 1.0,
                preserve: true,
            });
        }
    }
    if let Some(rf) = input.role_frame {
        if !rf.is_empty() {
            sections.push(Section {
                tag: SectionTag::RoleFrame,
                text: rf.to_string(),
                importance: 0.9,
                preserve: true,
            });
        }
    }
    if let Some(m) = input.mandate {
        if !m.is_empty() {
            sections.push(Section {
                tag: SectionTag::Mandate,
                text: m.to_string(),
                importance: 0.95,
                preserve: true,
            });
        }
    }
    if let Some(rc) = input.rejection_criteria {
        if !rc.is_empty() {
            sections.push(Section {
                tag: SectionTag::RejectionCriteria,
                text: rc.to_string(),
                importance: 0.9,
                preserve: true,
            });
        }
    }
    if !input.active_ctx.is_empty() {
        sections.push(Section {
            tag: SectionTag::ActiveCtx,
            text: input.active_ctx.to_string(),
            importance: 0.7,
            preserve: false,
        });
    }
    if let Some(ret) = input.retry_context {
        if !ret.is_empty() {
            sections.push(Section {
                tag: SectionTag::RetryContext,
                text: ret.to_string(),
                importance: 0.5,
                preserve: false,
            });
        }
    }
    if let Some(t) = input.tombstone {
        if !t.is_empty() {
            sections.push(Section {
                tag: SectionTag::Tombstone,
                text: t.to_string(),
                importance: 1.0,
                preserve: true,
            });
        }
    }
    if let Some(gk) = input.global_knowledge {
        if !gk.is_empty() {
            sections.push(Section {
                tag: SectionTag::GlobalKnowledge,
                text: gk.to_string(),
                importance: 1.0,
                preserve: true,
            });
        }
    }
    if let Some(tk) = input.topic_knowledge {
        if !tk.is_empty() {
            sections.push(Section {
                tag: SectionTag::TopicKnowledge,
                text: tk.to_string(),
                importance: 0.8,
                preserve: false,
            });
        }
    }
    if let Some(ct) = input.constraint_tensions {
        if !ct.is_empty() {
            sections.push(Section {
                tag: SectionTag::ConstraintTension,
                text: ct.to_string(),
                importance: 0.85,
                preserve: false,
            });
        }
    }
    if let Some(cl) = input.compliance_checklist {
        if !cl.is_empty() {
            sections.push(Section {
                tag: SectionTag::ComplianceChecklist,
                text: cl.to_string(),
                importance: 1.0,
                preserve: true,
            });
        }
    }
    sections
}

/// Adjust section importance scores based on content signals.
///
/// Both adjustments apply to the same base score independently:
/// +0.15 if the section contains a constraint ID (C-NNN);
/// -0.30 if the section text is a verbatim substring of `prev_wave_text`
/// (exact match — not semantic similarity).
/// When both fire, net effect is base - 0.15 (penalty dominates).
#[must_use]
pub fn score_sections(mut sections: Vec<Section>, prev_wave_text: Option<&str>) -> Vec<Section> {
    let re = constraint_regex();
    for s in &mut sections {
        if s.preserve {
            continue;
        }
        let base = s.importance;
        let mut adjusted = base;
        if re.is_match(&s.text) {
            adjusted = (adjusted + 0.15).min(1.0);
        }
        if let Some(prev) = prev_wave_text {
            if prev.contains(s.text.as_str()) {
                adjusted = (base - 0.3).max(0.0);
            }
        }
        s.importance = adjusted;
    }
    sections
}

#[derive(Debug)]
pub struct RulePassInput<'a> {
    pub prev_wave_blob: Option<&'a AssembledContext>,
    pub wave: u32,
}

/// Apply rule-based transforms in-place to non-preserved sections.
/// Returns true if any section was delta-replaced from the previous wave.
#[must_use]
pub fn rule_pass(sections: &mut [Section], input: RulePassInput<'_>) -> bool {
    let mut delta_applied = false;
    for section in sections.iter_mut() {
        if section.preserve {
            continue;
        }
        // 1. Cross-wave delta: if section text appears verbatim in prev blob, replace with marker.
        if let Some(prev) = input.prev_wave_blob {
            if !section.text.is_empty() && prev.text.contains(section.text.as_str()) {
                section.text = format!(
                    "[WAVE {} CONTEXT — unchanged, omitted for token efficiency]",
                    input.wave.saturating_sub(1)
                );
                delta_applied = true;
                continue;
            }
        }
        // 2. Block deduplication (normalize_whitespace is called internally by dedup_blocks)
        section.text = dedup_blocks(&section.text);
    }
    delta_applied
}

fn dedup_blocks(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() < 8 {
        return normalize_whitespace(text);
    }
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut result_lines: Vec<&str> = Vec::with_capacity(lines.len());
    let mut i = 0;
    while i < lines.len() {
        if i + 4 <= lines.len() {
            let block = lines[i..i + 4].join("\n");
            if seen.contains(&block) {
                result_lines.push("[duplicate block omitted]");
            } else {
                seen.insert(block);
                result_lines.extend_from_slice(&lines[i..i + 4]);
            }
            i += 4;
        } else {
            result_lines.push(lines[i]);
            i += 1;
        }
    }
    normalize_whitespace(&result_lines.join("\n"))
}

fn normalize_whitespace(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut blank_count = 0usize;
    for line in text.lines() {
        if line.trim().is_empty() {
            blank_count += 1;
            if blank_count <= 1 {
                result.push('\n');
            }
        } else {
            blank_count = 0;
            result.push_str(line.trim_end());
            result.push('\n');
        }
    }
    result.trim_end().to_string()
}

/// Trim non-preserved sections by importance ascending until token estimate ≤ budget.
///
/// Each targeted section is reduced to 60% of its current length, trimmed to the
/// nearest sentence boundary (". ") if one exists within the trimmed range.
/// Preserved sections are never touched.
#[must_use]
pub fn importance_trim(mut sections: Vec<Section>, budget: usize) -> Vec<Section> {
    let mut target_indices: Vec<usize> = sections
        .iter()
        .enumerate()
        .filter(|(_, s)| !s.preserve)
        .map(|(i, _)| i)
        .collect();
    target_indices.sort_by(|&a, &b| {
        sections[a]
            .importance
            .partial_cmp(&sections[b].importance)
            .unwrap()
    });

    loop {
        let current_tokens: usize = sections.iter().map(|s| estimate_tokens(&s.text)).sum();
        if current_tokens <= budget {
            break;
        }

        let mut trimmed_any = false;
        for idx in &target_indices {
            if sections[*idx].text.is_empty() {
                continue;
            }
            let text = &sections[*idx].text;
            let keep_bytes = (text.len() as f64 * 0.6) as usize;
            if keep_bytes == 0 {
                continue;
            }
            // Find the char boundary at or before keep_bytes
            let keep_bytes = text
                .char_indices()
                .map(|(i, _)| i)
                .take_while(|&i| i <= keep_bytes)
                .last()
                .unwrap_or(0);
            let search_range = &text[..keep_bytes];
            let trimmed = if let Some(pos) = search_range.rfind(". ") {
                text[..=pos].to_string()
            } else {
                search_range.to_string()
            };
            if trimmed != *text {
                sections[*idx].text = trimmed;
                trimmed_any = true;
                break;
            }
        }

        if !trimmed_any {
            break;
        }
    }
    sections
}

/// Returns true if compression has been too aggressive and should be stopped.
/// `original_len` and `compressed_len` are token estimates.
/// Returns true when `compressed_len / original_len < threshold`.
#[must_use]
pub fn quality_guard(original_len: usize, compressed_len: usize, threshold: f32) -> bool {
    if original_len == 0 {
        return false;
    }
    let ratio = compressed_len as f32 / original_len as f32;
    ratio < threshold
}

pub mod stable_cache {
    use std::collections::HashMap;
    use std::sync::Mutex;

    #[derive(Debug, Default)]
    pub struct StableContextCache {
        entries: Mutex<HashMap<u64, CachedSection>>,
    }

    #[derive(Debug, Clone)]
    pub struct CachedSection {
        pub compressed_text: String,
        pub original_token_estimate: usize,
        pub compressed_token_estimate: usize,
        pub hit_count: u64,
    }

    impl StableContextCache {
        #[must_use]
        pub fn new() -> Self {
            Self::default()
        }

        pub fn get(&self, key: u64) -> Option<CachedSection> {
            self.entries.lock().unwrap().get(&key).cloned()
        }

        pub fn insert(&self, key: u64, entry: CachedSection) {
            self.entries.lock().unwrap().insert(key, entry);
        }

        pub fn record_hit(&self, key: u64) {
            let mut map = self.entries.lock().unwrap();
            if let Some(e) = map.get_mut(&key) {
                e.hit_count += 1;
            }
        }

        pub fn len(&self) -> usize {
            self.entries.lock().unwrap().len()
        }

        pub fn is_empty(&self) -> bool {
            self.len() == 0
        }
    }
}
