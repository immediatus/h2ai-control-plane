use crate::types::{CompositeOp, ConstraintPredicate, NumericOp, VocabularyMode};

/// Evaluate a sync predicate against an output string.
///
/// Returns a score in [0.0, 1.0].
/// - `LlmJudge`: returns 1.0 (pass-through — use the async LLM path).
/// - `OracleExecution`: returns 0.0 (safe degradation — use `eval_async` for oracle predicates).
pub fn eval_sync(pred: &ConstraintPredicate, output: &str) -> f64 {
    let lower = output.to_lowercase();
    match pred {
        ConstraintPredicate::VocabularyPresence { mode, terms } => {
            eval_vocabulary(mode, terms, &lower)
        }
        ConstraintPredicate::NegativeKeyword { terms } => {
            eval_vocabulary(&VocabularyMode::NoneOf, terms, &lower)
        }
        ConstraintPredicate::RegexMatch {
            pattern,
            must_match,
        } => match regex::Regex::new(pattern) {
            Ok(re) => {
                let matched = re.is_match(output);
                if matched == *must_match {
                    1.0
                } else {
                    0.0
                }
            }
            Err(_) => 0.0,
        },
        ConstraintPredicate::NumericThreshold {
            field_pattern,
            op,
            value,
        } => eval_numeric(field_pattern, op, *value, output),
        ConstraintPredicate::LlmJudge { .. } => {
            // Must be evaluated via async path; sync path is a pass-through.
            1.0
        }
        ConstraintPredicate::OracleExecution { .. } => {
            // Requires an async HTTP call; sync path always returns 0.0 (safe degradation).
            // Use eval_async in h2ai-orchestrator::verification for oracle predicates.
            0.0
        }
        ConstraintPredicate::JsonSchema { schema } => {
            let instance = match serde_json::from_str::<serde_json::Value>(output) {
                Ok(v) => v,
                Err(_) => return 0.0,
            };
            match jsonschema::validator_for(schema) {
                Ok(validator) => {
                    if validator.is_valid(&instance) {
                        1.0
                    } else {
                        0.0
                    }
                }
                Err(_) => 0.0,
            }
        }
        ConstraintPredicate::LengthRange {
            min_chars,
            max_chars,
        } => {
            let len = output.chars().count();
            let ok = min_chars.map_or(true, |m| len >= m) && max_chars.map_or(true, |m| len <= m);
            if ok {
                1.0
            } else {
                0.0
            }
        }
        ConstraintPredicate::Composite { op, children } => {
            let scores: Vec<f64> = children.iter().map(|c| eval_sync(c, output)).collect();
            match op {
                CompositeOp::And => scores.iter().cloned().fold(1.0_f64, f64::min),
                CompositeOp::Or => scores.iter().cloned().fold(0.0_f64, f64::max),
                CompositeOp::Not => {
                    let child_score = scores.first().copied().unwrap_or(0.0);
                    1.0 - child_score
                }
            }
        }
    }
}

fn eval_vocabulary(mode: &VocabularyMode, terms: &[String], lower_output: &str) -> f64 {
    if terms.is_empty() {
        return 1.0;
    }
    let hit_count = terms
        .iter()
        .filter(|t| lower_output.contains(t.to_lowercase().as_str()))
        .count();
    match mode {
        VocabularyMode::AllOf => hit_count as f64 / terms.len() as f64,
        VocabularyMode::AnyOf => {
            if hit_count > 0 {
                1.0
            } else {
                0.0
            }
        }
        VocabularyMode::NoneOf => {
            if hit_count == 0 {
                1.0
            } else {
                0.0
            }
        }
    }
}

fn eval_numeric(
    field_pattern: &str,
    op: &crate::types::NumericOp,
    threshold: f64,
    output: &str,
) -> f64 {
    let Ok(re) = regex::Regex::new(field_pattern) else {
        return 0.0;
    };
    let Some(cap) = re.captures(output) else {
        return 0.0;
    };
    let Some(num_str) = cap.get(1).or_else(|| cap.get(0)) else {
        return 0.0;
    };
    let Ok(v) = num_str.as_str().parse::<f64>() else {
        return 0.0;
    };
    let passes = match op {
        NumericOp::Lt => v < threshold,
        NumericOp::Le => v <= threshold,
        NumericOp::Eq => (v - threshold).abs() < 1e-9,
        NumericOp::Ge => v >= threshold,
        NumericOp::Gt => v > threshold,
    };
    if passes {
        1.0
    } else {
        0.0
    }
}
