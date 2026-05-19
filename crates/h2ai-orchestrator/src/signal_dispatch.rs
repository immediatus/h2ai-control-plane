use h2ai_types::signal::SignalPayload;

pub enum ResumeAction {
    ContinueToNextWave {
        grounding: Option<String>,
        mandate_override: Option<String>,
    },
    Finalize {
        approved: bool,
        reviewer_note: Option<String>,
        operator_id: String,
    },
    Ignore,
}

pub fn resolve_action(payload: SignalPayload) -> ResumeAction {
    match payload {
        SignalPayload::WaveContinue(s) => ResumeAction::ContinueToNextWave {
            grounding: s.grounding,
            mandate_override: s.mandate_override,
        },
        SignalPayload::Approve(s) => ResumeAction::Finalize {
            approved: s.approved,
            reviewer_note: s.reviewer_note,
            operator_id: s.operator_id,
        },
        _ => ResumeAction::Ignore,
    }
}
