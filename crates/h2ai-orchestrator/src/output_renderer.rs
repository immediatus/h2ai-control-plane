use crate::provenance::{DocumentConfidence, ProvenanceMap, ProvisionConfidence};

fn confidence_label(dc: &DocumentConfidence) -> &'static str {
    match dc {
        DocumentConfidence::High => "High",
        DocumentConfidence::ReviewRecommended => "ReviewRecommended",
        DocumentConfidence::RequiresReview => "RequiresReview",
        DocumentConfidence::Unverified => "Unverified",
    }
}

fn provision_label(pc: &ProvisionConfidence) -> &'static str {
    match pc {
        ProvisionConfidence::Verified => "Verified",
        ProvisionConfidence::AutoCorrected => "AutoCorrected",
        ProvisionConfidence::ReviewRecommended => "ReviewRecommended",
        ProvisionConfidence::RequiresReview => "RequiresReview",
        ProvisionConfidence::Unverified => "Unverified",
    }
}

/// Renders the final output string with confidence metadata.
///
/// `mode = "passthrough"`: returns text unchanged; fabrication/provenance data is available
///   in the ProvenanceRecordedEvent but the output itself carries no annotation.
/// `mode = "clean"`: prepends a single-line blockquote confidence header.
/// `mode = "audit"`: prepends header, appends inline provision annotations and footer.
///
/// The `text` argument must be the exact bytes that will be committed to NATS JetStream.
pub fn render_output(text: &str, map: &ProvenanceMap, mode: &str) -> String {
    if mode == "passthrough" {
        return text.to_string();
    }

    let dc = map.document_confidence();
    let label = confidence_label(&dc);

    let header = format!(
        "> **Epistemic Confidence: {}** — h2ai epistemic quality stage\n\n",
        label
    );

    if mode != "audit" {
        return format!("{}{}", header, text);
    }

    // Audit mode: inline per-provision annotations
    let annotations: String = map
        .provisions()
        .iter()
        .filter(|p| !matches!(p.confidence, ProvisionConfidence::Verified))
        .map(|p| {
            let gaps = if p.gap_ids.is_empty() {
                String::new()
            } else {
                format!(" [gaps: {}]", p.gap_ids.join(", "))
            };
            format!(
                "> ⚠ **{}** — {}{}\n",
                p.provision_label,
                provision_label(&p.confidence),
                gaps
            )
        })
        .collect();

    let footer = format!(
        "\n\n---\n> **Epistemic Footer** | Document confidence: {} | Provisions reviewed: {}\n",
        label,
        map.provisions().len()
    );

    format!("{}{}{}{}", header, annotations, text, footer)
}
