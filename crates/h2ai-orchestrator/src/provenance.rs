use h2ai_types::events::CheckVerdict;

/// Per-provision confidence level. The five states form a strict dominance order:
/// Unverified > RequiresReview > ReviewRecommended > AutoCorrected > Verified
/// (Unverified is worst; Verified is best).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ProvisionConfidence {
    /// Passed verification with score == 1.0 and all binary checks PRESENT.
    Verified,
    /// Passed verification but MicroExplorerResolver auto-corrected a gap.
    AutoCorrected,
    /// Passed with score < 1.0 (soft constraint partial credit); human review recommended.
    ReviewRecommended,
    /// Gap detected and not resolved; manual review required before delivery.
    RequiresReview,
    /// No verification data available for this provision.
    Unverified,
}

/// Document-level confidence: determined by the worst provision's confidence state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DocumentConfidence {
    High,
    ReviewRecommended,
    RequiresReview,
    Unverified,
}

/// Provenance record for one named provision within the output.
#[derive(Debug, Clone)]
pub struct ProvisionProvenance {
    /// Heading or label identifying this provision (e.g., "Section 1", "SECTION 2").
    pub provision_label: String,
    pub confidence: ProvisionConfidence,
    /// Per-check verdicts from the LlmJudge covering this provision.
    pub verdicts: Vec<CheckVerdict>,
    /// Gap IDs that targeted this provision (empty when no gaps detected).
    pub gap_ids: Vec<String>,
}

/// Complete epistemic provenance for all provisions in a single task output.
#[derive(Debug, Default, Clone)]
pub struct ProvenanceMap {
    provisions: Vec<ProvisionProvenance>,
}

impl ProvenanceMap {
    pub fn new() -> Self {
        Self {
            provisions: Vec::new(),
        }
    }

    pub fn add_provision(&mut self, prov: ProvisionProvenance) {
        self.provisions.push(prov);
    }

    pub fn provisions(&self) -> &[ProvisionProvenance] {
        &self.provisions
    }

    /// Returns the document-level confidence: worst provision confidence wins.
    pub fn document_confidence(&self) -> DocumentConfidence {
        if self.provisions.is_empty() {
            return DocumentConfidence::Unverified;
        }
        let worst = self.provisions.iter().map(|p| &p.confidence).max();
        match worst {
            Some(ProvisionConfidence::Unverified) => DocumentConfidence::Unverified,
            Some(ProvisionConfidence::RequiresReview) => DocumentConfidence::RequiresReview,
            Some(ProvisionConfidence::ReviewRecommended) => DocumentConfidence::ReviewRecommended,
            _ => DocumentConfidence::High,
        }
    }
}
