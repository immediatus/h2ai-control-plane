use crate::state::AppState;
use h2ai_types::events::{
    AuditDomainDemotedEvent, AuditDomainPromotedEvent, ShadowAuditorResultEvent,
};
use std::collections::HashMap;
use std::sync::Arc;

/// Rolling window for one domain: tracks the last `window` agree/disagree decisions.
struct DomainWindow {
    window: usize,
    /// Ring buffer: true = disagreement.
    buf: std::collections::VecDeque<bool>,
}

impl DomainWindow {
    fn new(window: usize) -> Self {
        Self {
            window,
            buf: std::collections::VecDeque::new(),
        }
    }

    fn push(&mut self, disagreement: bool) {
        if self.buf.len() == self.window {
            self.buf.pop_front();
        }
        self.buf.push_back(disagreement);
    }

    fn n_observations(&self) -> usize {
        self.buf.len()
    }

    fn disagreement_rate(&self) -> f64 {
        if self.buf.is_empty() {
            return 0.0;
        }
        let count = self.buf.iter().filter(|&&d| d).count();
        count as f64 / self.buf.len() as f64
    }
}

/// Accumulates shadow audit results per domain, manages promotion and demotion.
///
/// Spawn one instance per API node at startup. Feed events from engine runs via
/// `process()`. Persists the promoted-domain set to NATS KV after every change.
pub struct ShadowAuditorAccumulator {
    state: Arc<AppState>,
    promotion_threshold: f64,
    promotion_window: usize,
    demotion_threshold: f64,
    demotion_window: usize,
    auto_demotion: bool,
    windows: HashMap<String, DomainWindow>,
}

impl ShadowAuditorAccumulator {
    pub fn new(state: Arc<AppState>) -> Self {
        let promotion_threshold = state.cfg.safety.shadow_auditor.promotion_threshold;
        let promotion_window = state.cfg.safety.shadow_auditor.promotion_window;
        let auto_demotion = state.cfg.safety.shadow_auditor.auto_demotion;
        Self {
            state,
            promotion_threshold,
            promotion_window,
            demotion_threshold: promotion_threshold / 2.0,
            demotion_window: promotion_window * 2,
            auto_demotion,
            windows: HashMap::new(),
        }
    }

    /// Process a batch of shadow audit events from one engine run.
    ///
    /// Updates per-domain windows and fires promotion/demotion when thresholds
    /// are crossed. Persists promoted-domain set to NATS KV after any change.
    pub async fn process(&mut self, events: Vec<ShadowAuditorResultEvent>) {
        if events.is_empty() {
            return;
        }
        let mut changed = false;

        for ev in &events {
            let window = self
                .windows
                .entry(ev.domain.clone())
                .or_insert_with(|| DomainWindow::new(self.promotion_window));
            window.push(ev.disagreement);

            let n = window.n_observations();
            let rate = window.disagreement_rate();
            let currently_promoted = self
                .state
                .promoted_audit_domains
                .read()
                .await
                .contains(&ev.domain);

            if !currently_promoted && n >= self.promotion_window && rate > self.promotion_threshold
            {
                self.state
                    .promoted_audit_domains
                    .write()
                    .await
                    .insert(ev.domain.clone());
                changed = true;
                tracing::warn!(
                    domain = %ev.domain,
                    rate = rate,
                    n = n,
                    "shadow_auditor: domain promoted to AND-vote mode"
                );
                // Publish promotion event to NATS (best-effort).
                let promote_ev =
                    h2ai_types::events::H2AIEvent::AuditDomainPromoted(AuditDomainPromotedEvent {
                        domain: ev.domain.clone(),
                        disagreement_rate: rate,
                        n_observations: n,
                        timestamp_ms: chrono::Utc::now().timestamp_millis() as u64,
                    });
                let task_id_sentinel = h2ai_types::identity::TaskId::from_uuid(uuid::Uuid::nil());
                self.state
                    .nats
                    .publish_event(&task_id_sentinel, &promote_ev)
                    .await
                    .ok();
            } else if currently_promoted && self.auto_demotion {
                // Switch window to demotion window size on first demote check.
                let demote_window = self
                    .windows
                    .entry(ev.domain.clone())
                    .or_insert_with(|| DomainWindow::new(self.demotion_window));
                let dn = demote_window.n_observations();
                let dr = demote_window.disagreement_rate();
                if dn >= self.demotion_window && dr < self.demotion_threshold {
                    self.state
                        .promoted_audit_domains
                        .write()
                        .await
                        .remove(&ev.domain);
                    changed = true;
                    tracing::info!(
                        domain = %ev.domain,
                        rate = dr,
                        n = dn,
                        "shadow_auditor: domain demoted from AND-vote mode"
                    );
                    let demote_ev = h2ai_types::events::H2AIEvent::AuditDomainDemoted(
                        AuditDomainDemotedEvent {
                            domain: ev.domain.clone(),
                            disagreement_rate: dr,
                            n_observations: dn,
                            timestamp_ms: chrono::Utc::now().timestamp_millis() as u64,
                        },
                    );
                    let task_id_sentinel =
                        h2ai_types::identity::TaskId::from_uuid(uuid::Uuid::nil());
                    self.state
                        .nats
                        .publish_event(&task_id_sentinel, &demote_ev)
                        .await
                        .ok();
                }
            }
        }

        if changed {
            let domains: std::collections::HashSet<String> =
                self.state.promoted_audit_domains.read().await.clone();
            if let Err(e) = self.state.nats.put_shadow_promoted_domains(&domains).await {
                tracing::warn!(error = %e, "shadow_auditor: failed to persist promoted domains");
            }
        }

        // Update Prometheus metrics.
        let total_disagreements: u64 = events.iter().filter(|e| e.disagreement).count() as u64;
        let total_events = events.len() as u64;
        let promoted_count = self.state.promoted_audit_domains.read().await.len();
        let mut metrics = self.state.metrics.write().await;
        metrics.shadow_audit_total += total_events;
        metrics.shadow_audit_disagreements += total_disagreements;
        metrics.shadow_audit_promoted_domains = promoted_count;
        if metrics.shadow_audit_total > 0 {
            metrics.shadow_audit_disagreement_rate =
                metrics.shadow_audit_disagreements as f64 / metrics.shadow_audit_total as f64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domain_window_disagreement_rate_empty() {
        let w = DomainWindow::new(10);
        assert_eq!(w.disagreement_rate(), 0.0);
        assert_eq!(w.n_observations(), 0);
    }

    #[test]
    fn domain_window_rates_correctly() {
        let mut w = DomainWindow::new(4);
        w.push(true);
        w.push(false);
        w.push(true);
        w.push(false);
        assert_eq!(w.n_observations(), 4);
        assert!((w.disagreement_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn domain_window_evicts_oldest_on_overflow() {
        let mut w = DomainWindow::new(3);
        // fill with disagreements
        w.push(true);
        w.push(true);
        w.push(true);
        // push agreement — evicts oldest (true), window is now [true, true, false]
        w.push(false);
        assert_eq!(w.n_observations(), 3);
        assert!((w.disagreement_rate() - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn domain_window_all_agreements_rate_zero() {
        let mut w = DomainWindow::new(5);
        for _ in 0..5 {
            w.push(false);
        }
        assert_eq!(w.disagreement_rate(), 0.0);
    }

    #[test]
    fn domain_window_all_disagreements_rate_one() {
        let mut w = DomainWindow::new(5);
        for _ in 0..5 {
            w.push(true);
        }
        assert_eq!(w.disagreement_rate(), 1.0);
    }

    #[test]
    fn domain_window_partial_fill_does_not_count_empty_slots() {
        let mut w = DomainWindow::new(10);
        w.push(true);
        w.push(false);
        // only 2 observations, not 10
        assert_eq!(w.n_observations(), 2);
        assert!((w.disagreement_rate() - 0.5).abs() < 1e-9);
    }
}
