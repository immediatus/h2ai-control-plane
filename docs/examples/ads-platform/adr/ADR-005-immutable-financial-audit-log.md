# ADR-005: Dual-Ledger Architecture — Immutable Financial Audit Log

## Status

Accepted

## Context

The platform bills advertisers for every ad impression delivered. At 1M QPS, billing events accumulate at a rate that makes financial audit a non-trivial problem. Two compliance requirements create a constraint the operational database alone cannot satisfy:

**SOX (Sarbanes-Oxley) compliance:** Financial records must be non-repudiable — records cannot be altered or deleted after they are written. CockroachDB, as a mutable relational database, allows rows to be updated, deleted, or modified by database administrators. This violates SOX's tamper-evidence requirement.

**Tax record retention:** Financial records must be retained for 7 years. A mutable operational database that supports row deletion cannot provide this guarantee without additional enforcement.

**Dispute resolution:** Advertisers dispute charges. Without a verifiable, immutable record of exactly when each impression was delivered and what price was charged, disputes cannot be resolved with confidence. "What does the database say" is not sufficient when the database is mutable.

The operational database (CockroachDB) is deliberately mutable — it is optimized for real-time budget enforcement, corrections, and operational efficiency. Making the operational database immutable would require removing standard database operations and adding significant complexity without improving serving latency.

## Decision

**Dual-ledger architecture:**

- **Operational Ledger (CockroachDB):** Mutable. Optimized for real-time budget checks (≤3ms read latency), campaign configuration, billing writes. Rows can be updated for corrections and deleted during cleanup. This is the live operational system.

- **Immutable Audit Log (Kafka → ClickHouse):** Append-only. Every financial event — budget deductions, charges, refunds, corrections — is published to the Kafka `financial-events` topic and consumed into ClickHouse append-only MergeTree tables. Records are never updated or deleted. Hash chaining between records detects any tampering. Retained for 7 years minimum.

**Event flow:**
Every operation that affects advertiser billing publishes an event to Kafka `financial-events` synchronously as part of the operation. The event reaches ClickHouse within seconds via the Kafka consumer. A daily reconciliation job compares total spend in CockroachDB against the sum of events in ClickHouse and alerts on discrepancies exceeding 0.01%.

**Infrastructure cost:** Kafka cluster + ClickHouse deployment adds approximately 15–20% to database infrastructure budget. This is accepted as the cost of regulatory compliance and audit confidence.

## Consequences

**Easier:** Auditor requests are answered from ClickHouse — no impact on the operational database. Advertiser disputes have a verifiable, tamper-evident record. SOX and tax retention compliance is satisfied without modifying the operational database design.

**Harder:** Every billing operation now has two write paths (CockroachDB + Kafka). The reconciliation job must be maintained and its alerts must be actionable. ClickHouse requires a separate operational team or expertise.

## Constraints

- Every financial event — ad impression billing, budget deduction, refund, correction, adjustment — must be published to the Kafka `financial-events` topic as part of the transaction that created it. Publishing to Kafka is not optional or best-effort for financial events.
- Financial events must not be deleted or updated in ClickHouse under any circumstances. Corrections are appended as new events (e.g., a `BillingCorrection` event that references the original event), never mutations to existing records.
- ClickHouse MergeTree tables storing financial events must have a `TTL` of at minimum 7 years. Shorter TTL configurations are prohibited.
- The daily reconciliation job must compare total spend per advertiser per day between CockroachDB and ClickHouse. It must alert when the discrepancy exceeds 0.01% of daily spend for any advertiser.
- The operational CockroachDB ledger may be modified by application code for corrections and adjustments. All such modifications must generate a corresponding corrective event published to Kafka.
- Access to ClickHouse financial data must be read-only for all application services. Only the Kafka consumer (the ClickHouse writer) may write to financial event tables. No application service may issue INSERT, UPDATE, or DELETE statements against ClickHouse financial tables directly.
- Hash chaining between ClickHouse financial records must be implemented and validated by the reconciliation job. A broken hash chain must trigger an immediate alert — it indicates record tampering or data loss.

## References

- Series: "Architecting Real-Time Ads Platform", Part 1 — Financial Integrity: Immutable Audit Log
- Series: Part 3 — Immutable Financial Audit Log: Compliance Architecture
- Compliance: SOX (Sarbanes-Oxley) tamper-evidence requirement
- Compliance: Tax record retention — 7 years minimum
