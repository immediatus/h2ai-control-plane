# CONSTRAINT-007: Tiered Data Consistency — Different Consistency Models per Data Type

## Status

Accepted

## Context

The platform handles data with radically different correctness requirements. Applying a single consistency model across all data types forces a false choice: either accept strong consistency for everything (which makes user profile reads 10ms+ when they could be sub-millisecond) or accept eventual consistency for everything (which permits incorrect billing and budget enforcement).

Three data types represent distinct positions on the consistency/latency tradeoff:

**Advertiser budgets:** If the budget counter reads stale, the platform over-delivers. Over-delivery above 1% is a billing accuracy violation. Strong consistency is non-negotiable.

**User targeting profiles:** If a user's profile update (new interest signal, demographic change) takes 5–10 seconds to propagate to all Ad Servers, the platform serves a slightly less-targeted ad for those seconds. The revenue impact is negligible — a targeting signal that is 5 seconds stale does not materially affect CTR prediction. Eventual consistency with seconds of lag is acceptable.

**ML features:** Feature staleness of several minutes has minimal impact on GBDT CTR prediction accuracy. The model was trained on features that are minutes-to-hours old by design (feature computation pipelines run on batch schedules). Sub-minute feature freshness provides no measurable lift.

**Campaign configurations:** An advertiser pauses a campaign. The change must take effect immediately. If Ad Servers continue serving from a stale configuration for 30 seconds, the advertiser is charged for impressions they explicitly stopped. Strong consistency is required.

**Billing events:** For SOX compliance and audit, billing events require linearizable ordering — there must be a total order on all financial events, with no ambiguity about which event happened before another.

## Decision

**Three-tier consistency model:**

| Data type | Consistency | Storage | Rationale |
|---|---|---|---|
| Advertiser budgets | Strong (atomic Redis counters) | Redis + CockroachDB | Financial accuracy ≤1% overspend |
| Campaign configurations | Strong (immediate visibility) | CockroachDB | Pausing a campaign must take effect immediately |
| Billing events | Linearizable (HLC timestamps) | CockroachDB + Kafka → ClickHouse | SOX audit trail requires total ordering |
| User targeting profiles | Eventual (seconds of lag) | L1 in-process cache + L2 Redis | Profile updates have negligible targeting impact at seconds of staleness |
| ML features | Eventual (minutes of lag) | L1 in-process cache + L2 Redis + Feature Store | Model trained on batch features; sub-minute freshness provides no measurable lift |

**L1/L2 cache hierarchy for eventually-consistent data:**
- L1 (in-process, per-instance): sub-millisecond reads, TTL ≤60s. Never authoritative.
- L2 (Redis cluster, shared): 1–2ms reads, TTL appropriate to data type. Read-through from source of truth on miss.
- Cache invalidation: TTL-based expiry. Event-driven invalidation is not used — the latency of invalidation messages is not meaningfully different from TTL expiry at the consistency granularity these data types require.

**Schema evolution:**
Schema changes to strongly-consistent stores (CockroachDB) follow the expand-contract pattern: add new columns/indexes online with `CONCURRENTLY` (non-blocking), deploy application code that writes to both old and new schema, validate, then remove old schema. Direct table rewrites or dual-write patterns are prohibited outside of schema migration windows.

## Consequences

**Easier:** User profile reads are sub-millisecond (L1 cache hit). Feature reads are sub-millisecond. Budget checks are sub-millisecond (Redis atomic counter read). Campaign configuration changes take effect within a single CockroachDB replication round-trip.

**Harder:** Engineers must know which consistency tier applies to each data type. Making budget decisions using stale cache data would be a critical bug. The system must prevent cache-inconsistent budget checks by design — budget enforcement reads must bypass cache entirely.

## Constraints

- Advertiser budget checks must read from Redis atomic counters directly. Budget checks must not read from L1 or L2 cache. Serving an ad based on a cached (potentially stale) budget balance is prohibited.
- Campaign configuration changes (pause, resume, budget modification) must be read from CockroachDB within one replication round-trip of the write. Ad Servers must not serve impressions for paused campaigns based on cached configuration. The configuration cache TTL must not exceed 5 seconds.
- User profile data may be served from L1 or L2 cache with TTL ≤60 seconds. Serving user profile data that is more than 60 seconds stale is a configuration violation.
- ML features may be served from L1 or L2 cache with TTL ≤5 minutes. Feature staleness beyond 5 minutes must trigger a cache miss and a fetch from the Feature Store.
- Billing events must be written with HLC (Hybrid Logical Clock) timestamps. Wall clock timestamps are not permitted for billing events — clock skew between nodes would break linearizability of the financial audit trail.
- Schema changes to CockroachDB must use the expand-contract pattern. `ALTER TABLE` with blocking DDL is prohibited in production. Index creation must use `CONCURRENTLY`. Partition restructuring requiring dual-write must be scoped to a documented migration window.
- A service must not apply the wrong consistency model to a data type. A component that reads advertiser budget from an L1 or L2 cache rather than from Redis atomic counters is a critical correctness violation and must be caught in code review.

## References

- Series: "Architecting Real-Time Ads Platform", Part 1 — Consistency Requirements by Data Type
- Series: Part 3 — Distributed Caching Architecture; Budget Pacing
- Series: Part 4 — Schema Evolution: Zero-Downtime Data Migration
- Financial accuracy: ≤1% budget overspend (drives strong consistency for budgets)
