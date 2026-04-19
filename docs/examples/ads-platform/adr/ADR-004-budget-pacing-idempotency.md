# ADR-004: Budget Pacing — Pre-Allocation with Idempotency Protection

## Status

Accepted

## Context

Advertisers set daily budgets. The platform must not deliver more than the advertiser's budget — over-delivery is a billing accuracy violation with financial and legal consequences. The platform serves 1M QPS across hundreds of Ad Server instances. Two naive approaches both fail at this scale:

**Naive approach A — Check CockroachDB on every request:**
- CockroachDB read latency: ~10ms
- At 1M QPS: 10M ms of database time consumed per second — 10,000× over capacity
- CockroachDB max throughput: ~100K QPS for this workload
- Result: database becomes the bottleneck at 10% of target QPS

**Naive approach B — No coordination, check later:**
- At 1M QPS with 100 Ad Server instances: over-delivery scales linearly with instance count
- With no coordination, each instance is unaware of what others are spending
- Result: unbounded over-delivery, violates the ≤1% accuracy constraint

**Budget idempotency gap:** Even with a correctly designed budget check, distributed systems have a critical failure scenario. An Ad Server can crash after successfully debiting the budget but before delivering the ad or returning a response. The client retries the request. Without idempotency protection, the retry debits the budget again — double-billing for a single impression.

At 1M QPS with 0.1% retry rate: 1,000 retries/second. Without idempotency: 100% of retries double-bill. Impact: 0.1% of traffic at 2× billing = systematic violation of the ≤1% accuracy constraint at 10× the allowed magnitude.

## Decision

**Budget pacing: pre-allocation with Redis atomic counters.**

1. **Atomic Pacing Service** pre-allocates budget chunks to Ad Server instances using Redis `DECRBY` (atomic).
2. Ad Servers spend from their local allocation atomically — no coordination per request.
3. Every 30 seconds, Ad Servers reconcile: return unused budget to the Atomic Pacing Service via `INCRBY`.
4. CockroachDB records all spend events with HLC (Hybrid Logical Clock) timestamps as the authoritative billing ledger.
5. A Timeout Monitor releases stale allocations after 5 minutes to handle server crashes.
6. When remaining budget drops below 10%, allocation chunk size is reduced dynamically: `A_new = B_remaining / (S × 10)`. This bounds max over-delivery to ≤1% of daily budget.

**Idempotency protection: Redis Lua atomic check-and-set.**

Every budget deduction is protected by an idempotency key. The Redis Lua script atomically:
1. Checks if `idem:campaign_{id}:{client_request_id}` exists.
2. If yes: returns the cached result without debiting. The budget is not touched again.
3. If no: debits the budget AND stores the idempotency key with 30-second TTL in the same atomic operation.

Key naming: `idem:campaign_{campaign_id}:{client_request_id}_{timestamp_bucket}`

The `campaign_id` prefix ensures Redis cluster sharding keeps keys co-located with the campaign's budget counter. The 30-second TTL prevents memory accumulation. The `timestamp_bucket` prevents cross-window collisions.

**Mathematical bound:**

Maximum over-delivery: `OverDelivery_max = S × A` where S = server count, A = allocation chunk size.

With dynamic chunk sizing at <10% remaining: `A_new = B_remaining / (S × 10)` → max over-delivery reduces to ~1% of budget.

## Consequences

**Easier:** Per-request budget check is sub-millisecond (Redis local allocation). No CockroachDB read on the critical path. Over-delivery is mathematically bounded. Double-billing is prevented by construction.

**Harder:** System has three moving parts (Redis counters, CockroachDB ledger, Kafka audit log) that must stay consistent. Daily reconciliation job must catch and alert on discrepancies between Redis state and CockroachDB ledger. Idempotency key TTL (30s) means retries arriving after 30s are not protected — but this is acceptable since the network timeout for the original request is shorter.

## Constraints

- Budget checks on the ad serving critical path must use Redis atomic counters, not CockroachDB reads. CockroachDB must not be queried per ad request for budget enforcement.
- All budget deductions must use the idempotency key mechanism. Budget deductions without idempotency key protection are prohibited. This applies to both the Atomic Pacing Service and any service that performs a budget-affecting operation.
- The Redis Lua script that deducts budget must perform atomic check-and-set: check the idempotency key AND deduct the budget in a single Lua transaction. Separating these into two Redis commands is prohibited — it creates a race condition between the key check and the deduction.
- Idempotency key TTL must be 30 seconds. Keys with no TTL are prohibited — they will exhaust Redis memory.
- Idempotency key naming must follow the pattern `idem:campaign_{campaign_id}:{client_request_id}_{timestamp_bucket}`. Flat key naming without `campaign_id` prefix is prohibited — it prevents Redis cluster sharding from co-locating keys with their budget counters.
- When budget remaining drops below 10%, the Atomic Pacing Service must reduce allocation chunk size using the formula `A_new = B_remaining / (S × 10)`. Maintaining full allocation size at <10% remaining is prohibited.
- The Timeout Monitor must release stale allocations after 5 minutes. Allocations must not be held indefinitely — they prevent budget from being reallocated to other Ad Server instances after a crash.
- Every spend event must be published to the Kafka `financial-events` topic and consumed into the ClickHouse audit log. Budget deductions recorded only in Redis and not in the immutable audit log violate the dual-ledger compliance requirement (see ADR-005).

## References

- Series: "Architecting Real-Time Ads Platform", Part 3 — Budget Pacing: Distributed Spend Control
- Series: Part 3 — Idempotency Protection: Defending Against Double-Debits
- Financial accuracy requirement: ≤1% budget overspend
