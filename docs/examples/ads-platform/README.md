# Example Project: Real-Time Ads Platform

This example is derived from the blog series **"Architecting Real-Time Ads Platform"** by Yuriy Polyulya. The series documents architectural decisions for a system serving 400M+ DAU at 1M+ QPS with 150ms P95 latency.

The constraint documents here capture the actual decisions and their rationale from the series. They are structured in `ConstraintDoc` format so that H2AI Control Plane can use them as the Dark Knowledge corpus for integration testing.

## System summary

A real-time advertising platform with the following characteristics:

- **Scale:** 400M DAU, 1M+ QPS peak (1.5M QPS capacity with 50% headroom)
- **Latency:** 150ms P95 end-to-end; 100ms RTB budget; ≤10ms for internal service calls
- **Financial accuracy:** ≤1% budget overspend; SOX-compliant immutable audit trail
- **Architecture:** Stateless request services, dual-source auction (ML internal + RTB external), tiered consistency model

## Services

| Service | Role |
|---|---|
| Ad Server Orchestrator | Coordinates request flow, fans out to ML and RTB in parallel |
| Auction Engine | Runs eCPM-based first-price auction across all bids |
| ML Inference Service | GBDT-based CTR prediction using pre-materialized feature vectors |
| RTB Gateway | OpenRTB 2.5 fanout to 50+ DSPs with adaptive per-DSP timeouts |
| Atomic Pacing Service | Budget pre-allocation and idempotent spend enforcement via Redis |
| Feature Store | Materialized user targeting features, L1/L2 cache hierarchy |
| User Profile Service | User identity and targeting attributes |

## Constraint corpus

| Constraint | Decision | Key constraints |
|---|---|---|
| [CONSTRAINT-001](constraints/CONSTRAINT-001-stateless-request-services.md) | All request services are stateless | No per-user state across requests; no sticky sessions; L1 cache TTL ≤60s |
| [CONSTRAINT-002](constraints/CONSTRAINT-002-service-communication-protocols.md) | gRPC internal, REST external | No REST between internal services; no gRPC for external; no async on critical path |
| [CONSTRAINT-003](constraints/CONSTRAINT-003-rtb-timeout-strategy.md) | Adaptive per-DSP timeouts via HdrHistogram | T_global=100ms; no exact-percentile substitutes; in-process only; ≥100 samples before activation |
| [CONSTRAINT-004](constraints/CONSTRAINT-004-budget-pacing-idempotency.md) | Pre-allocation + Redis atomic check-and-set | No CockroachDB reads on critical path; atomic Lua only; TTL=30s on idempotency keys |
| [CONSTRAINT-005](constraints/CONSTRAINT-005-immutable-financial-audit-log.md) | Dual-ledger: operational (CockroachDB) + immutable (Kafka → ClickHouse) | Every billing event to Kafka; no ClickHouse mutations; 7-year retention |
| [CONSTRAINT-006](constraints/CONSTRAINT-006-java-zgc-runtime.md) | Java 21 + Generational ZGC, 32GB heap | No G1GC; heap exactly 32GB; virtual threads required; separate gRPC thread pool |
| [CONSTRAINT-007](constraints/CONSTRAINT-007-tiered-data-consistency.md) | Different consistency per data type | Budget checks bypass cache; config TTL ≤5s; ML features TTL ≤5min; HLC for billing |

## Task manifests

| Task | Tests | Expected behavior |
|---|---|---|
| [task-dsp-onboarding.json](tasks/task-dsp-onboarding.json) | CONSTRAINT-003 timeout constraints | Proposals raising T_global or activating before 100 samples are pruned |
| [task-budget-enforcement-crash-recovery.json](tasks/task-budget-enforcement-crash-recovery.json) | CONSTRAINT-004 + CONSTRAINT-005 idempotency | Proposals using non-atomic check-then-act or missing Kafka publish are pruned |
| [task-ml-feature-latency.json](tasks/task-ml-feature-latency.json) | CONSTRAINT-001 + CONSTRAINT-006 + CONSTRAINT-007 | Proposals touching heap size, caching budget data, or using platform threads are pruned |

## Running as integration tests

```bash
# Copy constraint corpus into the configured path
cp -r docs/examples/ads-platform/constraints/ /path/to/constraints/

# Run calibration
curl -X POST http://localhost:8080/calibrate

# Submit each task and verify outcomes
for task in docs/examples/ads-platform/tasks/*.json; do
  echo "Submitting $task..."
  curl -s -X POST http://localhost:8080/tasks \
    -H "Content-Type: application/json" \
    -d @"$task" | jq .
done

# Or run the integration test suite
cargo nextest run --test integration -- --test-threads=1
```

## Source series

- Part 1: System Foundation & Latency Engineering
- Part 2: Dual-Source Revenue Engine — OpenRTB & ML Inference Pipeline
- Part 3: Caching, Auctions & Budget Control
- Part 4: Production Operations — Fraud, Multi-Region & Operational Excellence
- Part 5: Complete Implementation Blueprint — Technology Stack & Architecture Guide
