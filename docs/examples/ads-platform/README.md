# Example Project: Real-Time Ads Platform

This example is derived from the blog series **"Architecting Real-Time Ads Platform"** by Yuriy Polyulya. The series documents architectural decisions for a system serving 400M+ DAU at 1M+ QPS with 150ms P95 latency.

The constraint documents here capture the actual decisions and their rationale from the series. They are structured in `ConstraintDoc` YAML format so that H2AI Control Plane can use them as the Dark Knowledge corpus for integration testing.

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

All constraints are in YAML format and loaded automatically when `constraint_wiki.enabled = true` in `h2ai.toml`.

### Platform constraints

| ID | Title | Domains | Key rule |
|---|---|---|---|
| [CONSTRAINT-001](constraints/CONSTRAINT-001-stateless-request-services.yaml) | Stateless Request-Handling Services | availability, scalability | No per-user state; no sticky sessions; L1 cache TTL ≤60s |
| [CONSTRAINT-002](constraints/CONSTRAINT-002-service-communication-protocols.yaml) | Service Communication Protocols | performance, consistency | gRPC internal, REST external; no async on critical path |
| [CONSTRAINT-003](constraints/CONSTRAINT-003-rtb-timeout-strategy.yaml) | RTB Timeout Strategy | latency, availability | T_global=100ms; HdrHistogram P95; ≥100 samples before activation |
| [CONSTRAINT-004](constraints/CONSTRAINT-004-budget-pacing-idempotency.yaml) | Budget Pacing Idempotency | financial-accuracy, consistency | Pre-allocation + Redis atomic CAS; no CockroachDB on critical path |
| [CONSTRAINT-005](constraints/CONSTRAINT-005-immutable-financial-audit-log.yaml) | Immutable Financial Audit Log | compliance, durability | Every billing event to Kafka; no ClickHouse mutations; 7-year retention |
| [CONSTRAINT-006](constraints/CONSTRAINT-006-java-zgc-runtime.yaml) | Java ZGC Runtime | performance, reliability | Java 21 + Generational ZGC; 32GB heap exactly; virtual threads required |
| [CONSTRAINT-007](constraints/CONSTRAINT-007-tiered-data-consistency.yaml) | Tiered Data Consistency | consistency, performance | Budget checks bypass cache; config TTL ≤5s; HLC for billing |

### Framework implementation constraints

These constraints were used to guide H2AI's own evaluation cache implementation and serve as a self-referential test of the framework.

| ID | Title | Domains | Key rule |
|---|---|---|---|
| [CONSTRAINT-CACHE-1](constraints/CONSTRAINT-CACHE-1-similarity-function.yaml) | Similarity Function | performance, correctness | Use `repetition::similarity` (Jaccard); no cosine/embedding on hot path |
| [CONSTRAINT-CACHE-2](constraints/CONSTRAINT-CACHE-2-concurrent-map.yaml) | Concurrent Map | performance, correctness | DashMap, not `Arc<Mutex<HashMap>>`; no blocking across `.await` |
| [CONSTRAINT-CACHE-3](constraints/CONSTRAINT-CACHE-3-task-scoped-lifetime.yaml) | Task-Scoped Lifetime | correctness, memory | Per-task cache; no global/static storage; drop on task completion |
| [CONSTRAINT-CACHE-4](constraints/CONSTRAINT-CACHE-4-typed-telemetry.yaml) | Typed Telemetry | observability | Cache hits → `VerificationScoredEvent` field; not log-only |

## Task manifests

| Task | Constraints exercised | Expected behavior |
|---|---|---|
| [task-dsp-onboarding.json](tasks/task-dsp-onboarding.json) | CONSTRAINT-003 | Proposals raising T_global or activating before 100 samples are pruned |
| [task-budget-enforcement-crash-recovery.json](tasks/task-budget-enforcement-crash-recovery.json) | CONSTRAINT-004 + CONSTRAINT-005 | Proposals using non-atomic check-then-act or missing Kafka publish are pruned |
| [task-ml-feature-latency.json](tasks/task-ml-feature-latency.json) | CONSTRAINT-001 + CONSTRAINT-006 + CONSTRAINT-007 | Proposals touching heap size, caching budget data, or using platform threads are pruned |

## Running the e2e test

```bash
# 1. Start NATS with JetStream (if not already running)
/tmp/nats-server -js -p 4222 &

# 2. Start the API server with the local config
H2AI_CONFIG=h2ai.toml cargo run --release --bin h2ai-control-plane
# Calibration runs automatically at startup — wait for "startup calibration complete" log line.

# 3. Submit a task and stream events
curl -s -X POST http://localhost:8080/tasks \
  -H "Content-Type: application/json" \
  -d @docs/examples/ads-platform/tasks/task-dsp-onboarding.json | jq .

# 4. Stream the task events (replace TASK_ID)
curl -N http://localhost:8080/tasks/TASK_ID/events
```

## Source series

- Part 1: System Foundation & Latency Engineering
- Part 2: Dual-Source Revenue Engine — OpenRTB & ML Inference Pipeline
- Part 3: Caching, Auctions & Budget Control
- Part 4: Production Operations — Fraud, Multi-Region & Operational Excellence
- Part 5: Complete Implementation Blueprint — Technology Stack & Architecture Guide
