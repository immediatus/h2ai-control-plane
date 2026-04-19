# ADR-001: Stateless Request-Handling Services

## Status

Accepted

## Context

The platform serves 400M+ daily active users at 1M+ QPS peak with a 150ms P95 latency budget. Early prototypes explored stateful service designs (session affinity, in-process caches of user context, per-instance bid history). These created three operational failure modes at scale:

1. **Deployment friction** — stateful services required session draining before pod shutdown. Rolling deployments that took seconds for stateless services took hours for stateful ones.
2. **Instance failure cascades** — when a stateful Ad Server instance died, all in-flight requests it was processing were lost with no recovery path, since context was not externalized.
3. **Scaling bottleneck** — adding instances to a stateful tier required redistributing state, which required coordination, which added latency proportional to the number of existing instances.

At 1M QPS, any architectural property that makes horizontal scaling expensive is a load-bearing failure.

## Decision

All request-handling services — **Ad Server Orchestrator, Auction Engine, ML Inference Service, RTB Gateway, User Profile Service** — are stateless. They hold no session state, no per-request context, and no in-process mutable state that is not rebuilt from external storage on every request.

State lives exclusively in dedicated storage layers:
- **User profiles and ML features** — distributed cache (L1 in-process read-through, L2 Redis cluster)
- **Advertiser budgets and campaign configs** — CockroachDB (strongly consistent)
- **Budget counters** — Redis atomic counters
- **Financial events** — Kafka → ClickHouse append-only log

Request-handling service instances are interchangeable. Any instance can handle any request. Load balancers make no routing decisions based on user identity or prior request history.

## Consequences

**Easier:**
- Horizontal scaling: add instances at any time with no coordination
- Zero-downtime deployments: rolling update with no session draining
- Fault tolerance: failed instance is replaced without state recovery; next request goes to any other instance
- Load balancing: pure round-robin or least-connections, no sticky sessions

**Harder:**
- Every request must reconstruct context from storage. Cache hit rate becomes load-bearing — a cache miss on the critical path adds 10ms+ database latency to every request that misses.
- Local in-process caches (L1) must be treated as pure read-through caches, never as authoritative state. Updates from other instances must invalidate L1 within a bounded time window.

## Constraints

- Ad Server Orchestrator, Auction Engine, ML Inference Service, RTB Gateway, and User Profile Service must not store any per-user or per-session state in process memory beyond the lifetime of a single request.
- These services must not use sticky-session load balancing. Any instance must be able to serve any request without prior request history.
- In-process L1 caches are permitted as read-through caches only. They must have a TTL no greater than 60 seconds. They must not be the authoritative source of any data that affects billing or auction decisions.
- Service instances must be replaceable without state migration. Killing any instance must not require a warm-up period exceeding 60 seconds before the replacement instance serves production traffic.
- No request handler may write to a local file or local database. All writes go to the distributed storage tier.

## References

- Series: "Architecting Real-Time Ads Platform", Part 1 — Stateless Design Philosophy
- Latency budget: 150ms P95 end-to-end; storage tier latency budget ≤10ms for profile + feature reads
