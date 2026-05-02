# CONSTRAINT-002: Service Communication Protocols — gRPC Internal, REST External

## Status

Accepted

## Context

The ad serving critical path makes 6–8 service-to-service calls within the 150ms latency budget. Each call must complete within 5–10ms. At 1M QPS, the serialization overhead of each call is not a rounding error — it is a material fraction of the latency budget.

Two communication patterns exist in the system: calls between internal services we own, and calls to external DSPs (Demand-Side Platforms) that implement the OpenRTB industry standard.

JSON over HTTP/1.1 was the original implementation choice (simplicity). Under load testing:
- JSON parsing added 2–5ms per internal call
- At 6 internal calls per request: 12–30ms consumed by serialization alone
- This consumed 8–20% of the 150ms P95 budget before any business logic ran

The 150ms P95 budget does not accommodate per-call JSON parsing overhead across the internal call chain.

## Decision

**Internal service-to-service communication:** gRPC over HTTP/2 with Protocol Buffers serialization.

**External communication (DSPs, publisher SDK, advertiser API):** REST with JSON over HTTP/1.1 or HTTP/2.

The split is enforced at the service boundary: internal APIs are defined only in `.proto` files. External APIs are defined only as OpenAPI specs. No internal service exposes a REST endpoint to another internal service.

**Why gRPC internally:**
- Protocol Buffers serialization: 3–10× smaller payloads than JSON, sub-millisecond serialization overhead vs 2–5ms for JSON
- HTTP/2 multiplexing: multiple concurrent RPCs share a single TCP connection, avoiding connection setup overhead
- Schema-based contracts: proto definitions provide compile-time validation between services and catch API drift at build time
- At 5,000 QPS per Ad Server instance with 32 persistent gRPC connections per downstream service: ~156 req/s per connection, fully reusing connections

**Why REST externally:**
- OpenRTB protocol mandates JSON over HTTP — DSPs cannot be asked to implement gRPC
- External parties cannot be required to share proto schema definitions
- JSON is human-readable, which is valuable for integration debugging with external DSP engineering teams
- The 100ms RTB network round-trip dwarfs 2–5ms JSON parsing overhead, making it negligible

**Why not async messaging on the critical path:**
The 150ms P95 budget does not accommodate message queue hops. Each hop adds minimum 5–20ms (queue write + consumer poll). Async messaging is used for off-critical-path workflows only: billing events, analytics pipelines, ML feature computation, audit log.

## Consequences

**Easier:** Internal call latency stays under 10ms. Schema evolution is caught at compile time via proto compatibility checks. HTTP/2 connection reuse eliminates TLS handshake overhead on hot paths.

**Harder:** Developers must maintain `.proto` definitions for every internal API. Proto schema evolution requires backward-compatibility discipline (field numbering, optional vs required). New engineers unfamiliar with gRPC face a learning curve.

**Retry policy:** gRPC retries are permitted only on `UNAVAILABLE` status code (service temporarily down). Retrying `DEADLINE_EXCEEDED` (timeout) is prohibited — retrying timed-out requests amplifies cascading failures under load.

## Constraints

- Internal service-to-service calls must use gRPC with Protocol Buffers. REST/JSON is not permitted for internal APIs.
- External integrations (DSPs, publisher SDK, advertiser management API) must use REST with JSON. gRPC is not permitted for external APIs.
- The async message bus (Kafka/NATS) must not be used on the ad serving critical path. Message bus is permitted only for: billing events, audit log writes, ML feature pipeline, analytics.
- gRPC retry policy must not retry on `DEADLINE_EXCEEDED`. Only `UNAVAILABLE` may be retried, with a maximum of 2 attempts and exponential backoff (10ms → 50ms initial).
- Internal services must not expose REST endpoints to other internal services. REST exposure is only for external-facing APIs.
- gRPC keepalive must be configured: ping interval 60s, timeout 20s. This detects dead connections before requests fail.
- Message size limit on gRPC calls: 4MB. Requests exceeding this limit must be rejected at the framework level, not passed to application logic.

## References

- Series: "Architecting Real-Time Ads Platform", Part 1 — Communication Architecture
- Series: Part 5 — gRPC Configuration, Linkerd service mesh
- Latency budget: service-to-service calls ≤10ms; JSON serialization overhead 2–5ms per call
