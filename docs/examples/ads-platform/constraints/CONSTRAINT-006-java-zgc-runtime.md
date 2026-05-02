# CONSTRAINT-006: Java 21 + Generational ZGC for Request-Path Services

## Status

Accepted

## Context

The platform's P99 tail latency requirement is directly threatened by garbage collection pauses. At 1M QPS, the P99 represents 10,000 requests per second. A GC pause that stops the JVM for 50ms causes every one of those 10,000 requests to time out during the pause.

GC analysis across candidate runtimes:

| Runtime | GC pause at P99.9 | Impact at 1M QPS |
|---|---|---|
| Java 17 + G1GC | 41–55ms stop-the-world | 410–550 requests timeout per pause |
| Java 21 + Shenandoah | <10ms concurrent | ~100 requests affected, 15–20% CPU overhead |
| Java 21 + Generational ZGC | <2ms P99.9 (32GB heap) | ~20 requests affected, 10% CPU overhead |
| Go | Sub-millisecond, but no JIT warmup | Cold start latency; limited ads ecosystem |
| Rust | No GC | Requires rewriting entire ad stack; hiring risk |

G1GC's 41–55ms stop-the-world pauses consume up to 37% of the 150ms P95 latency budget in a single GC event. Shenandoah is concurrent but imposes 15–20% CPU overhead at 1M QPS, translating to ~$500K/year in additional compute cost at the platform's scale. Rust eliminates GC entirely but requires rewriting the ad stack and multiplies hiring difficulty — the ads ecosystem tooling (OpenRTB libraries, feature store SDKs, ML serving frameworks) is Java-native.

Netflix production data (March 2024, JDK 21, Generational ZGC): "no explicit tuning required" for critical streaming services, <2ms P99.9 pauses on 32GB heaps under sustained load.

## Decision

Java 21 with Generational ZGC for all request-path services: Ad Server Orchestrator, Auction Engine, ML Inference Service, RTB Gateway.

**Heap sizing: 32GB per instance.**

Derived from allocation rate analysis: at 5,000 QPS per instance, average request creates ~50KB of objects → allocation rate = 250 MB/sec. With ZGC's concurrent collection, the heap cycles every ~2 minutes at 50% utilization on a 32GB heap. This leaves sufficient headroom for allocation bursts without triggering emergency GC.

**Thread configuration:**
- Request threads: 200 virtual threads (Java 21 Project Loom) — lightweight, no OS thread exhaustion under concurrent request load
- gRPC I/O threads: 32 threads (2× CPU cores) — dedicated network I/O pool, never shared with request processing
- Background tasks: 16 threads — event publishing, cache warming, async operations

**Validation:**
With G1GC's 41–55ms pauses at P99.9, 410–550 requests would timeout per pause event. With ZGC's <2ms P99.9 pauses, only ~20 requests are affected — a 98% reduction in GC-caused timeouts.

## Consequences

**Easier:** GC pauses are no longer a source of P99 latency spikes. ZGC's concurrent operation eliminates stop-the-world pauses from the latency budget calculation. Java ecosystem gives access to battle-tested OpenRTB, protobuf, gRPC, and ML serving libraries.

**Harder:** 32GB heap per instance raises instance memory requirements. EC2 instance selection must target 64GB RAM (32GB JVM + 32GB OS page cache for Redis client and file I/O). Engineers must understand ZGC behavior — specifically, that ZGC trades CPU for pause reduction. CPU utilization runs ~10% higher than G1GC.

## Constraints

- Request-path services (Ad Server Orchestrator, Auction Engine, ML Inference Service, RTB Gateway) must use Java 21 or later with Generational ZGC enabled (`-XX:+UseZGC -XX:+ZGenerational`). G1GC and Shenandoah are not permitted for these services.
- Heap size for request-path services must be 32GB (`-Xms32g -Xmx32g`). Smaller heaps increase GC frequency; larger heaps increase evacuation pause duration. Both directions risk violating the P99 tail latency constraint.
- Request handling must use Java 21 virtual threads (Project Loom), not platform threads. Platform thread pools with fixed sizes cannot handle concurrent request spikes without thread exhaustion.
- gRPC I/O must run on a dedicated thread pool of 32 threads, separate from the virtual thread request pool. Sharing I/O threads with request processing creates head-of-line blocking.
- Background tasks (event publishing, cache warming, ML feature computation) must not run on the request thread pool. They must use the dedicated 16-thread background pool.
- The JVM must be configured with explicit heap settings: `-Xms32g -Xmx32g`. Allowing the JVM to size the heap automatically is prohibited — automatic sizing under load can select sub-optimal heap sizes.
- GC pause duration at P99.9 must be monitored via the Prometheus `jvm_gc_pause_seconds` histogram. An alert must fire when P99.9 exceeds 5ms — this indicates ZGC is not operating within design parameters and investigation is required.

## References

- Series: "Architecting Real-Time Ads Platform", Part 5 — Runtime & Garbage Collection: Java 21 + ZGC
- Series: Part 1 — P99 Tail Latency Defense: The Unacceptable Tail
- Netflix ZGC production report: JDK 21, March 2024
- Latency budget: GC pause budget ≤2ms P99.9 to stay within 150ms P95 SLO
