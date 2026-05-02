# CONSTRAINT-003: RTB Timeout Strategy — Adaptive Per-DSP Timeouts with HdrHistogram

## Status

Accepted

## Context

The RTB Gateway fans out bid requests to 50+ external DSPs simultaneously. The global RTB timeout is 100ms — the industry-standard maximum after which an auction must proceed with whatever bids have arrived. DSPs that have not responded by the deadline are excluded from the auction.

Three approaches were considered for handling the 100ms deadline:

**Strategy A — Hard global timeout:** All DSPs get exactly 100ms. Simple to implement. Problem: fast DSPs (that typically respond in 60–70ms) are held to the same 100ms wall clock. The auction cannot complete until the timeout fires, even when all fast DSPs have already responded.

**Strategy B — Adaptive per-DSP timeouts:** Each DSP gets a timeout based on its observed P95 latency. Fast DSPs get lower timeouts (60–70ms). Slow or variable DSPs get the global max (100ms). The auction can proceed as soon as all fast DSPs respond, without waiting for the global wall clock.

**Strategy C — Progressive auction:** Run a preliminary auction at 80ms with available bids, update the winner if late arrivals (up to 100ms) beat the current best bid.

Load testing showed that DSP response latency distributions are highly heterogeneous: some DSPs are consistently under 70ms, others are consistently over 90ms, and some are bimodal (fast most of the time, slow under load). A single global timeout optimizes for none of them.

## Decision

Adaptive per-DSP timeouts (Strategy B) are the primary timeout mechanism.

Each Ad Server instance maintains an in-memory latency histogram per DSP using **HdrHistogram** (High Dynamic Range Histogram). The per-DSP timeout is:

```
T_dsp = min(P95(histogram_dsp), T_global)
```

where `T_global = 100ms`.

**Why HdrHistogram over t-digest:**
HdrHistogram provides exact percentile calculations with bounded memory (O(1) per recording). t-digest uses approximation. For timeout decisions that directly affect auction revenue, precision is required — approximate percentiles can systematically over- or under-cut fast DSPs.

**Histogram configuration:**
- Range: 1–1000ms
- Precision: 2 significant digits
- Memory: ~2KB per DSP histogram (50 DSPs × 2KB = 100KB per Ad Server instance)
- Window: Rolling 5-minute window (balances responsiveness to DSP latency changes vs. stability against outliers)
- Storage: In-process per Ad Server instance. Not shared via Redis — each instance builds its own view from live traffic.

**Cold start handling:**
- New DSP or instance restart: use `T_global = 100ms` until 100 samples collected
- Minimum sample size of 100 prevents a single outlier from setting the timeout

**Revenue monitoring:**
A `timeout_revenue_loss` metric tracks bids that arrived after the per-DSP timeout but before `T_global` and would have won the auction. This quantifies the revenue cost of the adaptive timeout relative to Strategy A.

## Consequences

**Easier:** Fast DSPs contribute to lower overall auction latency, saving 20–30ms on the critical path in typical conditions. Platform can return ad responses earlier when the high-bid DSPs happen to be fast ones.

**Harder:** Per-DSP timeout logic requires histogram maintenance and cold-start handling. The `timeout_revenue_loss` metric must be monitored and its threshold must be defined as an operational SLO. New DSP onboarding requires explicit cold-start period before histogram-based timeouts activate.

## Constraints

- The RTB Gateway must maintain a per-DSP latency histogram using HdrHistogram. No other histogram library is permitted for this purpose — approximation algorithms (t-digest, etc.) must not be used for timeout computation.
- The global RTB timeout `T_global` is 100ms. Per-DSP timeouts must never exceed `T_global`.
- Per-DSP adaptive timeouts must not activate until 100 samples have been collected for that DSP. Before 100 samples, the DSP receives `T_global`.
- Histograms must use a rolling 5-minute window. Unbounded accumulation is prohibited — it prevents the system from adapting to DSP latency changes.
- DSP histogram data must not be stored in Redis or any external store. It is per-instance in-process state that rebuilds from live traffic within minutes after instance restart.
- The `timeout_revenue_loss` metric must be instrumented and exposed on the Prometheus `/metrics` endpoint. An alert must fire when this metric exceeds a defined revenue threshold per hour.
- After the per-DSP timeout fires, the DSP's response must be discarded even if it arrives within `T_global`. The auction runs with bids collected at per-DSP cutoff.
- RTB bid requests must be cancelled at the network layer when the timeout fires. Open HTTP connections to DSPs that have timed out must not be kept alive waiting for a response.

## References

- Series: "Architecting Real-Time Ads Platform", Part 2 — RTB Timeout Handling and Partial Auctions
- Series: Part 1 — Latency Budget Decomposition (RTB budget: 100ms)
- HdrHistogram: https://github.com/HdrHistogram/HdrHistogram
