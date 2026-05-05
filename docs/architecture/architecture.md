# H2AI Architecture

H2AI Control Plane is a Rust runtime that coordinates pools of LLM adapters as an *adversarial committee*: independent generators, an independent verifier, and an independent auditor produce a resolved output that is more reliable than any single adapter. The runtime treats this committee as a physical system — an ensemble whose throughput, diversity, and quality are computable, calibrated, and bounded.

This document is the system-level map: phases, components, and how they fit together. The math is in [`math.md`](math.md). The HTTP/event/config surface is in [`reference.md`](reference.md). Operational details are in [`operations.md`](operations.md). Open questions are in [`research-state.md`](research-state.md).

---

## 1. What the system is

The control plane orchestrates a single task as a 6-phase pipeline. Each phase is event-sourced to NATS JetStream — every state transition is replayable, and every retry decision is auditable. Two independent diversity signals govern execution:

- **Hamming Common Ground (CG)**: pairwise constraint-satisfaction agreement across the adapter pool, measured during calibration. Drives `β_eff = β₀ × (1 − CG_mean)` and the USL ceiling `N_max = round(√((1 − α) / β_eff))`.
- **Cosine N_eff**: participation-ratio diversity from the eigendecomposition of the embedding cosine kernel. A pool-level `n_eff_cosine_prior` is the Bayesian prior at calibration; a task-level `n_eff_cosine_actual` is computed at every MAPE-K decision point.

The two signals are not redundant. Hamming CG measures *behavioural* agreement on the constraint corpus. Cosine N_eff measures *semantic* independence at generation time. Both flow through the planner, the multiplication-condition gate, and the MAPE-K retry loop.

---

## 2. Execution phases

A task moves through six phases. Each phase emits one or more events, and every retry restarts at Phase 2.

### Phase 1 — Bootstrap

The orchestrator compiles the task description and the active constraint corpus into an immutable `system_context`. The `J_eff` gate enforces a minimum context-fill fraction; tasks below the threshold are rejected with `ContextUnderflow` rather than run with insufficient grounding. Emits `TaskBootstrapped`.

### Phase 2 — Topology Provisioning

The planner selects topology, explorer roles, and merge strategy from the calibration result and the task's Pareto weights. Outputs:

- `topology_kind` ∈ {Ensemble, HierarchicalTree, TeamSwarmHybrid}
- N explorer configs with τ values
- One auditor config
- `merge_strategy` chosen by `MergeStrategy::from_role_costs` (ScoreOrdered / ConsensusMedian / OutlierResistant{f})
- `n_max`, `interface_n_max`, `beta_eff` snapshots
- A `constraint_tombstone` field — populated only when retrying after `ConstrainedExploration`

Emits `TopologyProvisioned`.

### Phase 2.5 — Multiplication Condition Gate

Three conditions must hold before the system commits compute. All three are evaluated against the calibrated `EnsembleCalibration`:

1. `p_mean > min_competence` — adapters are above chance.
2. `rho_mean < max_correlation` — error correlation is below the saturation point.
3. `cg_mean ≥ θ_coord` — the Common Ground floor.

Failure produces `MultiplicationConditionFailed` with one of `InsufficientCompetence`, `InsufficientDecorrelation`, or `CommonGroundBelowFloor`. The retry policy then selects the next topology or fails the task.

### Phase 2.6 — Pool Diversity Guard

A separate gate, evaluated only when `cfg.diversity_threshold > 0`. Compares the calibration's `n_eff_cosine_prior` against `1.0 + diversity_threshold`. When the pool's effective independent-adapter count is below the floor, the engine emits a synthetic `ZeroSurvival` with `failure_mode = ModeCollapse` (driving adapter rotation) and routes through `RetryPolicy`. This is the fourth multiplication condition: `InsufficientPoolDiversity`. It exists because Hamming CG can mark constraint-profile agreement as "high coordination" while the pool remains semantically near-degenerate (correlated hallucination risk).

### Phase 3 — Parallel Generation (TAO)

N explorers run their TAO (Thought–Action–Observation) loops in parallel through the Tokio executor. Each explorer independently:

- Receives the immutable `system_context`.
- Iterates up to `tao_config.max_turns` times, emitting `TaoIteration` per turn.
- Produces a `Proposal` event with raw output and token cost — or a `ProposalFailed` event on timeout, OOM, or adapter error.

`GenerationPhaseCompleted` summarises success/failure counts. Adapter rotation offset (set by `ModeCollapse` retries) is applied at adapter selection time so a retry sees a rotated subset of the pool.

### Phase 3.5 — Verification

A dedicated verification adapter (LLM-as-Judge) scores every proposal against the constraint corpus. Each scoring emits `VerificationScored {score, reason, passed}`. Proposals that fail verification become `BranchPruned` with their `violated_constraints` recorded.

### Phase 4 — Auditor Gate

A separate auditor adapter (typically a stronger reasoning model than the verifier) is the final non-negotiable gate. Its output is required to be JSON `{approved, reason}`. Non-JSON output is treated as rejection (fail-safe). Rejected proposals become additional `BranchPruned` events.

### Phase 5 — Merge

Surviving proposals enter `MergeEngine::resolve` with the strategy chosen at Phase 2:

- **ScoreOrdered**: pick the highest verification score.
- **ConsensusMedian**: pick the proposal with highest mean Jaccard similarity to the rest. *Not Byzantine-resistant — vulnerable to coordinated proposals at f ≥ n/2.*
- **OutlierResistant{f}**: smallest sum of distances to its `n − f − 2` nearest neighbours in Jaccard-distance space (Krum-style). Requires `n ≥ 2f + 3`.
- **MultiOutlierResistant{f, m}**: iteratively select m survivors via OutlierResistant, then take the highest verification score.

Emits `SemilatticeCompiled` (with valid/pruned lists, strategy, elapsed time, input count) and either `MergeResolved` (success) or `ZeroSurvival` (zero-survival → MAPE-K retry).

### Phase 5a — Synthesis (optional)

When `synthesis_enabled` and at least `synthesis_min_proposals` have survived audit, the synthesis adapter performs a critique→synthesis→re-verify pass over the candidate set. The re-verified score is compared against `max(individual_scores)`; the difference is recorded as `synthesis_gain` on `HarnessAttribution`. If synthesis improves the maximum, its output replaces the merge result.

### MAPE-K loop on zero survival

When Phase 5 reports `ZeroSurvival`:

1. **Monitor**: compute `n_eff_cosine_actual` from the wave's raw outputs (when an `EmbeddingModel` is configured).
2. **Analyse**: classify via `classify_failure_mode(n_eff, n_requested, diversity_threshold)`:
   - `n_eff > diversity_threshold × n_requested` → `ConstrainedExploration` (diverse exploration, none satisfied constraints).
   - Otherwise → `ModeCollapse` (correlated hallucination).
3. **Plan**:
   - On `ConstrainedExploration`: synthesise a Constraint Violation Tombstone — dense IDs+severity only, never raw text — and stage it for the next `TopologyProvisioned`.
   - On `ModeCollapse`: increment `adapter_rotation_offset` by 1 modulo pool size.
4. **Execute**: `RetryPolicy::decide` selects the next topology / τ-reduction / hint from the available retry actions.

Both interventions are bookkept as Prometheus counters with a `failure_mode` label (`mode_collapse` and `constrained_exploration`).

### Post-merge async event

After `MergeResolved`, the engine spawns an async task that publishes `EpistemicYield {n_eff_cosine_actual, n_eff_prior, yield_ratio, adapters}`. `yield_ratio = n_eff_actual / N_requested` — the "financial yield": you paid for N adapters, you received `n_eff_actual` independent perspectives. This event never blocks task close.

---

## 3. Component map

The workspace contains 16 crates, organised by responsibility. Every crate compiles standalone; cross-crate communication is event-typed.

```
h2ai-types          Pure value types + math primitives (USL, EigenCalibration, EnsembleCalibration,
                    MergeStrategy, MultiplicationConditionFailure, EpistemicYieldEvent, FailureMode,
                    H2AIEvent enum).
h2ai-config         Layered config loading (reference.toml + env overrides). Single source of truth.
h2ai-adapters       Adapter trait + per-provider implementations (Anthropic, OpenAI, Gemini, Ollama,
                    llama.cpp, Mock). Tokio-native via async-trait.
h2ai-context        EmbeddingModel trait, fastembed wrapper, cosine_similarity utilities.
h2ai-constraints    Constraint corpus parser (markdown ADR format), predicate types
                    (VocabularyPresence, AllOf, AnyOf, ...), severity weights.
h2ai-autonomic      Calibration harness, epistemic diagnostics (compute_n_eff_cosine,
                    classify_failure_mode, synthesize_tombstone), ensemble calibration plumbing,
                    Talagrand rank histogram, Thompson Sampling bandit over N.
h2ai-memory         InMemoryCache + NatsKvStore implementations of the SessionMemory trait.
h2ai-nats           NATS JetStream client, stream/KV creation, event publish/subscribe.
h2ai-orchestrator   ExecutionEngine — the 6-phase MAPE-K loop. MergeEngine. Verification phase.
                    Synthesis phase. RetryPolicy, MultiplicationChecker, SelfOptimizer.
h2ai-planner        Pareto-weighted topology selection, role assignment, τ spread, role error costs.
h2ai-provisioner    Static / NATS / Kubernetes agent providers.
h2ai-state          CRDT-friendly TaskState, ProposalSet (LUB by generation, then score),
                    snapshot/replay machinery.
h2ai-telemetry      tracing→OTLP plumbing, structured spans for every phase.
h2ai-tools          CLI utilities (calibration triggers, manifest validation, corpus linting).
h2ai-agent          Edge-side agent runtime (the container that runs an explorer in production).
h2ai-api            Axum HTTP server: POST /tasks, SSE event stream, calibration endpoints,
                    health/ready/metrics, Merge Authority UI assets.
```

A concrete request flow:

```
POST /tasks               h2ai-api  ──► h2ai-orchestrator (ExecutionEngine::run_offline)
  Phase 1                 h2ai-constraints + h2ai-context
  Phase 2                 h2ai-planner reads h2ai-config + last CalibrationCompleted
  Phase 2.5/2.6           h2ai-orchestrator (gates: MultiplicationCondition, diversity guard)
  Phase 3                 h2ai-adapters (parallel, via Tokio + spawn_blocking for FFI)
  Phase 3.5               h2ai-orchestrator::verification + verification adapter
  Phase 4                 h2ai-orchestrator + auditor adapter
  Phase 5/5a              h2ai-orchestrator::merge + h2ai-orchestrator::synthesis
  Each phase event ──►    h2ai-nats (publish to h2ai.tasks.{task_id})
  Each retry decision ──► h2ai-orchestrator (RetryPolicy + MAPE-K bivariate routing)
GET /tasks/:id/events     h2ai-api SSE consumes from NATS subject h2ai.tasks.{task_id}
```

---

## 4. Event sourcing model

Every state transition is an `H2AIEvent` published to `h2ai.tasks.{task_id}`. Crash recovery is replay from the last snapshot offset; SSE clients reconnect with `Last-Event-ID`. Full event enumeration is in [`reference.md`](reference.md#event-vocabulary). Event payload schemas are stable: every field added since the initial release uses `#[serde(default)]` so old serialised events continue to deserialise.

The authoritative log is NATS JetStream stream `H2AI_TASKS` (file-backed, replicated). Calibration data lives in the `H2AI_CALIBRATION` KV store. Snapshots are written to `H2AI_SNAPSHOTS` periodically — recovery loads the latest snapshot and replays only events with `sequence > last_sequence`.

---

## 5. What H2AI does *not* do better

The control plane is honest about its boundaries. The system does *not* compete with:

- **Single-shot inference latency.** A direct call to one model endpoint will always be cheaper and faster. H2AI buys reliability, not speed.
- **Generic agentic frameworks.** Frameworks that compose tools and memory solve a different problem; H2AI orchestrates an adversarial committee with calibrated physics. The two are complementary, not competing.
- **Specialised serving stacks** (vLLM, TGI). Those optimise per-request throughput. H2AI delegates to them via adapters.
- **Tasks where ground truth is hidden from verification.** The auditor and verifier need to observe the constraint surface. Tasks with hidden oracles get no benefit from the adversarial committee — at best, the system reduces to its single best adapter.
- **Workloads with a single dominant adapter.** When `n_eff_cosine_prior → 1.0`, the multiplication condition fails and the system correctly refuses to run. Buying more capacity from one model family does not produce a committee.

The reliability gain is real only when the calibrated adapter pool is genuinely diverse — both in constraint behaviour (Hamming CG) and in semantic embedding (cosine N_eff). When the pool is monoculture, no amount of orchestration recovers the missing independence.
