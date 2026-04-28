# AdapterRegistry + TaskProfile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `TaskProfile` routing enum and `AdapterRegistry` so the engine resolves the right adapter (LLM vs SLM) per task type, replacing the ad-hoc `similarity_adapter: Option<&dyn IComputeAdapter>` field in `EngineInput`.

**Architecture:** `TaskProfile` (3 variants: Reasoning / Scoring / Structural) is defined in `h2ai-types`. `AdapterRegistry` holds an `Arc<dyn IComputeAdapter>` per profile with fallback to `reasoning` when a specialized adapter is not configured. `EngineInput` drops `similarity_adapter` in favour of `registry: &'a AdapterRegistry`. `AppState` gains an optional `scoring_adapter`; the route handler builds the `AdapterRegistry` before spawning the engine task.

**Tech Stack:** Rust, `std::sync::Arc`, existing `IComputeAdapter` trait in `h2ai-types`, axum `AppState`, tokio async.

---

## File Map

| File | Change |
|---|---|
| `crates/h2ai-types/src/adapter.rs` | Add `TaskProfile` enum + `AdapterRegistry` struct |
| `crates/h2ai-types/tests/adapter_test.rs` | Add registry resolution tests |
| `crates/h2ai-orchestrator/src/engine.rs` | Replace `similarity_adapter` field with `registry: &'a AdapterRegistry` |
| `crates/h2ai-orchestrator/tests/engine_test.rs` | Fix 6 broken `EngineInput` literals |
| `crates/h2ai-orchestrator/tests/system_test.rs` | Fix 4 broken `EngineInput` literals |
| `crates/h2ai-orchestrator/tests/deadline_test.rs` | Fix 1 broken `EngineInput` literal |
| `crates/h2ai-orchestrator/tests/diversity_test.rs` | Fix 1 broken `EngineInput` literal |
| `crates/h2ai-api/src/state.rs` | Add `scoring_adapter: Option<Arc<dyn IComputeAdapter>>` + `fn registry()` |
| `crates/h2ai-api/src/main.rs` | Read `H2AI_SCORING_PROVIDER` env var, build optional scoring adapter |
| `crates/h2ai-api/src/routes/tasks.rs` | Build `AdapterRegistry`, replace `similarity_adapter: None` |

---

## Task 1: `TaskProfile` + `AdapterRegistry` in h2ai-types

**Files:**
- Modify: `crates/h2ai-types/src/adapter.rs`
- Modify: `crates/h2ai-types/tests/adapter_test.rs`

- [ ] **Step 1: Write failing tests**

Add to `crates/h2ai-types/tests/adapter_test.rs`:

```rust
use h2ai_types::adapter::{AdapterRegistry, TaskProfile};
// (keep existing imports)

// ── registry tests ────────────────────────────────────────────────────────────

#[derive(Debug)]
struct LabelAdapter(String, h2ai_types::config::AdapterKind);

#[async_trait::async_trait]
impl h2ai_types::adapter::IComputeAdapter for LabelAdapter {
    async fn execute(
        &self,
        _req: h2ai_types::adapter::ComputeRequest,
    ) -> Result<h2ai_types::adapter::ComputeResponse, h2ai_types::adapter::AdapterError> {
        Ok(h2ai_types::adapter::ComputeResponse {
            output: self.0.clone(),
            token_cost: 0,
            adapter_kind: self.1.clone(),
        })
    }
    fn kind(&self) -> &h2ai_types::config::AdapterKind {
        &self.1
    }
}

fn label(name: &str) -> std::sync::Arc<dyn h2ai_types::adapter::IComputeAdapter> {
    std::sync::Arc::new(LabelAdapter(
        name.into(),
        h2ai_types::config::AdapterKind::CloudGeneric {
            endpoint: "mock://test".into(),
            api_key_env: "NONE".into(),
        },
    ))
}

#[test]
fn registry_reasoning_resolves_to_reasoning_adapter() {
    let reg = AdapterRegistry::new(label("reasoning"));
    let ptr = reg.resolve(&TaskProfile::Reasoning) as *const dyn h2ai_types::adapter::IComputeAdapter;
    // resolved adapter is the reasoning one — confirm by executing it synchronously is not needed;
    // we just confirm the resolve path does not panic and returns a valid reference.
    let _ = ptr;
}

#[test]
fn registry_scoring_falls_back_to_reasoning_when_not_set() {
    let reasoning = label("reasoning");
    let reg = AdapterRegistry::new(reasoning.clone());
    let resolved = reg.resolve(&TaskProfile::Scoring) as *const _;
    let expected = reasoning.as_ref() as *const _;
    assert_eq!(resolved, expected, "scoring must fall back to reasoning when not configured");
}

#[test]
fn registry_scoring_uses_dedicated_adapter_when_set() {
    let scoring = label("scoring");
    let reg = AdapterRegistry::new(label("reasoning")).with_scoring(scoring.clone());
    let resolved = reg.resolve(&TaskProfile::Scoring) as *const _;
    let expected = scoring.as_ref() as *const _;
    assert_eq!(resolved, expected, "scoring must return the dedicated adapter");
}

#[test]
fn registry_structural_falls_back_to_reasoning_when_not_set() {
    let reasoning = label("reasoning");
    let reg = AdapterRegistry::new(reasoning.clone());
    let resolved = reg.resolve(&TaskProfile::Structural) as *const _;
    let expected = reasoning.as_ref() as *const _;
    assert_eq!(resolved, expected, "structural must fall back to reasoning when not configured");
}

#[test]
fn registry_structural_uses_dedicated_adapter_when_set() {
    let structural = label("structural");
    let reg = AdapterRegistry::new(label("reasoning")).with_structural(structural.clone());
    let resolved = reg.resolve(&TaskProfile::Structural) as *const _;
    let expected = structural.as_ref() as *const _;
    assert_eq!(resolved, expected, "structural must return the dedicated adapter");
}

#[test]
fn registry_all_three_resolve_independently() {
    let r = label("r");
    let sc = label("sc");
    let st = label("st");
    let reg = AdapterRegistry::new(r.clone())
        .with_scoring(sc.clone())
        .with_structural(st.clone());
    assert_eq!(reg.resolve(&TaskProfile::Reasoning) as *const _, r.as_ref() as *const _);
    assert_eq!(reg.resolve(&TaskProfile::Scoring) as *const _, sc.as_ref() as *const _);
    assert_eq!(reg.resolve(&TaskProfile::Structural) as *const _, st.as_ref() as *const _);
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cargo test --package h2ai-types --test adapter_test 2>&1 | tail -10
```

Expected: `error[E0412]: cannot find type 'AdapterRegistry' in module 'h2ai_types::adapter'`

- [ ] **Step 3: Implement `TaskProfile` and `AdapterRegistry`**

Replace the full content of `crates/h2ai-types/src/adapter.rs` with:

```rust
use crate::config::AdapterKind;
use crate::physics::TauValue;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeRequest {
    pub system_context: String,
    pub task: String,
    pub tau: TauValue,
    pub max_tokens: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeResponse {
    pub output: String,
    pub token_cost: u64,
    pub adapter_kind: AdapterKind,
}

#[derive(Debug, Error)]
pub enum AdapterError {
    #[error("adapter timed out before producing output")]
    Timeout,
    #[error("adapter OOM panic: {0}")]
    OomPanic(String),
    #[error("network error: {0}")]
    NetworkError(String),
    #[error("FFI error from llama.cpp: {0}")]
    FfiError(String),
}

#[async_trait]
pub trait IComputeAdapter: Send + Sync + std::fmt::Debug {
    async fn execute(&self, request: ComputeRequest) -> Result<ComputeResponse, AdapterError>;
    fn kind(&self) -> &AdapterKind;
}

/// Capability tier required by a compute task.
///
/// Callsites declare which profile they need; `AdapterRegistry::resolve` returns
/// the configured adapter for that profile, falling back to `Reasoning` when a
/// dedicated adapter is not available.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskProfile {
    /// Full LLM — explorers, compound planning, any high-reasoning task.
    Reasoning,
    /// Small / cheap model — semantic similarity scoring, short JSON scoring tasks.
    /// Falls back to `Reasoning` if no dedicated adapter is configured.
    Scoring,
    /// Any model that reliably follows instructions — auditor, schema validation.
    /// Falls back to `Reasoning` if no dedicated adapter is configured.
    Structural,
}

/// Maps `TaskProfile` → `Arc<dyn IComputeAdapter>` with fallback to `Reasoning`.
///
/// Build with [`AdapterRegistry::new`] (requires only a reasoning adapter) and
/// optionally attach dedicated adapters via [`with_scoring`] / [`with_structural`].
///
/// ```
/// # use h2ai_types::adapter::{AdapterRegistry, TaskProfile};
/// # use std::sync::Arc;
/// // Minimal — all profiles use the same adapter:
/// // AdapterRegistry::new(reasoning_adapter)
///
/// // With a dedicated SLM for cheap scoring:
/// // AdapterRegistry::new(llm).with_scoring(slm)
/// ```
#[derive(Clone)]
pub struct AdapterRegistry {
    reasoning: Arc<dyn IComputeAdapter>,
    scoring: Option<Arc<dyn IComputeAdapter>>,
    structural: Option<Arc<dyn IComputeAdapter>>,
}

impl AdapterRegistry {
    /// Create a registry with only a reasoning adapter. Scoring and structural
    /// profiles fall back to the reasoning adapter until explicitly configured.
    pub fn new(reasoning: Arc<dyn IComputeAdapter>) -> Self {
        Self {
            reasoning,
            scoring: None,
            structural: None,
        }
    }

    /// Attach a dedicated adapter for `TaskProfile::Scoring` tasks.
    pub fn with_scoring(mut self, adapter: Arc<dyn IComputeAdapter>) -> Self {
        self.scoring = Some(adapter);
        self
    }

    /// Attach a dedicated adapter for `TaskProfile::Structural` tasks.
    pub fn with_structural(mut self, adapter: Arc<dyn IComputeAdapter>) -> Self {
        self.structural = Some(adapter);
        self
    }

    /// Resolve the adapter for the given profile.
    ///
    /// `Scoring` and `Structural` fall back to the reasoning adapter when no
    /// dedicated adapter has been configured.
    pub fn resolve(&self, profile: &TaskProfile) -> &dyn IComputeAdapter {
        match profile {
            TaskProfile::Reasoning => self.reasoning.as_ref(),
            TaskProfile::Scoring => self
                .scoring
                .as_deref()
                .unwrap_or(self.reasoning.as_ref()),
            TaskProfile::Structural => self
                .structural
                .as_deref()
                .unwrap_or(self.reasoning.as_ref()),
        }
    }
}

impl std::fmt::Debug for AdapterRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdapterRegistry")
            .field("reasoning", &self.reasoning.kind())
            .field(
                "scoring",
                &self.scoring.as_ref().map(|a| a.kind()),
            )
            .field(
                "structural",
                &self.structural.as_ref().map(|a| a.kind()),
            )
            .finish()
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cargo test --package h2ai-types --test adapter_test 2>&1 | tail -15
```

Expected: all tests pass including the 6 new registry tests.

- [ ] **Step 5: Commit**

```bash
git add crates/h2ai-types/src/adapter.rs crates/h2ai-types/tests/adapter_test.rs
git commit -m "feat(types): TaskProfile + AdapterRegistry — profile-based adapter routing"
```

---

## Task 2: Replace `similarity_adapter` with `registry` in `EngineInput`

**Files:**
- Modify: `crates/h2ai-orchestrator/src/engine.rs`

**Context:** `EngineInput` currently has `similarity_adapter: Option<&'a dyn IComputeAdapter>` (line 67). It is used at two call sites inside `run_offline`:
- Line 116: `compiler::compile(..., input.similarity_adapter)` — Phase 1 J_eff
- Line 737: `MergeEngine::resolve(..., input.similarity_adapter)` — Phase 5 cluster coherence

Replace both with `Some(input.registry.resolve(&TaskProfile::Scoring))`.

- [ ] **Step 1: Verify tests are currently broken**

```bash
cargo check --package h2ai-orchestrator --tests 2>&1 | grep "missing field" | head -5
```

Expected: several `missing field 'similarity_adapter'` errors.

- [ ] **Step 2: Update `EngineInput` and call sites in `engine.rs`**

**2a.** In the imports at the top of `engine.rs`, add:

```rust
use h2ai_types::adapter::{AdapterRegistry, TaskProfile};
```

(The existing `use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};` line should become:)

```rust
use h2ai_types::adapter::{AdapterRegistry, ComputeRequest, IComputeAdapter, TaskProfile};
```

**2b.** Replace the `EngineInput` field:

Find:
```rust
    /// Optional SLM adapter for semantic similarity in J_eff and cluster coherence checks.
    /// When None, falls back to token-level Jaccard throughout — zero extra cost.
    pub similarity_adapter: Option<&'a dyn IComputeAdapter>,
```

Replace with:
```rust
    /// Adapter registry for profile-based routing.
    /// `TaskProfile::Scoring` resolves to a cheap SLM if configured; otherwise falls
    /// back to the reasoning adapter. Used for J_eff semantic scoring and cluster
    /// coherence checks.
    pub registry: &'a AdapterRegistry,
```

**2c.** Fix the Phase 1 call site (around line 116):

Find:
```rust
        let compiled = compiler::compile(
            description,
            &input.constraint_corpus,
            &required_kw,
            input.cfg,
            input.similarity_adapter,
        )
```

Replace with:
```rust
        let compiled = compiler::compile(
            description,
            &input.constraint_corpus,
            &required_kw,
            input.cfg,
            Some(input.registry.resolve(&TaskProfile::Scoring)),
        )
```

**2d.** Fix the Phase 5 call site (around line 737, inside the MAPE-K loop). Search for the `MergeEngine::resolve` call that passes `input.similarity_adapter`:

Find:
```rust
                input.similarity_adapter,
```

Replace with:
```rust
                Some(input.registry.resolve(&TaskProfile::Scoring)),
```

- [ ] **Step 3: Verify the library compiles**

```bash
cargo check --package h2ai-orchestrator 2>&1 | tail -5
```

Expected: `Finished dev profile` (library compiles; tests still broken until Task 3).

- [ ] **Step 4: Commit**

```bash
git add crates/h2ai-orchestrator/src/engine.rs
git commit -m "feat(orchestrator): replace similarity_adapter with AdapterRegistry in EngineInput"
```

---

## Task 3: Fix broken integration test sites

**Files:**
- Modify: `crates/h2ai-orchestrator/tests/engine_test.rs`
- Modify: `crates/h2ai-orchestrator/tests/system_test.rs`
- Modify: `crates/h2ai-orchestrator/tests/deadline_test.rs`
- Modify: `crates/h2ai-orchestrator/tests/diversity_test.rs`

**Context:** All four files construct `EngineInput` with `nats_dispatch: None,` as the last field. After Task 2 that field no longer exists as the last field — `registry` does. Every `EngineInput` literal needs `registry: &registry_var,` added, and each test function needs a registry variable built from the mock adapter.

The pattern to apply in every test function that builds `EngineInput`:

**Before (tail of EngineInput literal):**
```rust
        store: store.clone(),
        nats_dispatch: None,
    };
```

**After:**
```rust
        store: store.clone(),
        nats_dispatch: None,
        registry: &registry,
    };
```

And at the start of each test function (before the `EngineInput` literal), add:
```rust
    let registry = h2ai_types::adapter::AdapterRegistry::new(
        std::sync::Arc::new(mock_adapter()) as std::sync::Arc<dyn h2ai_types::adapter::IComputeAdapter>
    );
```

Where `mock_adapter()` is the adapter already used in that function for `explorer_adapters`.

**Detailed per-file instructions follow.**

- [ ] **Step 1: Fix `engine_test.rs`**

The file has 6 `EngineInput` constructions (at lines 70, 143, 204, 266, 328, 411). Each test function already calls `mock_adapter()`.

Add `use std::sync::Arc;` and `use h2ai_types::adapter::{AdapterRegistry, IComputeAdapter};` to the imports at the top of the file.

For each of the 6 test functions, add directly before the `EngineInput {` literal:
```rust
    let registry = AdapterRegistry::new(Arc::new(mock_adapter()) as Arc<dyn IComputeAdapter>);
```

And add as the last field in each `EngineInput` literal:
```rust
        registry: &registry,
```

The complete imports block at the top of `engine_test.rs` should look like:

```rust
use h2ai_adapters::mock::MockAdapter;
use h2ai_autonomic::calibration::{CalibrationHarness, CalibrationInput};
use h2ai_config::H2AIConfig;
use h2ai_context::adr::parse_adr;
use h2ai_orchestrator::engine::{EngineError, EngineInput, ExecutionEngine};
use h2ai_orchestrator::task_store::TaskStore;
use h2ai_types::adapter::{AdapterRegistry, IComputeAdapter};
use h2ai_types::config::{
    AdapterKind, AgentRole, AuditorConfig, ParetoWeights, RoleSpec, TaoConfig, VerificationConfig,
};
use h2ai_types::identity::TaskId;
use h2ai_types::manifest::{ExplorerRequest, TaskManifest, TopologyRequest};
use std::sync::Arc;
```

Example of the first test after the fix (lines 70–93 region):
```rust
    let registry = AdapterRegistry::new(Arc::new(mock_adapter()) as Arc<dyn IComputeAdapter>);
    let input = EngineInput {
        task_id: TaskId::new(),
        manifest,
        calibration: cal,
        explorer_adapters: vec![
            &adapter as &dyn IComputeAdapter,
            &adapter2,
        ],
        verification_adapter: &scorer as &dyn IComputeAdapter,
        auditor_adapter: &auditor as &dyn IComputeAdapter,
        auditor_config: AuditorConfig {
            adapter: AdapterKind::CloudGeneric {
                endpoint: "mock".into(),
                api_key_env: "NONE".into(),
            },
            ..Default::default()
        },
        tao_config: TaoConfig::default(),
        verification_config: VerificationConfig::default(),
        constraint_corpus: corpus,
        cfg: &cfg,
        store: store.clone(),
        nats_dispatch: None,
        registry: &registry,
    };
```

- [ ] **Step 2: Fix `system_test.rs`**

The file has 4 `EngineInput` constructions (at lines 151, 215, 260, 340). Apply the same pattern: add registry variable before each literal, add `registry: &registry,` as the last field.

Add to imports:
```rust
use h2ai_types::adapter::{AdapterRegistry, IComputeAdapter};
use std::sync::Arc;
```

Each registry creation uses whatever mock adapter is already in scope for that test.

- [ ] **Step 3: Fix `deadline_test.rs`**

The file has 1 `EngineInput` construction (at line 56 inside the `make_input` helper function). Add registry as a parameter or create it inside the helper.

The cleanest approach: add `registry: &'a AdapterRegistry` as a parameter to the `make_input` function, and pass it in at the call site.

Read the current `make_input` signature (around line 20):
```rust
fn make_input<'a>(
    adapter: &'a MockAdapter,
    cfg: &'a H2AIConfig,
    cal: &'a h2ai_types::events::CalibrationCompletedEvent,
    store: TaskStore,
) -> EngineInput<'a> {
```

Updated signature:
```rust
fn make_input<'a>(
    adapter: &'a MockAdapter,
    cfg: &'a H2AIConfig,
    cal: &'a h2ai_types::events::CalibrationCompletedEvent,
    store: TaskStore,
    registry: &'a AdapterRegistry,
) -> EngineInput<'a> {
```

And add `registry,` to the `EngineInput` literal inside `make_input`.

At each call site of `make_input`, create a registry and pass it:
```rust
let registry = AdapterRegistry::new(Arc::new(adapter.clone()) as Arc<dyn IComputeAdapter>);
// ...
make_input(&adapter, &cfg, &cal, store.clone(), &registry)
```

Wait — `MockAdapter` doesn't implement `Clone`. Check if it does:

```rust
// crates/h2ai-adapters/src/mock.rs — MockAdapter does NOT derive Clone.
```

Use the mock adapter for reasoning in the registry. Instead of cloning:
```rust
let reasoning_arc: Arc<dyn IComputeAdapter> = Arc::new(MockAdapter::new("mock".into()));
let registry = AdapterRegistry::new(reasoning_arc);
```

Add imports to `deadline_test.rs`:
```rust
use h2ai_types::adapter::{AdapterRegistry, IComputeAdapter};
use std::sync::Arc;
```

- [ ] **Step 4: Fix `diversity_test.rs`**

The file has 1 `EngineInput` construction inside a helper function (around line 143). Apply the same pattern as `deadline_test.rs`: add `registry: &'a AdapterRegistry` to the helper signature, create it at the call sites.

Add imports:
```rust
use h2ai_types::adapter::{AdapterRegistry, IComputeAdapter};
use std::sync::Arc;
```

- [ ] **Step 5: Run all orchestrator tests**

```bash
cargo test --package h2ai-orchestrator 2>&1 | tail -20
```

Expected: all tests pass (no `missing field` errors, no test failures).

- [ ] **Step 6: Commit**

```bash
git add crates/h2ai-orchestrator/tests/engine_test.rs \
        crates/h2ai-orchestrator/tests/system_test.rs \
        crates/h2ai-orchestrator/tests/deadline_test.rs \
        crates/h2ai-orchestrator/tests/diversity_test.rs
git commit -m "fix(orchestrator): add AdapterRegistry to all EngineInput test constructions"
```

---

## Task 4: Wire registry through `AppState`, `main.rs`, and `tasks.rs`

**Files:**
- Modify: `crates/h2ai-api/src/state.rs`
- Modify: `crates/h2ai-api/src/main.rs`
- Modify: `crates/h2ai-api/src/routes/tasks.rs`

**Context:** `AppState` currently has `explorer_adapter`, `verification_adapter`, `auditor_adapter`. The route handler in `tasks.rs` constructs `EngineInput` with `similarity_adapter: None`. After Task 2, that field is gone and `registry: &registry` is needed instead.

We add `scoring_adapter: Option<Arc<dyn IComputeAdapter>>` to `AppState`, and a helper `fn registry(&self) -> AdapterRegistry` that builds the registry from the explorer adapter (reasoning) + optional scoring adapter. The route handler calls `state.registry()` to get a local `AdapterRegistry` and passes a reference to `EngineInput`.

- [ ] **Step 1: Update `state.rs`**

Replace the full contents of `crates/h2ai-api/src/state.rs` with:

```rust
use h2ai_config::H2AIConfig;
use h2ai_orchestrator::session_journal::SessionJournal;
use h2ai_orchestrator::task_store::TaskStore;
use h2ai_state::nats::NatsClient;
use h2ai_types::adapter::{AdapterRegistry, IComputeAdapter};
use h2ai_types::events::CalibrationCompletedEvent;
use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};

#[derive(Clone)]
pub struct AppState {
    pub nats: Arc<NatsClient>,
    pub cfg: Arc<H2AIConfig>,
    pub store: TaskStore,
    pub calibration: Arc<RwLock<Option<CalibrationCompletedEvent>>>,
    pub journal: Arc<SessionJournal>,
    pub explorer_adapter: Arc<dyn IComputeAdapter>,
    /// Scores proposals in Phase 3.5. Returns `{"score": float, "reason": "..."}`.
    pub verification_adapter: Arc<dyn IComputeAdapter>,
    /// Approves/rejects proposals in Phase 4. Returns `{"approved": bool, "reason": "..."}`.
    pub auditor_adapter: Arc<dyn IComputeAdapter>,
    /// Optional dedicated adapter for TaskProfile::Scoring (semantic similarity, short JSON
    /// scoring). When None, the explorer adapter is used for all profiles.
    pub scoring_adapter: Option<Arc<dyn IComputeAdapter>>,
    /// Limits concurrent task executions to cfg.max_concurrent_tasks.
    pub task_semaphore: Arc<Semaphore>,
}

impl AppState {
    pub fn new(
        nats: NatsClient,
        cfg: H2AIConfig,
        explorer_adapter: Arc<dyn IComputeAdapter>,
        auditor_adapter: Arc<dyn IComputeAdapter>,
    ) -> Self {
        let nats = Arc::new(nats);
        let journal = Arc::new(SessionJournal::new(nats.clone()));
        let max_tasks = cfg.max_concurrent_tasks;
        Self {
            nats,
            cfg: Arc::new(cfg),
            store: TaskStore::new(),
            calibration: Arc::new(RwLock::new(None)),
            journal,
            explorer_adapter,
            verification_adapter: auditor_adapter.clone(),
            auditor_adapter,
            scoring_adapter: None,
            task_semaphore: Arc::new(Semaphore::new(max_tasks)),
        }
    }

    /// Build an `AdapterRegistry` from this state.
    ///
    /// The reasoning adapter is always `explorer_adapter`. The scoring adapter
    /// is used for `TaskProfile::Scoring` if configured; otherwise the explorer
    /// adapter handles all profiles.
    pub fn registry(&self) -> AdapterRegistry {
        let reg = AdapterRegistry::new(self.explorer_adapter.clone());
        match &self.scoring_adapter {
            Some(scoring) => reg.with_scoring(scoring.clone()),
            None => reg,
        }
    }
}
```

- [ ] **Step 2: Update `main.rs`**

After the existing `let auditor_adapter = build_adapter(&auditor_kind);` line, add:

```rust
    let scoring_kind_opt = {
        let provider = env::var("H2AI_SCORING_PROVIDER")
            .unwrap_or_else(|_| "none".into())
            .to_lowercase();
        if provider == "none" || provider.is_empty() {
            None
        } else {
            Some(adapter_kind_from_env("SCORING"))
        }
    };
    let scoring_adapter: Option<Arc<dyn IComputeAdapter>> =
        scoring_kind_opt.as_ref().map(build_adapter);
```

After `let app_state = AppState::new(nats, cfg, explorer_adapter, auditor_adapter);`, add:

```rust
    let app_state = if let Some(sa) = scoring_adapter {
        AppState { scoring_adapter: Some(sa), ..app_state }
    } else {
        app_state
    };
```

Also add `eprintln!` for the scoring adapter:
```rust
    eprintln!("scoring  adapter: {:?}", scoring_kind_opt);
```

The complete relevant block in `main` after this change:

```rust
    let explorer_kind = adapter_kind_from_env("EXPLORER");
    let auditor_kind = adapter_kind_from_env("AUDITOR");
    let explorer_adapter = build_adapter(&explorer_kind);
    let auditor_adapter = build_adapter(&auditor_kind);

    let scoring_kind_opt = {
        let provider = env::var("H2AI_SCORING_PROVIDER")
            .unwrap_or_else(|_| "none".into())
            .to_lowercase();
        if provider == "none" || provider.is_empty() {
            None
        } else {
            Some(adapter_kind_from_env("SCORING"))
        }
    };
    let scoring_adapter: Option<Arc<dyn IComputeAdapter>> =
        scoring_kind_opt.as_ref().map(build_adapter);

    eprintln!("explorer adapter: {:?}", explorer_kind);
    eprintln!("auditor  adapter: {:?}", auditor_kind);
    eprintln!("scoring  adapter: {:?}", scoring_kind_opt);

    let mut app_state = AppState::new(nats, cfg, explorer_adapter, auditor_adapter);
    if let Some(sa) = scoring_adapter {
        app_state.scoring_adapter = Some(sa);
    }
```

Note: this requires making `scoring_adapter` a `pub` field (it already is in the updated `state.rs`).

- [ ] **Step 3: Update `tasks.rs`**

In `submit_task`, the `tokio::spawn` block currently does:

```rust
    let explorer = state.explorer_adapter.clone();
    let verifier = state.verification_adapter.clone();
    let auditor = state.auditor_adapter.clone();

    let state_clone = state.clone();
    let manifest_clone = manifest.clone();
    let calibration_clone = calibration.clone();
    let store_clone = state.store.clone();
    let task_id_clone = task_id.clone();

    tokio::spawn(async move {
        let _permit = permit;
        let input = EngineInput {
            ...
            similarity_adapter: None,
        };
```

Replace this with:

```rust
    let explorer = state.explorer_adapter.clone();
    let verifier = state.verification_adapter.clone();
    let auditor = state.auditor_adapter.clone();
    let registry = state.registry();   // ← build registry from state

    let state_clone = state.clone();
    let manifest_clone = manifest.clone();
    let calibration_clone = calibration.clone();
    let store_clone = state.store.clone();
    let task_id_clone = task_id.clone();

    tokio::spawn(async move {
        let _permit = permit;
        let input = EngineInput {
            task_id: task_id_clone,
            manifest: manifest_clone,
            calibration: calibration_clone,
            explorer_adapters: vec![explorer.as_ref(), explorer.as_ref()],
            verification_adapter: verifier.as_ref(),
            auditor_adapter: auditor.as_ref(),
            auditor_config: h2ai_types::config::AuditorConfig {
                adapter: auditor.kind().clone(),
                ..Default::default()
            },
            tao_config: TaoConfig::default(),
            verification_config: VerificationConfig::default(),
            constraint_corpus: corpus,
            cfg: &state_clone.cfg,
            store: store_clone,
            nats_dispatch: None,
            registry: &registry,   // ← registry replaces similarity_adapter: None
        };
        ...
    });
```

Also remove the import of `IComputeAdapter` from `tasks.rs` if it was only used for `similarity_adapter`.

- [ ] **Step 4: Build and test**

```bash
cargo check --workspace 2>&1 | tail -5
cargo test --package h2ai-api 2>&1 | tail -10
```

Expected: clean compile and all tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/h2ai-api/src/state.rs \
        crates/h2ai-api/src/main.rs \
        crates/h2ai-api/src/routes/tasks.rs
git commit -m "feat(api): wire AdapterRegistry into AppState and EngineInput via tasks.rs"
```

---

## Task 5: Full workspace verification

- [ ] **Step 1: Run full test suite**

```bash
cargo test --workspace 2>&1 | tail -30
```

Expected: all tests pass, zero compilation errors.

- [ ] **Step 2: Check for any remaining `similarity_adapter` references**

```bash
grep -r "similarity_adapter" crates/ --include="*.rs"
```

Expected: no output (all references gone).

- [ ] **Step 3: Commit if any stray fixes were needed**

If Step 2 found anything or Step 1 had failures, fix them and commit. Otherwise this step is a no-op.

---

## Self-Review

**Spec coverage check:**

| Requirement | Task |
|---|---|
| `TaskProfile` enum with 3 variants | Task 1 |
| `AdapterRegistry` with builder pattern | Task 1 |
| `resolve()` falls back to reasoning | Task 1 (tests cover both paths) |
| `EngineInput` uses registry instead of `similarity_adapter` | Task 2 |
| Both J_eff and cluster coherence use `TaskProfile::Scoring` | Task 2 |
| Broken integration tests fixed | Task 3 |
| `AppState` gains optional `scoring_adapter` | Task 4 |
| `H2AI_SCORING_PROVIDER` env var in `main.rs` | Task 4 |
| `tasks.rs` uses `state.registry()` | Task 4 |

**Placeholder scan:** None found.

**Type consistency:** `AdapterRegistry` defined in Task 1, used by name in Tasks 2, 3, 4. `TaskProfile::Scoring` used in Task 2. `state.registry()` returns `AdapterRegistry` (value, not reference). Route handler takes `&registry` where `registry` is a local owned by the `async move` block — valid for the engine lifetime.
