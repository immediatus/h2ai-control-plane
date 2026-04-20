# Constraint Corpus Guide

The Dark Knowledge Compiler is only as good as the constraints it reads. This guide explains the typed constraint system, what makes an effective corpus, how the compiler reads ADRs and `ConstraintDoc` files, and how to diagnose `ContextUnderflowError`.

---

## Why constraints matter

The `J_eff` gate exists because LLM agents hallucinate architectural decisions that the human never externalized. An agent asked to design a service will make choices about authentication, database access, service boundaries, and error handling — and without explicit constraints, it will guess.

When it guesses wrong, the Auditor rejects the proposal (`BranchPrunedEvent`). If every Explorer guesses wrong in the same way, you get `ZeroSurvivalEvent` followed by exhausted retries and `TaskFailedEvent`.

Constraints are the mechanism by which your team's tacit knowledge becomes explicit, machine-checkable rules. The compiler reads them, indexes their constraint coverage, and computes `J_eff` as the Jaccard overlap between what you provided and what the task requires.

**A corpus of well-written constraints turns `ContextUnderflowError` into `202 Accepted`.**

---

## Constraint document types

### Option A — ADR format (backward-compatible)

The compiler recognizes ADRs in standard format. The `## Constraints` section is what matters for Auditor enforcement; everything else is context.

```markdown
# ADR-{number}: {short title}

## Status
{Proposed | Accepted | Deprecated | Superseded by ADR-N}

## Context
What is the situation that forced this decision? Be explicit about constraints —
compliance requirements, team conventions, integration dependencies.

## Decision
The decision, stated as an active sentence.

## Consequences
What becomes easier? What becomes harder?

## Constraints

(strongly recommended)
Bullet list of hard rules that agents must not violate:
- Service X must not write directly to the database.
- All tokens expire in ≤ 15 minutes.
- No synchronous calls across service boundary Y.
```

The `## Constraints` heading produces a `ConstraintDoc` with `Hard { threshold: 0.8 }` severity and a `VocabularyPresence { AllOf }` predicate built from the bullet terms. This is the backward-compatible path — **no existing ADR files need to be changed**.

### Option B — Typed ConstraintDoc format

For precise, machine-checkable rules, you can use the full typed format with explicit severity and predicate sections:

```markdown
# CONSTRAINT-{id}: {short title}

## Severity
Hard threshold=0.9

## Predicate
VocabularyPresence AllOf
- stateless
- jwt
- no session state

## Remediation
Ensure the proposal explicitly states that auth is stateless and does not store
session state. Reference ADR-001.
```

The `## Severity` heading accepts:
- `Hard threshold=<float>` — blocks merge if score < threshold; compliance → 0.0 if any Hard constraint fails
- `Soft weight=<float>` — contributes to weighted soft score
- `Advisory` — informational only; never blocks

The `## Predicate` heading accepts one of:
- `VocabularyPresence AllOf|AnyOf|NoneOf` + bullet terms
- `NegativeKeyword` + bullet terms (fails if any term appears)
- `RegexMatch must_match=true|false` + a single regex pattern bullet
- `NumericThreshold field=<regex> op=lt|le|eq|ge|gt value=<float>`
- `LlmJudge` + rubric text (evaluated async via the auditor adapter)

The `## Remediation` section provides the `remediation_hint` that the MAPE-K `RetryWithHints` action surfaces to the next generation round.

---

## Compliance formula

For a given proposal, the Auditor evaluates every `ConstraintDoc` in the corpus:

```
score_i        ∈ [0.0, 1.0]   (per predicate; VocabularyPresence AllOf is fractional)
hard_gate      = all Hard predicates have score_i ≥ threshold_i
soft_score     = Σ(w_i × score_i) / Σw_i   (Soft constraints only)
compliance     = if hard_gate { soft_score } else { 0.0 }
error_cost     = 1.0 − compliance            (recorded in BranchPrunedEvent)
```

`VocabularyMode` semantics:
- **`AllOf`** — fractional: `hits / total_terms`. All terms must appear for score = 1.0.
- **`AnyOf`** — binary 1.0 if any term appears, 0.0 otherwise.
- **`NoneOf`** — binary 1.0 if no term appears (negative keyword gate, same as `NegativeKeyword`).

Empty `Composite { And }` nodes return 1.0 (vacuously true). Empty `Composite { Or }` nodes return 0.0 (vacuously false).

---

## MAPE-K retry hints

When a proposal is pruned, `BranchPrunedEvent.violated_constraints` carries one `ConstraintViolation` per failed constraint:

```json
{
  "constraint_id": "ADR-001",
  "score": 0.2,
  "severity_label": "Hard",
  "remediation_hint": "Ensure the proposal explicitly states that auth is stateless."
}
```

The `RetryPolicy::decide` in `h2ai-autonomic` collects `remediation_hint` values from Hard violations first. When hints are present it returns `RetryWithHints` — the next generation wave receives targeted repair guidance rather than generic τ adjustments. Write clear remediation hints on Hard constraints to get the most useful MAPE-K retry behaviour.

---

## What the compiler extracts

The compiler builds a vocabulary index from each `ConstraintDoc`. For `VocabularyPresence` predicates:
- **Terms list** — used to compute `J_eff` (Jaccard overlap with task keywords) and to check proposals
- **Document ID** — referenced in `BranchPrunedEvent.constraint_id`

For `LlmJudge` predicates the rubric is compiled into `system_context` and executed by the auditor adapter at validation time.

---

## Writing effective constraints

### Be explicit about prohibitions

Vague:
```markdown
## Decision
We use JWT authentication.
```

Effective:
```markdown
## Decision
All services authenticate via JWT. No service stores session state.

## Constraints
- Services must not write session tokens to any storage medium.
- Services must not maintain a session revocation list (use short expiry instead).
- Token expiry must not exceed 15 minutes for access tokens.
- Refresh token rotation must invalidate the previous token on use.
```

The explicit prohibitions are what the Auditor checks proposals against. "We use JWT" tells an Explorer what to use. "Must not store session state" tells the Auditor what to reject.

### Name the affected scope

Vague:
```markdown
We don't allow direct database access.
```

Effective:
```markdown
## Constraints
- The `api-gateway` service must not query the `orders` or `inventory`
  databases directly. All reads go through `order-service` or
  `inventory-service` via gRPC.
```

### Always add a remediation hint to Hard constraints

```markdown
## Remediation
Add explicit language stating all reads go through the service boundary.
Reference ADR-003 gRPC requirement.
```

Remediation hints power `RetryWithHints` — without them, the MAPE-K loop falls back to generic τ adjustment or keyword-based hallucination detection, which is less precise.

### Reference compliance requirements

```markdown
## Constraints
- Personal data (as defined by GDPR Article 4) must not be logged in
  plaintext. Log entries must use pseudonymized identifiers only.
  [Compliance: GDPR-LOG-001]
```

Compliance references increase `J_eff` for tasks in regulated domains.

---

## Corpus organization

```
adr/
├── architecture/
│   ├── ADR-001-stateless-auth.md
│   ├── ADR-002-event-sourced-state.md
│   └── ADR-007-no-direct-db-access.md
├── security/
│   ├── ADR-010-gdpr-logging.md
│   └── CONSTRAINT-SEC-001-injection-prevention.md
├── infrastructure/
│   └── ADR-021-nats-as-event-log.md
└── deprecated/
    └── ADR-005-redis-session-store.md  # Status: Deprecated
```

The compiler scans the entire directory recursively. Deprecated ADRs are still indexed — the `Status: Deprecated` tag teaches the Auditor that a historical decision was explicitly reversed. Both ADR-format and typed ConstraintDoc-format files are loaded via the same `load_corpus` call.

---

## Diagnosing low J_eff

When `POST /tasks` returns `ContextUnderflowError`, the response includes `missing_coverage`:

```json
{
  "error": "ContextUnderflowError",
  "j_eff": 0.18,
  "threshold": 0.4,
  "missing_coverage": [
    "authentication strategy",
    "database access policy",
    "error propagation between services"
  ]
}
```

For each item in `missing_coverage`:

1. Check if a constraint doc exists but is poorly worded — rewrite the `## Constraints` or `## Predicate` section.
2. If no constraint doc exists — this is Dark Knowledge. Write the ADR or ConstraintDoc before resubmitting.
3. As a short-term workaround, add the constraint directly to the task manifest `context` field.

### The manifest context field

For task-specific constraints not worth a permanent document:

```json
{
  "description": "...",
  "context": "This task is scoped to the checkout flow only. The payment service must not be modified. All changes must be backward-compatible with the v2 checkout API."
}
```

This raises `J_eff` for the task without adding noise to the corpus.

---

## Corpus maintenance

**After every architectural decision:** Write the ADR or ConstraintDoc immediately. Dark Knowledge accretes fastest in the gap between "we decided" and "we wrote it down."

**After an incident:** Add a Hard constraint with a remediation hint that would have caught it. `BranchPrunedEvent.violated_constraints` from a `ZeroSurvivalEvent` shows exactly which constraints fired — extend them with remediation hints so future retries are targeted.

**After deprecating a pattern:** Mark the ADR `Status: Deprecated` and add a `Superseded by ADR-N` line. Do not delete it — the compiler uses deprecated ADRs to teach the Auditor about explicitly reversed decisions.

**Quarterly review:** Run `POST /calibrate` after significant corpus changes. If `κ_eff` drops, your Common Ground improved — agents share more knowledge and coordinate cheaper. If it rises, new constraints introduced epistemic distance between adapters.

---

## Minimum viable corpus

For a team starting from zero, these five ADRs cover the most common `ContextUnderflowError` causes:

1. **Authentication and session management** — how users are identified, token lifecycle, revocation policy
2. **Database access policy** — which services can read/write which databases, direct vs. service-mediated access
3. **Service boundary rules** — what crosses a service boundary, synchronous vs. async communication
4. **Error handling and propagation** — how errors surface to callers, retry policies, circuit breakers
5. **Sensitive data handling** — what is PII, where it may be stored, how it must be logged

With these five, most tasks will clear the `J_eff` threshold without needing the `context` workaround. Add remediation hints to each so that `ZeroSurvivalEvent` retries are targeted rather than generic.
