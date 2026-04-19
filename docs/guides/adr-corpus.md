# ADR Corpus Guide

The Dark Knowledge Compiler is only as good as the ADRs it reads. This guide explains what makes an ADR effective for `J_eff` computation, how to structure your corpus, and common mistakes that cause `ContextUnderflowError`.

---

## Why ADRs matter

The `J_eff` gate exists because LLM agents hallucinate architectural decisions that the human never externalized. An agent asked to design a service will make choices about authentication, database access, service boundaries, and error handling — and without explicit constraints, it will guess.

When it guesses wrong, the Auditor rejects the proposal (`BranchPrunedEvent`). If every Explorer guesses wrong in the same way, you get `ZeroSurvivalEvent` followed by exhausted retries and `TaskFailedEvent`.

ADRs are the mechanism by which your team's tacit knowledge becomes explicit context. The compiler reads them, indexes their constraint coverage, and computes `J_eff` as the Jaccard overlap between what you provided and what the task requires.

**A corpus of well-written ADRs turns `ContextUnderflowError` into `202 Accepted`.**

---

## ADR structure

The compiler recognizes ADRs in standard format. The `## Status`, `## Decision`, and `## Consequences` sections are required. Everything else is optional but valuable.

```markdown
# ADR-{number}: {short title}

## Status

{Proposed | Accepted | Deprecated | Superseded by ADR-N}

## Context

What is the situation that forced this decision? What forces are at play?
Be explicit about constraints — compliance requirements, team conventions,
integration dependencies, performance envelopes.

## Decision

The decision, stated as an active sentence.

## Consequences

What becomes easier? What becomes harder? What invariants must other
decisions respect?

## Constraints

(optional but strongly recommended)
Bullet list of hard rules that agents must not violate:
- Service X must not write directly to the database.
- All tokens expire in ≤ 15 minutes.
- No synchronous calls across service boundary Y.
```

---

## What the compiler extracts

The compiler builds a constraint index from each ADR. It extracts:

- **Service and component names** — used to identify which parts of the system the ADR governs
- **Prohibition statements** — phrases like "must not", "is forbidden", "never", "is prohibited"
- **Requirement statements** — phrases like "must", "is required", "always", "shall"
- **Named patterns** — design pattern references (stateless, event-sourced, hexagonal, etc.)
- **Compliance references** — regulatory identifiers (GDPR, SOC2, HIPAA, internal policy IDs)

A task description is then matched against this index. `J_eff` measures how much of the task's required knowledge domain is covered by the indexed constraints.

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
All services authenticate via JWT. No service stores session state in any
database, cache, or in-process store. Token validation is stateless —
signature verification only against the shared public key.

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
- The `reporting-service` is the only service permitted read-only access
  to the `analytics` replica.
```

Scope specificity increases `J_eff` for tasks that involve the named services.

### Include the rationale

The rationale is not just for humans — the compiler uses it to identify the *why* behind constraints, which helps the Auditor distinguish intentional violations from accidental ones.

```markdown
## Context
Direct database access from multiple services caused three incidents in
Q3 2025 where schema migrations broke unannounced consumers. CR-2025-09
requires service boundary enforcement.
```

### Reference compliance requirements

```markdown
## Constraints
- Personal data (as defined by GDPR Article 4) must not be logged in
  plaintext. Log entries must use pseudonymized identifiers only.
  [Compliance: GDPR-LOG-001]
- Audit logs must be written to the immutable append-only store at
  `audit.internal`. [Compliance: SOC2-CC7.2]
```

Compliance references increase `J_eff` for tasks in regulated domains.

---

## Corpus organization

```
adr/
├── architecture/
│   ├── ADR-001-stateless-auth.md
│   ├── ADR-002-event-sourced-state.md
│   ├── ADR-003-service-boundaries.md
│   └── ADR-007-no-direct-db-access.md
├── security/
│   ├── ADR-010-gdpr-logging.md
│   └── ADR-011-secret-management.md
├── infrastructure/
│   ├── ADR-020-kubernetes-profiles.md
│   └── ADR-021-nats-as-event-log.md
└── deprecated/
    └── ADR-005-redis-session-store.md  # Status: Deprecated
```

The compiler scans the entire directory recursively. Deprecated ADRs are still indexed — the `Status: Deprecated` tag teaches the Auditor that a historical decision was explicitly reversed.

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

1. Check if an ADR exists but is poorly worded — rewrite the constraint section.
2. If no ADR exists — this is Dark Knowledge. Write the ADR before resubmitting.
3. As a short-term workaround, add the constraint directly to the task manifest `context` field.

### The manifest context field

For task-specific constraints not worth a permanent ADR:

```json
{
  "description": "...",
  "context": "This task is scoped to the checkout flow only. The payment service must not be modified. All changes must be backward-compatible with the v2 checkout API."
}
```

This raises `J_eff` for the task without adding noise to the corpus.

---

## Corpus maintenance

**After every architectural decision:** Write the ADR immediately, not weeks later. Dark Knowledge accretes fastest in the gap between "we decided" and "we wrote it down."

**After an incident:** Add a constraint that would have caught it. `BranchPrunedEvent` logs from a `ZeroSurvivalEvent` are direct evidence of what your corpus was missing.

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

With these five, most tasks will clear the `J_eff` threshold without needing the `context` workaround.
