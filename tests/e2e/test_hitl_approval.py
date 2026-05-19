#!/usr/bin/env python3
"""
Standalone HITL approval e2e test.

Validates the full approval callback flow:
  1. Submit task with require_approval=true to a specific tenant
  2. Receive PendingApprovalEvent — assert task_id matches, proposed_output is present
     and contains task-domain content (the output must be about the actual task)
  3. Derive a reviewer_note from the actual proposed_output content
     (proves the reviewer read the callback, not just blindly ACK'd it)
  4. POST /signal with Approve payload addressed to exact task_id + tenant_id
  5. Assert ApprovalResolvedEvent echoes back operator_id, reviewer_note, approved=True
  6. Assert task reaches MergeResolved

Usage (server must be running with HITL enabled):
  python3 tests/e2e/test_hitl_approval.py

Start server first:
  H2AI_CONFIG=tests/e2e/scenarios/features/03-hitl/h2ai.toml ./target/release/h2ai-control-plane
"""

import sys
import time
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from client import submit_task, stream_events, submit_signal, wait_for_health

TENANT_ID = "compliance-team"
OPERATOR_ID = "e2e-operator"

TASK = {
    "description": (
        "Design an automated approval workflow for high-value wire transfers. "
        "Route transfers over $500K to a compliance officer within 60 seconds. "
        "Auto-approve transfers from the pre-approved counterparty whitelist under $2M. "
        "Hard-block any transfer where the beneficiary appears on the OFAC SDN list."
    ),
    "require_approval": True,
    "pareto_weights": {"diversity": 0.4, "containment": 0.5, "throughput": 0.1},
    "explorers": {"count": 2, "tau_min": 0.3, "tau_max": 0.7},
    "context": (
        "Whitelisted counterparties: 47 pre-approved correspondent banks. "
        "OFAC SDN list: updated daily at 02:00 UTC. "
        "Audit trail must be written before the transfer is executed."
    ),
}

# Task-domain keywords that must appear in the proposed_output.
# These are the core requirements from the task description — a valid output
# must address each of these topics for the reviewer to be able to evaluate it.
REQUIRED_CONTENT_KEYWORDS: list[tuple[str, list[str]]] = [
    ("OFAC / SDN block",       ["OFAC", "SDN", "sanction"]),
    ("whitelist / counterparty", ["whitelist", "counterparty", "correspondent"]),
    ("audit trail",            ["audit", "ledger", "log"]),
    ("compliance routing",     ["compliance", "officer", "route", "500"]),
]

TIMEOUT_S = 2400


# ── Assertion helpers ──────────────────────────────────────────────────────────

def assert_eq(label: str, actual, expected) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_not_none(label: str, actual) -> None:
    if actual is None:
        raise AssertionError(f"{label}: expected a value, got None")


def check_domain_content(proposed_output: str) -> str:
    """Validate proposed_output covers required task topics.

    Returns a reviewer_note derived from the actual content found,
    proving the reviewer read the output before approving.
    Raises AssertionError if any required topic is missing.
    """
    text_lower = proposed_output.lower()
    covered: list[str] = []
    missing: list[str] = []

    for topic, keywords in REQUIRED_CONTENT_KEYWORDS:
        if any(kw.lower() in text_lower for kw in keywords):
            covered.append(topic)
        else:
            missing.append(topic)

    if missing:
        raise AssertionError(
            f"proposed_output is missing required topic(s): {missing}. "
            f"Covered: {covered}. "
            f"Output preview: {proposed_output[:300]!r}"
        )

    # Build a reviewer_note grounded in the actual output content —
    # this proves the signal response is a reaction to the callback content,
    # not a blind ACK.
    return (
        f"Reviewed output (len={len(proposed_output)}). "
        f"Confirmed coverage: {', '.join(covered)}. "
        f"Approved for production."
    )


# ── Main test ─────────────────────────────────────────────────────────────────

def run() -> None:
    print("=== HITL Approval Signal E2E Test ===")
    print(f"  tenant_id: {TENANT_ID}")

    wait_for_health(timeout_s=30)

    task_id = submit_task(TASK, tenant_id=TENANT_ID)
    print(f"  task_id:   {task_id}")
    print()

    pending_approval_event: dict | None = None
    approval_resolved_event: dict | None = None
    reviewer_note_sent: str = ""
    terminal_kind: str = ""

    for event in stream_events(task_id, tenant_id=TENANT_ID, timeout_s=TIMEOUT_S):
        kind = event.get("kind", "")

        # ── PendingApproval ────────────────────────────────────────────────────
        if kind == "PendingApproval":
            pending_approval_event = event
            print("[HITL] PendingApproval received — validating callback content")

            # 1. The event must be for the exact task we submitted
            ev_task_id = str(event.get("task_id", ""))
            assert_eq("PendingApproval.task_id", ev_task_id, task_id)
            print(f"  task_id:          PASS  ({ev_task_id})")

            # 2. The proposed_output must be present and non-empty
            proposed_output: str = event.get("proposed_output") or ""
            if not proposed_output.strip():
                raise AssertionError(
                    "PendingApproval.proposed_output is empty — engine sent nothing to review"
                )
            print(f"  proposed_output:  PASS  (len={len(proposed_output)})")

            # 3. The callback must still be live (not already expired)
            timeout_at_ms: int = event.get("timeout_at_ms") or 0
            now_ms = int(time.time() * 1000)
            assert_not_none("PendingApproval.timeout_at_ms", timeout_at_ms or None)
            if timeout_at_ms <= now_ms:
                raise AssertionError(
                    f"PendingApproval.timeout_at_ms={timeout_at_ms} already expired "
                    f"(now={now_ms}); cannot respond"
                )
            remaining_s = (timeout_at_ms - now_ms) // 1000
            print(f"  timeout_at_ms:    PASS  ({remaining_s}s remaining)")

            # 4. The proposed_output must address the actual task requirements —
            #    validates the engine produced a meaningful response, not filler.
            #    The reviewer_note is derived from what was found, not hardcoded.
            print("  content check:")
            reviewer_note_sent = check_domain_content(proposed_output)
            for topic, keywords in REQUIRED_CONTENT_KEYWORDS:
                text_lower = proposed_output.lower()
                found = next((kw for kw in keywords if kw.lower() in text_lower), None)
                print(f"    {topic:30s}  PASS  (matched {found!r})")

            print()
            print(f"  reviewer_note derived from output content:")
            print(f"    {reviewer_note_sent!r}")
            print()

            # 5. Submit the signal — response must be addressed to the correct
            #    task_id and tenant_id; operator_id and reviewer_note are the
            #    payload we expect to see echoed in ApprovalResolvedEvent.
            print(f"[HITL] Submitting Approve signal")
            print(f"  POST /v1/{TENANT_ID}/tasks/{task_id}/signal")

            resp = submit_signal(
                task_id=task_id,
                tenant_id=TENANT_ID,
                payload={
                    "kind": "Approve",
                    "data": {
                        "approved": True,
                        "reviewer_note": reviewer_note_sent,
                        "operator_id": OPERATOR_ID,
                    },
                },
            )
            assert_eq("submit_signal.status", resp.get("status"), "signal_queued")
            print(f"  response:  PASS  (status={resp.get('status')!r})")
            print()

        # ── ApprovalResolved ───────────────────────────────────────────────────
        elif kind == "ApprovalResolved":
            approval_resolved_event = event
            print("[HITL] ApprovalResolvedEvent received — verifying echo")

            ev_task_id   = str(event.get("task_id", ""))
            ev_operator  = event.get("operator_id", "")
            ev_note      = event.get("reviewer_note")
            ev_approved  = event.get("approved")
            ev_decided   = event.get("decided_at_ms")

            # Every field we sent must come back verbatim — the engine must not
            # alter or drop the operator context.
            assert_eq("ApprovalResolved.task_id",     ev_task_id,  task_id)
            assert_eq("ApprovalResolved.operator_id", ev_operator, OPERATOR_ID)
            assert_eq("ApprovalResolved.reviewer_note", ev_note,   reviewer_note_sent)
            assert_eq("ApprovalResolved.approved",    ev_approved, True)
            assert_not_none("ApprovalResolved.decided_at_ms", ev_decided)

            print(f"  task_id:       PASS  ({ev_task_id})")
            print(f"  operator_id:   PASS  ({ev_operator!r})")
            print(f"  reviewer_note: PASS  (matches derived note)")
            print(f"  approved:      PASS  ({ev_approved})")
            print(f"  decided_at_ms: PASS  ({ev_decided})")
            print()

        elif kind in ("MergeResolved", "TaskFailed"):
            terminal_kind = kind
            j_eff = event.get("j_eff")
            print(f"[terminal] {kind}" + (f"  j_eff={j_eff:.3f}" if j_eff else ""))
            break

        else:
            score = event.get("score")
            suffix = f"  score={score:.2f}" if score is not None else ""
            print(f"  event: {kind}{suffix}")

    # ── Final assertions ───────────────────────────────────────────────────────
    print()
    print("=== Final Assertions ===")
    errors: list[str] = []

    if pending_approval_event is None:
        errors.append("PendingApprovalEvent never fired — HITL gate did not activate")

    if approval_resolved_event is None:
        errors.append("ApprovalResolvedEvent never received — signal may not have reached engine")

    if terminal_kind != "MergeResolved":
        errors.append(f"terminal={terminal_kind!r}, expected 'MergeResolved'")

    if errors:
        print("FAIL")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)

    print("PASS")
    print(f"  ✓ PendingApproval for task={task_id} tenant={TENANT_ID!r}")
    print(f"  ✓ proposed_output covers all required domain topics")
    print(f"  ✓ reviewer_note derived from callback content (not hardcoded)")
    print(f"  ✓ signal routed to correct tenant_id + task_id")
    print(f"  ✓ ApprovalResolved echoes operator_id, reviewer_note, approved=True")
    print(f"  ✓ task completed: MergeResolved")


if __name__ == "__main__":
    run()
