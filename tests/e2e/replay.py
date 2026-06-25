#!/usr/bin/env python3
"""
H2AI scenario replay — regression and improvement analysis.

Starts the server with the scenario-specific config, submits the task,
captures the full SSE event stream, checks assertions, saves artifacts.

Usage:
  python3 tests/e2e/replay.py                                    # all scenarios
  python3 tests/e2e/replay.py benchmark                          # single scenario by name
  python3 tests/e2e/replay.py features/01-thinking-loop          # feature isolation scenario
  python3 tests/e2e/replay.py --list                             # list available scenarios
  python3 tests/e2e/replay.py --baseline benchmark               # direct LLM (no H2AI)
  python3 tests/e2e/replay.py --config baseline.toml benchmark   # run with alternate config
  python3 tests/e2e/replay.py --trials 3 benchmark               # run k times, report pass^k
  python3 tests/e2e/replay.py --compare benchmark                # h2ai vs baseline delta table
  python3 tests/e2e/replay.py --triple features/09-full-stack    # 3-way: LLM vs LLM+RAG vs H2AI

Output per run:
  tests/e2e/results/<scenario>/<timestamp>/
    events.jsonl   — raw SSE event stream (one JSON object per line)
    output.txt     — merged output text from MergeResolved
    summary.json   — signals, assertion results, pass/fail
"""

import argparse
import collections
import dataclasses
import datetime
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
import tomllib
import traceback
import urllib.request

from client import submit_task, stream_events, submit_signal, wait_for_health, trigger_calibration_and_wait, DEFAULT_TENANT


# ── Per-event structured tracer ───────────────────────────────────────────────

def _elapsed(start: float) -> str:
    s = int(time.monotonic() - start)
    return f"+{s // 60:02d}:{s % 60:02d}"


def _shorten(s: str, n: int = 90) -> str:
    if not s:
        return ""
    s = s.replace("\n", " ")
    return s[:n] + "…" if len(s) > n else s


def _trace_event(event: dict, start: float) -> None:
    """Print a structured, annotated trace line (+ optional detail lines) for one SSE event."""
    kind = event.get("kind", "unknown")
    elapsed = _elapsed(start)
    header = ""
    details: list[str] = []

    if kind == "AwarenessProbeCompleted":
        verdicts = event.get("verdicts") or []
        n_gated = sum(1 for v in verdicts if v.get("gated"))
        flag = "⚠ " if n_gated else ""
        header = f"{flag}mode={event.get('mode')}  n={len(verdicts)}  gated={n_gated}  re_iterated={event.get('re_iterated', False)}"
        for v in verdicts:
            verdict = v.get("verdict", "?")
            cid = v.get("constraint_id", "?")
            gated_tag = "  [GATED]" if v.get("gated") else ""
            icon = "✓" if verdict == "ACKNOWLEDGED" else "✗"
            details.append(f"    {icon} {cid}  {verdict}{gated_tag}")
            if verdict != "ACKNOWLEDGED":
                details.append(f"       {_shorten(v.get('rationale') or '', 100)}")

    elif kind == "ThinkingLoopCompleted":
        header = (
            f"enabled={event.get('enabled')}  iterations={event.get('iterations_run')}"
            f"  coverage={event.get('coverage_score', 0.0):.2f}"
            f"  shared_len={event.get('shared_understanding_len', 0)}"
        )
        archetypes = event.get("archetypes") or []
        if archetypes:
            details.append(f"    archetypes: {', '.join(str(a) for a in archetypes[:4])}")

    elif kind == "ComplexityProbe":
        complexity = event.get("complexity", "?")
        decompose = event.get("decompose_recommended", False)
        latency_ms = event.get("probe_latency_ms", 0)
        flag = "⚠ " if (isinstance(complexity, int) and complexity >= 4) else ""
        header = f"{flag}complexity={complexity}/5  decompose={decompose}  latency={latency_ms}ms"
        rationale = event.get("rationale") or ""
        if rationale:
            details.append(f"    {_shorten(rationale, 110)}")

    elif kind == "TaskComplexityAssessed":
        tcc = event.get("tcc_effective")
        quadrant = event.get("task_quadrant", "?")
        n_eff = event.get("n_eff_pool")
        skip_reason = event.get("probe_skip_reason") or ""
        tcc_str = f"{tcc:.3f}" if isinstance(tcc, float) else "—"
        header = f"quadrant={quadrant}  tcc_eff={tcc_str}  n_eff={n_eff}" + (f"  skip={skip_reason}" if skip_reason else "")

    elif kind == "VerificationScored":
        score = event.get("score", 0.0)
        passed = event.get("passed", False)
        reason = (event.get("reason") or "").strip()
        eid = str(event.get("explorer_id") or "")[:8]
        icon = "✓" if passed else "✗"
        header = f"{icon} explorer={eid}  score={score:.2f}  {'PASS' if passed else 'FAIL'}"
        if reason:
            details.append(f"    violated: {reason}")

    elif kind == "BranchPruned":
        constraints = [v.get("constraint_id", "?") for v in event.get("violated_constraints") or []]
        wave = event.get("retry_count", "?")
        eid = str(event.get("explorer_id") or "")[:8]
        header = f"⚠ wave={wave}  explorer={eid}  constraints=[{', '.join(constraints)}]"
        for v in (event.get("violated_constraints") or []):
            cid = v.get("constraint_id", "?")
            score = v.get("score", 0.0)
            severity = v.get("severity_label", "?")
            hint = _shorten(v.get("remediation_hint") or "", 80)
            details.append(f"    ✗ {cid}  score={score:.2f}  severity={severity}" + (f"  hint: {hint}" if hint else ""))

    elif kind == "ZeroSurvival":
        wave = event.get("wave", "?")
        reason = _shorten(event.get("reason") or "", 80)
        header = f"⚠ wave={wave}  all proposals pruned — MAPE-K retry"
        if reason:
            details.append(f"    {reason}")

    elif kind == "SelectionResolved":
        valid = event.get("valid_proposals") or []
        pruned = event.get("pruned_proposals") or []
        n_in = event.get("n_input_proposals", 0)
        n_failed = event.get("n_failed_proposals", 0)
        strategy = event.get("merge_strategy", "?")
        header = f"valid={len(valid)}/{n_in}  pruned={len(pruned)}  failed={n_failed}  strategy={strategy}"
        for p in pruned:
            eid = str(p[0])[:8] if isinstance(p, list) else str(p)[:8]
            reason = p[1] if (isinstance(p, list) and len(p) > 1) else "?"
            details.append(f"    pruned: {eid}  ({reason})")

    elif kind == "MergeResolved":
        j_eff = event.get("j_eff")
        out_len = len(event.get("resolved_output") or event.get("output") or "")
        j_str = f"{j_eff:.3f}" if j_eff is not None else "—"
        header = f"j_eff={j_str}  output={out_len} chars"

    elif kind == "TaskFailed":
        reason = _shorten(event.get("reason") or "", 100)
        header = f"⚠ {reason}"

    elif kind == "CorrelatedFabrication":
        cfi = event.get("cfi", 0.0)
        pressure = event.get("injection_pressure", 0.0)
        entities = event.get("shared_ungrounded_entities") or []
        hint_injected = event.get("hint_injected", False)
        flag = "⚠ " if cfi > 0.5 else ""
        header = f"{flag}CFI={cfi:.3f}  pressure={pressure:.3f}  hint_injected={hint_injected}"
        if entities:
            details.append(f"    shared entities: {', '.join(str(e) for e in entities[:6])}")

    elif kind == "ResearcherGrounding":
        slot = event.get("slot") or "?"
        source = event.get("source") or "?"
        assumption = _shorten(event.get("shared_assumption") or "", 80)
        lit = _shorten(event.get("literature_summary") or "", 110)
        header = f"slot={slot}  source={source}"
        if assumption:
            details.append(f"    assumption: {assumption}")
        if lit:
            details.append(f"    verdict: {lit}")

    elif kind == "GenerationKnowledge":
        injected = event.get("knowledge_injected", False)
        nodes = event.get("skill_nodes_count", 0)
        q = event.get("q_confidence", 0.0)
        header = f"injected={injected}  skill_nodes={nodes}  q_conf={q:.3f}"

    elif kind == "TaskAttribution":
        q = event.get("q_confidence", 0.0)
        basis = event.get("prediction_basis") or "?"
        waste = event.get("waste_ratio", 0.0)
        tokens = event.get("tokens_used", 0)
        skill_nodes = event.get("skill_nodes_injected", 0)
        header = f"q_conf={q:.3f}  basis={basis}  waste={waste:.2f}  tokens={tokens}  skill_nodes={skill_nodes}"

    elif kind == "OracleCalibrationPatched":
        pass_rate = event.get("oracle_pass_rate", 0.0)
        p_before = event.get("p_mean_before", 0.0)
        p_after = event.get("p_mean_after", 0.0)
        header = f"pass_rate={pass_rate:.2f}  p_mean {p_before:.3f}→{p_after:.3f}"

    elif kind == "LeaderElected":
        term = event.get("term")
        leader = str(event.get("leader_explorer_id") or "")[:8]
        cred = event.get("credibility_score", 0.0)
        header = f"term={term}  leader={leader}  credibility={cred:.2f}"

    elif kind == "SocraticDiagnosis":
        term = event.get("term")
        eig_rank = event.get("eig_rank")
        question = _shorten(event.get("question") or "", 90)
        header = f"term={term}  eig_rank={eig_rank}"
        if question:
            details.append(f"    Q: {question}")

    elif kind == "PendingApproval":
        header = "→ submitting auto-approval signal"

    elif kind == "ApprovalResolved":
        approved = event.get("approved", False)
        op = event.get("operator_id") or "?"
        note = _shorten(event.get("reviewer_note") or "", 60)
        header = f"approved={approved}  operator={op!r}" + (f"  note={note!r}" if note else "")

    elif kind == "VerifierFrozen":
        cid = event.get("constraint_id") or "?"
        reason = _shorten(event.get("reason") or "", 90)
        header = f"⚠ {cid}  {reason}"

    elif kind == "ConstraintAmbiguityDetected":
        cid = event.get("constraint_id") or "?"
        score = event.get("ambiguity_score") or 0.0
        rewrite = _shorten(event.get("suggested_rewrite") or "", 80)
        header = f"⚠ {cid}  ambiguity_score={score:.3f}"
        if rewrite:
            details.append(f"    suggested_rewrite: {rewrite}")

    elif kind == "ConstraintAmbiguity":
        cid = event.get("constraint_id") or "?"
        score = event.get("disagreement_rate") or event.get("score") or 0.0
        header = f"⚠ {cid}  disagreement_rate={score:.3f}"

    elif kind == "VerifierInstability":
        cid = event.get("constraint_id") or "?"
        wave = event.get("wave") or "?"
        header = f"⚠ {cid}  wave={wave}  verifier rejection reasons diverging"

    elif kind == "VerifierReasonContradiction":
        cid = event.get("constraint_id") or "?"
        wave = event.get("wave") or "?"
        header = f"⚠ {cid}  wave={wave}  contradictory verifier reasons detected"

    elif kind == "ComplexityCeilingDetected":
        wave = event.get("wave") or "?"
        header = f"⚠ wave={wave}  intra-retry complexity ceiling fired"

    elif kind == "BudgetExhausted":
        header = "⚠ per-task budget exhausted — retries blocked"

    elif kind == "CostThresholdWarning":
        used = (event.get("used_fraction") or 0.0)
        threshold = (event.get("threshold_fraction") or 0.0)
        header = f"⚠ budget {used:.0%} used (threshold {threshold:.0%})"

    elif kind == "TieredExit":
        k_accepted = event.get("k_accepted") or 0
        n_total = event.get("n_total") or 0
        header = f"tiered exit: {k_accepted}/{n_total} proposals accepted early"

    elif kind == "ConvergenceGate":
        n = event.get("n_proposals") or 0
        header = f"convergence gate: {n} proposals semantically equivalent"

    elif kind == "CoherenceIncomplete":
        domains = event.get("uncovered_domains") or []
        header = f"⚠ uncovered_domains=[{', '.join(str(d) for d in domains[:6])}]"

    elif kind == "EpistemicYield":
        yield_score = event.get("yield_score") or 0.0
        header = f"yield={yield_score:.3f}"

    elif kind == "ProvenanceRecorded":
        dc = event.get("document_confidence") or "?"
        provisions = event.get("provision_count") or 0
        open_gaps = event.get("open_gap_count") or 0
        flag = "⚠ " if open_gaps > 0 else ""
        header = f"{flag}document_confidence={dc}  provisions={provisions}  open_gaps={open_gaps}"

    elif kind == "TopologyProvisioned":
        n_exp = event.get("n_explorers") or event.get("explorer_count") or "?"
        topology = event.get("topology_kind") or "?"
        header = f"topology={topology}  n_explorers={n_exp}"

    elif kind == "TaskBootstrapped":
        n_constraints = event.get("n_constraints") or "?"
        n_checks = event.get("n_checks") or "?"
        header = f"n_constraints={n_constraints}  n_checks={n_checks}"

    elif kind == "GenerationPhaseCompleted":
        n_proposals = event.get("n_proposals") or "?"
        n_failed = event.get("n_failed") or 0
        header = f"n_proposals={n_proposals}  n_failed={n_failed}"

    elif kind == "ProposalFailed":
        eid = str(event.get("explorer_id") or "")[:8]
        reason = _shorten(event.get("reason") or "", 80)
        header = f"⚠ explorer={eid}  {reason}"

    elif kind == "MultiplicationConditionFailed":
        reason = _shorten(event.get("reason") or "", 90)
        header = f"⚠ multiplication condition rejected topology — {reason}"

    elif kind == "OproTriggered":
        j_eff = event.get("j_eff_ema") or event.get("j_eff") or 0.0
        header = f"j_eff_ema={j_eff:.3f}  OPRO optimization triggered"

    elif kind == "PromptVariantPromoted":
        variant = _shorten(event.get("variant_id") or "?", 24)
        delta = event.get("j_eff_delta") or 0.0
        header = f"variant={variant}  j_eff_delta={delta:.3f}"

    elif kind == "OracleGateResult":
        passed_count = len(event.get("passed_proposals") or [])
        failed_count = len(event.get("failed_proposals") or [])
        header = f"passed={passed_count}  failed={failed_count}"

    elif kind == "ConsensusRequired":
        header = f"⚠ BFT error costs exceeded threshold — switching to consensus merge"

    elif kind == "ConstraintRepairAttempted":
        cid = event.get("constraint_id") or "?"
        header = f"constraint repair started: {cid}"

    elif kind == "ConstraintVersionCreated":
        cid = event.get("constraint_id") or "?"
        version = event.get("version") or "?"
        header = f"constraint version created: {cid}  v{version}"

    elif kind == "ConstraintRepairFailed":
        cid = event.get("constraint_id") or "?"
        reason = _shorten(event.get("reason") or "", 80)
        header = f"⚠ constraint repair failed: {cid}  {reason}"

    elif kind == "CalibrationCompleted":
        n_max = event.get("n_max") or event.get("n_max_computed")
        n_it = event.get("n_it_optimal")
        beta = event.get("beta_eff") or event.get("beta_effective")
        source = event.get("calibration_source") or event.get("source") or "?"
        flag = "⚠ " if (isinstance(n_max, (int, float)) and n_max < 3) else ""
        parts: list[str] = []
        if n_max is not None:
            parts.append(f"N_max={n_max}")
        if n_it is not None:
            parts.append(f"n_it={n_it}")
        if isinstance(beta, float):
            parts.append(f"β_eff={beta:.4f}")
        parts.append(f"source={source}")
        header = flag + "  ".join(parts)

    elif kind == "DiversityGuardDegraded":
        reason = _shorten(event.get("reason") or "", 90)
        header = f"⚠ {reason}"

    elif kind == "InductionTriggered":
        n_hints = event.get("n_hints") or event.get("hints_loaded") or "?"
        wave = event.get("wave") or "?"
        header = f"wave={wave}  n_hints={n_hints}  cross-task priming loaded"

    elif kind == "KnowledgeGapDetected":
        domains = event.get("domains") or []
        n_cold = event.get("n_cold_checks") or "?"
        header = f"n_cold_checks={n_cold}  domains=[{', '.join(str(d) for d in domains[:4])}]"

    elif kind == "RepairContextBuilt":
        n_targets = event.get("n_targets") or "?"
        wave = event.get("wave") or "?"
        best_score = event.get("global_best_score")
        score_str = f"  best_score={best_score:.2f}" if isinstance(best_score, float) else ""
        meta_repair = "  [MetaRepair]" if event.get("meta_repair_injected") else ""
        header = f"wave={wave}  n_targets={n_targets}{score_str}{meta_repair}"

    elif kind == "InductionResult":
        n_patterns = event.get("n_patterns_matched") or "?"
        hint_len = len(event.get("hint_text") or "")
        header = f"n_patterns={n_patterns}  hint_len={hint_len}"

    elif kind == "OomGuardFired":
        rss_mb = event.get("rss_mb") or "?"
        threshold_mb = event.get("threshold_mb") or "?"
        header = f"⚠ RSS={rss_mb}MB ≥ threshold={threshold_mb}MB — retries blocked"

    elif kind == "GenerationTimeout":
        n_timed_out = event.get("n_timed_out") or "?"
        n_completed = event.get("n_completed") or "?"
        header = f"⚠ timed_out={n_timed_out}  completed={n_completed}"

    elif kind == "VerifierFreezeBypass":
        cid = event.get("constraint_id") or "?"
        waves_frozen = event.get("waves_frozen") or "?"
        header = f"⚠ {cid}  bypass active after {waves_frozen} frozen waves"

    elif kind == "DppmClusterSolveCompleted":
        n_clusters = event.get("n_clusters") or "?"
        n_solved = event.get("n_solved") or "?"
        header = f"clusters={n_clusters}  solved={n_solved}"

    elif kind == "IntegrationWaveCompleted":
        score = event.get("score")
        score_str = f"{score:.2f}" if isinstance(score, float) else "—"
        header = f"integration_score={score_str}"

    else:
        skip = {"kind", "task_id", "timestamp", "explorer_id", "tenant_id"}
        interesting = {k: v for k, v in event.items() if k not in skip and v is not None}
        if interesting:
            header = "  ".join(f"{k}={v}" for k, v in list(interesting.items())[:5])

    print(f"  [{elapsed}] {kind:<36} {header}", flush=True)
    for dl in details:
        print(dl, flush=True)


# ── Per-wave execution tracker ────────────────────────────────────────────────

@dataclasses.dataclass
class _WaveSummary:
    wave: int
    n_proposed: int = 0
    n_passed: int = 0
    n_pruned: int = 0
    violated: list = dataclasses.field(default_factory=list)
    scores: list = dataclasses.field(default_factory=list)
    zero_survival: bool = False

    @property
    def avg_score(self) -> "float | None":
        return sum(self.scores) / len(self.scores) if self.scores else None


class ExecutionTracker:
    """Accumulates cross-event state to produce wave-level summaries and anomaly diagnosis."""

    def __init__(self, start: float) -> None:
        self.start = start
        self.waves: list[_WaveSummary] = [_WaveSummary(wave=0)]
        self.phase_log: list[tuple[str, str]] = []  # (elapsed, label)
        self.violation_counts: collections.Counter = collections.Counter()
        self.grounding_cfi_history: list[float] = []
        self.calibration: "dict | None" = None

    @property
    def current_wave(self) -> _WaveSummary:
        return self.waves[-1]

    def observe(self, event: dict) -> None:
        kind = event.get("kind", "")
        if kind == "TaskBootstrapped":
            self.phase_log.append((_elapsed(self.start), "Bootstrapped"))
        elif kind == "ThinkingLoopCompleted":
            self.phase_log.append((_elapsed(self.start), "ThinkingDone"))
        elif kind == "TopologyProvisioned":
            self.phase_log.append((_elapsed(self.start), "TopologyReady"))
        elif kind == "GenerationPhaseCompleted":
            self.phase_log.append((_elapsed(self.start), "GenerationDone"))
            self.current_wave.n_proposed += event.get("n_proposals") or 0
        elif kind == "VerificationScored":
            score = event.get("score", 0.0)
            self.current_wave.scores.append(score)
            if event.get("passed", False):
                self.current_wave.n_passed += 1
        elif kind == "BranchPruned":
            self.current_wave.n_pruned += 1
            for v in event.get("violated_constraints") or []:
                cid = v.get("constraint_id", "")
                if cid:
                    self.current_wave.violated.append(cid)
                    self.violation_counts[cid] += 1
        elif kind == "ZeroSurvival":
            self.current_wave.zero_survival = True
            nxt = len(self.waves)
            self.waves.append(_WaveSummary(wave=nxt))
            self.phase_log.append((_elapsed(self.start), f"ZeroSurvival→wave{nxt}"))
        elif kind == "MergeResolved":
            self.phase_log.append((_elapsed(self.start), "MergeResolved"))
        elif kind == "TaskFailed":
            self.phase_log.append((_elapsed(self.start), "TaskFailed"))
        elif kind == "CorrelatedFabrication":
            self.grounding_cfi_history.append(event.get("cfi", 0.0))
        elif kind == "CalibrationCompleted":
            self.calibration = event


def _print_wave_summary(tracker: ExecutionTracker) -> None:
    active = [w for w in tracker.waves if w.n_proposed > 0 or w.scores or w.zero_survival]
    if not active:
        return
    print("  ── wave breakdown ──────────────────────────────────────────────────────")
    print(f"  {'wave':<5} {'prop':>5} {'pass':>5} {'prune':>6} {'avg_score':>10}  top violated constraints")
    for w in active:
        avg = f"{w.avg_score:.2f}" if w.avg_score is not None else "  —"
        zs = " ← ZeroSurvival" if w.zero_survival else ""
        seen: list[str] = []
        for c in w.violated:
            if c not in seen:
                seen.append(c)
            if len(seen) == 3:
                break
        top_v = ", ".join(seen)
        print(f"  {w.wave:<5} {w.n_proposed:>5} {w.n_passed:>5} {w.n_pruned:>6} {avg:>10}  {top_v}{zs}")
    if tracker.violation_counts:
        top = tracker.violation_counts.most_common(6)
        parts = "  ".join(f"{cid}×{cnt}" for cid, cnt in top)
        print(f"  violation totals: {parts}")
    print()


def _print_phase_timeline(tracker: ExecutionTracker) -> None:
    if not tracker.phase_log:
        return
    phases = "  →  ".join(f"{label}({t})" for t, label in tracker.phase_log)
    print(f"  timeline: {phases}")
    print()


def _scan_server_log_issues(log_path: "pathlib.Path") -> list[str]:
    """Return ERROR / WARN / PANIC lines from the server log."""
    if not log_path.exists():
        return []
    issues: list[str] = []
    try:
        with open(log_path) as f:
            for line in f:
                lower = line.lower()
                if any(m in lower for m in ("error", "panic", "thread 'main' panicked")):
                    issues.append(line.rstrip())
                elif " warn " in lower:
                    issues.append(line.rstrip())
    except OSError:
        pass
    return issues


def _print_server_log_tail(log_path: "pathlib.Path", n: int = 40) -> None:
    if not log_path.exists():
        return
    try:
        lines = log_path.read_text().splitlines()
        tail = lines[-n:] if len(lines) > n else lines
        print(f"  ── server log tail ({log_path.name}) {'─'*40}")
        for line in tail:
            print(f"  {line}")
        print()
    except OSError:
        pass


def _diagnose_run(tracker: ExecutionTracker, terminal_kind: str, j_eff: "float | None") -> None:
    issues: list[str] = []

    # Constraints that keep failing across proposals / waves
    for cid, cnt in tracker.violation_counts.most_common():
        if cnt >= 2:
            issues.append(
                f"constraint {cid} violated {cnt} times total — "
                "repair context may not be addressing the root cause"
            )

    # Wave 0 ZeroSurvival: nothing passed in the first wave
    if tracker.waves and tracker.waves[0].zero_survival:
        issues.append("wave 0 ended in ZeroSurvival — all initial proposals pruned before any passed")

    # Flat compliance scores across waves: repair is not improving anything
    avgs = [w.avg_score for w in tracker.waves if w.avg_score is not None]
    if len(avgs) >= 2:
        spread = max(avgs) - min(avgs)
        if spread < 0.05:
            avg_str = ", ".join(f"{a:.2f}" for a in avgs)
            issues.append(
                f"compliance scores flat across {len(avgs)} waves [{avg_str}] "
                "— repair context is not improving constraint pass rate"
            )

    # CFI rising: grounding not working
    hist = tracker.grounding_cfi_history
    if len(hist) >= 2 and hist[-1] > hist[0] + 0.10:
        issues.append(
            f"CFI rising across waves: {hist[0]:.2f} → {hist[-1]:.2f} "
            "— grounding hints are not reducing correlated fabrication"
        )

    # TaskFailed: surface it as a diagnosis item too
    if terminal_kind == "TaskFailed":
        issues.append("task terminated with TaskFailed — check server log (shown above) for root cause")

    # Low j_eff
    if j_eff is not None and j_eff < 0.40:
        issues.append(
            f"j_eff={j_eff:.3f} below 0.40 — ensemble efficiency degraded "
            "(low diversity, high pruning rate, or insufficient passing proposals)"
        )

    print("  ── diagnosis ───────────────────────────────────────────────────────────")
    if issues:
        for issue in issues:
            print(f"  ⚠ {issue}")
    else:
        print("  ✓ no anomalies detected")
    print()


REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
SCENARIOS_DIR = pathlib.Path(__file__).parent / "scenarios"
RESULTS_DIR = pathlib.Path(__file__).parent / "results"
SERVER_BIN = REPO_ROOT / "target2" / "release" / "h2ai-control-plane"


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=REPO_ROOT, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


GIT_SHA = _git_sha()
MULTIFAMILY = os.environ.get("H2AI_E2E_MULTIFAMILY", "").strip() == "1"
# Single model token budget for all LLM calls in the e2e harness.
# Mirrors model_max_tokens in reference.toml.
MODEL_MAX_TOKENS = 32768


# ── Scenario loading ──────────────────────────────────────────────────────────

def load_scenarios(names: list[str] | None) -> list[tuple[str, pathlib.Path, dict]]:
    """Return list of (display_name, scenario_dir, task_dict).

    Scans recursively under SCENARIOS_DIR so any subdirs are discovered
    automatically.  The display name is the path relative to SCENARIOS_DIR
    with slashes preserved (e.g. "complexity-routing").
    Names passed on the CLI can match either the full relative path or the
    leaf directory name.
    """
    paths = sorted(SCENARIOS_DIR.glob("**/task.json"))
    if not paths:
        raise RuntimeError(f"no scenarios under {SCENARIOS_DIR}")
    result = []
    for path in paths:
        scenario_dir = path.parent
        rel = scenario_dir.relative_to(SCENARIOS_DIR)
        display_name = str(rel)
        leaf_name = scenario_dir.name
        if names:
            if display_name not in names and leaf_name not in names:
                continue
        result.append((display_name, scenario_dir, json.loads(path.read_text())))
    if names:
        found_display = {n for n, _, _ in result}
        found_leaf = {d.name for _, d, _ in result}
        missing = [n for n in names if n not in found_display and n not in found_leaf]
        if missing:
            available = [str(p.parent.relative_to(SCENARIOS_DIR)) for p in paths]
            raise RuntimeError(f"scenario(s) not found: {missing}. Available: {available}")
    return result


# ── Server lifecycle ──────────────────────────────────────────────────────────

def check_no_server_running() -> None:
    try:
        with urllib.request.urlopen("http://localhost:8080/health", timeout=2):
            raise RuntimeError(
                "server already running on :8080 — unknown config, test isolation violated.\n"
                "  Stop it first:  kill $(lsof -ti:8080)"
            )
    except urllib.error.URLError:
        pass  # nothing listening — good


def start_server(scenario_dir: pathlib.Path, config_file: str = "h2ai.toml") -> subprocess.Popen:
    check_no_server_running()
    if not SERVER_BIN.exists():
        raise RuntimeError(f"binary not found: {SERVER_BIN} — run: cargo build --release")
    config_path = scenario_dir / config_file
    if not config_path.exists():
        raise RuntimeError(f"config not found: {config_path}")
    env = os.environ.copy()
    env["H2AI_CONFIG"] = str(config_path)
    log_path = REPO_ROOT / "tests" / "e2e" / "results" / "_server_logs" / f"{config_path.stem}-{scenario_dir.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        [str(SERVER_BIN)],
        env=env,
        cwd=REPO_ROOT,
        stdout=log_fh,
        stderr=log_fh,
    )
    proc._log_fh = log_fh  # type: ignore[attr-defined]
    proc._log_path = log_path  # type: ignore[attr-defined]
    print(f"  server pid={proc.pid}  config={config_path.relative_to(REPO_ROOT)}  log={log_path.relative_to(REPO_ROOT)}")
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    fh = getattr(proc, "_log_fh", None)
    if fh:
        fh.close()


# ── H2AI replay ───────────────────────────────────────────────────────────────

def run_scenario(scenario_name: str, task: dict) -> dict:
    tenant_id = task.get("_tenant_id", DEFAULT_TENANT)
    payload = {k: v for k, v in task.items() if not k.startswith("_")}
    task_id = submit_task(payload, tenant_id=tenant_id)
    print(f"  tenant_id: {tenant_id}  task_id: {task_id}")

    timeout_s = task.get("_timeout_s", 1800)
    run_start = time.monotonic()
    tracker = ExecutionTracker(run_start)
    events_raw: list[dict] = []
    verification_scores: list[float] = []
    grounding_events: list[dict] = []
    pruned_constraints: list[str] = []
    j_eff: float | None = None
    merged_output = ""
    terminal_kind = ""
    terminal = False
    thinking_loop_event: dict | None = None
    prediction_basis_final: str | None = None
    oracle_calibration_patched: dict | None = None
    hitl_gate_fired = False
    approval_signal_context: dict | None = None
    approval_resolved_event: dict | None = None
    leader_elected = False
    leader_elected_events: list[dict] = []
    provenance_recorded_event: dict | None = None

    # Wall-clock deadline: starts at task submission, resets after
    # ThinkingLoopCompleted so the wave always gets the full timeout_s budget
    # even when a long thinking loop runs before the first wave.
    wave_deadline = time.monotonic() + timeout_s

    for event in stream_events(task_id, tenant_id=tenant_id, timeout_s=300):
        kind = event.get("kind", "")

        # Thinking loop completing marks the start of wave execution.
        # Reset the deadline so the wave gets the full timeout_s budget.
        if kind == "ThinkingLoopCompleted":
            wave_deadline = time.monotonic() + timeout_s

        if time.monotonic() > wave_deadline:
            break

        events_raw.append(event)

        # ── state tracking (unchanged from before) ──
        if kind == "PendingApproval":
            event_task_id = str(event.get("task_id", ""))
            if event_task_id != task_id:
                raise AssertionError(
                    f"PendingApproval task_id mismatch: got {event_task_id!r}, expected {task_id!r}"
                )
            operator_id = "e2e-harness"
            reviewer_note = "auto-approved by e2e harness"
            approval_signal_context = {
                "task_id": task_id,
                "tenant_id": tenant_id,
                "operator_id": operator_id,
                "reviewer_note": reviewer_note,
            }
            _trace_event(event, run_start)
            try:
                resp = submit_signal(
                    task_id=task_id,
                    tenant_id=tenant_id,
                    payload={
                        "kind": "Approve",
                        "data": {
                            "approved": True,
                            "reviewer_note": reviewer_note,
                            "operator_id": operator_id,
                        },
                    },
                )
                print(f"             → approval signal queued (status={resp.get('status')})", flush=True)
            except Exception as exc:
                print(f"             ⚠ approval signal failed: {exc}", flush=True)
            hitl_gate_fired = True

        elif kind == "ApprovalResolved":
            approval_resolved_event = event
            ev_task_id = str(event.get("task_id", ""))
            ev_op = event.get("operator_id", "")
            ev_note = event.get("reviewer_note")
            ev_approved = event.get("approved")
            ctx = approval_signal_context or {}
            errors = []
            if ev_task_id != task_id:
                errors.append(f"task_id {ev_task_id!r} != {task_id!r}")
            if ctx and ev_op != ctx.get("operator_id"):
                errors.append(f"operator_id {ev_op!r} != {ctx.get('operator_id')!r}")
            if ctx and ev_note != ctx.get("reviewer_note"):
                errors.append(f"reviewer_note {ev_note!r} != {ctx.get('reviewer_note')!r}")
            if not ev_approved:
                errors.append(f"approved={ev_approved!r}, expected True")
            if errors:
                raise AssertionError("ApprovalResolved context mismatch: " + "; ".join(errors))
            _trace_event(event, run_start)

        elif kind == "VerificationScored":
            score = event.get("score", 0.0)
            verification_scores.append(score)
            _trace_event(event, run_start)

        elif kind == "BranchPruned":
            for v in event.get("violated_constraints", []):
                cid = v.get("constraint_id", "")
                if cid:
                    pruned_constraints.append(cid)
            _trace_event(event, run_start)

        elif kind == "CorrelatedFabrication":
            grounding_events.append(event)
            _trace_event(event, run_start)

        elif kind == "ThinkingLoopCompleted":
            thinking_loop_event = event
            _trace_event(event, run_start)

        elif kind == "TaskAttribution":
            prediction_basis_final = event.get("prediction_basis")
            _trace_event(event, run_start)

        elif kind == "OracleCalibrationPatched":
            oracle_calibration_patched = event
            _trace_event(event, run_start)

        elif kind == "LeaderElected":
            leader_elected = True
            leader_elected_events.append(event)
            _trace_event(event, run_start)

        elif kind == "ProvenanceRecorded":
            provenance_recorded_event = event
            _trace_event(event, run_start)

        elif kind == "MergeResolved":
            j_eff = event.get("j_eff")
            merged_output = event.get("resolved_output", event.get("output", ""))
            terminal = True
            terminal_kind = kind
            _trace_event(event, run_start)

        elif kind == "TaskFailed":
            terminal = True
            terminal_kind = kind
            _trace_event(event, run_start)

        else:
            _trace_event(event, run_start)

        tracker.observe(event)

        if terminal:
            break

    avg_score = sum(verification_scores) / len(verification_scores) if verification_scores else 0.0
    elapsed_total = int(time.monotonic() - run_start)
    print(
        f"  ── run complete in {elapsed_total}s:"
        f"  terminal={terminal_kind}  verified={len(verification_scores)}  avg_score={avg_score:.3f}",
        end="",
    )
    if j_eff is not None:
        print(f"  j_eff={j_eff:.3f}", end="")
    if thinking_loop_event:
        print(f"  thinking_iters={thinking_loop_event.get('iterations_run')}", end="")
    if leader_elected:
        print(f"  leader_elections={len(leader_elected_events)}", end="")
    print()
    print()
    _print_phase_timeline(tracker)
    _print_wave_summary(tracker)
    _diagnose_run(tracker, terminal_kind, j_eff)

    return {
        "task_id": task_id,
        "terminal": terminal,
        "terminal_kind": terminal_kind,
        "j_eff": j_eff,
        "verification_scores": verification_scores,
        "avg_verification_score": avg_score,
        "grounding_events": grounding_events,
        "pruned_constraints": pruned_constraints,
        "merged_output": merged_output,
        "events_raw": events_raw,
        "thinking_loop_event": thinking_loop_event,
        "prediction_basis_final": prediction_basis_final,
        "oracle_calibration_patched": oracle_calibration_patched,
        "hitl_gate_fired": hitl_gate_fired,
        "approval_signal_context": approval_signal_context,
        "approval_resolved_event": approval_resolved_event,
        "leader_elected": leader_elected,
        "leader_elected_events": leader_elected_events,
        "provenance_recorded_event": provenance_recorded_event,
        "tracker": tracker,
    }


# ── Assertions ────────────────────────────────────────────────────────────────

def check_assertions(result: dict, expected: dict, task_json: dict) -> dict[str, dict]:
    """Evaluate _expected assertions against a run result.

    Supported assertion keys (all opt-in — only evaluated when present in _expected):
    - terminal: exact match against terminal_kind ("MergeResolved", "TaskFailed", …)
    - valid_proposals_min: len(verification_scores) >= N
    - j_eff_min: j_eff >= diversity_weight * coverage_score
    - should_prune (MULTIFAMILY only): constraint IDs present in pruned_constraints
    - provenance_recorded: bool — ProvenanceRecordedEvent was emitted during the run
    - document_confidence_not_verified: bool — document_confidence != "High" (some provisions uncertain)
    - open_gap_count_min: int — open_gap_count from ProvenanceRecordedEvent >= N
    """
    out: dict[str, dict] = {}

    terminal_kind = result.get("terminal_kind", "")

    # terminal is opt-in — only assert when the scenario explicitly requires it
    if "terminal" in expected:
        exp = expected["terminal"]
        terminal_ok = terminal_kind == exp
        out["terminal"] = {"expected": exp, "actual": terminal_kind, "pass": terminal_ok}

    if "valid_proposals_min" in expected:
        actual = len(result["verification_scores"])
        exp = expected["valid_proposals_min"]
        out["valid_proposals_min"] = {"expected": exp, "actual": actual, "pass": actual >= exp}

    if "j_eff_min" in expected:
        diversity_weight = task_json["pareto_weights"]["diversity"]
        tl = result.get("thinking_loop_event")
        coverage = tl.get("coverage_score", 0.0) if tl else 0.0
        computed_min = diversity_weight * coverage
        actual = result["j_eff"]
        out["j_eff_min"] = {
            "expected": round(computed_min, 4),
            "actual": actual,
            "pass": actual is not None and actual >= computed_min,
            "computed_from": f"diversity={diversity_weight} × coverage={coverage:.3f}",
        }

    if MULTIFAMILY and "should_prune" in expected:
        for item in expected["should_prune"]:
            constraint_id = item.get("violates", "").split("—")[0].strip()
            found = any(constraint_id in c for c in result["pruned_constraints"])
            out[f"prune_{constraint_id}"] = {"expected": True, "actual": found, "pass": found}

    if "provenance_recorded" in expected:
        actual = result.get("provenance_recorded_event") is not None
        exp = expected["provenance_recorded"]
        out["provenance_recorded"] = {"expected": exp, "actual": actual, "pass": actual == exp}

    if "document_confidence_not_verified" in expected and expected["document_confidence_not_verified"]:
        pev = result.get("provenance_recorded_event")
        if pev is None:
            out["document_confidence_not_verified"] = {
                "expected": "not High", "actual": "no ProvenanceRecorded event", "pass": False,
            }
        else:
            dc = pev.get("document_confidence", "")
            out["document_confidence_not_verified"] = {
                "expected": "not High", "actual": dc, "pass": dc != "High",
            }

    if "open_gap_count_min" in expected:
        pev = result.get("provenance_recorded_event")
        if pev is None:
            out["open_gap_count_min"] = {
                "expected": expected["open_gap_count_min"], "actual": None, "pass": False,
            }
        else:
            actual = pev.get("open_gap_count", 0)
            exp = expected["open_gap_count_min"]
            out["open_gap_count_min"] = {"expected": exp, "actual": actual, "pass": actual >= exp}

    return out


# ── Result persistence ────────────────────────────────────────────────────────

def save_results(scenario_name: str, task: dict, result: dict, assertions: dict) -> pathlib.Path:
    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = RESULTS_DIR / scenario_name / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "events.jsonl", "w") as f:
        for ev in result["events_raw"]:
            f.write(json.dumps(ev) + "\n")

    if result["merged_output"]:
        (out_dir / "output.txt").write_text(result["merged_output"])

    tl = result.get("thinking_loop_event")
    ocp = result.get("oracle_calibration_patched")
    summary = {
        "scenario": scenario_name,
        "benchmark": task.get("_benchmark"),
        "git_sha": GIT_SHA,
        "timestamp": ts,
        "task_id": result["task_id"],
        "terminal_kind": result["terminal_kind"],
        "j_eff": result["j_eff"],
        "verification_scores": result["verification_scores"],
        "avg_verification_score": result["avg_verification_score"],
        "grounding_events_count": len(result["grounding_events"]),
        "grounding_cfi": result["grounding_events"][0].get("cfi") if result["grounding_events"] else None,
        "pruned_constraints": result["pruned_constraints"],
        # Thinking loop signals
        "thinking_loop_enabled": tl.get("enabled") if tl else None,
        "thinking_loop_iterations": tl.get("iterations_run") if tl else None,
        "thinking_loop_coverage": tl.get("coverage_score") if tl else None,
        "thinking_loop_understanding_len": tl.get("shared_understanding_len") if tl else None,
        # ρ EMA / calibration signals
        "prediction_basis_final": result.get("prediction_basis_final"),
        # Oracle p_mean patch
        "oracle_p_patched": ocp is not None,
        "oracle_pass_rate": ocp.get("oracle_pass_rate") if ocp else None,
        "oracle_p_mean_before": ocp.get("p_mean_before") if ocp else None,
        "oracle_p_mean_after": ocp.get("p_mean_after") if ocp else None,
        "hitl_gate_fired": result.get("hitl_gate_fired", False),
        # Epistemic leader signals
        "leader_elected": result.get("leader_elected", False),
        "leader_election_count": len(result.get("leader_elected_events", [])),
        "assertions": assertions,
        "pass": all(c["pass"] for c in assertions.values()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return out_dir


# ── Constraint context loader ─────────────────────────────────────────────────

def _load_constraint_context(scenario_dir: pathlib.Path, task: dict) -> str:
    """Load constraint definitions from the wiki corpus and format them for LLM context.

    Used by --context-augmented to give the bare LLM the same constraint knowledge
    that h2ai injects into explorer context via the constraint wiki.  This isolates
    whether h2ai's improvement comes from constraint knowledge or from orchestration.

    Returns a formatted string, empty string if no constraints found.
    """
    import re

    constraint_ids: list[str] = task.get("constraints", [])
    if not constraint_ids:
        return ""

    # Resolve corpus path from the scenario's h2ai.toml
    toml_path = scenario_dir / "h2ai.toml"
    if not toml_path.exists():
        return ""
    cfg = tomllib.loads(toml_path.read_text())
    wiki_cfg = cfg.get("constraint_wiki", {})
    corpus_path_str = wiki_cfg.get("corpus_path", "")
    if not corpus_path_str:
        return ""

    corpus_dir = REPO_ROOT / corpus_path_str

    blocks: list[str] = []
    for cid in constraint_ids:
        # Match e.g. CONSTRAINT-004-budget-pacing-idempotency.yaml
        matches = sorted(corpus_dir.glob(f"{cid}*.yaml"))
        if not matches:
            continue
        raw = matches[0].read_text()

        # Parse key fields with simple regex (avoids pyyaml dependency)
        def _field(key: str) -> str:
            m = re.search(rf'^{key}:\s*["\']?(.*?)["\']?\s*$', raw, re.MULTILINE)
            return m.group(1).strip() if m else ""

        def _block(key: str) -> str:
            # Find "  key: |" line and collect only the block content lines
            # (lines whose indentation exceeds the key's own indentation level)
            lines = raw.splitlines()
            key_indent = None
            content: list[str] = []
            in_block = False
            for line in lines:
                if not in_block:
                    stripped = line.lstrip()
                    if stripped.startswith(f"{key}:") and "|" in line:
                        key_indent = len(line) - len(stripped)
                        in_block = True
                else:
                    if line.strip() == "":
                        content.append("")
                        continue
                    line_indent = len(line) - len(line.lstrip())
                    if line_indent <= key_indent:
                        break  # back to sibling or parent level — stop
                    content.append(line)
            if not content:
                return ""
            min_indent = min((len(l) - len(l.lstrip()) for l in content if l.strip()), default=0)
            return "\n".join(l[min_indent:] for l in content).strip()

        title = _field("title")
        pass_criteria = _block("pass") or _field("pass")
        hint = _field("remediation_hint")

        block = f"### {cid}: {title}\n**Required:** {pass_criteria}"
        if hint:
            block += f"\n**Guidance:** {hint}"
        blocks.append(block)

    if not blocks:
        return ""

    return (
        "The following constraints MUST ALL be satisfied in your design. "
        "Violating any one of them makes the solution unacceptable:\n\n"
        + "\n\n".join(blocks)
    )


def _wait_for_llm_ready(scenario_name: str, timeout_s: int = 300) -> None:
    """Poll the LLM /models endpoint until it responds without 503, then warm it up.

    A 122B model reloads after being evicted between long runs. Without this wait,
    the second arm in --compare sees 503 on every call and produces zero proposals.
    """
    base_url = _llm_base_url_for_scenario(scenario_name)
    models_url = f"{base_url}/models"
    deadline = time.time() + timeout_s
    poll = 5
    print(f"  [llm-ready] polling {models_url} (timeout {timeout_s}s) …", flush=True)
    while time.time() < deadline:
        try:
            req = urllib.request.Request(models_url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    print(f"  [llm-ready] LLM ready", flush=True)
                    return
        except urllib.error.HTTPError as exc:
            if exc.code == 503:
                body = exc.read()[:120].decode(errors="replace")
                print(f"  [llm-ready] 503 ({body}) — waiting {poll}s …", flush=True)
            else:
                print(f"  [llm-ready] HTTP {exc.code} — waiting {poll}s …", flush=True)
        except Exception as exc:
            print(f"  [llm-ready] {exc} — waiting {poll}s …", flush=True)
        time.sleep(poll)
    print(f"  [llm-ready] WARNING: LLM did not become ready within {timeout_s}s; proceeding anyway")


def _llm_base_url_for_scenario(scenario_name: str, config_name: str = "h2ai.toml") -> str:
    """Return the base URL (without path) of the first adapter profile."""
    toml_path = SCENARIOS_DIR / scenario_name.split("/")[-1] / config_name
    if not toml_path.exists():
        toml_path = SCENARIOS_DIR / scenario_name / config_name
    if not toml_path.exists():
        toml_path = SCENARIOS_DIR / scenario_name.split("/")[-1] / "h2ai.toml"
    if not toml_path.exists():
        toml_path = SCENARIOS_DIR / scenario_name / "h2ai.toml"
    if toml_path.exists():
        cfg = tomllib.loads(toml_path.read_text())
        for profile in cfg.get("adapter_profiles", []):
            kind = profile.get("kind", {})
            for adapter_cfg in kind.values():
                if isinstance(adapter_cfg, dict) and "endpoint" in adapter_cfg:
                    return adapter_cfg["endpoint"].rstrip("/")
    return "http://host.docker.internal:8080/v1"


# ── Baseline mode (direct LLM, no H2AI) ──────────────────────────────────────

def _llm_endpoint_for_scenario(scenario_name: str, config_name: str = "h2ai.toml") -> tuple[str, str, int, str]:
    """Return (endpoint_url, model_name, max_tokens, api_key) for the scenario's first adapter profile."""
    toml_path = SCENARIOS_DIR / scenario_name.split("/")[-1] / config_name
    if not toml_path.exists():
        toml_path = SCENARIOS_DIR / scenario_name / config_name
    if not toml_path.exists():
        toml_path = SCENARIOS_DIR / scenario_name.split("/")[-1] / "h2ai.toml"
    if not toml_path.exists():
        toml_path = SCENARIOS_DIR / scenario_name / "h2ai.toml"
    if toml_path.exists():
        cfg = tomllib.loads(toml_path.read_text())
        max_tokens = int(cfg.get("model_max_tokens", MODEL_MAX_TOKENS))
        for profile in cfg.get("adapter_profiles", []):
            kind = profile.get("kind", {})
            for adapter_cfg in kind.values():
                if isinstance(adapter_cfg, dict) and "endpoint" in adapter_cfg:
                    base = adapter_cfg["endpoint"].rstrip("/")
                    model = adapter_cfg.get("model") or profile.get("name", "local")
                    api_key_env = adapter_cfg.get("api_key_env", "")
                    api_key = os.environ.get(api_key_env, "") if api_key_env else ""
                    return f"{base}/chat/completions", model, max_tokens, api_key

    return "http://host.docker.internal:8080/v1/chat/completions", "local", MODEL_MAX_TOKENS, ""


def _llm_call(endpoint: str, model: str, messages: list[dict], max_tokens: int = MODEL_MAX_TOKENS, api_key: str = "") -> str:
    payload = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.6}).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    delay = 5
    for attempt in range(10):
        req = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read())
                content = data["choices"][0]["message"].get("content") or ""
                return content
        except urllib.error.HTTPError as exc:
            body = exc.read()[:300]
            if exc.code == 429:
                print(f"  [rate-limited 429, retry {attempt+1}/10 in {delay}s]")
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            raise RuntimeError(f"HTTP {exc.code} from {endpoint}: {body}") from exc
        except (ConnectionRefusedError, urllib.error.URLError) as exc:
            raise RuntimeError(
                f"LLM endpoint unreachable: {endpoint}\n"
                f"  Make sure your LLM server is running, or set:\n"
                f"  H2AI_LLM_ENDPOINT=http://<host>:<port>/v1/chat/completions"
            ) from None
    raise RuntimeError(f"Rate limit exceeded after 10 retries: {endpoint}")


def _print_checks_for_review(checks: list[dict], out_dir: pathlib.Path) -> None:
    """Print check questions to stdout so a human reviewer knows what to evaluate."""
    if not checks:
        return
    print(f"  review output at: {out_dir / 'output.txt'}")
    print(f"  evaluate manually against {len(checks)} check(s):")
    for check in checks:
        print(f"    [{check['id']}] {check['text']}")


def run_baseline(scenario_name: str, task: dict, constraint_context: str = "", config_name: str = "h2ai.toml") -> dict:
    endpoint, model, max_tokens, api_key = _llm_endpoint_for_scenario(scenario_name, config_name)
    mode_label = "context-augmented" if constraint_context else "bare LLM"
    print(f"  LLM endpoint: {endpoint}  model: {model}  max_tokens: {max_tokens}  mode: {mode_label}")

    system_parts = ["You are a senior distributed systems engineer. Be concrete and precise."]
    if constraint_context:
        system_parts.append(constraint_context)
    system_prompt = "\n\n".join(system_parts)

    user_content = task["description"]
    if task.get("context"):
        user_content += "\n\nContext: " + task["context"]

    print(f"  Generating {mode_label} answer via LLM…")
    if constraint_context:
        print(f"  Constraints injected: {len(task.get('constraints', []))} definitions ({len(constraint_context)} chars)")
    t0 = time.time()
    answer = _llm_call(endpoint, model, [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ], max_tokens=max_tokens, api_key=api_key)
    elapsed = time.time() - t0
    print(f"  Answer: {len(answer)} chars in {elapsed:.0f}s")

    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = RESULTS_DIR / scenario_name / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "output.txt").write_text(answer)
    summary = {
        "scenario": scenario_name,
        "timestamp": ts,
        "mode": "context-augmented" if constraint_context else "baseline",
        "constraints_injected": task.get("constraints", []) if constraint_context else [],
        "llm_endpoint": endpoint,
        "answer_chars": len(answer),
        "elapsed_s": round(elapsed, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    checks = task.get("_expected", {}).get("_review_checklist", [])
    _print_checks_for_review(checks, out_dir)
    return summary


# ── Entry point ───────────────────────────────────────────────────────────────

def _run_h2ai_trials(
    scenario_name: str,
    scenario_dir: pathlib.Path,
    task: dict,
    config_file: str,
    trials: int,
) -> dict:
    """Run h2ai for `trials` iterations, return aggregated metrics."""
    results = []
    for trial in range(1, trials + 1):
        if trials > 1:
            print(f"  ── trial {trial}/{trials} ──")
        proc = None
        try:
            proc = start_server(scenario_dir, config_file)
            wait_for_health()
            print("  server ready")
            required_n = task.get("explorers", {}).get("count", 1)
            trigger_calibration_and_wait(min_n_max=required_n, timeout_s=900)
            result = run_scenario(scenario_name, task)
            assertions = check_assertions(result, task.get("_expected", {}), task)
            out_dir = save_results(scenario_name, task, result, assertions)
            checks = task.get("_expected", {}).get("_review_checklist", [])
            _print_checks_for_review(checks, out_dir)
            passed = all(c["pass"] for c in assertions.values())
            results.append({"passed": passed, "result": result, "assertions": assertions, "out_dir": str(out_dir)})
            mark = "PASS" if passed else "FAIL"
            failed = [k for k, v in assertions.items() if not v["pass"]]
            print(f"  → {mark}" + (f"  failed: {', '.join(failed)}" if failed else ""))
        except Exception as e:
            results.append({"passed": False, "error": str(e)})
            print(f"  → ERROR: {e}")
            traceback.print_exc()
            if proc:
                log_path = getattr(proc, "_log_path", None)
                if log_path:
                    issues = _scan_server_log_issues(log_path)
                    if issues:
                        print(f"  ── server log issues ({len(issues)} lines) ─────────────────────────────────")
                        for line in issues[-30:]:
                            print(f"  {line}")
                        print()
                    _print_server_log_tail(log_path, n=30)
        finally:
            if proc:
                stop_server(proc)
                print("  server stopped")

    passing = sum(1 for r in results if r["passed"])
    pass_k = passing / trials
    last = next((r for r in reversed(results) if "result" in r), {})
    last_result = last.get("result", {})
    merged = last_result.get("merged_output", "")
    return {
        "trials": trials,
        "passing": passing,
        "pass_k": pass_k,
        "j_eff": last_result.get("j_eff"),
        "avg_verification_score": last_result.get("avg_verification_score", 0.0),
        "valid_proposals": len(last_result.get("verification_scores", [])),
        "thinking_loop_iters": (last_result.get("thinking_loop_event") or {}).get("iterations_run"),
        "hitl_fired": last_result.get("hitl_gate_fired", False),
        "leader_elected": last_result.get("leader_elected", False),
        "leader_election_count": len(last_result.get("leader_elected_events", [])),
        "grounding_events": len(last_result.get("grounding_events", [])),
        "answer_chars": len(merged) if merged else None,
        "elapsed_s": None,
    }


def _baseline_summary(result: dict) -> dict:
    """Wrap run_baseline() result in the same shape as _run_h2ai_trials() metrics."""
    return {
        "trials": 1,
        "passing": None,
        "pass_k": None,
        "j_eff": None,
        "avg_verification_score": None,
        "valid_proposals": 0,
        "thinking_loop_iters": None,
        "hitl_fired": False,
        "leader_elected": None,
        "leader_election_count": 0,
        "grounding_events": 0,
        "answer_chars": result.get("answer_chars"),
        "elapsed_s": result.get("elapsed_s"),
    }


def _print_triple_table(
    h2ai_metrics: dict,
    llm_metrics: dict,
    rag_metrics: dict,
) -> None:
    def fmt(v):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.3f}"
        if isinstance(v, bool):
            return "yes" if v else "no"
        return str(v)

    def delta(a, b):
        if a is None or b is None:
            return "—"
        if isinstance(a, float) and isinstance(b, float):
            d = a - b
            return f"{'+' if d >= 0 else ''}{d:.3f}"
        if isinstance(a, bool) and isinstance(b, bool):
            return "✓" if a and not b else ("✗" if not a and b else "=")
        if isinstance(a, int) and isinstance(b, int):
            d = a - b
            return f"{'+' if d >= 0 else ''}{d}"
        return "—"

    rows = [
        ("output_chars",   llm_metrics.get("answer_chars"),       rag_metrics.get("answer_chars"),       h2ai_metrics.get("answer_chars")),
        ("elapsed_s",      llm_metrics.get("elapsed_s"),          rag_metrics.get("elapsed_s"),          h2ai_metrics.get("elapsed_s")),
        ("avg_verif_score",None,                                   None,                                  h2ai_metrics["avg_verification_score"]),
        ("j_eff",          None,                                   None,                                  h2ai_metrics["j_eff"]),
        ("thinking_iters", None,                                   None,                                  h2ai_metrics["thinking_loop_iters"]),
        ("grounding_events", llm_metrics["grounding_events"],     rag_metrics["grounding_events"],       h2ai_metrics["grounding_events"]),
        ("leader_elected", None,                                   None,                                  h2ai_metrics.get("leader_elected")),
        ("hitl_fired",     llm_metrics["hitl_fired"],             rag_metrics["hitl_fired"],             h2ai_metrics["hitl_fired"]),
    ]
    col_w = 14
    sep = "─" * (col_w * 4 + 14)
    print(f"\n{sep}")
    print(f"  {'Metric':<{col_w}}  {'bare LLM':>{col_w}}  {'LLM+RAG':>{col_w}}  {'H2AI':>{col_w}}  {'Δ(H2AI-RAG)':>{col_w}}")
    print(sep)
    for name, llm_v, rag_v, h2ai_v in rows:
        print(
            f"  {name:<{col_w}}  {fmt(llm_v):>{col_w}}  {fmt(rag_v):>{col_w}}"
            f"  {fmt(h2ai_v):>{col_w}}  {delta(h2ai_v, rag_v):>{col_w}}"
        )
    print(f"{sep}\n")


def _print_delta_table(h2ai_metrics: dict, baseline_metrics: dict) -> None:
    def fmt(v):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.3f}"
        if isinstance(v, bool):
            return "yes" if v else "no"
        return str(v)

    def delta(a, b):
        if a is None or b is None:
            return "—"
        if isinstance(a, float) and isinstance(b, float):
            d = a - b
            return f"{'+' if d >= 0 else ''}{d:.3f}"
        if isinstance(a, bool) and isinstance(b, bool):
            return "✓" if a and not b else ("✗" if not a and b else "=")
        if isinstance(a, int) and isinstance(b, int):
            d = a - b
            return f"{'+' if d >= 0 else ''}{d}"
        return "—"

    rows = [
        ("pass^k",          h2ai_metrics["pass_k"],                baseline_metrics["pass_k"]),
        ("j_eff",           h2ai_metrics["j_eff"],                 baseline_metrics["j_eff"]),
        ("avg_verif_score", h2ai_metrics["avg_verification_score"],None),
        ("valid_proposals", h2ai_metrics["valid_proposals"],       baseline_metrics["valid_proposals"]),
        ("thinking_iters",  h2ai_metrics["thinking_loop_iters"],   baseline_metrics["thinking_loop_iters"]),
        ("hitl_fired",      h2ai_metrics["hitl_fired"],            baseline_metrics["hitl_fired"]),
        ("leader_elected",  h2ai_metrics.get("leader_elected"),    baseline_metrics.get("leader_elected")),
        ("grounding_events", h2ai_metrics["grounding_events"],     baseline_metrics["grounding_events"]),
    ]
    col_w = 18
    print(f"\n{'─'*72}")
    print(f"  {'Metric':<{col_w}}  {'baseline':>{col_w}}  {'h2ai':>{col_w}}  {'delta':>{col_w}}")
    print(f"{'─'*72}")
    for name, h2ai_val, base_val in rows:
        print(f"  {name:<{col_w}}  {fmt(base_val):>{col_w}}  {fmt(h2ai_val):>{col_w}}  {delta(h2ai_val, base_val):>{col_w}}")
    print(f"{'─'*72}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay H2AI scenarios, capture results for regression analysis.")
    parser.add_argument("tasks", nargs="*", metavar="SCENARIO",
                        help="scenario name(s) (e.g. benchmark, features/01-thinking-loop); default: all")
    parser.add_argument("--list", action="store_true", help="list available scenarios and exit")
    parser.add_argument("--baseline", action="store_true",
                        help="direct LLM mode — call LLM without H2AI, score against _expected.checks")
    parser.add_argument("--context-augmented", action="store_true", dest="context_augmented",
                        help="bare LLM + constraint definitions injected into system prompt; isolates framework value from constraint knowledge")
    parser.add_argument("--config", default="h2ai.toml", metavar="FILE",
                        help="toml config filename to load from scenario dir (default: h2ai.toml)")
    parser.add_argument("--trials", type=int, default=1, metavar="K",
                        help="run each scenario K times and report pass^k (default: 1)")
    parser.add_argument("--compare", action="store_true",
                        help="run h2ai.toml then baseline.toml and print delta table")
    parser.add_argument("--triple", action="store_true",
                        help="3-way comparison: bare LLM vs LLM+constraints (RAG) vs H2AI full")
    args = parser.parse_args()

    if args.list:
        for path in sorted(SCENARIOS_DIR.glob("**/task.json")):
            t = json.loads(path.read_text())
            rel = path.parent.relative_to(SCENARIOS_DIR)
            n_checks = len(t.get("_expected", {}).get("_review_checklist", []))
            has_baseline = (path.parent / "baseline.toml").exists()
            print(f"  {str(rel):<45}  checks={n_checks}  {'[baseline.toml]' if has_baseline else ''}")
        return

    scenarios = load_scenarios(args.tasks or None)
    overall: dict[str, str] = {}

    if args.baseline or args.context_augmented:
        for scenario_name, scenario_dir, task in scenarios:
            mode = "CONTEXT-AUGMENTED" if args.context_augmented else "BASELINE (bare LLM)"
            print(f"{'='*60}")
            print(f"{mode}: {scenario_name}")
            print(f"{'='*60}")
            try:
                constraint_ctx = _load_constraint_context(scenario_dir, task) if args.context_augmented else ""
                run_baseline(scenario_name, task, constraint_context=constraint_ctx, config_name=args.config or "h2ai.toml")
                overall[scenario_name] = "DONE"
            except Exception as e:
                overall[scenario_name] = f"ERROR: {e}"
                print(f"  → ERROR: {e}")
                traceback.print_exc()
            print()

    elif args.compare:
        for scenario_name, scenario_dir, task in scenarios:
            print(f"{'='*60}")
            print(f"COMPARE: {scenario_name}")
            print(f"{'='*60}")
            try:
                print(f"\n[h2ai.toml — framework]")
                _wait_for_llm_ready(scenario_name)
                h2ai_m = _run_h2ai_trials(scenario_name, scenario_dir, task, "h2ai.toml", args.trials)
                print(f"\n[baseline.toml — feature OFF]")
                _wait_for_llm_ready(scenario_name)
                base_m = _run_h2ai_trials(scenario_name, scenario_dir, task, "baseline.toml", args.trials)
                _print_delta_table(h2ai_m, base_m)
                compare_semantics = task.get("_compare_semantics", "strict")
                if compare_semantics == "no_regression":
                    # Feature adds safety/correctness without measurable output uplift.
                    # Pass when h2ai passes majority of trials AND is at least as good as baseline.
                    verdict = "PASS" if h2ai_m["pass_k"] >= 0.5 and h2ai_m["pass_k"] >= base_m["pass_k"] else "FAIL/WORSE"
                else:
                    verdict = "PASS" if h2ai_m["pass_k"] > base_m["pass_k"] else "SAME/WORSE"
                overall[scenario_name] = verdict
            except Exception as e:
                overall[scenario_name] = f"ERROR: {e}"
                print(f"  → ERROR: {e}")
                traceback.print_exc()
            print()

    elif args.triple:
        for scenario_name, scenario_dir, task in scenarios:
            print(f"{'='*60}")
            print(f"TRIPLE: {scenario_name}")
            print(f"  bare LLM  →  LLM+RAG (constraints injected)  →  H2AI full")
            print(f"{'='*60}")
            try:
                print(f"\n[1/3 — bare LLM]")
                llm_result = run_baseline(scenario_name, task, constraint_context="", config_name=args.config or "h2ai.toml")
                llm_m = _baseline_summary(llm_result)

                print(f"\n[2/3 — LLM+RAG (constraints injected into system prompt)]")
                constraint_ctx = _load_constraint_context(scenario_dir, task)
                rag_result = run_baseline(scenario_name, task, constraint_context=constraint_ctx, config_name=args.config or "h2ai.toml")
                rag_m = _baseline_summary(rag_result)

                h2ai_cfg = args.config or "h2ai.toml"
                print(f"\n[3/3 — H2AI full ({h2ai_cfg})]")
                h2ai_m = _run_h2ai_trials(scenario_name, scenario_dir, task, h2ai_cfg, args.trials)

                _print_triple_table(h2ai_m, llm_m, rag_m)

                # Triple mode: H2AI must pass majority of trials (≥50%).
                # Content quality is left for human review of output.txt files.
                verdict = "PASS" if h2ai_m["pass_k"] >= 0.5 else "FAIL"
                overall[scenario_name] = verdict
            except Exception as e:
                overall[scenario_name] = f"ERROR: {e}"
                print(f"  → ERROR: {e}")
                traceback.print_exc()
            print()

    else:
        for scenario_name, scenario_dir, task in scenarios:
            print(f"{'='*60}")
            print(f"SCENARIO: {scenario_name}  config={args.config}  trials={args.trials}")
            print(f"{'='*60}")
            try:
                metrics = _run_h2ai_trials(scenario_name, scenario_dir, task, args.config, args.trials)
                if args.trials > 1:
                    print(f"  pass^{args.trials} = {metrics['passing']}/{args.trials} = {metrics['pass_k']:.2f}")
                overall[scenario_name] = "PASS" if metrics["pass_k"] >= 1.0 / args.trials else "FAIL"
            except Exception as e:
                overall[scenario_name] = f"ERROR: {e}"
                print(f"  → ERROR: {e}")
                traceback.print_exc()
            print()

    print(f"{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    failed = 0
    for name, verdict in overall.items():
        mark = "✓" if verdict == "PASS" else "✗"
        print(f"  {mark} {name}: {verdict}")
        if verdict != "PASS":
            failed += 1
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
