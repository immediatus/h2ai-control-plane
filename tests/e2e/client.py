"""Shared HTTP client for H2AI e2e tests."""

import json
import time
import urllib.request
import urllib.error
from typing import Iterator

BASE_URL = "http://localhost:8080"
API_PREFIX = "/v1"
DEFAULT_TENANT = "default"


def _task_url(tenant_id: str, *parts: str) -> str:
    path = f"{API_PREFIX}/{tenant_id}/tasks"
    if parts:
        path += "/" + "/".join(parts)
    return f"{BASE_URL}{path}"


def submit_task(task: dict, tenant_id: str = DEFAULT_TENANT) -> str:
    payload = json.dumps(task).encode()
    req = urllib.request.Request(
        _task_url(tenant_id),
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        err_body = e.read().decode(errors="replace")
        raise urllib.error.HTTPError(e.url, e.code, f"{e.msg} — {err_body}", e.headers, None) from None
    task_id = body.get("task_id")
    if not task_id:
        raise RuntimeError(f"no task_id in response: {body}")
    return task_id


def stream_events(task_id: str, tenant_id: str = DEFAULT_TENANT, timeout_s: int = 300) -> Iterator[dict]:
    """Stream SSE events for a task.

    timeout_s is a per-read socket timeout: if no bytes arrive (including SSE
    keepalive comment lines) for this many seconds, the connection is presumed
    dead and a socket.timeout exception propagates to the caller.  Wall-clock
    deadline management (e.g. resetting after ThinkingLoopCompleted) is the
    caller's responsibility so that pre-wave phases do not eat into the budget.
    """
    url = _task_url(tenant_id, task_id, "events")
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload:
                ev = json.loads(payload)
                # H2AIEvent serializes as {event_type: "...", payload: {...}}
                # Normalize to a flat dict with a "kind" key for test assertions.
                if "event_type" in ev and "payload" in ev:
                    flat = {"kind": ev["event_type"]}
                    flat.update(ev["payload"])
                    yield flat
                else:
                    yield ev


def submit_signal(
    task_id: str,
    payload: dict,
    tenant_id: str = DEFAULT_TENANT,
    timeout_ms: int | None = None,
) -> dict:
    """POST /signal — returns parsed response body.

    payload must be a SignalPayload dict, e.g.:
      {"kind": "Approve", "data": {"approved": True, "reviewer_note": "...", "operator_id": "..."}}
      {"kind": "WaveContinue", "data": {"grounding": "...", "mandate_override": None}}
    """
    body: dict = {"payload": payload}
    if timeout_ms is not None:
        body["timeout_ms"] = timeout_ms
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        _task_url(tenant_id, task_id, "signal"),
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def wait_for_health(timeout_s: int = 120) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError(f"server not healthy after {timeout_s}s")


def trigger_calibration_and_wait(min_n_max: int = 1, timeout_s: int = 300) -> float:
    """POST /v1/calibrate to start a fresh calibration, then poll until a fresh
    calibration with n_max >= min_n_max is current.

    Strategy: record the calibration_id that was current BEFORE posting, then wait
    until the current_id changes (any new calibration is acceptable — both the
    startup calibration and our triggered one use the fixed n_eff_cosine_prior code).
    This avoids the race where the startup calibration completes first and blocks
    the triggered_id from ever becoming current within the timeout.
    """
    # Snapshot the stale calibration_id before triggering.
    stale_id: str | None = None
    try:
        with urllib.request.urlopen(f"{BASE_URL}/v1/calibrate/current", timeout=10) as resp:
            stale_id = json.loads(resp.read()).get("calibration_id")
    except Exception:
        pass  # no current calibration yet

    req = urllib.request.Request(
        f"{BASE_URL}/v1/calibrate",
        data=b"",
        method="POST",
    )
    triggered_id: str | None = None
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
        triggered_id = body.get("calibration_id")
        print(f"  calibration triggered: id={triggered_id or '?'}  adapters={body.get('adapter_count')}")
    except Exception as e:
        print(f"  calibration trigger warning (proceeding anyway): {e}")

    deadline = time.time() + timeout_s
    last_n_max: float | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE_URL}/v1/calibrate/current", timeout=10) as resp:
                data = json.loads(resp.read())
            n_max = float(data.get("n_max", 0))
            current_id = data.get("calibration_id")
            if n_max != last_n_max:
                print(f"  calibration: n_max={n_max:.0f}  (need ≥{min_n_max})")
                last_n_max = n_max
            # Accept any calibration that is newer than the one we saw before POSTing.
            # This handles both: (a) triggered_id becoming current, and (b) the startup
            # calibration completing first (which is also fresh and uses fixed code).
            is_fresh = current_id != stale_id or stale_id is None
            if is_fresh and n_max >= min_n_max:
                return n_max
        except urllib.error.HTTPError:
            pass  # 503 CalibrationRequired — not ready yet
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError(f"calibration did not reach n_max≥{min_n_max} within {timeout_s}s (last={last_n_max})")
