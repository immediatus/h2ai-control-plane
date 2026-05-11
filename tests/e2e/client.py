"""Shared HTTP client for H2AI e2e tests."""

import json
import time
import urllib.request
import urllib.error
from typing import Iterator

BASE_URL = "http://localhost:8080"
API_PREFIX = "/v1"


def submit_task(task: dict) -> str:
    payload = json.dumps(task).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{API_PREFIX}/tasks",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read())
    task_id = body.get("task_id")
    if not task_id:
        raise RuntimeError(f"no task_id in response: {body}")
    return task_id


def stream_events(task_id: str, timeout_s: int = 300) -> Iterator[dict]:
    url = f"{BASE_URL}{API_PREFIX}/tasks/{task_id}/events"
    req = urllib.request.Request(url)
    deadline = time.time() + timeout_s
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        for raw in resp:
            if time.time() > deadline:
                break
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
