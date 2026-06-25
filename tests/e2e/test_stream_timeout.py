"""Unit tests for stream_events deadline semantics.

Bug: stream_events uses deadline = time.time() + timeout_s checked after every
SSE line (including keepalives). For tasks with long pre-wave phases (thinking
loop), the deadline fires before the wave completes, causing false FAIL results.

Fix: remove wall-clock deadline from stream_events (keep socket-level timeout
only). Let run_scenario manage the wall-clock deadline and reset it after
ThinkingLoopCompleted so the wave always gets the full timeout_s budget.
"""

import json
import pathlib
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from client import stream_events


# ── SSE helpers ───────────────────────────────────────────────────────────────

def _sse_keepalive() -> bytes:
    return b": keepalive\n"


def _sse_event(kind: str, **payload) -> bytes:
    ev = {"event_type": kind, "payload": payload}
    return f"data: {json.dumps(ev)}\n".encode()


class _FakeSSEResponse:
    """Synchronous fake SSE HTTP response that yields pre-built lines."""

    def __init__(self, lines: list[bytes]):
        self._lines = iter(lines)

    def __enter__(self) -> "_FakeSSEResponse":
        return self

    def __exit__(self, *args) -> None:
        pass

    def __iter__(self):
        yield from self._lines


# ── tests ─────────────────────────────────────────────────────────────────────

class TestStreamEventsDropsEventAfterDeadline(unittest.TestCase):
    """FAILING before fix: deadline in stream_events fires after timeout_s
    from the call start, dropping events that arrive after the deadline even
    when the server is still sending keepalives.

    Fake time advances by 1 second per time.time() call:
      call 1 → 1000.0  (deadline = 1000 + 2 = 1002)
      call 2 → 1001.0  (check after keepalive 1: 1001 > 1002? No)
      call 3 → 1002.0  (check after keepalive 2: 1002 > 1002? No — strictly >)
      call 4 → 1003.0  (check after data line:   1003 > 1002? YES → break — BUG)
    """

    def _fake_time_sequence(self):
        """Returns successive fake timestamps, 1 second apart, starting at 1000."""
        t = 1000.0
        while True:
            yield t
            t += 1.0

    def test_late_event_is_dropped_before_fix(self):
        """The MergeResolved event arrives after the deadline — currently dropped."""
        fake_time = self._fake_time_sequence()
        lines = [
            _sse_keepalive(),
            _sse_keepalive(),
            _sse_event("MergeResolved", task_id="t1", resolved_output="ok"),
        ]
        fake_resp = _FakeSSEResponse(lines)

        with patch("client.time") as mock_time, \
             patch("urllib.request.urlopen", return_value=fake_resp):
            mock_time.time.side_effect = lambda: next(fake_time)

            events = list(stream_events("task-1", timeout_s=2))

        # After fix: MergeResolved IS received (deadline removed from stream_events).
        # Before fix: MergeResolved is dropped (deadline fires, loop breaks).
        self.assertEqual(len(events), 1, "MergeResolved should not be dropped by the stream_events deadline")
        self.assertEqual(events[0]["kind"], "MergeResolved")

    def test_event_before_deadline_is_always_received(self):
        """Events that arrive before the deadline are received both before and after fix."""
        fake_time = self._fake_time_sequence()
        lines = [
            _sse_event("ThinkingLoopCompleted", task_id="t1", iterations_run=3),
            _sse_event("MergeResolved", task_id="t1", resolved_output="ok"),
        ]
        fake_resp = _FakeSSEResponse(lines)

        with patch("client.time") as mock_time, \
             patch("urllib.request.urlopen", return_value=fake_resp):
            mock_time.time.side_effect = lambda: next(fake_time)

            events = list(stream_events("task-1", timeout_s=60))

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["kind"], "ThinkingLoopCompleted")
        self.assertEqual(events[1]["kind"], "MergeResolved")


class TestStreamEventsSocketTimeout(unittest.TestCase):
    """After fix: stream_events uses urlopen(timeout=timeout_s) as a socket
    timeout only. When the server responds within the socket timeout, all
    events are delivered."""

    def test_all_events_delivered_when_server_responds_fast(self):
        """All events are yielded when no deadline interrupts the stream."""
        lines = [
            _sse_keepalive(),
            _sse_event("ThinkingLoopCompleted", task_id="t1", iterations_run=2),
            _sse_keepalive(),
            _sse_event("ComplexityProbe", task_id="t1", complexity=3),
            _sse_event("MergeResolved", task_id="t1", resolved_output="final"),
        ]
        fake_resp = _FakeSSEResponse(lines)

        with patch("urllib.request.urlopen", return_value=fake_resp):
            events = list(stream_events("task-1", timeout_s=300))

        kinds = [e["kind"] for e in events]
        self.assertEqual(kinds, ["ThinkingLoopCompleted", "ComplexityProbe", "MergeResolved"])


if __name__ == "__main__":
    unittest.main()
