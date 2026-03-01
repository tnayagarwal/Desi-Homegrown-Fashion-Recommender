"""
Unit tests for the metrics tracker decorator and summary function.
All tests use stdlib only — no external dependencies required.
"""
import time
from src.monitoring.metrics import track, get_summary, increment, record_latency, _counters, _latencies


def reset():
    _counters["request_count"] = 0
    _counters["error_count"] = 0
    _latencies.clear()


def test_track_increments_request_count():
    reset()
    @track
    def noop(): return True
    noop()
    assert _counters["request_count"] == 1


def test_track_increments_error_count_on_exception():
    reset()
    @track
    def boom(): raise ValueError("test error")
    try: boom()
    except ValueError: pass
    assert _counters["error_count"] == 1
    assert _counters["request_count"] == 1


def test_track_records_latency():
    reset()
    @track
    def slow(): time.sleep(0.01)
    slow()
    assert len(_latencies) == 1
    assert _latencies[0] >= 0.01


def test_summary_error_rate():
    reset()
    _counters["request_count"] = 10
    _counters["error_count"] = 2
    summary = get_summary()
    assert summary["error_rate"] == 0.2


def test_summary_latency_percentiles():
    reset()
    for ms in range(1, 101):  # 1ms to 100ms
        record_latency(ms / 1000.0)
    summary = get_summary()
    assert summary["latency_p50_ms"] > 0
    assert summary["latency_p99_ms"] >= summary["latency_p50_ms"]
