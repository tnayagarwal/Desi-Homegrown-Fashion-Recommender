"""
Prometheus-Compatible Metrics Tracker
=======================================
Lightweight in-memory metrics store compatible with the Prometheus
data model. Replace the counter/histogram classes with
`prometheus_client` equivalents for production scraping.

Usage:
    from src.monitoring.metrics import track, get_summary

    @track
    def my_function():
        ...

    summary = get_summary()
"""
import time
import logging
from functools import wraps
from typing import Callable, Any

logger = logging.getLogger(__name__)

_counters: dict = {"request_count": 0, "error_count": 0}
_latencies: list = []


def increment(metric: str, value: int = 1) -> None:
    """Increment a named counter."""
    if metric in _counters:
        _counters[metric] += value


def record_latency(seconds: float) -> None:
    """Append a latency observation, capped at 10k samples."""
    _latencies.append(seconds)
    if len(_latencies) > 10_000:
        del _latencies[:5_000]


def get_summary() -> dict:
    """Return a snapshot of all current metrics."""
    lats = sorted(_latencies)
    n = len(lats)
    return {
        "request_count": _counters["request_count"],
        "error_count": _counters["error_count"],
        "error_rate": round(_counters["error_count"] / max(_counters["request_count"], 1), 4),
        "latency_p50_ms": round(lats[n // 2] * 1000, 2) if n else 0,
        "latency_p99_ms": round(lats[int(n * 0.99)] * 1000, 2) if n else 0,
        "latency_mean_ms": round(sum(lats) / max(n, 1) * 1000, 2),
        "sample_count": n,
    }


def track(fn: Callable) -> Callable:
    """
    Decorator: automatically track request count, errors, and latency for any function.

    Example:
        @track
        def process(data):
            return heavy_computation(data)
    """
    @wraps(fn)
    def wrapper(*args, **kwargs) -> Any:
        increment("request_count")
        start = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
            return result
        except Exception as e:
            increment("error_count")
            logger.error("[metrics] Error in %s: %s", fn.__name__, e)
            raise
        finally:
            record_latency(time.perf_counter() - start)
    return wrapper
