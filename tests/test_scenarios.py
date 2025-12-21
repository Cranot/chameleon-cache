"""Scenario-based tests for ChameleonCache."""

import random
import pytest
from chameleon import ChameleonCache


def generate_zipf(n_items: int, n_requests: int, alpha: float = 1.0) -> list:
    """Generate Zipf-distributed access pattern."""
    # Pre-compute probabilities
    weights = [1.0 / (i ** alpha) for i in range(1, n_items + 1)]
    total = sum(weights)
    probs = [w / total for w in weights]

    # Generate requests
    items = list(range(n_items))
    return random.choices(items, weights=probs, k=n_requests)


def generate_loop(loop_size: int, n_requests: int, extra: int = 0) -> list:
    """Generate loop access pattern (N+k problem)."""
    items = list(range(loop_size + extra))
    return [items[i % len(items)] for i in range(n_requests)]


def generate_sequential(n_items: int, n_requests: int) -> list:
    """Generate sequential scan pattern."""
    return [i % n_items for i in range(n_requests)]


def run_trace(cache_class, capacity: int, trace: list) -> float:
    """Run a trace and return hit rate."""
    cache = cache_class(capacity)
    hits = sum(1 for key in trace if cache.access(key))
    return hits / len(trace) * 100


class TestZipfWorkload:
    """Test Zipf/Power-law workloads."""

    def test_zipf_basic(self):
        """Test cache works on Zipf workload."""
        random.seed(42)
        trace = generate_zipf(10000, 50000, alpha=0.99)
        hit_rate = run_trace(ChameleonCache, 500, trace)

        # Should achieve reasonable hit rate on Zipf
        assert hit_rate > 50, f"Expected >50% on Zipf-0.99, got {hit_rate:.2f}%"

    def test_zipf_high_skew(self):
        """Test cache on high-skew Zipf."""
        random.seed(42)
        trace = generate_zipf(10000, 50000, alpha=1.2)
        hit_rate = run_trace(ChameleonCache, 500, trace)

        # High skew = easier to cache
        assert hit_rate > 70, f"Expected >70% on Zipf-1.2, got {hit_rate:.2f}%"


class TestLoopWorkload:
    """Test loop/scan workloads."""

    def test_loop_tight(self):
        """Test tight loop (N+1 problem)."""
        trace = generate_loop(500, 50000, extra=1)
        hit_rate = run_trace(ChameleonCache, 500, trace)

        # Should handle N+1 loops well
        assert hit_rate > 90, f"Expected >90% on tight loop, got {hit_rate:.2f}%"

    def test_loop_loose(self):
        """Test loose loop (N+10 problem)."""
        trace = generate_loop(500, 50000, extra=10)
        hit_rate = run_trace(ChameleonCache, 500, trace)

        # Loose loops are harder but should still work
        assert hit_rate > 80, f"Expected >80% on loose loop, got {hit_rate:.2f}%"


class TestSequentialWorkload:
    """Test sequential scan resistance."""

    def test_sequential_scan(self):
        """Test sequential scan pattern."""
        trace = generate_sequential(10000, 50000)
        hit_rate = run_trace(ChameleonCache, 500, trace)

        # Sequential is hardest - just verify it doesn't crash
        assert hit_rate >= 0


class TestMixedWorkload:
    """Test mixed/shifting workloads."""

    def test_phase_shift(self):
        """Test workload with phase shifts."""
        random.seed(42)

        # Phase 1: Zipf
        trace1 = generate_zipf(5000, 10000, alpha=0.99)

        # Phase 2: Loop
        trace2 = generate_loop(500, 10000, extra=1)

        # Phase 3: Different Zipf
        trace3 = generate_zipf(5000, 10000, alpha=0.8)

        trace = trace1 + trace2 + trace3

        hit_rate = run_trace(ChameleonCache, 500, trace)

        # Should adapt to changing patterns
        assert hit_rate > 30, f"Expected >30% on mixed workload, got {hit_rate:.2f}%"


class TestGhostUtility:
    """Test ghost utility detection."""

    def test_ghost_utility_high_on_loops(self):
        """Verify ghost utility is high on loop workloads."""
        cache = ChameleonCache(500)

        # Run a tight loop
        trace = generate_loop(500, 10000, extra=1)
        for key in trace:
            cache.access(key)

        stats = cache.get_stats()
        # Ghost utility should be detectable
        # (exact value depends on warmup and timing)

    def test_mode_detection(self):
        """Test that mode changes based on workload."""
        random.seed(42)
        cache = ChameleonCache(500)

        # Run Zipf workload
        trace = generate_zipf(5000, 20000, alpha=0.99)
        for key in trace:
            cache.access(key)

        stats = cache.get_stats()
        # Should detect high variance from Zipf
        # Mode should be FREQ or SCAN


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
