#!/usr/bin/env python3
"""
VERIFICATION: Does skip-decay overfit?
======================================

Test the skip-decay approach on MULTIPLE workload types to ensure
we haven't overfit to the Corda->Loop->Corda pattern.

Workloads to test:
1. Zipf (power law) - frequency matters
2. Loop patterns - recency matters
3. Temporal (shifting) - adaptability matters
4. Sequential scan - pollution resistance matters
"""

import random
from collections import OrderedDict

CACHE_SIZE = 1000


class BaselineTinyLFU:
    """Standard TinyLFU with normal decay."""

    def __init__(self, cap):
        self.cap = cap
        self.win_cap = max(1, int(cap * 0.01))
        self.main_cap = cap - self.win_cap
        self.ghost_cap = cap * 2

        self.window = OrderedDict()
        self.main = OrderedDict()
        self.freq = {}
        self.ghost = OrderedDict()

        self.ops = 0
        self.decay_at = cap * 10

    def access(self, k):
        self.ops += 1
        self.freq[k] = min(15, self.freq.get(k, 0) + 1)

        if k in self.window:
            self.window.move_to_end(k)
            return True

        if k in self.main:
            self.main.move_to_end(k)
            return True

        ghost_data = self.ghost.pop(k, None)
        if ghost_data is not None:
            ghost_freq, _ = ghost_data
            self.freq[k] = min(15, self.freq[k] + ghost_freq)

        self._add_to_window(k)

        if self.ops >= self.decay_at:
            self._decay()

        return False

    def _add_to_window(self, k):
        if len(self.window) >= self.win_cap:
            evicted, _ = self.window.popitem(last=False)
            self._try_promote(evicted)
        self.window[k] = 1

    def _try_promote(self, k):
        k_freq = self.freq.get(k, 0)
        if len(self.main) < self.main_cap:
            self.main[k] = 1
            return
        if self.main_cap == 0:
            self._add_ghost(k)
            return
        victim, _ = self.main.popitem(last=False)
        v_freq = self.freq.get(victim, 0)
        if k_freq > v_freq:
            self._add_ghost(victim)
            self.main[k] = 1
        else:
            self.main[victim] = 1
            self._add_ghost(k)

    def _add_ghost(self, k):
        if len(self.ghost) >= self.ghost_cap:
            old, _ = self.ghost.popitem(last=False)
            self.freq.pop(old, None)
        self.ghost[k] = (self.freq.get(k, 0), self.ops)

    def _decay(self):
        self.ops = 0
        self.freq = {k: v >> 1 for k, v in self.freq.items() if v > 1}


class SkipDecayTinyLFU:
    """TinyLFU with adaptive skip-decay when hit rate is high."""

    def __init__(self, cap):
        self.cap = cap
        self.win_cap = max(1, int(cap * 0.01))
        self.main_cap = cap - self.win_cap
        self.ghost_cap = cap * 2

        self.window = OrderedDict()
        self.main = OrderedDict()
        self.freq = {}
        self.ghost = OrderedDict()

        self.ops = 0
        self.decay_at = cap * 10

        self.stats_interval = max(100, cap // 2)
        self.last_stats_reset = 0
        self.recent_hits = 0
        self.recent_accesses = 0
        self.recent_hit_rate = 0.0

    def access(self, k):
        self.ops += 1
        self.recent_accesses += 1
        self.freq[k] = min(15, self.freq.get(k, 0) + 1)

        if k in self.window:
            self.recent_hits += 1
            self.window.move_to_end(k)
            return True

        if k in self.main:
            self.recent_hits += 1
            self.main.move_to_end(k)
            return True

        ghost_data = self.ghost.pop(k, None)
        if ghost_data is not None:
            ghost_freq, _ = ghost_data
            self.freq[k] = min(15, self.freq[k] + ghost_freq)

        self._add_to_window(k)

        if self.ops - self.last_stats_reset >= self.stats_interval:
            if self.recent_accesses > 0:
                self.recent_hit_rate = self.recent_hits / self.recent_accesses
            self.recent_hits = 0
            self.recent_accesses = 0
            self.last_stats_reset = self.ops

        if self.ops >= self.decay_at:
            self._decay()

        return False

    def _add_to_window(self, k):
        if len(self.window) >= self.win_cap:
            evicted, _ = self.window.popitem(last=False)
            self._try_promote(evicted)
        self.window[k] = 1

    def _try_promote(self, k):
        k_freq = self.freq.get(k, 0)
        if len(self.main) < self.main_cap:
            self.main[k] = 1
            return
        if self.main_cap == 0:
            self._add_ghost(k)
            return
        victim, _ = self.main.popitem(last=False)
        v_freq = self.freq.get(victim, 0)
        if k_freq > v_freq:
            self._add_ghost(victim)
            self.main[k] = 1
        else:
            self.main[victim] = 1
            self._add_ghost(k)

    def _add_ghost(self, k):
        if len(self.ghost) >= self.ghost_cap:
            old, _ = self.ghost.popitem(last=False)
            self.freq.pop(old, None)
        self.ghost[k] = (self.freq.get(k, 0), self.ops)

    def _decay(self):
        # SKIP decay if cache is working well
        if self.recent_hit_rate > 0.40:
            self.ops = 0
            self.last_stats_reset = 0
            return

        self.ops = 0
        self.last_stats_reset = 0
        self.freq = {k: v >> 1 for k, v in self.freq.items() if v > 1}


def generate_zipf(n, alpha=0.99, unique_keys=10000):
    """Generate Zipf-distributed accesses."""
    weights = [1.0 / (i ** alpha) for i in range(1, unique_keys + 1)]
    total = sum(weights)
    probs = [w / total for w in weights]

    trace = []
    cumulative = []
    cum = 0
    for p in probs:
        cum += p
        cumulative.append(cum)

    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                trace.append(i)
                break
    return trace


def generate_loop(n, loop_size, cache_size):
    """Generate loop pattern: cycling through more items than cache can hold."""
    return [i % loop_size for i in range(n)]


def generate_temporal(n, phases=5, keys_per_phase=2000):
    """Generate temporal/shifting workload."""
    trace = []
    phase_len = n // phases
    for phase in range(phases):
        base = phase * (keys_per_phase // 2)  # Some overlap between phases
        for _ in range(phase_len):
            # Zipf within each phase
            key = base + int(random.paretovariate(1.5)) % keys_per_phase
            trace.append(key)
    return trace


def generate_sequential(n, unique_keys=50000):
    """Generate sequential scan (worst case for caches)."""
    return [i % unique_keys for i in range(n)]


def run_test(cache_cls, trace):
    cache = cache_cls(CACHE_SIZE)
    hits = sum(1 for k in trace if cache.access(k))
    return hits / len(trace) * 100


def main():
    random.seed(42)

    print("="*70)
    print("OVERFITTING VERIFICATION")
    print("="*70)
    print(f"Cache size: {CACHE_SIZE}")
    print()

    workloads = {
        'zipf_0.8': generate_zipf(100000, alpha=0.8),
        'zipf_0.99': generate_zipf(100000, alpha=0.99),
        'zipf_1.2': generate_zipf(100000, alpha=1.2),
        'loop_n+10': generate_loop(100000, CACHE_SIZE + 10, CACHE_SIZE),
        'loop_n+100': generate_loop(100000, CACHE_SIZE + 100, CACHE_SIZE),
        'loop_2n': generate_loop(100000, CACHE_SIZE * 2, CACHE_SIZE),
        'temporal': generate_temporal(100000),
        'sequential': generate_sequential(100000),
    }

    print(f"{'Workload':<15} | {'Baseline':>10} | {'Skip-Decay':>10} | {'Diff':>8} | {'Winner':<10}")
    print("-" * 70)

    baseline_wins = 0
    skipdecay_wins = 0
    ties = 0
    total_baseline = 0
    total_skipdecay = 0

    for name, trace in workloads.items():
        baseline_rate = run_test(BaselineTinyLFU, trace)
        skipdecay_rate = run_test(SkipDecayTinyLFU, trace)
        diff = skipdecay_rate - baseline_rate

        total_baseline += baseline_rate
        total_skipdecay += skipdecay_rate

        if abs(diff) < 0.1:
            winner = "TIE"
            ties += 1
        elif diff > 0:
            winner = "SKIP-DECAY"
            skipdecay_wins += 1
        else:
            winner = "BASELINE"
            baseline_wins += 1

        print(f"{name:<15} | {baseline_rate:>9.2f}% | {skipdecay_rate:>9.2f}% | {diff:>+7.2f}pp | {winner:<10}")

    print("-" * 70)
    avg_baseline = total_baseline / len(workloads)
    avg_skipdecay = total_skipdecay / len(workloads)
    avg_diff = avg_skipdecay - avg_baseline

    print(f"{'AVERAGE':<15} | {avg_baseline:>9.2f}% | {avg_skipdecay:>9.2f}% | {avg_diff:>+7.2f}pp |")
    print()
    print(f"Summary: Baseline wins {baseline_wins}, Skip-Decay wins {skipdecay_wins}, Ties {ties}")

    if avg_diff >= 0 and baseline_wins <= skipdecay_wins:
        print("\n[OK] Skip-decay is NOT overfitting - it helps or is neutral across workloads")
    elif baseline_wins > skipdecay_wins:
        print("\n[WARNING] Skip-decay might be overfitting - baseline wins more often")
    else:
        print("\n[OK] Skip-decay shows mixed results but positive average")


if __name__ == "__main__":
    main()
