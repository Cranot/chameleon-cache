#!/usr/bin/env python3
"""
Verify the combined SkipDecay + HillClimb approach across multiple workloads.
"""

import random
from collections import OrderedDict

CACHE_SIZE = 1000


class BaselineTinyLFU:
    """Standard TinyLFU."""

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


class SkipDecayPlusHillClimb:
    """Combined approach: skip-decay + hill climbing."""

    def __init__(self, cap):
        self.cap = cap
        self.win_pct = 0.01
        self.win_cap = max(1, int(cap * self.win_pct))
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

        self.sample_window = 1000
        self.window_hits = 0
        self.window_ops = 0
        self.last_hit_rate = 0.0
        self.best_hit_rate = 0.0
        self.last_adjustment = 0
        self.adjustment_interval = 2000
        self.direction = 1
        self.step_size = 0.02
        self.worse_count = 0

    def access(self, k):
        self.ops += 1
        self.recent_accesses += 1
        self.freq[k] = min(15, self.freq.get(k, 0) + 1)

        if k in self.window:
            self.recent_hits += 1
            self.window_hits += 1
            self.window_ops += 1
            self.window.move_to_end(k)
            return True

        if k in self.main:
            self.recent_hits += 1
            self.window_hits += 1
            self.window_ops += 1
            self.main.move_to_end(k)
            return True

        self.window_ops += 1

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

        if self.window_ops >= self.sample_window:
            self._sample_and_adjust()

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

    def _sample_and_adjust(self):
        current_hit_rate = self.window_hits / self.window_ops if self.window_ops > 0 else 0

        if abs(current_hit_rate - self.last_hit_rate) > 0.15:
            self.best_hit_rate = current_hit_rate
            self.worse_count = 0
            self.direction = 1

        if self.ops - self.last_adjustment >= self.adjustment_interval:
            if current_hit_rate > 0.01:
                if current_hit_rate > self.best_hit_rate + 0.003:
                    self.best_hit_rate = current_hit_rate
                    self.worse_count = 0
                elif current_hit_rate < self.last_hit_rate - 0.003:
                    self.worse_count += 1
                    if self.worse_count >= 2:
                        self.direction *= -1
                        self.worse_count = 0
                else:
                    self.worse_count = 0

                new_pct = self.win_pct + (self.direction * self.step_size)
                new_pct = max(0.01, min(0.30, new_pct))

                if new_pct != self.win_pct:
                    self.win_pct = new_pct
                    self.win_cap = max(1, int(self.cap * self.win_pct))
                    self.main_cap = self.cap - self.win_cap

            self.last_hit_rate = current_hit_rate
            self.last_adjustment = self.ops

        self.window_hits = 0
        self.window_ops = 0

    def _decay(self):
        if self.recent_hit_rate > 0.40:
            self.ops = 0
            self.last_stats_reset = 0
            self.last_adjustment = 0
            return

        self.ops = 0
        self.last_stats_reset = 0
        self.last_adjustment = 0
        self.freq = {k: v >> 1 for k, v in self.freq.items() if v > 1}


def generate_zipf(n, alpha=0.99, unique_keys=10000):
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


def generate_loop(n, loop_size):
    return [i % loop_size for i in range(n)]


def generate_temporal(n, phases=5, keys_per_phase=2000):
    trace = []
    phase_len = n // phases
    for phase in range(phases):
        base = phase * (keys_per_phase // 2)
        for _ in range(phase_len):
            key = base + int(random.paretovariate(1.5)) % keys_per_phase
            trace.append(key)
    return trace


def generate_mixed(n, cache_size):
    """Mixed workload: zipf -> loop -> zipf -> scan -> zipf"""
    segment = n // 5
    trace = []
    # Zipf segment
    trace.extend(generate_zipf(segment, alpha=0.99, unique_keys=5000))
    # Loop segment
    trace.extend(generate_loop(segment, cache_size + 50))
    # Zipf again
    trace.extend(generate_zipf(segment, alpha=0.99, unique_keys=5000))
    # Sequential scan (pollution)
    trace.extend([i for i in range(segment)])
    # Zipf recovery
    trace.extend(generate_zipf(segment, alpha=0.99, unique_keys=5000))
    return trace


def run_test(cache_cls, trace):
    cache = cache_cls(CACHE_SIZE)
    hits = sum(1 for k in trace if cache.access(k))
    return hits / len(trace) * 100


def main():
    random.seed(42)

    print("=" * 70)
    print("COMBINED APPROACH VERIFICATION")
    print("=" * 70)
    print(f"Cache size: {CACHE_SIZE}")
    print()

    workloads = {
        'zipf_0.8': generate_zipf(100000, alpha=0.8),
        'zipf_0.99': generate_zipf(100000, alpha=0.99),
        'zipf_1.2': generate_zipf(100000, alpha=1.2),
        'loop_n+10': generate_loop(100000, CACHE_SIZE + 10),
        'loop_n+100': generate_loop(100000, CACHE_SIZE + 100),
        'loop_2n': generate_loop(100000, CACHE_SIZE * 2),
        'temporal': generate_temporal(100000),
        'mixed': generate_mixed(100000, CACHE_SIZE),
    }

    print(f"{'Workload':<15} | {'Baseline':>10} | {'Combined':>10} | {'Diff':>8} | {'Winner':<10}")
    print("-" * 70)

    baseline_wins = 0
    combined_wins = 0
    ties = 0
    total_baseline = 0
    total_combined = 0

    for name, trace in workloads.items():
        baseline_rate = run_test(BaselineTinyLFU, trace)
        combined_rate = run_test(SkipDecayPlusHillClimb, trace)
        diff = combined_rate - baseline_rate

        total_baseline += baseline_rate
        total_combined += combined_rate

        if abs(diff) < 0.1:
            winner = "TIE"
            ties += 1
        elif diff > 0:
            winner = "COMBINED"
            combined_wins += 1
        else:
            winner = "BASELINE"
            baseline_wins += 1

        print(f"{name:<15} | {baseline_rate:>9.2f}% | {combined_rate:>9.2f}% | {diff:>+7.2f}pp | {winner:<10}")

    print("-" * 70)
    avg_baseline = total_baseline / len(workloads)
    avg_combined = total_combined / len(workloads)
    avg_diff = avg_combined - avg_baseline

    print(f"{'AVERAGE':<15} | {avg_baseline:>9.2f}% | {avg_combined:>9.2f}% | {avg_diff:>+7.2f}pp |")
    print()
    print(f"Summary: Baseline wins {baseline_wins}, Combined wins {combined_wins}, Ties {ties}")

    if avg_diff >= 0 and baseline_wins <= combined_wins:
        print("\n[OK] Combined approach generalizes well across workloads")
    else:
        print("\n[WARNING] Combined approach may be overfitting")


if __name__ == "__main__":
    main()
