#!/usr/bin/env python3
"""
Auto-tuner for Chameleon ghost utility threshold.
Tests thresholds from 1% to 25% and finds the optimal value.
"""

import sys
sys.path.insert(0, '.')

from collections import OrderedDict
import random

# Import workload generators from bench.py
from bench import (
    load_trace, gen_loop, gen_zipf, gen_sequential, gen_temporal,
    ALGORITHMS
)

# Get TinyLFU class from registry
TinyLFU = ALGORITHMS['tinylfu'][0]

class ChameleonTunable:
    """Chameleon with configurable ghost utility threshold."""

    __slots__ = ('cap', 'window', 'main', 'freq', 'ghost', 'ops', 'decay_at',
                 'recent_keys', 'recent_pos', 'mode', 'last_check', 'check_interval',
                 'hits', 'misses', 'unique_count', 'win_cap', 'main_cap', 'ghost_cap',
                 'max_freq_seen', 'window_accesses', 'window_uniques', 'window_freq',
                 'is_high_variance', 'is_flat_variance', 'last_ghost_hit',
                 'ghost_hits', 'ghost_lookups', 'ghost_utility',
                 'loop_threshold')  # NEW: configurable threshold

    def __init__(self, cap, loop_threshold=0.10):
        self.cap = cap
        self.loop_threshold = loop_threshold  # Configurable!

        self.win_cap = max(1, cap // 100)
        self.main_cap = cap - self.win_cap
        self.ghost_cap = cap * 2

        self.window = OrderedDict()
        self.main = OrderedDict()
        self.freq = {}
        self.ghost = OrderedDict()
        self.ops = 0
        self.decay_at = cap * 10

        self.recent_keys = [None] * min(500, cap)
        self.recent_pos = 0
        self.mode = 'SCAN'
        self.last_check = 0
        self.check_interval = max(100, cap // 2)

        self.hits = 0
        self.misses = 0
        self.unique_count = 0

        self.max_freq_seen = 0
        self.window_accesses = 0
        self.window_uniques = set()
        self.window_freq = {}
        self.is_high_variance = False
        self.is_flat_variance = False
        self.last_ghost_hit = None

        self.ghost_hits = 0
        self.ghost_lookups = 0
        self.ghost_utility = 0.0

    def access(self, k):
        self.ops += 1
        old_freq = self.freq.get(k, 0)
        self.freq[k] = min(15, old_freq + 1)

        self.window_accesses += 1
        self.window_uniques.add(k)
        self.window_freq[k] = self.window_freq.get(k, 0) + 1
        if self.window_freq[k] > self.max_freq_seen:
            self.max_freq_seen = self.window_freq[k]

        if self.freq[k] == 1:
            self.unique_count += 1
        self.recent_keys[self.recent_pos] = k
        self.recent_pos = (self.recent_pos + 1) % len(self.recent_keys)

        if k in self.window:
            self.hits += 1
            self.window.move_to_end(k)
            return True

        if k in self.main:
            self.hits += 1
            self.main.move_to_end(k)
            return True

        self.misses += 1
        self.ghost_lookups += 1

        ghost_data = self.ghost.pop(k, None)
        if ghost_data is not None:
            self.ghost_hits += 1
            ghost_freq, evict_time = ghost_data
            recency = self.ops - evict_time
            if recency < self.decay_at // 2:
                boost = ghost_freq
            else:
                boost = max(1, ghost_freq >> 1)
            self.freq[k] = min(15, self.freq[k] + boost)
            self.last_ghost_hit = k
        else:
            self.last_ghost_hit = None

        self._add_to_window(k)

        if self.ops - self.last_check >= self.check_interval:
            self._detect_mode()
            self.last_check = self.ops

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

        victim, _ = self.main.popitem(last=False)
        v_freq = self.freq.get(victim, 0)

        should_admit = False
        has_returning_support = k_freq > 1

        ghost_is_useful = self.ghost_utility > 0.02
        ghost_is_loop = self.ghost_utility > self.loop_threshold  # USE CONFIGURABLE THRESHOLD

        if ghost_is_loop:
            allow_weak_tie_break = False
        else:
            allow_weak_tie_break = (
                not self.is_high_variance
                or k_freq > 1
                or ghost_is_useful
            )

        if self.mode == 'SCAN':
            if k_freq > v_freq:
                should_admit = True
            elif k_freq == v_freq and has_returning_support and not ghost_is_loop:
                should_admit = True
        elif self.mode == 'FREQ':
            if k_freq > v_freq:
                should_admit = True
            elif k_freq == v_freq and has_returning_support and allow_weak_tie_break:
                should_admit = True
        elif self.mode == 'RECENCY':
            should_admit = True
        else:
            if k_freq > v_freq:
                should_admit = True
            elif k_freq == v_freq and has_returning_support and allow_weak_tie_break:
                should_admit = True

        if should_admit:
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

    def _detect_mode(self):
        total = self.hits + self.misses
        if total < 100:
            return

        hit_rate = self.hits / total
        unique_rate = self.unique_count / total if total > 0 else 0
        warmup_complete = self.ops > self.check_interval * 5

        n_unique = len(self.window_uniques)
        if n_unique > 0 and self.window_accesses > 0:
            avg_freq = self.window_accesses / n_unique
            variance_ratio = self.max_freq_seen / avg_freq if avg_freq > 0 else 0
            self.is_high_variance = variance_ratio > 15
            self.is_flat_variance = variance_ratio < 5
        else:
            self.is_high_variance = False
            self.is_flat_variance = True

        if self.ghost_lookups > 0:
            self.ghost_utility = self.ghost_hits / self.ghost_lookups
        else:
            self.ghost_utility = 0.0

        self.window_uniques.clear()
        self.window_freq.clear()
        self.window_accesses = 0
        self.max_freq_seen = 0

        is_loop_pattern = self.ghost_utility > self.loop_threshold  # USE CONFIGURABLE THRESHOLD

        if not warmup_complete:
            self.mode = 'MIXED'
            self.win_cap = max(1, self.cap // 10)
        elif is_loop_pattern:
            self.mode = 'SCAN'
            self.win_cap = max(1, self.cap // 100)
        elif self.is_high_variance:
            self.mode = 'FREQ'
            self.win_cap = max(1, self.cap // 100)
        elif hit_rate < 0.20:
            self.mode = 'SCAN'
            self.win_cap = max(1, self.cap // 100)
        elif unique_rate < 0.3:
            self.mode = 'RECENCY'
            self.win_cap = max(1, self.cap // 5)
        elif hit_rate > 0.40:
            self.mode = 'FREQ'
            self.win_cap = max(1, self.cap // 20)
        else:
            self.mode = 'MIXED'
            self.win_cap = max(1, self.cap // 10)

        self.main_cap = self.cap - self.win_cap

        self.hits = 0
        self.misses = 0
        self.unique_count = 0
        self.ghost_hits = 0
        self.ghost_lookups = 0

    def _decay(self):
        self.ops = 0
        self.freq = {k: v >> 1 for k, v in self.freq.items() if v > 1}


def run_trace(trace, cache_class, cache_size, **kwargs):
    """Run a trace and return hit rate."""
    cache = cache_class(cache_size, **kwargs)
    hits = sum(1 for key in trace if cache.access(key))
    return hits / len(trace) * 100


def main():
    print("\n" + "=" * 60)
    print("   CHAMELEON GHOST UTILITY THRESHOLD AUTO-TUNER")
    print("=" * 60)

    # Load/generate workloads (smaller for speed)
    print("\nLoading workloads...")

    workloads = {}

    # Real traces (sampled for speed)
    hill = load_trace('hill', limit=50000)
    if hill:
        workloads['Hill'] = (hill, int(len(set(hill)) * 0.10))

    cloud = load_trace('cloud', limit=50000)
    if cloud:
        workloads['Cloud'] = (cloud, int(len(set(cloud)) * 0.10))

    twitter = load_trace('twitter', limit=200000)
    if twitter:
        workloads['Twitter'] = (twitter, int(len(set(twitter)) * 0.10))

    # Synthetic traces
    workloads['LOOP-N+1'] = (gen_loop(1000, 100000, extra=1), 1000)
    workloads['LOOP-N+10'] = (gen_loop(1000, 100000, extra=10), 1000)
    workloads['ZIPF-0.8'] = (gen_zipf(10000, 100000, alpha=0.8), 1000)
    workloads['ZIPF-0.99'] = (gen_zipf(10000, 100000, alpha=0.99), 1000)
    workloads['TEMPORAL'] = (gen_temporal(10000, 100000, phases=5), 1000)
    workloads['SEQUENTIAL'] = (gen_sequential(10000, 100000), 1000)

    print(f"  Loaded {len(workloads)} workloads")

    # Get TinyLFU baseline
    print("\nRunning TinyLFU baseline...")
    tinylfu_results = {}
    for name, (trace, cache_size) in workloads.items():
        tinylfu_results[name] = run_trace(trace, TinyLFU, cache_size)

    # Test thresholds from 1% to 25%
    thresholds = [i / 100 for i in range(1, 26)]
    results = {}

    print("\nTesting thresholds 1% to 25%...")
    print("-" * 60)

    for threshold in thresholds:
        pct = int(threshold * 100)
        results[threshold] = {}

        total_chameleon = 0
        total_tinylfu = 0
        wins = 0

        for name, (trace, cache_size) in workloads.items():
            hit_rate = run_trace(trace, ChameleonTunable, cache_size, loop_threshold=threshold)
            results[threshold][name] = hit_rate
            total_chameleon += hit_rate
            total_tinylfu += tinylfu_results[name]
            if hit_rate > tinylfu_results[name]:
                wins += 1

        avg_chameleon = total_chameleon / len(workloads)
        avg_tinylfu = total_tinylfu / len(workloads)
        diff = avg_chameleon - avg_tinylfu

        print(f"  {pct:2d}%: avg={avg_chameleon:.2f}% (vs TinyLFU: {diff:+.2f}pp) wins={wins}/{len(workloads)}")
        results[threshold]['_avg'] = avg_chameleon
        results[threshold]['_diff'] = diff
        results[threshold]['_wins'] = wins

    # Find best threshold
    print("\n" + "=" * 60)
    print("   RESULTS")
    print("=" * 60)

    best_avg = max(results.items(), key=lambda x: x[1]['_avg'])
    best_diff = max(results.items(), key=lambda x: x[1]['_diff'])
    best_wins = max(results.items(), key=lambda x: x[1]['_wins'])

    print(f"\n  Best Average:     {int(best_avg[0]*100)}% ({best_avg[1]['_avg']:.2f}%)")
    print(f"  Best vs TinyLFU:  {int(best_diff[0]*100)}% ({best_diff[1]['_diff']:+.2f}pp)")
    print(f"  Most Wins:        {int(best_wins[0]*100)}% ({best_wins[1]['_wins']}/{len(workloads)})")

    # Detailed breakdown of top 5
    print("\n  Top 5 Thresholds by Average:")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['_avg'], reverse=True)[:5]
    for thresh, data in sorted_results:
        print(f"    {int(thresh*100):2d}%: {data['_avg']:.2f}% ({data['_diff']:+.2f}pp vs TinyLFU, {data['_wins']} wins)")

    # Per-workload breakdown for best threshold
    best_thresh = best_avg[0]
    print(f"\n  Breakdown at {int(best_thresh*100)}% threshold:")
    for name in workloads:
        cham = results[best_thresh][name]
        tiny = tinylfu_results[name]
        diff = cham - tiny
        winner = "WIN" if diff > 0 else ""
        print(f"    {name:12s}: {cham:5.2f}% vs {tiny:5.2f}% ({diff:+.2f}pp) {winner}")

    print("\n" + "=" * 60)
    print(f"   RECOMMENDATION: Use {int(best_avg[0]*100)}% threshold")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
