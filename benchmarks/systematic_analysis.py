#!/usr/bin/env python3
"""
SYSTEMATIC ANALYSIS: Finding the Remaining 2.2pp
=================================================

Current state:
- Theoretical max: 29.08%
- Hybrid hill climber: 26.88% (92.4% of optimal)
- Gap: 2.2pp

Where is the gap coming from?

Hypothesis: Frequency decay during the loop phase causes instability.
- Every 5120 ops, frequencies halve
- This creates temporary imbalances
- Wrong items might enter during these windows

Let's test: What if we SKIP decay when hit rate is high?
"""

import gzip
from pathlib import Path
from collections import OrderedDict

CACHE_SIZE = 512
TRACES_DIR = Path(__file__).parent.parent / "traces"


def load_traces():
    path = TRACES_DIR / "trace_vaultservice_large.gz"
    with gzip.open(path, 'rb') as f:
        data = f.read()
    corda = [data[i:i+16] for i in range(0, len(data), 16)]

    path = TRACES_DIR / "loop.trace.gz"
    with gzip.open(path, 'rt') as f:
        loop = [int(line.strip()) for line in f]

    return corda, loop


class AdaptiveDecayCache:
    """
    Key insight: Decay helps during TRANSITION (flush garbage) but
    HURTS during STABLE phases (causes churn).

    Strategy:
    - Low hit rate: Fast decay (flush garbage quickly)
    - High hit rate: Slow/skip decay (protect stability)
    """

    def __init__(self, cap, adaptive_decay=True):
        self.cap = cap
        self.adaptive_decay = adaptive_decay

        self.win_cap = max(1, int(cap * 0.01))  # 1% window
        self.main_cap = cap - self.win_cap
        self.ghost_cap = cap * 2

        self.window = OrderedDict()
        self.main = OrderedDict()
        self.freq = {}
        self.ghost = OrderedDict()

        self.ops = 0
        self.base_decay_at = cap * 10
        self.decay_at = self.base_decay_at

        # Ghost utility
        self.ghost_hits = 0
        self.ghost_lookups = 0
        self.ghost_utility = 0.0
        self.stats_interval = max(100, cap // 2)
        self.last_stats_reset = 0

        # Hit rate tracking for adaptive decay
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

        # Miss
        self.ghost_lookups += 1
        ghost_data = self.ghost.pop(k, None)
        if ghost_data is not None:
            self.ghost_hits += 1
            ghost_freq, _ = ghost_data
            self.freq[k] = min(15, self.freq[k] + ghost_freq)

        self._add_to_window(k)

        # Update stats
        if self.ops - self.last_stats_reset >= self.stats_interval:
            if self.ghost_lookups > 0:
                self.ghost_utility = self.ghost_hits / self.ghost_lookups
            if self.recent_accesses > 0:
                self.recent_hit_rate = self.recent_hits / self.recent_accesses
            self.ghost_hits = 0
            self.ghost_lookups = 0
            self.recent_hits = 0
            self.recent_accesses = 0
            self.last_stats_reset = self.ops

        # Decay
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

        # Strict frequency comparison
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
        if self.adaptive_decay:
            # ADAPTIVE DECAY: adjust based on hit rate
            if self.recent_hit_rate > 0.35:
                # High hit rate: slow down decay (2x period)
                self.decay_at = self.base_decay_at * 2
                # Still do the decay, just less often
            else:
                # Low hit rate: fast decay
                self.decay_at = self.base_decay_at

        self.ops = 0
        self.last_stats_reset = 0
        self.freq = {k: v >> 1 for k, v in self.freq.items() if v > 1}


class SkipDecayWhenStable:
    """
    More aggressive: SKIP decay entirely when hit rate is high.

    Rationale: If the cache is working well, don't risk destabilizing it.
    """

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

        self.ghost_hits = 0
        self.ghost_lookups = 0
        self.ghost_utility = 0.0
        self.stats_interval = max(100, cap // 2)
        self.last_stats_reset = 0

        self.recent_hits = 0
        self.recent_accesses = 0
        self.recent_hit_rate = 0.0

        # Track decays for analysis
        self.decays_performed = 0
        self.decays_skipped = 0

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

        self.ghost_lookups += 1
        ghost_data = self.ghost.pop(k, None)
        if ghost_data is not None:
            self.ghost_hits += 1
            ghost_freq, _ = ghost_data
            self.freq[k] = min(15, self.freq[k] + ghost_freq)

        self._add_to_window(k)

        if self.ops - self.last_stats_reset >= self.stats_interval:
            if self.ghost_lookups > 0:
                self.ghost_utility = self.ghost_hits / self.ghost_lookups
            if self.recent_accesses > 0:
                self.recent_hit_rate = self.recent_hits / self.recent_accesses
            self.ghost_hits = 0
            self.ghost_lookups = 0
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
        # SKIP decay if hit rate is high (cache is stable)
        if self.recent_hit_rate > 0.40:
            self.decays_skipped += 1
            self.ops = 0  # Just reset counter
            self.last_stats_reset = 0
            return

        self.decays_performed += 1
        self.ops = 0
        self.last_stats_reset = 0
        self.freq = {k: v >> 1 for k, v in self.freq.items() if v > 1}


class BaselineFixed:
    """Baseline: Fixed 1% window, normal decay."""

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


def run_test(cache, trace, name):
    hits = sum(1 for k in trace if cache.access(k))
    rate = hits / len(trace) * 100
    return rate


def main():
    corda, loop = load_traces()
    full_trace = list(corda) + list(loop * 5) + list(corda)

    print("="*70)
    print("SYSTEMATIC ANALYSIS: Adaptive Decay")
    print("="*70)
    print(f"Cache size: {CACHE_SIZE}")
    print(f"Trace: Corda ({len(corda):,}) -> Loop x5 ({len(loop*5):,}) -> Corda ({len(corda):,})")
    print(f"Total: {len(full_trace):,} accesses")
    print()

    # Run tests
    results = {}

    print("Running baseline (fixed 1% window, normal decay)...")
    baseline = BaselineFixed(CACHE_SIZE)
    results['baseline'] = run_test(baseline, full_trace, 'baseline')

    print("Running adaptive decay (slow decay when stable)...")
    adaptive = AdaptiveDecayCache(CACHE_SIZE, adaptive_decay=True)
    results['adaptive'] = run_test(adaptive, full_trace, 'adaptive')

    print("Running skip-decay (skip decay entirely when stable)...")
    skip = SkipDecayWhenStable(CACHE_SIZE)
    results['skip_decay'] = run_test(skip, full_trace, 'skip_decay')

    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    theoretical_max = 29.08

    for name, rate in sorted(results.items(), key=lambda x: -x[1]):
        efficiency = rate / theoretical_max * 100
        diff = rate - results['baseline']
        print(f"  {name:20s}: {rate:6.2f}% ({efficiency:5.1f}% of optimal) [{diff:+.2f}pp vs baseline]")

    # Show decay stats for skip_decay
    print()
    print(f"Skip-decay stats:")
    print(f"  Decays performed: {skip.decays_performed}")
    print(f"  Decays skipped:   {skip.decays_skipped}")
    print(f"  Skip ratio:       {skip.decays_skipped / (skip.decays_performed + skip.decays_skipped + 0.001):.1%}")

    # Test on loop phase only
    print()
    print("="*70)
    print("LOOP PHASE ONLY (where hits come from)")
    print("="*70)

    loop5 = loop * 5

    baseline_loop = BaselineFixed(CACHE_SIZE)
    rate_baseline = run_test(baseline_loop, loop5, 'baseline')

    skip_loop = SkipDecayWhenStable(CACHE_SIZE)
    rate_skip = run_test(skip_loop, loop5, 'skip')

    theoretical_loop = 50.62

    print(f"  Baseline:    {rate_baseline:.2f}% ({rate_baseline/theoretical_loop*100:.1f}% of optimal)")
    print(f"  Skip-decay:  {rate_skip:.2f}% ({rate_skip/theoretical_loop*100:.1f}% of optimal)")
    print(f"  Theoretical: {theoretical_loop:.2f}%")


if __name__ == "__main__":
    main()
