#!/usr/bin/env python3
"""
HILL CLIMBER EXPERIMENT
=======================
Tests whether a simple hill climbing optimizer for window size can
outperform heuristic-based mode switching.

Key insight from Ben Manes (Caffeine):
"Most adaptive approaches rely on heuristics that guess based on second
order effects (e.g. ARC's ghosts), whereas a hit rate hill climbing
optimizer is able to focus on the main goal."
"""

import gzip
import sys
from pathlib import Path
from collections import OrderedDict

CACHE_SIZE = 512
TRACES_DIR = Path(__file__).parent.parent / "traces"

def load_loop_trace():
    path = TRACES_DIR / "loop.trace.gz"
    with gzip.open(path, 'rt') as f:
        return [int(line.strip()) for line in f]

def load_corda_trace():
    path = TRACES_DIR / "trace_vaultservice_large.gz"
    with gzip.open(path, 'rb') as f:
        data = f.read()
    keys = []
    for i in range(0, len(data), 16):
        keys.append(data[i:i+16])
    return keys


class ChameleonHillClimber:
    """
    Chameleon variant that uses hill climbing for window size optimization.

    V2 improvements:
    - Momentum: require consecutive worse samples before reversing
    - Phase detection: reset on dramatic hit rate changes
    - Ignore zero phases: don't climb when hit rate is fundamentally low
    """

    def __init__(self, cap):
        self.cap = cap

        # Start with TinyLFU-like small window (1%)
        self.win_pct = 0.01
        self.win_cap = max(1, int(cap * self.win_pct))
        self.main_cap = cap - self.win_cap
        self.ghost_cap = cap * 2

        # Storage
        self.window = OrderedDict()
        self.main = OrderedDict()
        self.freq = {}
        self.ghost = OrderedDict()

        # Timing
        self.ops = 0
        self.decay_at = cap * 10

        # Hill climber state (v2)
        self.sample_window = 1000  # Sample every N ops
        self.window_hits = 0
        self.window_ops = 0
        self.last_hit_rate = 0.0
        self.best_hit_rate = 0.0
        self.best_win_pct = 0.01
        self.last_adjustment = 0
        self.adjustment_interval = 3000  # Try new size every N ops (faster)
        self.direction = 1  # +1 = growing, -1 = shrinking
        self.step_size = 0.02  # 2% steps
        self.min_pct = 0.01
        self.max_pct = 0.30  # Cap at 30% (TinyLFU-style)
        self.worse_count = 0  # Momentum counter
        self.momentum_threshold = 2  # Need 2 consecutive worse before reversing

        # Telemetry for analysis
        self.telemetry = {
            'ops': [],
            'hit_rate': [],
            'window_pct': [],
        }

    def access(self, k):
        self.ops += 1

        # Update frequency
        self.freq[k] = min(15, self.freq.get(k, 0) + 1)

        # Window hit
        if k in self.window:
            self.window_hits += 1
            self.window_ops += 1
            self.window.move_to_end(k)
            return True

        # Main hit
        if k in self.main:
            self.window_hits += 1
            self.window_ops += 1
            self.main.move_to_end(k)
            return True

        # Miss
        self.window_ops += 1

        # Ghost boost
        ghost_data = self.ghost.pop(k, None)
        if ghost_data is not None:
            ghost_freq, evict_time = ghost_data
            self.freq[k] = min(15, self.freq[k] + ghost_freq)

        # Add to window
        self._add_to_window(k)

        # Hill climbing: sample hit rate and adjust window size
        if self.window_ops >= self.sample_window:
            self._sample_and_maybe_adjust()

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

        # Simple TinyLFU-style admission: strict frequency comparison
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

    def _sample_and_maybe_adjust(self):
        """Sample hit rate and potentially adjust window size (v2 with momentum)."""
        current_hit_rate = self.window_hits / self.window_ops if self.window_ops > 0 else 0

        # Record telemetry
        self.telemetry['ops'].append(self.ops)
        self.telemetry['hit_rate'].append(current_hit_rate * 100)
        self.telemetry['window_pct'].append(self.win_pct * 100)

        # Phase transition detection: dramatic change in hit rate
        if abs(current_hit_rate - self.last_hit_rate) > 0.20:
            # Reset hill climber state
            self.best_hit_rate = current_hit_rate
            self.best_win_pct = self.win_pct
            self.worse_count = 0
            self.direction = 1  # Start exploring upward

        # Hill climbing with momentum
        if self.ops - self.last_adjustment >= self.adjustment_interval:
            # Only climb if we have meaningful hit rate (>1%)
            if current_hit_rate > 0.01:
                if current_hit_rate > self.best_hit_rate + 0.005:
                    # New best! Save it and continue
                    self.best_hit_rate = current_hit_rate
                    self.best_win_pct = self.win_pct
                    self.worse_count = 0
                elif current_hit_rate < self.last_hit_rate - 0.005:
                    # Got worse
                    self.worse_count += 1
                    if self.worse_count >= self.momentum_threshold:
                        # Reverse direction
                        self.direction *= -1
                        self.worse_count = 0
                        # Optionally: revert to best known
                        # self.win_pct = self.best_win_pct
                else:
                    # No significant change
                    self.worse_count = 0

                # Apply adjustment
                new_pct = self.win_pct + (self.direction * self.step_size)
                new_pct = max(self.min_pct, min(self.max_pct, new_pct))

                if new_pct != self.win_pct:
                    self.win_pct = new_pct
                    self.win_cap = max(1, int(self.cap * self.win_pct))
                    self.main_cap = self.cap - self.win_cap

            self.last_hit_rate = current_hit_rate
            self.last_adjustment = self.ops

        # Reset sample window
        self.window_hits = 0
        self.window_ops = 0

    def _decay(self):
        self.ops = 0
        self.last_adjustment = 0
        self.freq = {k: v >> 1 for k, v in self.freq.items() if v > 1}


class ChameleonOriginal:
    """Original Chameleon for comparison (with bug fix)."""

    def __init__(self, cap):
        self.cap = cap
        self.win_cap = max(1, cap // 100)
        self.main_cap = cap - self.win_cap
        self.ghost_cap = cap * 2

        self.window = OrderedDict()
        self.main = OrderedDict()
        self.freq = {}
        self.ghost = OrderedDict()

        self.ops = 0
        self.decay_at = cap * 10

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

        self.ghost_hits = 0
        self.ghost_lookups = 0
        self.ghost_utility = 0.0

        # Telemetry
        self.telemetry = {
            'ops': [],
            'hit_rate': [],
            'window_pct': [],
            'mode': [],
        }
        self._sample_hits = 0
        self._sample_ops = 0

    def access(self, k):
        self.ops += 1
        self._sample_ops += 1

        self.freq[k] = min(15, self.freq.get(k, 0) + 1)

        self.window_accesses += 1
        self.window_uniques.add(k)
        self.window_freq[k] = self.window_freq.get(k, 0) + 1
        if self.window_freq[k] > self.max_freq_seen:
            self.max_freq_seen = self.window_freq[k]

        if self.freq[k] == 1:
            self.unique_count += 1

        if k in self.window:
            self.hits += 1
            self._sample_hits += 1
            self.window.move_to_end(k)
            return True

        if k in self.main:
            self.hits += 1
            self._sample_hits += 1
            self.main.move_to_end(k)
            return True

        self.misses += 1
        self.ghost_lookups += 1

        ghost_data = self.ghost.pop(k, None)
        if ghost_data is not None:
            self.ghost_hits += 1
            ghost_freq, evict_time = ghost_data
            self.freq[k] = min(15, self.freq[k] + ghost_freq)

        self._add_to_window(k)

        if self.ops - self.last_check >= self.check_interval:
            self._detect_mode()
            self.last_check = self.ops

        if self.ops >= self.decay_at:
            self._decay()

        # Sample telemetry every 1000 ops
        if self._sample_ops >= 1000:
            hr = self._sample_hits / self._sample_ops * 100
            self.telemetry['ops'].append(self.ops)
            self.telemetry['hit_rate'].append(hr)
            self.telemetry['window_pct'].append(self.win_cap / self.cap * 100)
            self.telemetry['mode'].append(self.mode)
            self._sample_hits = 0
            self._sample_ops = 0

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

        ghost_is_loop = self.ghost_utility > 0.12

        should_admit = False
        has_returning_support = k_freq > 1

        if self.mode == 'SCAN':
            if k_freq > v_freq:
                should_admit = True
            elif k_freq == v_freq and has_returning_support and not ghost_is_loop:
                should_admit = True
        elif self.mode == 'FREQ':
            if k_freq > v_freq:
                should_admit = True
            elif k_freq == v_freq and has_returning_support:
                should_admit = True
        elif self.mode == 'RECENCY':
            should_admit = True
        else:  # MIXED
            if k_freq > v_freq:
                should_admit = True
            elif k_freq == v_freq and has_returning_support:
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
        else:
            self.is_high_variance = False

        if self.ghost_lookups > 0:
            self.ghost_utility = self.ghost_hits / self.ghost_lookups

        self.window_uniques.clear()
        self.window_freq.clear()
        self.window_accesses = 0
        self.max_freq_seen = 0

        is_loop_pattern = self.ghost_utility > 0.12

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
        self.last_check = 0
        self.freq = {k: v >> 1 for k, v in self.freq.items() if v > 1}


def build_chained_trace():
    corda = load_corda_trace()
    loop = load_loop_trace()

    chained = []
    chained.extend(corda)
    for _ in range(5):
        chained.extend(loop)
    chained.extend(corda)

    return chained, len(corda), len(corda) + len(loop) * 5


def run_benchmark(trace, cache):
    hits = 0
    for key in trace:
        if cache.access(key):
            hits += 1
    return hits / len(trace) * 100


def main():
    print("="*70)
    print("HILL CLIMBER EXPERIMENT")
    print("="*70)
    print(f"Cache size: {CACHE_SIZE}")
    print()

    # Load trace
    print("Loading traces...")
    trace, loop_start, corda2_start = build_chained_trace()
    print(f"  Total: {len(trace):,} accesses")
    print(f"  Phase boundaries: corda1=0, loop={loop_start:,}, corda2={corda2_start:,}")
    print()

    # Run original Chameleon
    print("Running Original Chameleon...")
    original = ChameleonOriginal(CACHE_SIZE)
    original_rate = run_benchmark(trace, original)
    print(f"  Hit rate: {original_rate:.2f}%")

    # Run Hill Climber Chameleon
    print("\nRunning Hill Climber Chameleon...")
    hill_climber = ChameleonHillClimber(CACHE_SIZE)
    hill_climber_rate = run_benchmark(trace, hill_climber)
    print(f"  Hit rate: {hill_climber_rate:.2f}%")

    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"  Original Chameleon: {original_rate:.2f}%")
    print(f"  Hill Climber:       {hill_climber_rate:.2f}%")
    print(f"  Difference:         {hill_climber_rate - original_rate:+.2f}pp")

    # Show window size evolution for hill climber
    print("\n" + "="*70)
    print("HILL CLIMBER WINDOW SIZE EVOLUTION (sampled)")
    print("="*70)

    tel = hill_climber.telemetry
    if tel['ops']:
        # Show first few, middle, and last few samples
        samples = []
        n = len(tel['ops'])
        # First 5
        for i in range(min(5, n)):
            samples.append(i)
        # Around phase transitions
        for target in [loop_start, corda2_start]:
            for i, ops in enumerate(tel['ops']):
                if abs(ops - target) < 10000:
                    if i not in samples:
                        samples.append(i)
        # Last 5
        for i in range(max(0, n-5), n):
            if i not in samples:
                samples.append(i)

        samples = sorted(set(samples))

        for i in samples:
            ops = tel['ops'][i]
            hr = tel['hit_rate'][i]
            wp = tel['window_pct'][i]
            phase = "corda1" if ops < loop_start else ("loop" if ops < corda2_start else "corda2")
            print(f"  ops {ops:>10,}: HR={hr:5.2f}%, Window={wp:5.1f}% [{phase}]")

    # Compare mode changes vs window size changes
    print("\n" + "="*70)
    print("ORIGINAL CHAMELEON MODE EVOLUTION (sampled)")
    print("="*70)

    tel = original.telemetry
    if tel['ops']:
        prev_mode = None
        for i, mode in enumerate(tel['mode']):
            if mode != prev_mode:
                ops = tel['ops'][i]
                hr = tel['hit_rate'][i]
                wp = tel['window_pct'][i]
                phase = "corda1" if ops < loop_start else ("loop" if ops < corda2_start else "corda2")
                print(f"  ops {ops:>10,}: {mode:8s} HR={hr:5.2f}%, Window={wp:5.1f}% [{phase}]")
                prev_mode = mode


if __name__ == "__main__":
    main()
