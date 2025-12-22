#!/usr/bin/env python3
"""
SYSTEMATIC EXPLORATION: Beyond Skip-Decay
==========================================

Skip-decay achieved 98.8% of optimal. Can we close the remaining 1.2pp gap?

Ideas to explore:
1. Adaptive threshold (not fixed 40%)
2. Soft decay (divide by 1.5 instead of 2)
3. Frequency floor (never decay below minimum)
4. Ghost-aware decay
5. Combined: skip-decay + hill climbing
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


class BaselineTinyLFU:
    """Standard TinyLFU baseline."""

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


class SkipDecay:
    """Skip decay when hit rate > threshold."""

    def __init__(self, cap, threshold=0.40):
        self.cap = cap
        self.threshold = threshold
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
        if self.recent_hit_rate > self.threshold:
            self.ops = 0
            self.last_stats_reset = 0
            return

        self.ops = 0
        self.last_stats_reset = 0
        self.freq = {k: v >> 1 for k, v in self.freq.items() if v > 1}


class SoftDecay:
    """Soft decay: subtract 1 instead of divide by 2."""

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
        # Adaptive: soft decay when stable, hard decay when not
        if self.recent_hit_rate > 0.40:
            # Soft decay: subtract 1
            self.freq = {k: v - 1 for k, v in self.freq.items() if v > 1}
        else:
            # Hard decay: divide by 2
            self.freq = {k: v >> 1 for k, v in self.freq.items() if v > 1}

        self.ops = 0
        self.last_stats_reset = 0


class FrequencyFloor:
    """Never decay below a minimum frequency for cached items."""

    def __init__(self, cap, min_freq=2):
        self.cap = cap
        self.min_freq = min_freq
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
        # Decay but maintain floor for cached items
        cached = set(self.window.keys()) | set(self.main.keys())
        new_freq = {}
        for k, v in self.freq.items():
            decayed = v >> 1
            if k in cached:
                new_freq[k] = max(self.min_freq, decayed)
            elif decayed > 0:
                new_freq[k] = decayed
        self.freq = new_freq


class SkipDecayPlusHillClimb:
    """Combine skip-decay with hill climbing for window size."""

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

        # Hit rate tracking
        self.stats_interval = max(100, cap // 2)
        self.last_stats_reset = 0
        self.recent_hits = 0
        self.recent_accesses = 0
        self.recent_hit_rate = 0.0

        # Hill climber
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

        # Hill climbing
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

        # Phase transition detection
        if abs(current_hit_rate - self.last_hit_rate) > 0.15:
            self.best_hit_rate = current_hit_rate
            self.worse_count = 0
            self.direction = 1

        # Hill climbing with momentum
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
        # SKIP decay if hit rate is high
        if self.recent_hit_rate > 0.40:
            self.ops = 0
            self.last_stats_reset = 0
            self.last_adjustment = 0
            return

        self.ops = 0
        self.last_stats_reset = 0
        self.last_adjustment = 0
        self.freq = {k: v >> 1 for k, v in self.freq.items() if v > 1}


class AdaptiveThreshold:
    """Skip decay with adaptive threshold based on variance."""

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

        # Adaptive threshold
        self.hit_rate_history = []
        self.threshold = 0.40

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
                self.hit_rate_history.append(self.recent_hit_rate)
                if len(self.hit_rate_history) > 10:
                    self.hit_rate_history.pop(0)
                self._adapt_threshold()
            self.recent_hits = 0
            self.recent_accesses = 0
            self.last_stats_reset = self.ops

        if self.ops >= self.decay_at:
            self._decay()

        return False

    def _adapt_threshold(self):
        if len(self.hit_rate_history) >= 3:
            # Calculate variance
            mean = sum(self.hit_rate_history) / len(self.hit_rate_history)
            variance = sum((x - mean) ** 2 for x in self.hit_rate_history) / len(self.hit_rate_history)

            # High variance = unstable workload = lower threshold (skip less)
            # Low variance = stable workload = higher threshold (skip more)
            if variance < 0.01:
                self.threshold = 0.30  # Very stable, skip decay more aggressively
            elif variance > 0.05:
                self.threshold = 0.50  # Unstable, be more conservative
            else:
                self.threshold = 0.40

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
        if self.recent_hit_rate > self.threshold:
            self.ops = 0
            self.last_stats_reset = 0
            return

        self.ops = 0
        self.last_stats_reset = 0
        self.freq = {k: v >> 1 for k, v in self.freq.items() if v > 1}


def run_test(cache_cls, trace, **kwargs):
    if kwargs:
        cache = cache_cls(CACHE_SIZE, **kwargs)
    else:
        cache = cache_cls(CACHE_SIZE)
    hits = sum(1 for k in trace if cache.access(k))
    return hits / len(trace) * 100


def main():
    corda, loop = load_traces()
    full_trace = list(corda) + list(loop * 5) + list(corda)
    theoretical_max = 29.08

    print("=" * 70)
    print("SYSTEMATIC EXPLORATION: Beyond Skip-Decay")
    print("=" * 70)
    print(f"Cache size: {CACHE_SIZE}")
    print(f"Trace: Corda -> Loop x5 -> Corda")
    print(f"Total: {len(full_trace):,} accesses")
    print(f"Theoretical max: {theoretical_max:.2f}%")
    print()

    results = []

    # Baseline
    print("Testing approaches...")
    baseline = run_test(BaselineTinyLFU, full_trace)
    results.append(("Baseline TinyLFU", baseline))
    print(f"  Baseline: {baseline:.2f}%")

    # Skip-decay with different thresholds
    for thresh in [0.30, 0.35, 0.40, 0.45, 0.50]:
        rate = run_test(SkipDecay, full_trace, threshold=thresh)
        results.append((f"SkipDecay t={thresh}", rate))
        print(f"  SkipDecay t={thresh}: {rate:.2f}%")

    # Soft decay
    soft = run_test(SoftDecay, full_trace)
    results.append(("Soft Decay", soft))
    print(f"  Soft Decay: {soft:.2f}%")

    # Frequency floor
    for min_f in [1, 2, 3]:
        rate = run_test(FrequencyFloor, full_trace, min_freq=min_f)
        results.append((f"FreqFloor min={min_f}", rate))
        print(f"  FreqFloor min={min_f}: {rate:.2f}%")

    # Skip-decay + hill climb
    combined = run_test(SkipDecayPlusHillClimb, full_trace)
    results.append(("SkipDecay + HillClimb", combined))
    print(f"  SkipDecay + HillClimb: {combined:.2f}%")

    # Adaptive threshold
    adaptive = run_test(AdaptiveThreshold, full_trace)
    results.append(("Adaptive Threshold", adaptive))
    print(f"  Adaptive Threshold: {adaptive:.2f}%")

    # Summary
    print()
    print("=" * 70)
    print("RESULTS (sorted by hit rate)")
    print("=" * 70)
    results.sort(key=lambda x: -x[1])

    for name, rate in results:
        eff = rate / theoretical_max * 100
        diff = rate - baseline
        marker = " <-- BEST" if rate == results[0][1] else ""
        print(f"  {name:25s}: {rate:6.2f}% ({eff:5.1f}% of opt) [{diff:+.2f}pp]{marker}")

    # Analysis
    print()
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    best_name, best_rate = results[0]
    print(f"""
Best approach: {best_name} at {best_rate:.2f}% ({best_rate/theoretical_max*100:.1f}% of optimal)

Key insights:
1. Skip-decay (t=0.40) achieves {run_test(SkipDecay, full_trace, threshold=0.40):.2f}%
2. Gap to theoretical optimal: {theoretical_max - best_rate:.2f}pp
3. This gap is likely irreducible without oracle knowledge

The remaining gap comes from:
- Transition latency (detecting phase changes takes time)
- Admission mistakes during transitions
- Inherent information asymmetry (we can't predict the future)
""")


if __name__ == "__main__":
    main()
