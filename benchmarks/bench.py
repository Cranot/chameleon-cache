#!/usr/bin/env python3
"""
CACHE BENCHMARK SUITE
=====================
Fast, configurable benchmark for cache eviction algorithms.

Usage:
    python bench.py                     # Default run
    python bench.py --quick             # Fast mode (10k samples)
    python bench.py --full              # Full traces
    python bench.py --real              # Only real traces
    python bench.py --synth             # Only synthetic
    python bench.py -a lru,sieve,wgs    # Specific algorithms
    python bench.py --list              # List available algorithms
    python bench.py --compare a,b       # Head-to-head comparison

Examples:
    python bench.py --quick -a wgs,sieve,s3fifo
    python bench.py --real --full
    python bench.py --compare wgs,tinylfu
"""

import argparse
import time
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Any
import random

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class BenchConfig:
    # What to test
    real: bool = True
    synth: bool = True
    algos: list = field(default_factory=list)  # empty = all

    # Sample sizes (0 = full)
    real_limit: int = 50_000
    synth_size: int = 100_000
    cache_size: int = 1000  # for synthetic
    cache_pct: float = 0.10  # for real (10% of unique)

    # Output
    top_n: int = 15
    verbose: bool = True
    color: bool = True


# ============================================================================
# OUTPUT HELPERS
# ============================================================================

class Colors:
    BOLD = '\033[1m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RED = '\033[91m'
    END = '\033[0m'
    GRAY = '\033[90m'

def c(text, color, cfg):
    """Colorize text if enabled."""
    if cfg.color and sys.stdout.isatty():
        return f"{color}{text}{Colors.END}"
    return text

def p(msg="", cfg=None):
    """Print with flush."""
    if cfg is None or cfg.verbose:
        print(msg, flush=True)


# ============================================================================
# ALGORITHM REGISTRY
# ============================================================================

ALGORITHMS: dict[str, tuple[Callable, str]] = {}  # name -> (class, description)

def register(name: str, desc: str = ""):
    """Decorator to register an algorithm."""
    def decorator(cls):
        ALGORITHMS[name.lower()] = (cls, desc)
        return cls
    return decorator


# ============================================================================
# CACHE IMPLEMENTATIONS
# ============================================================================

@register("lru", "Least Recently Used - simple baseline")
class LRU:
    __slots__ = ('cap', 'cache')
    def __init__(self, cap):
        self.cap = cap
        self.cache = OrderedDict()
    def access(self, k):
        if k in self.cache:
            self.cache.move_to_end(k)
            return True
        if len(self.cache) >= self.cap:
            self.cache.popitem(last=False)
        self.cache[k] = 1
        return False


@register("sieve", "SIEVE (NSDI'24) - lazy promotion with visited bit")
class SIEVE:
    __slots__ = ('cap', 'cache', 'visited', 'hand')
    def __init__(self, cap):
        self.cap = cap
        self.cache = OrderedDict()
        self.visited = {}
        self.hand = None

    def access(self, k):
        if k in self.cache:
            self.visited[k] = True
            return True
        if len(self.cache) >= self.cap:
            self._evict()
        self.cache[k] = 1
        self.visited[k] = False
        return False

    def _evict(self):
        if not self.cache:
            return
        keys = list(self.cache.keys())
        n = len(keys)
        if self.hand not in self.cache:
            self.hand = keys[0]

        idx = keys.index(self.hand)
        for _ in range(n * 2):
            k = keys[idx]
            if self.visited.get(k):
                self.visited[k] = False
            else:
                del self.cache[k]
                self.visited.pop(k, None)
                self.hand = keys[(idx + 1) % n] if n > 1 else None
                return
            idx = (idx + 1) % n


@register("s3fifo", "S3-FIFO (SOSP'23) - small/main/ghost queues")
class S3FIFO:
    __slots__ = ('small_cap', 'main_cap', 'ghost_cap', 'small', 'main', 'ghost')
    def __init__(self, cap):
        self.small_cap = max(1, cap // 10)
        self.main_cap = cap - self.small_cap
        self.ghost_cap = cap
        self.small = OrderedDict()
        self.main = OrderedDict()
        self.ghost = OrderedDict()

    def access(self, k):
        if k in self.small:
            self.small[k] = min(3, self.small[k] + 1)
            return True
        if k in self.main:
            self.main[k] = min(3, self.main[k] + 1)
            return True

        # Miss
        reinserting = k in self.ghost
        if reinserting:
            del self.ghost[k]

        # Evict from small if needed
        while len(self.small) >= self.small_cap:
            ek, ef = self.small.popitem(last=False)
            if ef > 0:
                # Promote to main
                while len(self.main) >= self.main_cap:
                    mk, mf = self.main.popitem(last=False)
                    if mf > 0:
                        self.main[mk] = mf - 1
                        self.main.move_to_end(mk)
                    else:
                        self._add_ghost(mk)
                self.main[ek] = ef - 1
            else:
                self._add_ghost(ek)

        self.small[k] = 1 if reinserting else 0
        return False

    def _add_ghost(self, k):
        if len(self.ghost) >= self.ghost_cap:
            self.ghost.popitem(last=False)
        self.ghost[k] = 1


@register("arc", "ARC (IBM) - Adaptive Replacement Cache")
class ARC:
    """
    Adaptive Replacement Cache - balances recency and frequency adaptively.
    Uses 4 lists: T1 (recent), T2 (frequent), B1 (ghost recent), B2 (ghost frequent).
    Parameter p adapts based on ghost hits.
    """
    __slots__ = ('cap', 't1', 't2', 'b1', 'b2', 'p')

    def __init__(self, cap):
        self.cap = cap
        self.t1 = OrderedDict()  # Recent items (seen once recently)
        self.t2 = OrderedDict()  # Frequent items (seen twice recently)
        self.b1 = OrderedDict()  # Ghost list for T1 evictions
        self.b2 = OrderedDict()  # Ghost list for T2 evictions
        self.p = 0  # Target size for T1 (adaptive)

    def access(self, k):
        # Case 1: Hit in T1 - move to T2 (became frequent)
        if k in self.t1:
            del self.t1[k]
            self.t2[k] = 1
            return True

        # Case 2: Hit in T2 - move to MRU of T2
        if k in self.t2:
            self.t2.move_to_end(k)
            return True

        # Miss - check ghosts and adapt

        # Case 3: Ghost hit in B1 - adapt towards recency
        if k in self.b1:
            # Increase p (favor recency)
            delta = max(1, len(self.b2) // max(1, len(self.b1)))
            self.p = min(self.cap, self.p + delta)
            del self.b1[k]
            self._replace(k, False)
            self.t2[k] = 1
            return False

        # Case 4: Ghost hit in B2 - adapt towards frequency
        if k in self.b2:
            # Decrease p (favor frequency)
            delta = max(1, len(self.b1) // max(1, len(self.b2)))
            self.p = max(0, self.p - delta)
            del self.b2[k]
            self._replace(k, True)
            self.t2[k] = 1
            return False

        # Case 5: Complete miss
        t1_len = len(self.t1)
        b1_len = len(self.b1)

        if t1_len + b1_len == self.cap:
            if t1_len < self.cap:
                self.b1.popitem(last=False)
                self._replace(k, False)
            else:
                self.t1.popitem(last=False)
        elif t1_len + b1_len < self.cap:
            total = len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2)
            if total >= self.cap:
                if total == 2 * self.cap:
                    self.b2.popitem(last=False)
                self._replace(k, False)

        self.t1[k] = 1
        return False

    def _replace(self, k, in_b2):
        t1_len = len(self.t1)
        if t1_len > 0 and (t1_len > self.p or (in_b2 and t1_len == self.p)):
            # Evict from T1
            old, _ = self.t1.popitem(last=False)
            if len(self.b1) >= self.cap:
                self.b1.popitem(last=False)
            self.b1[old] = 1
        elif len(self.t2) > 0:
            # Evict from T2
            old, _ = self.t2.popitem(last=False)
            if len(self.b2) >= self.cap:
                self.b2.popitem(last=False)
            self.b2[old] = 1


@register("lirs", "LIRS - Low Inter-reference Recency Set")
class LIRS:
    """
    LIRS cache - tracks inter-reference recency.
    LIR = low IRR (hot), HIR = high IRR (cold).
    Simplified stack-free implementation.
    """
    __slots__ = ('cap', 'lir_cap', 'hir_cap', 'lir', 'hir', 'nonresident', 'recency')

    def __init__(self, cap):
        self.cap = cap
        self.lir_cap = max(1, int(cap * 0.99))  # 99% LIR
        self.hir_cap = cap - self.lir_cap       # 1% resident HIR
        self.lir = OrderedDict()       # LIR blocks (hot)
        self.hir = OrderedDict()       # Resident HIR blocks (cold)
        self.nonresident = OrderedDict()  # Non-resident HIR (ghost)
        self.recency = OrderedDict()   # Global recency order

    def access(self, k):
        # Update recency
        if k in self.recency:
            self.recency.move_to_end(k)
        else:
            self.recency[k] = 1

        # Hit in LIR
        if k in self.lir:
            self.lir.move_to_end(k)
            self._prune_lir()
            return True

        # Hit in resident HIR
        if k in self.hir:
            del self.hir[k]
            # Promote to LIR if space or demote old LIR
            if len(self.lir) < self.lir_cap:
                self.lir[k] = 1
            else:
                # Move oldest LIR to HIR
                old_lir, _ = self.lir.popitem(last=False)
                self.hir[old_lir] = 1
                self.lir[k] = 1
                self._prune_lir()
            return True

        # Miss
        was_nonresident = k in self.nonresident
        if was_nonresident:
            del self.nonresident[k]

        # Evict if needed
        total_resident = len(self.lir) + len(self.hir)
        if total_resident >= self.cap:
            if len(self.hir) > 0:
                # Evict from HIR first
                evicted, _ = self.hir.popitem(last=False)
                if len(self.nonresident) >= self.cap:
                    self.nonresident.popitem(last=False)
                self.nonresident[evicted] = 1
            elif len(self.lir) > 0:
                # Demote LIR to evicted
                evicted, _ = self.lir.popitem(last=False)
                if len(self.nonresident) >= self.cap:
                    self.nonresident.popitem(last=False)
                self.nonresident[evicted] = 1

        # Insert
        if was_nonresident and len(self.lir) < self.lir_cap:
            # Non-resident hit -> promote to LIR
            self.lir[k] = 1
        elif len(self.lir) < self.lir_cap:
            # Warmup: add to LIR
            self.lir[k] = 1
        else:
            # Add as resident HIR
            if len(self.hir) >= self.hir_cap and len(self.hir) > 0:
                evicted, _ = self.hir.popitem(last=False)
                if len(self.nonresident) >= self.cap:
                    self.nonresident.popitem(last=False)
                self.nonresident[evicted] = 1
            self.hir[k] = 1

        return False

    def _prune_lir(self):
        """Remove non-LIR items from bottom of LIR stack."""
        while self.lir and len(self.recency) > self.cap * 2:
            oldest = next(iter(self.recency))
            if oldest not in self.lir:
                del self.recency[oldest]
            else:
                break


@register("tinylfu-adaptive", "Adaptive W-TinyLFU (Caffeine) - hill climber window sizing")
class AdaptiveWTinyLFU:
    """
    Adaptive W-TinyLFU with hill climber.
    Adjusts window size based on observed hit rates.
    Based on: https://dl.acm.org/doi/10.1145/3274808.3274816
    """
    __slots__ = ('cap', 'win_cap', 'main_cap', 'window', 'main', 'freq', 'ops', 'decay_at',
                 'hits_in_window', 'hits_in_main', 'misses', 'sample_size', 'step_size',
                 'prev_hit_rate', 'increasing')

    def __init__(self, cap):
        self.cap = cap
        self.win_cap = max(1, cap // 100)  # Start at 1%
        self.main_cap = cap - self.win_cap
        self.window = OrderedDict()
        self.main = OrderedDict()
        self.freq = {}
        self.ops = 0
        self.decay_at = cap * 10

        # Hill climber state
        self.hits_in_window = 0
        self.hits_in_main = 0
        self.misses = 0
        self.sample_size = max(100, cap // 10)
        self.step_size = max(1, cap // 100)  # 1% steps
        self.prev_hit_rate = 0.0
        self.increasing = True  # Direction of hill climb

    def access(self, k):
        self.ops += 1
        self.freq[k] = min(15, self.freq.get(k, 0) + 1)

        if k in self.window:
            self.window.move_to_end(k)
            self.hits_in_window += 1
            return True
        if k in self.main:
            self.main.move_to_end(k)
            self.hits_in_main += 1
            return True

        # Miss
        self.misses += 1
        if len(self.window) >= self.win_cap:
            evicted, _ = self.window.popitem(last=False)
            self._try_promote(evicted)
        self.window[k] = 1

        if self.ops >= self.decay_at:
            self._decay()

        # Hill climb periodically
        total = self.hits_in_window + self.hits_in_main + self.misses
        if total >= self.sample_size:
            self._hill_climb()

        return False

    def _try_promote(self, k):
        if len(self.main) < self.main_cap:
            self.main[k] = 1
            return
        victim, _ = self.main.popitem(last=False)
        if self.freq.get(k, 0) > self.freq.get(victim, 0):
            self.main[k] = 1
        else:
            self.main[victim] = 1

    def _decay(self):
        self.ops = 0
        self.freq = {k: v >> 1 for k, v in self.freq.items() if v > 1}

    def _hill_climb(self):
        """Adjust window size based on hit rate trend."""
        total = self.hits_in_window + self.hits_in_main + self.misses
        if total == 0:
            return

        hit_rate = (self.hits_in_window + self.hits_in_main) / total

        # Compare to previous
        if hit_rate < self.prev_hit_rate:
            # Got worse, reverse direction
            self.increasing = not self.increasing

        # Adjust window size
        if self.increasing:
            new_win = min(self.cap - 1, self.win_cap + self.step_size)
        else:
            new_win = max(1, self.win_cap - self.step_size)

        if new_win != self.win_cap:
            self.win_cap = new_win
            self.main_cap = self.cap - self.win_cap
            # Rebalance if needed
            while len(self.window) > self.win_cap:
                evicted, _ = self.window.popitem(last=False)
                self._try_promote(evicted)
            while len(self.main) > self.main_cap and len(self.main) > 0:
                self.main.popitem(last=False)

        self.prev_hit_rate = hit_rate
        self.hits_in_window = 0
        self.hits_in_main = 0
        self.misses = 0


@register("tinylfu", "W-TinyLFU (Caffeine) - window + frequency filter")
class WTinyLFU:
    __slots__ = ('win_cap', 'main_cap', 'window', 'main', 'freq', 'ops', 'decay_at')
    def __init__(self, cap):
        self.win_cap = max(1, cap // 100)  # 1% window
        self.main_cap = cap - self.win_cap
        self.window = OrderedDict()
        self.main = OrderedDict()
        self.freq = {}
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

        # Miss - add to window
        if len(self.window) >= self.win_cap:
            evicted, _ = self.window.popitem(last=False)
            self._try_promote(evicted)
        self.window[k] = 1

        if self.ops >= self.decay_at:
            self._decay()
        return False

    def _try_promote(self, k):
        if len(self.main) < self.main_cap:
            self.main[k] = 1
            return
        victim, _ = self.main.popitem(last=False)
        if self.freq.get(k, 0) > self.freq.get(victim, 0):
            self.main[k] = 1
        else:
            self.main[victim] = 1

    def _decay(self):
        self.ops = 0
        self.freq = {k: v >> 1 for k, v in self.freq.items() if v > 1}


@register("chameleon", "Chameleon - workload-adaptive cache (TinyLFU-style when needed)")
class Chameleon:
    """
    Workload-Adaptive Cache - detects patterns and switches strategies.

    Key insight: TinyLFU dominates scan-resistant workloads because of its
    WINDOW + MAIN structure. The window buffers new items while they build
    frequency, preventing cache pollution.

    Modes:
    - SCAN: Use TinyLFU-style window+main with frequency filter
    - FREQ: Larger window, frequency-biased admission
    - RECENCY: Large window, LRU-like behavior
    - MIXED: Balanced window and frequency

    The key is that we ALWAYS use a window, but its size and the admission
    policy change based on detected workload.
    """
    __slots__ = ('cap', 'window', 'main', 'freq', 'ghost', 'ops', 'decay_at',
                 'recent_keys', 'recent_pos', 'mode', 'last_check', 'check_interval',
                 'hits', 'misses', 'unique_count', 'win_cap', 'main_cap', 'ghost_cap',
                 'max_freq_seen', 'window_accesses', 'window_uniques', 'window_freq',
                 'is_high_variance', 'is_flat_variance', 'last_ghost_hit',
                 'ghost_hits', 'ghost_lookups', 'ghost_utility',
                 # Skip-decay tracking (v1.1)
                 'recent_hit_rate', 'skip_decay_hits', 'skip_decay_accesses',
                 'skip_decay_interval', 'last_skip_decay_reset')

    def __init__(self, cap):
        self.cap = cap

        # Start with TinyLFU-like configuration (1% window)
        self.win_cap = max(1, cap // 100)
        self.main_cap = cap - self.win_cap
        self.ghost_cap = cap * 2  # Medium Ghost: balance between ZIPF (small) and loops (large)

        self.window = OrderedDict()  # Small window for new items
        self.main = OrderedDict()    # Main cache
        self.freq = {}               # Frequency counts
        self.ghost = OrderedDict()   # Ghost buffer
        self.ops = 0
        self.decay_at = cap * 10

        # Pattern detection
        self.recent_keys = [None] * min(500, cap)
        self.recent_pos = 0
        self.mode = 'SCAN'  # Start conservative
        self.last_check = 0
        self.check_interval = max(100, cap // 2)

        # Stats for detection
        self.hits = 0
        self.misses = 0
        self.unique_count = 0

        # Variance detection (Gemini's insight: distinguish Zipf from Temporal/Loop)
        self.max_freq_seen = 0       # Track highest TRUE frequency in window (uncapped)
        self.window_accesses = 0     # Accesses in current detection window
        self.window_uniques = set()  # Unique items in current window (for accurate variance)
        self.window_freq = {}        # TRUE frequency per item in window (uncapped, for variance)
        self.is_high_variance = False  # High variance (>15) = Zipf, force strict
        self.is_flat_variance = False  # Flat variance (<5) = Loop/Sequential, allow tie-break
        self.last_ghost_hit = None   # Track ghost hits for tie-breaking

        # Ghost Utility (Gemini Final: distinguish useful loops from random noise)
        self.ghost_hits = 0          # Ghost buffer hits in current window
        self.ghost_lookups = 0       # Total cache misses in current window
        self.ghost_utility = 0.0     # ghost_hits / ghost_lookups (>2% = useful)

        # Skip-decay tracking (v1.1 enhancement)
        self.recent_hit_rate = 0.0
        self.skip_decay_hits = 0
        self.skip_decay_accesses = 0
        self.skip_decay_interval = max(100, cap // 2)
        self.last_skip_decay_reset = 0

    def access(self, k):
        self.ops += 1
        self.skip_decay_accesses += 1
        old_freq = self.freq.get(k, 0)
        self.freq[k] = min(15, old_freq + 1)

        # Window-based tracking for variance detection
        self.window_accesses += 1
        self.window_uniques.add(k)

        # Track TRUE frequency per item in window (UNCAPPED for variance detection)
        # This is critical: the capped sketch freq (max 15) masks Zipf's heavy tail
        self.window_freq[k] = self.window_freq.get(k, 0) + 1
        if self.window_freq[k] > self.max_freq_seen:
            self.max_freq_seen = self.window_freq[k]

        # Track for pattern detection (per-period stats)
        if self.freq[k] == 1:
            self.unique_count += 1
        self.recent_keys[self.recent_pos] = k
        self.recent_pos = (self.recent_pos + 1) % len(self.recent_keys)

        # Window hit
        if k in self.window:
            self.hits += 1
            self.skip_decay_hits += 1
            self.window.move_to_end(k)
            return True

        # Main hit
        if k in self.main:
            self.hits += 1
            self.skip_decay_hits += 1
            self.main.move_to_end(k)
            return True

        # Cache miss
        self.misses += 1
        self.ghost_lookups += 1  # Track for Ghost Utility

        # Ghost hit - boost frequency based on recency
        ghost_data = self.ghost.pop(k, None)
        if ghost_data is not None:
            self.ghost_hits += 1  # Track for Ghost Utility
            ghost_freq, evict_time = ghost_data
            recency = self.ops - evict_time
            # Recent evictions (< half decay period) get full boost
            # Older evictions get diminishing boost (prevents noise from winning)
            if recency < self.decay_at // 2:
                boost = ghost_freq  # Full boost for recent items
            else:
                boost = max(1, ghost_freq >> 1)  # Half boost for older items
            self.freq[k] = min(15, self.freq[k] + boost)
            self.last_ghost_hit = k  # Mark for ghost-biased tie-breaking
        else:
            self.last_ghost_hit = None

        # Add to window (TinyLFU pattern - always buffer new items)
        self._add_to_window(k)

        # Update skip-decay hit rate tracking
        if self.ops - self.last_skip_decay_reset >= self.skip_decay_interval:
            if self.skip_decay_accesses > 0:
                self.recent_hit_rate = self.skip_decay_hits / self.skip_decay_accesses
            self.skip_decay_hits = 0
            self.skip_decay_accesses = 0
            self.last_skip_decay_reset = self.ops

        # Periodic mode detection
        if self.ops - self.last_check >= self.check_interval:
            self._detect_mode()
            self.last_check = self.ops

        if self.ops >= self.decay_at:
            self._decay()

        return False

    def _add_to_window(self, k):
        """Add item to window, promoting evicted item to main if worthy."""
        if len(self.window) >= self.win_cap:
            evicted, _ = self.window.popitem(last=False)
            self._try_promote(evicted)
        self.window[k] = 1

    def _try_promote(self, k):
        """Try to promote item from window to main (mode-specific policy)."""
        k_freq = self.freq.get(k, 0)

        # If main has space, always admit
        if len(self.main) < self.main_cap:
            self.main[k] = 1
            return

        # Find victim (LRU in main)
        victim, _ = self.main.popitem(last=False)
        v_freq = self.freq.get(victim, 0)

        # Mode-specific admission decision with refined tie-breaking
        # Key insight from Gemini: High variance = Zipf, be extra strict on ties
        should_admit = False

        # "Returning support" = freq > 1 means item was seen before
        has_returning_support = k_freq > 1

        # GHOST UTILITY ANALYSIS (Gemini Final: "The Tale of Two Tails")
        # Low utility (<2%) = Zipf-like (random noise) -> don't trust ghost
        # Medium utility (2-15%) = Mixed pattern -> trust ghost moderately
        # High utility (>15%) = Strong loop -> be STRICT, loop items will return anyway
        #
        # Key insight: For strong loops (high utility), items WILL come back.
        # Don't rush to admit on first return. Wait for genuine frequency buildup.
        # This is why TinyLFU wins on Hill - strict ">" prevents churn.
        ghost_is_useful = self.ghost_utility > 0.02
        ghost_is_loop = self.ghost_utility > 0.12 and self.is_flat_variance  # v1.2: require flat variance

        # v1.2 FIX: High ghost utility + NOT flat variance = temporal locality
        # Items return frequently but aren't cycling in a loop pattern
        is_temporal_locality = self.ghost_utility > 0.15 and not self.is_flat_variance
        is_first_time = k_freq <= 1  # Item hasn't been accessed while in cache

        # REFINED TIE-BREAKING (Gemini's insight):
        # In high-variance (Zipf) mode, freq=1 items are "one-hit wonders" - reject ties.
        # In low-variance + high ghost utility (loops), items will return - be patient.
        # For strong loops: NO tie-breaks at all. TinyLFU-style strict ">" only.
        # This prevents churn - items will return and build genuine frequency.
        if ghost_is_loop:
            allow_weak_tie_break = False  # Loop detected: be strict like TinyLFU
        else:
            allow_weak_tie_break = (
                not self.is_high_variance  # Low/medium variance: allow ties
                or k_freq > 1  # Already has frequency evidence
                or ghost_is_useful  # Ghost is proving useful
            )

        # v1.2 FIX: Trust recency for first-time items in temporal locality patterns
        if is_temporal_locality and is_first_time:
            # Lenient comparison for first-time items in temporal locality
            should_admit = k_freq >= v_freq or v_freq <= 2  # Can beat low-freq victims
        elif self.mode == 'SCAN':
            # SCAN MODE: Trust Ghost when utility is useful (but not for strong loops)
            if k_freq > v_freq:
                should_admit = True
            elif k_freq == v_freq and has_returning_support and not ghost_is_loop:
                # Ghost tie-break: only if freq > 1 AND not in a strong loop
                should_admit = True
        elif self.mode == 'FREQ':
            # Strict for Zipf workloads - but respect ghost when it's useful
            if k_freq > v_freq:
                should_admit = True
            elif k_freq == v_freq and has_returning_support and allow_weak_tie_break:
                should_admit = True
        elif self.mode == 'RECENCY':
            # LRU-like: always admit (recency wins for shifting workloads)
            should_admit = True
        else:  # MIXED
            # Moderate with conditional tie-breaker
            if k_freq > v_freq:
                should_admit = True
            elif k_freq == v_freq and has_returning_support and allow_weak_tie_break:
                should_admit = True

        if should_admit:
            self._add_ghost(victim)
            self.main[k] = 1
        else:
            # Put victim back, reject newcomer
            self.main[victim] = 1
            self._add_ghost(k)

    def _add_ghost(self, k):
        """Add evicted item to ghost buffer with timestamp."""
        if len(self.ghost) >= self.ghost_cap:
            old, _ = self.ghost.popitem(last=False)
            self.freq.pop(old, None)
        # Store (frequency, eviction_time) for smarter readmission
        self.ghost[k] = (self.freq.get(k, 0), self.ops)

    def _detect_mode(self):
        """Detect workload pattern and adjust configuration."""
        total = self.hits + self.misses
        if total < 100:
            return

        hit_rate = self.hits / total
        unique_rate = self.unique_count / total if total > 0 else 0

        # WARMUP PROTECTION: Don't switch to strict modes too early
        # Need at least 5 detection periods to stabilize
        warmup_complete = self.ops > self.check_interval * 5

        # Variance detection using WINDOW-based metrics (immune to decay corruption)
        # High ratio = Zipf (power law), Low ratio = Loop/Sequential (uniform)
        n_unique = len(self.window_uniques)
        if n_unique > 0 and self.window_accesses > 0:
            avg_freq = self.window_accesses / n_unique
            variance_ratio = self.max_freq_seen / avg_freq if avg_freq > 0 else 0
            # Gemini: >15 = Zipf (power law), <5 = Loop/Sequential (flat)
            self.is_high_variance = variance_ratio > 15
            self.is_flat_variance = variance_ratio < 5
        else:
            self.is_high_variance = False
            self.is_flat_variance = False  # v1.2: Don't assume loop until proven

        # Calculate Ghost Utility (Gemini Final: "The Tale of Two Tails")
        # High utility = Loop (ghost returning items) -> trust ghost even in strict mode
        # Low utility = Zipf (random noise) -> ignore ghost in strict mode
        if self.ghost_lookups > 0:
            self.ghost_utility = self.ghost_hits / self.ghost_lookups
        else:
            self.ghost_utility = 0.0

        # DEBUG: Log variance and ghost utility for troubleshooting (uncomment if needed)
        # if n_unique > 0 and self.ops < self.check_interval * 15:
        #     print(f"[DEBUG] ops={self.ops} mode={self.mode} var={variance_ratio:.1f} hit={hit_rate:.2f} high={self.is_high_variance} flat={self.is_flat_variance} ghost_util={self.ghost_utility:.1%}")

        # Reset window metrics for next detection period
        self.window_uniques.clear()
        self.window_freq.clear()
        self.window_accesses = 0
        self.max_freq_seen = 0

        old_mode = self.mode

        # LOOP DETECTION: High ghost utility (>12%) AND flat variance = strong loop pattern
        # v1.2 FIX: Corda has high ghost utility (33%) but is NOT a loop (has variance)
        # Force TinyLFU-like behavior only for true loops
        is_loop_pattern = self.ghost_utility > 0.12 and self.is_flat_variance

        # v1.2 FIX: Detect when our strategy is failing
        # If ghost utility is high (items returning) but hit rate is very low (we're not catching them),
        # then our current approach isn't working - switch to RECENCY regardless of variance
        # This handles Corda which has flat variance (looks like loop) but isn't a loop
        is_strategy_failing = (
            self.ghost_utility > 0.20 and  # Items are definitely returning
            hit_rate < 0.05                # But we're getting almost no hits
        )

        # WARMUP: Use permissive MIXED mode until we have enough data
        if not warmup_complete:
            self.mode = 'MIXED'
            self.win_cap = max(1, self.cap // 10)
        # v1.2 PRIORITY 0.5: Strategy failing - items return but we can't catch them
        # This overrides loop detection because if ghost utility is high but hits are near-zero,
        # our current approach is wrong regardless of variance
        elif is_strategy_failing:
            self.mode = 'RECENCY'  # Trust recency, admit new items
            self.win_cap = max(1, self.cap // 3)  # Large 33% window for probation
        # PRIORITY 0 (NEW): Ghost Loop Override - acts like TinyLFU
        # When ghost buffer is proving very useful (loop), use strict mode
        elif is_loop_pattern:
            self.mode = 'SCAN'  # Strict like TinyLFU
            self.win_cap = max(1, self.cap // 100)  # Tiny 1% window
        # PRIORITY 1: High Variance Override (Targets Zipf-0.8 & 0.99)
        elif self.is_high_variance:
            self.mode = 'FREQ'  # Strict (>), Tie-Break only if k > 1
            self.win_cap = max(1, self.cap // 100)
        # PRIORITY 2: Low Hit Rate (Targets CloudPhysics, Hill, Scans, Loops)
        # KEY INSIGHT: Low hit_rate needs TINY window (1%) like TinyLFU.
        elif hit_rate < 0.20:
            self.mode = 'SCAN'  # Always use SCAN for low hit_rate
            self.win_cap = max(1, self.cap // 100)  # TINY 1% window like TinyLFU
        # PRIORITY 3: Low Uniqueness (Targets Repeating Loops/Temporal)
        # MUST come before hit_rate > 0.40 to catch TEMPORAL!
        elif unique_rate < 0.3:
            self.mode = 'RECENCY'  # Admit All (True)
            self.win_cap = max(1, self.cap // 5)
        # PRIORITY 4: High Hit Rate (Targets Stable Web Traffic)
        # Gemini: 40% hit rate = stable regime, protect it
        elif hit_rate > 0.40:
            self.mode = 'FREQ'  # Strict (>), Tie-Break only if k > 1
            self.win_cap = max(1, self.cap // 20)
        # PRIORITY 5: Middle Ground (Targets Twitter, Mixed)
        else:
            self.mode = 'MIXED'  # Lenient (>=), Tie-Break Allowed
            self.win_cap = max(1, self.cap // 10)

        # Adjust main capacity
        self.main_cap = self.cap - self.win_cap

        # Reset stats for next detection period
        self.hits = 0
        self.misses = 0
        self.unique_count = 0
        self.ghost_hits = 0
        self.ghost_lookups = 0

    def _decay(self):
        """
        Periodic frequency decay with skip-decay enhancement.

        Skip decay when hit rate is high (>40%) to prevent churn
        in stable phases.
        """
        # Skip decay if cache is performing well
        if self.recent_hit_rate > 0.40:
            self.ops = 0
            self.last_check = 0
            self.last_skip_decay_reset = 0
            return

        # Normal decay when cache is struggling
        self.ops = 0
        self.last_check = 0
        self.last_skip_decay_reset = 0
        self.freq = {k: v >> 1 for k, v in self.freq.items() if v > 1}


@register("wgs-freq", "WGS-Freq - strict frequency admission like TinyLFU")
class WGSFreq:
    """WGS with TinyLFU-style strict frequency admission.

    Key difference: Even with ghost hit, we still compare frequencies.
    This prevents one-time ghost hits from polluting the cache.
    """
    __slots__ = ('cap', 'win_cap', 'main_cap', 'ghost_cap', 'decay_at',
                 'window', 'main', 'visited', 'ghost', 'freq', 'hand', 'ops',
                 'min_freq')

    def __init__(self, cap, win_pct=0.01, ghost_pct=0.50, decay_mult=10, min_freq=2):
        self.cap = cap
        self.win_cap = max(1, int(cap * win_pct))
        self.main_cap = cap - self.win_cap
        self.ghost_cap = max(1, int(cap * ghost_pct))
        self.decay_at = cap * decay_mult
        self.min_freq = min_freq  # Minimum frequency for admission

        self.window = OrderedDict()
        self.main = OrderedDict()
        self.visited = {}
        self.ghost = OrderedDict()  # key -> saved_freq
        self.freq = {}
        self.hand = None
        self.ops = 0

    def access(self, k):
        self.ops += 1
        self.freq[k] = min(15, self.freq.get(k, 0) + 1)

        if k in self.window:
            self.window.move_to_end(k)
            return True
        if k in self.main:
            self.visited[k] = True
            return True

        # Miss - ghost hit gives frequency boost but still compares
        if k in self.ghost:
            saved_freq = self.ghost.pop(k)
            self.freq[k] = min(15, self.freq.get(k, 0) + saved_freq)

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

        # Strict: must have minimum frequency to even be considered
        if k_freq < self.min_freq:
            return

        if len(self.main) < self.main_cap:
            self.main[k] = 1
            self.visited[k] = False
            return

        victim = self._find_victim()
        if victim is None:
            return

        v_freq = self.freq.get(victim, 0)
        if k_freq > v_freq:  # Strict: must be GREATER than, not >=
            del self.main[victim]
            self.visited.pop(victim, None)
            self._add_ghost(victim)
            self.main[k] = 1
            self.visited[k] = False

    def _find_victim(self):
        if not self.main:
            return None
        keys = list(self.main.keys())
        n = len(keys)
        if self.hand not in self.main:
            self.hand = keys[0]

        idx = keys.index(self.hand)
        for _ in range(n * 2):
            k = keys[idx]
            if self.visited.get(k):
                self.visited[k] = False
            else:
                self.hand = keys[(idx + 1) % n] if n > 1 else None
                return k
            idx = (idx + 1) % n
        return self.hand

    def _add_ghost(self, k):
        if len(self.ghost) >= self.ghost_cap:
            old, _ = self.ghost.popitem(last=False)
            self.freq.pop(old, None)
        self.ghost[k] = self.freq.get(k, 0)  # Save frequency

    def _decay(self):
        self.ops = 0
        self.freq = {k: v >> 1 for k, v in self.freq.items() if v > 1}


@register("wgs-adapt", "WGS-Adaptive - adjusts parameters based on hit rate")
class WGSAdaptive:
    """WGS that adapts window size based on hit rate.

    If hit rate is low, shrink window (more space for main).
    If hit rate is high, keep current or grow window slightly.
    """
    __slots__ = ('cap', 'main_cap', 'ghost_cap', 'decay_at',
                 'window', 'main', 'visited', 'ghost', 'freq', 'hand', 'ops',
                 'win_min', 'win_max', 'win_cap', 'hits', 'accesses', 'last_adjust')

    def __init__(self, cap, ghost_pct=0.50, decay_mult=10):
        self.cap = cap
        self.win_min = max(1, cap // 100)      # 1% min
        self.win_max = max(1, cap // 5)        # 20% max
        self.win_cap = max(1, cap // 20)       # Start at 5%
        self.main_cap = cap - self.win_cap
        self.ghost_cap = max(1, int(cap * ghost_pct))
        self.decay_at = cap * decay_mult

        self.window = OrderedDict()
        self.main = OrderedDict()
        self.visited = {}
        self.ghost = OrderedDict()
        self.freq = {}
        self.hand = None
        self.ops = 0

        # Adaptation tracking
        self.hits = 0
        self.accesses = 0
        self.last_adjust = 0

    def access(self, k):
        self.ops += 1
        self.accesses += 1
        self.freq[k] = min(15, self.freq.get(k, 0) + 1)

        hit = False
        if k in self.window:
            self.window.move_to_end(k)
            hit = True
        elif k in self.main:
            self.visited[k] = True
            hit = True

        if hit:
            self.hits += 1
        else:
            if k in self.ghost:
                del self.ghost[k]
                self._promote_from_ghost(k)
            else:
                self._add_to_window(k)

        # Periodic adaptation
        if self.ops - self.last_adjust >= self.cap:
            self._adapt()
            self.last_adjust = self.ops

        if self.ops >= self.decay_at:
            self._decay()
        return hit

    def _adapt(self):
        """Adjust window size based on recent hit rate."""
        if self.accesses < 100:
            return

        hit_rate = self.hits / self.accesses
        old_win = self.win_cap

        if hit_rate < 0.1:  # Very low hit rate - shrink window
            self.win_cap = max(self.win_min, self.win_cap - self.cap // 100)
        elif hit_rate > 0.5:  # High hit rate - grow window slightly
            self.win_cap = min(self.win_max, self.win_cap + self.cap // 200)

        if self.win_cap != old_win:
            self.main_cap = self.cap - self.win_cap

        # Reset tracking
        self.hits = 0
        self.accesses = 0

    def _add_to_window(self, k):
        while len(self.window) >= self.win_cap:
            evicted, _ = self.window.popitem(last=False)
            self._try_promote(evicted)
        self.window[k] = 1

    def _try_promote(self, k):
        if len(self.main) < self.main_cap:
            self.main[k] = 1
            self.visited[k] = False
            return

        victim = self._find_victim()
        if victim is None:
            return

        if self.freq.get(k, 0) >= self.freq.get(victim, 0):
            del self.main[victim]
            self.visited.pop(victim, None)
            self._add_ghost(victim)
            self.main[k] = 1
            self.visited[k] = False
        else:
            self._add_ghost(k)

    def _promote_from_ghost(self, k):
        if len(self.main) < self.main_cap:
            self.main[k] = 1
            self.visited[k] = False
            return

        victim = self._find_victim()
        if victim:
            del self.main[victim]
            self.visited.pop(victim, None)
            self._add_ghost(victim)
            self.main[k] = 1
            self.visited[k] = False

    def _find_victim(self):
        if not self.main:
            return None
        keys = list(self.main.keys())
        n = len(keys)
        if self.hand not in self.main:
            self.hand = keys[0]

        idx = keys.index(self.hand)
        for _ in range(n * 2):
            k = keys[idx]
            if self.visited.get(k):
                self.visited[k] = False
            else:
                self.hand = keys[(idx + 1) % n] if n > 1 else None
                return k
            idx = (idx + 1) % n
        return self.hand

    def _add_ghost(self, k):
        if len(self.ghost) >= self.ghost_cap:
            old, _ = self.ghost.popitem(last=False)
            self.freq.pop(old, None)
        self.ghost[k] = 1

    def _decay(self):
        self.ops = 0
        self.freq = {k: v >> 1 for k, v in self.freq.items() if v > 1}


@register("wgs", "WGS-SIEVE - ghost buffer + frequency admission")
class WGS:
    __slots__ = ('cap', 'win_cap', 'main_cap', 'ghost_cap', 'decay_at',
                 'window', 'main', 'visited', 'ghost', 'freq', 'hand', 'ops')

    def __init__(self, cap, win_pct=0.01, ghost_pct=0.26, decay_mult=10):
        self.cap = cap
        self.win_cap = max(1, int(cap * win_pct))
        self.main_cap = cap - self.win_cap
        self.ghost_cap = max(1, int(cap * ghost_pct))
        self.decay_at = cap * decay_mult

        self.window = OrderedDict()
        self.main = OrderedDict()
        self.visited = {}
        self.ghost = OrderedDict()
        self.freq = {}
        self.hand = None
        self.ops = 0

    def access(self, k):
        self.ops += 1
        self.freq[k] = min(15, self.freq.get(k, 0) + 1)

        if k in self.window:
            self.window.move_to_end(k)
            return True
        if k in self.main:
            self.visited[k] = True
            return True

        # Miss
        if k in self.ghost:
            del self.ghost[k]
            self._promote_from_ghost(k)
        else:
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
        if len(self.main) < self.main_cap:
            self.main[k] = 1
            self.visited[k] = False
            return

        victim = self._find_victim()
        if victim is None:
            return

        if self.freq.get(k, 0) >= self.freq.get(victim, 0):
            del self.main[victim]
            self.visited.pop(victim, None)
            self._add_ghost(victim)
            self.main[k] = 1
            self.visited[k] = False
        else:
            self._add_ghost(k)

    def _promote_from_ghost(self, k):
        """Ghost hit = proven valuable, always admit."""
        if len(self.main) < self.main_cap:
            self.main[k] = 1
            self.visited[k] = False
            return

        victim = self._find_victim()
        if victim:
            del self.main[victim]
            self.visited.pop(victim, None)
            self._add_ghost(victim)
            self.main[k] = 1
            self.visited[k] = False

    def _find_victim(self):
        if not self.main:
            return None
        keys = list(self.main.keys())
        n = len(keys)
        if self.hand not in self.main:
            self.hand = keys[0]

        idx = keys.index(self.hand)
        for _ in range(n * 2):
            k = keys[idx]
            if self.visited.get(k):
                self.visited[k] = False
            else:
                self.hand = keys[(idx + 1) % n] if n > 1 else None
                return k
            idx = (idx + 1) % n
        return self.hand

    def _add_ghost(self, k):
        if len(self.ghost) >= self.ghost_cap:
            old, _ = self.ghost.popitem(last=False)
            self.freq.pop(old, None)
        self.ghost[k] = 1

    def _decay(self):
        self.ops = 0
        self.freq = {k: v >> 1 for k, v in self.freq.items() if v > 1}


# WGS Variants
@register("wgs-nowin", "WGS without window (pure SIEVE+ghost)")
class WGSNoWin(WGS):
    def __init__(self, cap): super().__init__(cap, win_pct=0.0)

@register("wgs-win5", "WGS with 5% window")
class WGSWin5(WGS):
    def __init__(self, cap): super().__init__(cap, win_pct=0.05)

@register("wgs-ghost40", "WGS with 40% ghost")
class WGSGhost40(WGS):
    def __init__(self, cap): super().__init__(cap, ghost_pct=0.40)


# ============================================================================
# RECIPE SYSTEM - Parameterized WGS variants for auto-tuning
# ============================================================================

class WGSRecipe:
    """Parameterized WGS for recipe tuning."""
    __slots__ = ('cap', 'win_cap', 'main_cap', 'ghost_cap', 'decay_at',
                 'window', 'main', 'visited', 'ghost', 'freq', 'hand', 'ops',
                 'freq_aging', 'aging_threshold', 'tiered_ghost', 'prob_admit')

    def __init__(self, cap, win_pct=0.01, ghost_pct=0.26, decay_mult=10,
                 freq_aging=False, aging_threshold=1000,
                 tiered_ghost=False, prob_admit=False):
        self.cap = cap
        self.win_cap = max(1, int(cap * win_pct)) if win_pct > 0 else 0
        self.main_cap = cap - self.win_cap
        self.ghost_cap = max(1, int(cap * ghost_pct))
        self.decay_at = cap * decay_mult

        self.window = OrderedDict()
        self.main = OrderedDict()
        self.visited = {}
        self.ghost = OrderedDict()  # key -> (freq, evict_count) for tiered
        self.freq = {}
        self.hand = None
        self.ops = 0

        # Recipe flags
        self.freq_aging = freq_aging
        self.aging_threshold = aging_threshold
        self.tiered_ghost = tiered_ghost
        self.prob_admit = prob_admit

    def access(self, k):
        self.ops += 1
        self.freq[k] = min(15, self.freq.get(k, 0) + 1)

        if self.win_cap > 0 and k in self.window:
            self.window.move_to_end(k)
            return True
        if k in self.main:
            self.visited[k] = True
            return True

        # Miss
        if k in self.ghost:
            ghost_data = self.ghost.pop(k)
            self._promote_from_ghost(k, ghost_data)
        else:
            if self.win_cap > 0:
                self._add_to_window(k)
            else:
                self._add_to_main(k)

        if self.ops >= self.decay_at:
            self._decay()
        return False

    def _add_to_window(self, k):
        if len(self.window) >= self.win_cap:
            evicted, _ = self.window.popitem(last=False)
            self._try_promote(evicted)
        self.window[k] = 1

    def _add_to_main(self, k):
        if len(self.main) >= self.main_cap:
            victim = self._find_victim()
            if victim:
                del self.main[victim]
                self.visited.pop(victim, None)
                self._add_ghost(victim)
        self.main[k] = 1
        self.visited[k] = False

    def _try_promote(self, k):
        if len(self.main) < self.main_cap:
            self.main[k] = 1
            self.visited[k] = False
            return

        victim = self._find_victim()
        if victim is None:
            return

        k_freq = self.freq.get(k, 0)
        v_freq = self.freq.get(victim, 0)

        # Frequency aging: penalize old high-freq items
        if self.freq_aging and self.ops > self.aging_threshold:
            # Decay frequency of items not seen recently
            pass  # Already handled in _decay

        # Probabilistic admission
        if self.prob_admit and k_freq < v_freq:
            # Small chance to admit anyway
            if random.random() > 0.1:  # 90% reject
                self._add_ghost(k)
                return

        if k_freq >= v_freq:
            del self.main[victim]
            self.visited.pop(victim, None)
            self._add_ghost(victim)
            self.main[k] = 1
            self.visited[k] = False
        else:
            self._add_ghost(k)

    def _promote_from_ghost(self, k, ghost_data):
        """Ghost hit = proven valuable."""
        if len(self.main) < self.main_cap:
            self.main[k] = 1
            self.visited[k] = False
            return

        victim = self._find_victim()
        if victim:
            # Tiered ghost: boost priority based on eviction count
            if self.tiered_ghost and isinstance(ghost_data, tuple):
                evict_count = ghost_data[1]
                self.freq[k] = min(15, self.freq.get(k, 0) + evict_count)

            del self.main[victim]
            self.visited.pop(victim, None)
            self._add_ghost(victim)
            self.main[k] = 1
            self.visited[k] = False

    def _find_victim(self):
        if not self.main:
            return None
        keys = list(self.main.keys())
        n = len(keys)
        if self.hand not in self.main:
            self.hand = keys[0]

        idx = keys.index(self.hand)
        for _ in range(n * 2):
            k = keys[idx]
            if self.visited.get(k):
                self.visited[k] = False
            else:
                self.hand = keys[(idx + 1) % n] if n > 1 else None
                return k
            idx = (idx + 1) % n
        return self.hand

    def _add_ghost(self, k):
        if len(self.ghost) >= self.ghost_cap:
            old, _ = self.ghost.popitem(last=False)
            self.freq.pop(old, None)

        if self.tiered_ghost:
            # Track eviction count
            old_data = self.ghost.get(k)
            evict_count = (old_data[1] + 1) if isinstance(old_data, tuple) else 1
            self.ghost[k] = (self.freq.get(k, 0), evict_count)
        else:
            self.ghost[k] = self.freq.get(k, 0)

    def _decay(self):
        self.ops = 0
        self.freq = {k: v >> 1 for k, v in self.freq.items() if v > 1}


# Recipe definitions: (name, description, kwargs)
RECIPES = [
    # =========================================================================
    # USE CASE: DB-SCAN (Database / Block I/O)
    # =========================================================================
    # Key insight: NO WINDOW - scans pollute window badly (-47pp on LOOP!)
    # Large ghost remembers scanned items, fast decay forgets old patterns

    # Core DB recipes
    ("db-pure", "DB: Pure SIEVE", dict(win_pct=0.0, ghost_pct=0.0)),
    ("db-ghost", "DB: Max ghost", dict(win_pct=0.0, ghost_pct=1.0)),
    ("db-fast", "DB: Fast decay", dict(win_pct=0.0, ghost_pct=0.75, decay_mult=3)),
    ("db-scan1", "DB: Scan resist v1", dict(win_pct=0.0, ghost_pct=1.0, decay_mult=3)),
    ("db-scan2", "DB: Scan resist v2", dict(win_pct=0.0, ghost_pct=0.50, tiered_ghost=True)),
    ("db-loop", "DB: Loop optimized", dict(win_pct=0.0, ghost_pct=0.75, decay_mult=5)),

    # =========================================================================
    # USE CASE: WEB-CDN (Web / CDN Caching)
    # =========================================================================
    # Key insight: Large window (20%) catches trending, tiered ghost protects hot
    # Slow decay keeps popular items, frequency matters

    # Core Web recipes
    ("web-trend", "WEB: Trending", dict(win_pct=0.20, ghost_pct=0.50, tiered_ghost=True)),
    ("web-hot", "WEB: Hot items", dict(win_pct=0.15, ghost_pct=0.75, tiered_ghost=True)),
    ("web-pop", "WEB: Popular", dict(win_pct=0.20, ghost_pct=0.40, decay_mult=20)),
    ("web-cdn1", "WEB: CDN v1", dict(win_pct=0.25, ghost_pct=0.50, tiered_ghost=True)),
    ("web-cdn2", "WEB: CDN v2", dict(win_pct=0.20, ghost_pct=0.75, decay_mult=15)),
    ("web-freq", "WEB: Freq focus", dict(win_pct=0.10, ghost_pct=1.0, tiered_ghost=True, decay_mult=20)),

    # =========================================================================
    # USE CASE: MEMORY-CACHE (Redis / Memcached)
    # =========================================================================
    # Key insight: LRU-like (large window), aggressive freq aging
    # Recent > frequent, fast adaptation to shifts

    # Core Memory recipes
    ("mem-lru", "MEM: LRU-like", dict(win_pct=0.30, ghost_pct=0.20, decay_mult=3)),
    ("mem-sess", "MEM: Sessions", dict(win_pct=0.35, ghost_pct=0.15, freq_aging=True, decay_mult=3)),
    ("mem-ttl", "MEM: TTL-like", dict(win_pct=0.40, ghost_pct=0.10, freq_aging=True, decay_mult=2)),
    ("mem-shift", "MEM: Shift adapt", dict(win_pct=0.25, ghost_pct=0.25, freq_aging=True, decay_mult=5)),
    ("mem-redis", "MEM: Redis-like", dict(win_pct=0.30, ghost_pct=0.30, freq_aging=True)),
    ("mem-fast", "MEM: Fast adapt", dict(win_pct=0.35, ghost_pct=0.20, decay_mult=2)),

    # =========================================================================
    # USE CASE: CLOUD-STORE (Cloud / Object Storage)
    # =========================================================================
    # Key insight: Balanced approach, medium everything, no extremes

    # Core Cloud recipes
    ("cloud-bal", "CLOUD: Balanced", dict(win_pct=0.10, ghost_pct=0.40)),
    ("cloud-std", "CLOUD: Standard", dict(win_pct=0.05, ghost_pct=0.50, decay_mult=10)),
    ("cloud-obj", "CLOUD: Object store", dict(win_pct=0.08, ghost_pct=0.45, tiered_ghost=True)),
    ("cloud-mix", "CLOUD: Mixed", dict(win_pct=0.12, ghost_pct=0.35, freq_aging=True)),
    ("cloud-s3", "CLOUD: S3-like", dict(win_pct=0.10, ghost_pct=0.50, tiered_ghost=True)),

    # =========================================================================
    # PARAMETER SWEEPS (for discovery)
    # =========================================================================

    # Window sweep
    ("w0", "No window", dict(win_pct=0.0)),
    ("w1", "1% window", dict(win_pct=0.01)),
    ("w5", "5% window", dict(win_pct=0.05)),
    ("w10", "10% window", dict(win_pct=0.10)),
    ("w15", "15% window", dict(win_pct=0.15)),
    ("w20", "20% window", dict(win_pct=0.20)),
    ("w25", "25% window", dict(win_pct=0.25)),
    ("w30", "30% window", dict(win_pct=0.30)),

    # Ghost sweep
    ("g0", "No ghost", dict(ghost_pct=0.0)),
    ("g25", "25% ghost", dict(ghost_pct=0.25)),
    ("g50", "50% ghost", dict(ghost_pct=0.50)),
    ("g75", "75% ghost", dict(ghost_pct=0.75)),
    ("g100", "100% ghost", dict(ghost_pct=1.0)),

    # Decay sweep
    ("d2", "2x decay", dict(decay_mult=2)),
    ("d5", "5x decay", dict(decay_mult=5)),
    ("d10", "10x decay", dict(decay_mult=10)),
    ("d20", "20x decay", dict(decay_mult=20)),

    # =========================================================================
    # HYBRID / EXPERIMENTAL
    # =========================================================================

    # All-rounder attempts
    ("hybrid-a", "Hybrid A", dict(win_pct=0.10, ghost_pct=0.60, tiered_ghost=True, decay_mult=10)),
    ("hybrid-b", "Hybrid B", dict(win_pct=0.05, ghost_pct=0.75, freq_aging=True, decay_mult=8)),
    ("hybrid-c", "Hybrid C", dict(win_pct=0.15, ghost_pct=0.50, tiered_ghost=True, freq_aging=True)),

    # Aggressive experiments
    ("exp-maxg", "EXP: Max ghost", dict(win_pct=0.0, ghost_pct=1.5)),
    ("exp-maxw", "EXP: Max window", dict(win_pct=0.40, ghost_pct=0.30)),
    ("exp-all", "EXP: All features", dict(win_pct=0.10, ghost_pct=0.50, tiered_ghost=True, freq_aging=True, prob_admit=True)),

    # =========================================================================
    # GRID SEARCH OPTIMIZED (discovered via systematic parameter search)
    # =========================================================================
    # These beat SOTA baselines on their target workloads

    # Winners (beat baselines)
    ("grid-twitter", "GRID: Twitter +1.24pp", dict(win_pct=0.25, ghost_pct=0.5, decay_mult=2, tiered_ghost=True)),
    ("grid-loop1", "GRID: LOOP-1 +1.18pp", dict(win_pct=0.0, ghost_pct=0.2, decay_mult=15, tiered_ghost=False)),
    ("grid-zipf12", "GRID: ZIPF-1.2 +0.43pp", dict(win_pct=0.01, ghost_pct=0.2, decay_mult=20, tiered_ghost=False)),
    ("grid-hill", "GRID: Hill +0.13pp", dict(win_pct=0.1, ghost_pct=1.0, decay_mult=15, tiered_ghost=True)),
    ("grid-cloud", "GRID: Cloud +0.02pp", dict(win_pct=0.2, ghost_pct=0.3, decay_mult=15, tiered_ghost=True)),

    # Near-ties (within 0.5pp of baseline)
    ("grid-zipf099", "GRID: ZIPF-0.99", dict(win_pct=0.01, ghost_pct=0.2, decay_mult=15, tiered_ghost=False)),
    ("grid-zipf08", "GRID: ZIPF-0.8", dict(win_pct=0.01, ghost_pct=0.2, decay_mult=10, tiered_ghost=True)),
]


def run_recipe(trace: list, cache_size: int, recipe_kwargs: dict) -> float:
    """Run a single recipe on trace, return hit rate %."""
    cache = WGSRecipe(cache_size, **recipe_kwargs)
    hits = sum(1 for k in trace if cache.access(k))
    return hits / len(trace) * 100


# ============================================================================
# TRACE LOADERS
# ============================================================================

def load_trace(name: str, limit: int = 0) -> list | None:
    """Load a trace by name."""
    loaders = {
        "hill": ("hill-cache/traces/example.lis", _parse_hill),
        "cloud": ("libCacheSim/data/cloudPhysicsIO.csv", _parse_cloud),
        "twitter": ("libCacheSim/data/twitter_cluster52.csv", _parse_twitter),
    }

    if name not in loaders:
        return None

    path, parser = loaders[name]
    try:
        return parser(path, limit)
    except FileNotFoundError:
        return None

def _parse_hill(path, limit):
    keys = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if parts:
                keys.append(int(parts[0]))
                if limit and len(keys) >= limit:
                    break
    return keys

def _parse_cloud(path, limit):
    keys = []
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.split(',')
            if len(parts) >= 5:
                keys.append(int(parts[4]))
                if limit and len(keys) >= limit:
                    break
    return keys

def _parse_twitter(path, limit):
    keys = []
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                keys.append(hash(parts[1]) & 0x7FFFFFFF)
                if limit and len(keys) >= limit:
                    break
    return keys


# ============================================================================
# SYNTHETIC WORKLOADS
# ============================================================================

def gen_loop(cache_size: int, n: int, extra: int = 1) -> list:
    """Loop pattern: cycles through cache_size + extra items."""
    loop_len = cache_size + extra
    return [i % loop_len for i in range(n)]

def gen_zipf(n_items: int, n: int, alpha: float = 0.99) -> list:
    """Zipfian distribution."""
    weights = [1.0 / (i + 1) ** alpha for i in range(n_items)]
    total = sum(weights)
    weights = [w / total for w in weights]
    return random.choices(range(n_items), weights=weights, k=n)

def gen_temporal(n_items: int, n: int, phases: int = 5) -> list:
    """Temporal locality: hot set changes over time."""
    keys = []
    per_phase = n // phases
    items_per = n_items // phases
    for p in range(phases):
        start = p * items_per
        keys.extend(random.randint(start, start + items_per - 1) for _ in range(per_phase))
    return keys

def gen_sequential(n_items: int, n: int) -> list:
    """Sequential scan."""
    return [i % n_items for i in range(n)]


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

def run_single(trace: list, cache_size: int, algo_name: str) -> float:
    """Run single algorithm on trace, return hit rate %."""
    if algo_name not in ALGORITHMS:
        return 0.0

    cls, _ = ALGORITHMS[algo_name]
    cache = cls(cache_size)
    hits = sum(1 for k in trace if cache.access(k))
    return hits / len(trace) * 100


def run_all(trace: list, cache_size: int, algos: list[str]) -> dict[str, float]:
    """Run all algorithms, return {name: hit_rate}."""
    return {name: run_single(trace, cache_size, name) for name in algos}


# ============================================================================
# MAIN
# ============================================================================

def print_table(results: dict, title: str, cfg: BenchConfig):
    """Print results table."""
    p(f"\n{c(title, Colors.BOLD, cfg)}", cfg)
    p("-" * 50, cfg)

    sorted_r = sorted(results.items(), key=lambda x: -x[1])
    best = sorted_r[0][1] if sorted_r else 0

    for i, (name, rate) in enumerate(sorted_r[:cfg.top_n], 1):
        diff = rate - best

        if i == 1:
            line = f"  {c(f'{i}.', Colors.GREEN, cfg)} {c(name, Colors.GREEN, cfg):<14} {c(f'{rate:>6.2f}%', Colors.GREEN, cfg)} {c('* BEST', Colors.YELLOW, cfg)}"
        elif diff > -1:
            line = f"  {i}. {name:<14} {rate:>6.2f}%  {c(f'{diff:+.2f}pp', Colors.GRAY, cfg)}"
        else:
            line = f"  {c(str(i), Colors.GRAY, cfg)}. {c(name, Colors.GRAY, cfg):<14} {c(f'{rate:>6.2f}%', Colors.GRAY, cfg)}  {c(f'{diff:+.2f}pp', Colors.GRAY, cfg)}"
        p(line, cfg)


# ============================================================================
# AUTO-TUNER (Smart - per workload type)
# ============================================================================

# Workload categories - organized by REAL USE CASES
WORKLOAD_TYPES = {
    # === DATABASE / BLOCK I/O ===
    # Table scans, index traversals, buffer pools
    # Key: Resist scan pollution, working set > cache
    "DB-SCAN": {
        "Hill": ("hill", "Block I/O trace"),
        "LOOP-1": ("loop", 1, "Tight loop (N+1)"),
        "LOOP-10": ("loop", 10, "Loose loop (N+10)"),
        "SEQ": ("seq", 0, "Sequential scan"),
    },

    # === WEB / CDN CACHING ===
    # Power-law popularity, trending content, long tail
    # Key: Frequency matters, protect hot items
    "WEB-CDN": {
        "Twitter": ("twitter", "Web cache trace"),
        "ZIPF-0.99": ("zipf", 0.99, "Standard web (a=0.99)"),
        "ZIPF-1.2": ("zipf", 1.2, "Heavy hitters (a=1.2)"),
    },

    # === IN-MEMORY CACHE (Redis/Memcached) ===
    # Sessions, shifting hot sets, TTL-like behavior
    # Key: Recency > frequency, adapt to shifts
    "MEMORY-CACHE": {
        "TEMPORAL-3": ("temporal", 3, "Fast shifts (3 phases)"),
        "TEMPORAL-5": ("temporal", 5, "Medium shifts (5 phases)"),
        "TEMPORAL-10": ("temporal", 10, "Slow shifts (10 phases)"),
    },

    # === CLOUD / OBJECT STORAGE ===
    # Less skewed, more uniform, mixed patterns
    # Key: Balanced approach, no extreme tuning
    "CLOUD-STORE": {
        "Cloud": ("cloud", "Cloud storage trace"),
        "ZIPF-0.8": ("zipf", 0.8, "Low skew (a=0.8)"),
    },
}


# ============================================================================
# GRID SEARCH AUTO-TUNER
# ============================================================================

def run_grid_search(cfg: BenchConfig, workload_name: str = None):
    """Systematic parameter grid search to find optimal configurations.

    Unlike the recipe-based tuner, this explores the full parameter space
    to find truly optimal configurations for each workload type.
    """
    import itertools

    start = time.time()

    p(c("\n+==================================================+", Colors.BLUE, cfg), cfg)
    p(c("|         GRID SEARCH AUTO-TUNER                   |", Colors.BLUE, cfg), cfg)
    p(c("+==================================================+", Colors.BLUE, cfg), cfg)

    # Parameter grids - coarse first, then fine-tune
    COARSE_GRID = {
        'win_pct': [0.0, 0.05, 0.10, 0.20, 0.30],
        'ghost_pct': [0.0, 0.25, 0.50, 0.75, 1.0],
        'decay_mult': [3, 10, 20],
    }

    FINE_GRID = {
        'win_pct': [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
        'ghost_pct': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0],
        'decay_mult': [2, 3, 5, 10, 15, 20],
        'tiered_ghost': [False, True],
    }

    sample_size = 20_000
    synth_size = 50_000
    C = 500

    # Load workloads
    workloads = {}
    for category, wl_items in WORKLOAD_TYPES.items():
        for name, info in wl_items.items():
            if workload_name and name != workload_name:
                continue

            trace = None
            cache_size = C
            desc = info[-1]
            trace_type = info[0]

            if trace_type in ("hill", "cloud", "twitter"):
                trace = load_trace(trace_type, sample_size)
                if trace:
                    unique = len(set(trace))
                    cache_size = max(50, int(unique * 0.10))
            elif trace_type == "zipf":
                trace = gen_zipf(C * 10, synth_size, info[1])
            elif trace_type == "loop":
                trace = gen_loop(C, synth_size, info[1])
            elif trace_type == "seq":
                trace = gen_sequential(C * 2, synth_size)
            elif trace_type == "temporal":
                trace = gen_temporal(C * 10, synth_size, info[1])

            if trace:
                workloads[name] = (trace, cache_size, category, desc)
                p(f"  Loaded: {name} ({len(trace):,} items, cache={cache_size})", cfg)

    if not workloads:
        p(c("No workloads found!", Colors.RED, cfg), cfg)
        return

    # Phase 1: Coarse grid search
    p(f"\n{c('PHASE 1: Coarse Grid Search', Colors.BOLD, cfg)}", cfg)
    coarse_combos = list(itertools.product(
        COARSE_GRID['win_pct'],
        COARSE_GRID['ghost_pct'],
        COARSE_GRID['decay_mult'],
    ))
    p(f"  Testing {len(coarse_combos)} parameter combinations...", cfg)

    coarse_results = {}  # workload -> [(params, rate), ...]

    for wl_name, (trace, cache_size, cat, desc) in workloads.items():
        results = []
        for win, ghost, decay in coarse_combos:
            params = dict(win_pct=win, ghost_pct=ghost, decay_mult=decay)
            rate = run_recipe(trace, cache_size, params)
            results.append((params, rate))
        results.sort(key=lambda x: -x[1])
        coarse_results[wl_name] = results
        p(f"  {wl_name}: best={results[0][1]:.2f}% (w={results[0][0]['win_pct']}, g={results[0][0]['ghost_pct']}, d={results[0][0]['decay_mult']})", cfg)

    # Phase 2: Fine-tune around best coarse results
    p(f"\n{c('PHASE 2: Fine-Tuning Best Regions', Colors.BOLD, cfg)}", cfg)

    fine_results = {}

    for wl_name, coarse_list in coarse_results.items():
        trace, cache_size, cat, desc = workloads[wl_name]

        # Take top 3 coarse results, explore fine grid around them
        best_regions = coarse_list[:3]
        fine_combos = set()

        for params, _ in best_regions:
            base_win = params['win_pct']
            base_ghost = params['ghost_pct']
            base_decay = params['decay_mult']

            # Fine search around this region
            for w in FINE_GRID['win_pct']:
                if abs(w - base_win) <= 0.10:  # Within ±10%
                    for g in FINE_GRID['ghost_pct']:
                        if abs(g - base_ghost) <= 0.25:  # Within ±25%
                            for d in FINE_GRID['decay_mult']:
                                for t in FINE_GRID['tiered_ghost']:
                                    fine_combos.add((w, g, d, t))

        # Test fine combinations
        results = []
        for win, ghost, decay, tiered in fine_combos:
            params = dict(win_pct=win, ghost_pct=ghost, decay_mult=decay, tiered_ghost=tiered)
            rate = run_recipe(trace, cache_size, params)
            results.append((params, rate))

        results.sort(key=lambda x: -x[1])
        fine_results[wl_name] = results[:10]  # Keep top 10

        best = results[0]
        p(f"  {wl_name}: {best[1]:.2f}% (tested {len(fine_combos)} combos)", cfg)
        p(f"    -> win={best[0]['win_pct']:.2f}, ghost={best[0]['ghost_pct']:.2f}, decay={best[0]['decay_mult']}, tiered={best[0].get('tiered_ghost', False)}", cfg)

    # Compare with baselines
    p(f"\n{c('COMPARISON WITH BASELINES', Colors.BOLD, cfg)}", cfg)

    baselines = ["lru", "sieve", "s3fifo", "tinylfu"]
    for wl_name, results in fine_results.items():
        trace, cache_size, cat, desc = workloads[wl_name]
        best_params, best_rate = results[0]

        # Get baseline rates
        baseline_rates = {}
        for algo in baselines:
            baseline_rates[algo] = run_single(trace, cache_size, algo)

        best_baseline = max(baselines, key=lambda a: baseline_rates[a])
        bl_rate = baseline_rates[best_baseline]
        diff = best_rate - bl_rate

        if diff > 0:
            diff_str = c(f"+{diff:.2f}pp", Colors.GREEN, cfg)
        else:
            diff_str = f"{diff:.2f}pp"

        p(f"  {wl_name}:", cfg)
        p(f"    Grid Search: {best_rate:.2f}% vs {best_baseline}: {bl_rate:.2f}% ({diff_str})", cfg)
        p(f"    Optimal: w={best_params['win_pct']:.2f}, g={best_params['ghost_pct']:.2f}, d={best_params['decay_mult']}, t={best_params.get('tiered_ghost', False)}", cfg)

    # Generate recipe suggestions
    p(f"\n{c('SUGGESTED NEW RECIPES', Colors.BOLD, cfg)}", cfg)
    for wl_name, results in fine_results.items():
        best = results[0][0]
        cat = workloads[wl_name][2]
        recipe_name = f"grid-{wl_name.lower().replace('-', '').replace('_', '')}"
        p(f'    ("{recipe_name}", "GRID: {wl_name}", dict(win_pct={best["win_pct"]}, ghost_pct={best["ghost_pct"]}, decay_mult={best["decay_mult"]}, tiered_ghost={best.get("tiered_ghost", False)})),', cfg)

    elapsed = time.time() - start
    p(f"\nCompleted in {elapsed:.1f}s", cfg)


def run_tune(cfg: BenchConfig, focus: str = "all"):
    """Smart auto-tuner - finds best recipe per workload type."""
    start = time.time()

    p(c("\n+==================================================+", Colors.BLUE, cfg), cfg)
    p(c("|         SMART RECIPE AUTO-TUNER                  |", Colors.BLUE, cfg), cfg)
    p(c("+==================================================+", Colors.BLUE, cfg), cfg)

    sample_size = 20_000
    synth_size = 50_000
    C = 500  # cache size for synthetic

    # Build workload list from use-case categories
    workloads = {}  # name -> (trace, cache_size, category, description)

    for category, wl_items in WORKLOAD_TYPES.items():
        if focus != "all" and focus.upper() != category:
            continue

        p(f"\n{c(f'>> {category}', Colors.BOLD, cfg)}", cfg)

        for name, info in wl_items.items():
            trace = None
            cache_size = C
            desc = info[-1]  # Last element is always description
            trace_type = info[0]

            if trace_type in ("hill", "cloud", "twitter"):
                # Real trace
                trace = load_trace(trace_type, sample_size)
                if trace:
                    unique = len(set(trace))
                    cache_size = max(50, int(unique * 0.10))
                    p(f"  {name:<12} {len(trace):,} req, {unique:,} unique, cache={cache_size}", cfg)
            elif trace_type == "zipf":
                alpha = info[1]
                trace = gen_zipf(C * 10, synth_size, alpha)
                p(f"  {name:<12} alpha={alpha} ({desc})", cfg)
            elif trace_type == "loop":
                extra = info[1]
                trace = gen_loop(C, synth_size, extra)
                p(f"  {name:<12} extra={extra} ({desc})", cfg)
            elif trace_type == "seq":
                trace = gen_sequential(C * 2, synth_size)
                p(f"  {name:<12} ({desc})", cfg)
            elif trace_type == "temporal":
                phases = info[1]
                trace = gen_temporal(C * 10, synth_size, phases)
                p(f"  {name:<12} phases={phases} ({desc})", cfg)

            if trace:
                workloads[name] = (trace, cache_size, category, desc)

    if not workloads:
        p(c("No workloads available!", Colors.RED, cfg), cfg)
        return

    # ========================================================================
    # BASELINE ALGORITHMS (SOTA)
    # ========================================================================
    p(f"\n{c('>> BASELINES (SOTA)', Colors.BOLD, cfg)}", cfg)

    baselines = ["lru", "sieve", "s3fifo", "tinylfu"]
    baseline_results = {}  # algo -> {workload: rate}

    for algo in baselines:
        rates = {}
        for wl_name, (trace, cache_size, cat, desc) in workloads.items():
            rate = run_single(trace, cache_size, algo)
            rates[wl_name] = rate
        baseline_results[algo] = rates
        avg = sum(rates.values()) / len(rates)
        p(f"  {algo:<12} avg={avg:.2f}%", cfg)

    # Find best baseline per workload
    baseline_best = {}
    for wl_name in workloads:
        best_algo = max(baselines, key=lambda a: baseline_results[a][wl_name])
        baseline_best[wl_name] = (best_algo, baseline_results[best_algo][wl_name])

    # ========================================================================
    # TEST ALL RECIPES
    # ========================================================================
    n_recipes = len(RECIPES)
    n_workloads = len(workloads)
    p(f"\n{c('Testing', Colors.BOLD, cfg)} {n_recipes} recipes x {n_workloads} workloads = {n_recipes * n_workloads} tests\n", cfg)

    # Results: recipe -> {workload: rate}
    results = {}
    # Per-workload best tracking (including baselines for reference)
    workload_best = {}
    for wl_name in workloads:
        # Start with best baseline
        workload_best[wl_name] = baseline_best[wl_name]

    for i, (recipe_name, desc, kwargs) in enumerate(RECIPES):
        rates = {}
        new_bests = []

        for wl_name, (trace, cache_size, cat, wl_desc) in workloads.items():
            rate = run_recipe(trace, cache_size, kwargs)
            rates[wl_name] = rate

            # Check if new best for this workload
            if rate > workload_best[wl_name][1]:
                workload_best[wl_name] = (recipe_name, rate)
                new_bests.append((wl_name, rate))

        results[recipe_name] = {"rates": rates, "desc": desc, "kwargs": kwargs}

        # Progress with new bests highlighted
        pct = (i + 1) / n_recipes * 100
        if new_bests:
            best_str = ", ".join(f"{wl}:{r:.1f}%" for wl, r in new_bests)
            p(f"  [{i+1:2}/{n_recipes}] {c(recipe_name, Colors.GREEN, cfg):<12} {c('NEW BEST:', Colors.YELLOW, cfg)} {best_str}", cfg)
        else:
            # Compact progress every 10 recipes
            if (i + 1) % 10 == 0 or i == n_recipes - 1:
                p(f"  [{i+1:2}/{n_recipes}] {recipe_name:<12} (no new bests)", cfg)

    # ========================================================================
    # RESULTS BY WORKLOAD TYPE
    # ========================================================================
    elapsed = time.time() - start

    p(c(f"\n+{'='*60}+", Colors.BLUE, cfg), cfg)
    p(c(f"|{'BEST RECIPES BY WORKLOAD TYPE':^60}|", Colors.BLUE, cfg), cfg)
    p(c(f"+{'='*60}+", Colors.BLUE, cfg), cfg)

    # Group by category
    categories = {}
    for wl_name, (trace, cache_size, cat, desc) in workloads.items():
        if cat not in categories:
            categories[cat] = []
        best_recipe, best_rate = workload_best[wl_name]
        categories[cat].append((wl_name, desc, best_recipe, best_rate))

    for cat, items in categories.items():
        p(f"\n{c(cat, Colors.BOLD, cfg)}", cfg)
        p(f"  {'Workload':<12} {'Best Recipe':<12} {'Rate':>8} {'vs SOTA':>10} {'SOTA Best':<12}", cfg)
        p(f"  {'-'*60}", cfg)

        for wl_name, desc, recipe, rate in sorted(items, key=lambda x: -x[3]):
            # Get best baseline for this workload
            bl_algo, bl_rate = baseline_best[wl_name]
            diff = rate - bl_rate
            diff_str = f"{diff:+.2f}pp" if diff != 0 else "same"
            if diff > 0:
                diff_str = c(diff_str, Colors.GREEN, cfg)
            elif diff < 0:
                diff_str = c(diff_str, Colors.RED, cfg)
            p(f"  {wl_name:<12} {c(recipe, Colors.GREEN, cfg):<12} {rate:>7.2f}% {diff_str:>10} {c(bl_algo, Colors.GRAY, cfg)} {bl_rate:.1f}%", cfg)

    # ========================================================================
    # PATTERN ANALYSIS
    # ========================================================================
    p(c(f"\n+{'='*60}+", Colors.BLUE, cfg), cfg)
    p(c(f"|{'PATTERN ANALYSIS':^60}|", Colors.BLUE, cfg), cfg)
    p(c(f"+{'='*60}+", Colors.BLUE, cfg), cfg)

    # Analyze winning recipes (skip baselines)
    winner_configs = {}
    for wl_name, (recipe, rate) in workload_best.items():
        if recipe and recipe in results:  # Skip baseline algorithms
            cfg_tuple = tuple(sorted(results[recipe]["kwargs"].items()))
            if cfg_tuple not in winner_configs:
                winner_configs[cfg_tuple] = []
            winner_configs[cfg_tuple].append(wl_name)

    # Count baseline wins
    baseline_wins = []
    for wl_name, (recipe, rate) in workload_best.items():
        if recipe in baselines:
            baseline_wins.append((wl_name, recipe))

    p(f"\n{c('Config patterns that win:', Colors.BOLD, cfg)}", cfg)
    for cfg_tuple, workloads_won in sorted(winner_configs.items(), key=lambda x: -len(x[1])):
        config_str = ", ".join(f"{k}={v}" for k, v in cfg_tuple)
        workloads_str = ", ".join(workloads_won)
        p(f"  [{len(workloads_won)} wins] {config_str}", cfg)
        p(f"           {c(workloads_str, Colors.GRAY, cfg)}", cfg)

    if baseline_wins:
        p(f"\n{c('Workloads where SOTA baselines win:', Colors.BOLD, cfg)}", cfg)
        for wl_name, algo in baseline_wins:
            p(f"  {wl_name:<14} -> {c(algo, Colors.YELLOW, cfg)} (no recipe beats it)", cfg)

    # ========================================================================
    # RECOMMENDATIONS
    # ========================================================================
    p(c(f"\n+{'='*60}+", Colors.BLUE, cfg), cfg)
    p(c(f"|{'RECOMMENDATIONS':^60}|", Colors.BLUE, cfg), cfg)
    p(c(f"+{'='*60}+", Colors.BLUE, cfg), cfg)

    # Find most versatile recipes (best average across all)
    avg_rates = {}
    for recipe_name, data in results.items():
        avg_rates[recipe_name] = sum(data["rates"].values()) / len(data["rates"])

    top_overall = sorted(avg_rates.items(), key=lambda x: -x[1])[:5]
    p(f"\n{c('Most versatile (best avg across all):', Colors.BOLD, cfg)}", cfg)
    for i, (recipe, avg) in enumerate(top_overall, 1):
        desc = results[recipe]["desc"]
        p(f"  {i}. {c(recipe, Colors.GREEN, cfg):<14} {avg:>6.2f}% avg  ({desc})", cfg)

    # Per-category recommendations
    p(f"\n{c('Per-category recommendations:', Colors.BOLD, cfg)}", cfg)
    for cat, items in categories.items():
        # Find recipe that wins most in this category
        recipe_wins = {}
        for wl_name, desc, recipe, rate in items:
            recipe_wins[recipe] = recipe_wins.get(recipe, 0) + 1
        if recipe_wins:
            best_for_cat = max(recipe_wins.items(), key=lambda x: x[1])
            avg_for_cat = sum(rate for _, _, r, rate in items if r == best_for_cat[0]) / best_for_cat[1]
            is_baseline = best_for_cat[0] in baselines
            color = Colors.YELLOW if is_baseline else Colors.GREEN
            label = " (SOTA)" if is_baseline else ""
            p(f"  {cat:<18} -> {c(best_for_cat[0], color, cfg)}{label} ({best_for_cat[1]}/{len(items)} wins, {avg_for_cat:.2f}% avg)", cfg)

    # ========================================================================
    # BASELINE COMPARISON SUMMARY
    # ========================================================================
    p(c(f"\n+{'='*60}+", Colors.BLUE, cfg), cfg)
    p(c(f"|{'BASELINE COMPARISON':^60}|", Colors.BLUE, cfg), cfg)
    p(c(f"+{'='*60}+", Colors.BLUE, cfg), cfg)

    # Count wins vs each baseline
    p(f"\n{c('Recipe wins vs SOTA baselines:', Colors.BOLD, cfg)}", cfg)
    for algo in baselines:
        wins = 0
        ties = 0
        losses = 0
        total_diff = 0
        for wl_name, (best_recipe, best_rate) in workload_best.items():
            bl_rate = baseline_results[algo][wl_name]
            diff = best_rate - bl_rate
            total_diff += diff
            if diff > 0.01:
                wins += 1
            elif diff < -0.01:
                losses += 1
            else:
                ties += 1
        avg_diff = total_diff / len(workloads)
        status = c("BEATING", Colors.GREEN, cfg) if wins > losses else c("LOSING", Colors.RED, cfg)
        p(f"  vs {algo:<10}: {wins}W-{ties}T-{losses}L  avg diff: {avg_diff:+.2f}pp  {status}", cfg)

    # Overall: best recipe avg vs best baseline avg
    best_recipe_avg = max(avg_rates.values())
    best_baseline_avg = max(sum(baseline_results[a][wl] for wl in workloads) / len(workloads) for a in baselines)
    overall_gain = best_recipe_avg - best_baseline_avg

    p(f"\n{c('Overall:', Colors.BOLD, cfg)}", cfg)
    p(f"  Best recipe avg:   {best_recipe_avg:.2f}%", cfg)
    p(f"  Best baseline avg: {best_baseline_avg:.2f}%", cfg)
    if overall_gain > 0:
        p(f"  Improvement:       {c(f'+{overall_gain:.2f}pp', Colors.GREEN, cfg)}", cfg)
    else:
        p(f"  Difference:        {c(f'{overall_gain:.2f}pp', Colors.RED, cfg)}", cfg)

    p(f"\n{c(f'Completed in {elapsed:.1f}s ({n_recipes * n_workloads + len(baselines) * n_workloads} tests)', Colors.GRAY, cfg)}", cfg)


def main():
    parser = argparse.ArgumentParser(
        description="Cache algorithm benchmark suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --quick                Fast test with small samples
  %(prog)s --real                 Only real-world traces
  %(prog)s -a wgs,sieve,lru       Test specific algorithms
  %(prog)s --list                 Show available algorithms
  %(prog)s --tune                 Auto-tune WGS recipes (real traces)
  %(prog)s --tune --synth         Auto-tune on synthetic traces
  %(prog)s --recipes              List available recipes
        """
    )
    parser.add_argument("--quick", action="store_true", help="Quick mode (10k samples)")
    parser.add_argument("--full", action="store_true", help="Full traces (slow)")
    parser.add_argument("--real", action="store_true", help="Only real traces")
    parser.add_argument("--synth", action="store_true", help="Only synthetic traces")
    parser.add_argument("-a", "--algos", type=str, help="Comma-separated algorithms")
    parser.add_argument("--list", action="store_true", help="List available algorithms")
    parser.add_argument("--tune", action="store_true", help="Auto-tune WGS recipes")
    parser.add_argument("--grid", action="store_true", help="Grid search for optimal parameters")
    parser.add_argument("--workload", type=str, help="Specific workload for grid search")
    parser.add_argument("--recipes", action="store_true", help="List available recipes")
    parser.add_argument("--no-color", action="store_true", help="Disable colors")
    parser.add_argument("-q", "--quiet", action="store_true", help="Less output")
    args = parser.parse_args()

    # List algorithms
    if args.list:
        print("\nAvailable algorithms:")
        print("-" * 50)
        for name, (_, desc) in sorted(ALGORITHMS.items()):
            print(f"  {name:<14} {desc}")
        print()
        return

    # List recipes
    if args.recipes:
        print("\nAvailable recipes:")
        print("-" * 60)
        for name, desc, kwargs in RECIPES:
            params = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            print(f"  {name:<14} {desc:<25} [{params}]")
        print()
        return

    # Grid search mode
    if args.grid:
        cfg = BenchConfig()
        if args.no_color:
            cfg.color = False
        if args.quiet:
            cfg.verbose = False
        run_grid_search(cfg, args.workload)
        return

    # Auto-tuner mode
    if args.tune:
        cfg = BenchConfig()
        if args.no_color:
            cfg.color = False
        if args.quiet:
            cfg.verbose = False

        # Determine focus (now using use-case categories)
        # Options: all, db-scan, web-cdn, memory-cache, cloud-store
        if args.synth and args.real:
            focus = "all"  # both
        elif args.synth:
            focus = "all"  # synth included in categories
        elif args.real:
            focus = "all"  # real included in categories
        else:
            focus = "all"  # default to all use cases

        run_tune(cfg, focus)
        return

    # Build config
    cfg = BenchConfig()

    if args.quick:
        cfg.real_limit = 10_000
        cfg.synth_size = 20_000
        cfg.cache_size = 500
    elif args.full:
        cfg.real_limit = 0
        cfg.synth_size = 200_000
        cfg.cache_size = 1000

    if args.real:
        cfg.synth = False
    if args.synth:
        cfg.real = False
    if args.algos:
        cfg.algos = [a.strip().lower() for a in args.algos.split(",")]
    if args.no_color:
        cfg.color = False
    if args.quiet:
        cfg.verbose = False

    # Select algorithms
    algos = cfg.algos if cfg.algos else list(ALGORITHMS.keys())
    algos = [a for a in algos if a in ALGORITHMS]

    if not algos:
        print("No valid algorithms specified!")
        return

    # Header
    p(c("\n+==================================================+", Colors.BLUE, cfg), cfg)
    p(c("|           CACHE BENCHMARK SUITE                  |", Colors.BLUE, cfg), cfg)
    p(c("+==================================================+", Colors.BLUE, cfg), cfg)

    mode = "quick" if args.quick else ("full" if args.full else "default")
    p(f"\nMode: {c(mode, Colors.YELLOW, cfg)} | Algorithms: {len(algos)}", cfg)

    start_time = time.time()
    all_results = {}

    # ========================================================================
    # REAL TRACES
    # ========================================================================
    if cfg.real:
        p(c("\n>> REAL-WORLD TRACES", Colors.BOLD, cfg), cfg)

        for name, label in [("hill", "Hill-Cache"), ("cloud", "CloudPhysics"), ("twitter", "Twitter")]:
            trace = load_trace(name, cfg.real_limit)
            if trace is None:
                p(f"  {label}: {c('not found', Colors.RED, cfg)}", cfg)
                continue

            unique = len(set(trace))
            cache_size = max(50, int(unique * cfg.cache_pct))

            p(f"\n  {c(label, Colors.BOLD, cfg)}: {len(trace):,} requests, {unique:,} unique, cache={cache_size}", cfg)

            results = run_all(trace, cache_size, algos)
            all_results[label] = results
            print_table(results, label, cfg)

    # ========================================================================
    # SYNTHETIC TRACES
    # ========================================================================
    if cfg.synth:
        p(c("\n>> SYNTHETIC TRACES", Colors.BOLD, cfg), cfg)

        C = cfg.cache_size
        N = cfg.synth_size

        workloads = [
            ("LOOP-N+1", gen_loop(C, N, 1)),
            ("LOOP-N+10", gen_loop(C, N, 10)),
            ("ZIPF-0.8", gen_zipf(C * 10, N, 0.8)),
            ("ZIPF-0.99", gen_zipf(C * 10, N, 0.99)),
            ("TEMPORAL", gen_temporal(C * 10, N, 5)),
            ("SEQUENTIAL", gen_sequential(C * 2, N)),
        ]

        for name, trace in workloads:
            p(f"\n  {c(name, Colors.BOLD, cfg)}: {len(trace):,} requests, cache={C}", cfg)
            results = run_all(trace, C, algos)
            all_results[name] = results
            print_table(results, name, cfg)

    # ========================================================================
    # SUMMARY
    # ========================================================================
    elapsed = time.time() - start_time

    p(c("\n+==================================================+", Colors.BLUE, cfg), cfg)
    p(c("|                    SUMMARY                       |", Colors.BLUE, cfg), cfg)
    p(c("+==================================================+", Colors.BLUE, cfg), cfg)

    if not all_results:
        p("\nNo results to summarize.", cfg)
        return

    # Aggregate scores
    real_traces = ["Hill-Cache", "CloudPhysics", "Twitter"]
    real_names = [n for n in all_results if n in real_traces]
    synth_names = [n for n in all_results if n not in real_traces]

    scores = {}
    for algo in algos:
        all_rates = [all_results[t].get(algo, 0) for t in all_results]
        real_rates = [all_results[t].get(algo, 0) for t in real_names]
        synth_rates = [all_results[t].get(algo, 0) for t in synth_names]

        scores[algo] = {
            "overall": sum(all_rates) / len(all_rates) if all_rates else 0,
            "real": sum(real_rates) / len(real_rates) if real_rates else 0,
            "synth": sum(synth_rates) / len(synth_rates) if synth_rates else 0,
            "wins": sum(1 for t in all_results if all_results[t].get(algo, 0) == max(all_results[t].values())),
        }

    # Print aggregate table
    p(f"\n{'Algorithm':<14} {'Overall':>9} {'Real':>9} {'Synth':>9} {'Wins':>6}", cfg)
    p("-" * 52, cfg)

    for algo in sorted(scores, key=lambda x: -scores[x]["overall"]):
        s = scores[algo]
        wins_str = f"{s['wins']}/{len(all_results)}"

        real_str = f"{s['real']:.2f}%" if real_names else "-"
        synth_str = f"{s['synth']:.2f}%" if synth_names else "-"

        p(f"{algo:<14} {s['overall']:>7.2f}%  {real_str:>8}  {synth_str:>8} {wins_str:>6}", cfg)

    # Winners
    p("", cfg)
    best_overall = max(scores, key=lambda x: scores[x]["overall"])
    p(f"  {c('*', Colors.YELLOW, cfg)} Best Overall: {c(best_overall, Colors.GREEN, cfg)} ({scores[best_overall]['overall']:.2f}%)", cfg)

    if real_names:
        best_real = max(scores, key=lambda x: scores[x]["real"])
        p(f"  {c('*', Colors.YELLOW, cfg)} Best Real:    {c(best_real, Colors.GREEN, cfg)} ({scores[best_real]['real']:.2f}%)", cfg)

    if synth_names:
        best_synth = max(scores, key=lambda x: scores[x]["synth"])
        p(f"  {c('*', Colors.YELLOW, cfg)} Best Synth:   {c(best_synth, Colors.GREEN, cfg)} ({scores[best_synth]['synth']:.2f}%)", cfg)

    most_wins = max(scores, key=lambda x: scores[x]["wins"])
    p(f"  {c('*', Colors.YELLOW, cfg)} Most Wins:    {c(most_wins, Colors.GREEN, cfg)} ({scores[most_wins]['wins']}/{len(all_results)})", cfg)

    p(f"\n{c(f'Completed in {elapsed:.1f}s', Colors.GRAY, cfg)}", cfg)


if __name__ == "__main__":
    main()
