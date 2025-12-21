#!/usr/bin/env python3
"""
CAFFEINE STRESS TEST
====================
Replicates Caffeine's simulator stress test configuration:
- Cache size: 512
- Trace chain: corda -> loop x5 -> corda

This tests cache behavior under phase transitions:
1. Low-locality (corda) - nearly all unique keys
2. Tight loop (loop x5) - cycling through 1011 items
3. Low-locality again (corda) - recovery after loop

Reference: https://github.com/ben-manes/caffeine/wiki/Simulator
"""

import gzip
import sys
from pathlib import Path
from collections import OrderedDict

# Add parent dir to import bench algorithms
sys.path.insert(0, str(Path(__file__).parent))
from bench import ALGORITHMS

CACHE_SIZE = 512
TRACES_DIR = Path(__file__).parent.parent / "traces"

def load_loop_trace():
    """Load LIRS loop.trace.gz - sequential integers 0-1010 looping."""
    path = TRACES_DIR / "loop.trace.gz"
    with gzip.open(path, 'rt') as f:
        return [int(line.strip()) for line in f]

def load_corda_trace():
    """Load Corda trace_vaultservice_large.gz - 16-byte binary keys."""
    path = TRACES_DIR / "trace_vaultservice_large.gz"
    with gzip.open(path, 'rb') as f:
        data = f.read()
    # Each entry is 16 bytes (UUID-like)
    keys = []
    for i in range(0, len(data), 16):
        keys.append(data[i:i+16])
    return keys

def build_chained_trace():
    """Build the stress test trace: corda → loop x5 → corda"""
    print("Loading traces...")
    corda = load_corda_trace()
    loop = load_loop_trace()

    print(f"  Corda: {len(corda):,} accesses, {len(set(corda)):,} unique keys")
    print(f"  Loop:  {len(loop):,} accesses, {len(set(loop)):,} unique keys (0-1010)")

    # Chain: corda → loop x5 → corda
    chained = []
    chained.extend(corda)
    for _ in range(5):
        chained.extend(loop)
    chained.extend(corda)

    print(f"  Chained: {len(chained):,} total accesses")
    return chained

def run_benchmark(trace, algorithms):
    """Run all algorithms on the trace."""
    results = {}

    for name in algorithms:
        if name not in ALGORITHMS:
            print(f"  Unknown algorithm: {name}")
            continue

        cls, desc = ALGORITHMS[name]
        cache = cls(CACHE_SIZE)

        hits = 0
        for key in trace:
            if cache.access(key):
                hits += 1

        hit_rate = (hits / len(trace)) * 100
        results[name] = hit_rate
        print(f"  {name:20s}: {hit_rate:6.2f}% ({hits:,} hits / {len(trace):,})")

    return results

def analyze_phases(algorithms):
    """Run each phase separately to understand behavior."""
    corda = load_corda_trace()
    loop = load_loop_trace()

    print("\n" + "="*60)
    print("PHASE ANALYSIS (each phase runs on fresh cache)")
    print("="*60)

    # Phase 1: Corda only
    print("\n[Phase 1] Corda (low locality):")
    for name in algorithms:
        if name not in ALGORITHMS:
            continue
        cls, _ = ALGORITHMS[name]
        cache = cls(CACHE_SIZE)
        hits = sum(1 for k in corda if cache.access(k))
        print(f"  {name:20s}: {hits/len(corda)*100:6.2f}%")

    # Phase 2: Loop only
    print("\n[Phase 2] Loop x1 (tight loop, 1011 items in cache of 512):")
    for name in algorithms:
        if name not in ALGORITHMS:
            continue
        cls, _ = ALGORITHMS[name]
        cache = cls(CACHE_SIZE)
        hits = sum(1 for k in loop if cache.access(k))
        print(f"  {name:20s}: {hits/len(loop)*100:6.2f}%")

    # Phase 3: Loop x5 (longer loop)
    print("\n[Phase 3] Loop x5 (sustained loop):")
    loop5 = loop * 5
    for name in algorithms:
        if name not in ALGORITHMS:
            continue
        cls, _ = ALGORITHMS[name]
        cache = cls(CACHE_SIZE)
        hits = sum(1 for k in loop5 if cache.access(k))
        print(f"  {name:20s}: {hits/len(loop5)*100:6.2f}%")

def main():
    # Test key algorithms
    algorithms = ["chameleon", "tinylfu", "tinylfu-adaptive", "arc", "lirs", "lru", "sieve"]

    print("="*60)
    print("CAFFEINE STRESS TEST")
    print("="*60)
    print(f"Cache size: {CACHE_SIZE}")
    print(f"Trace chain: corda -> loop x5 -> corda")
    print()

    # Build and run on chained trace
    trace = build_chained_trace()

    print("\n" + "="*60)
    print("FULL CHAINED BENCHMARK")
    print("="*60 + "\n")

    results = run_benchmark(trace, algorithms)

    # Sort by hit rate
    print("\n" + "="*60)
    print("RESULTS (sorted)")
    print("="*60)
    for name, rate in sorted(results.items(), key=lambda x: -x[1]):
        marker = "  <-- WINNER" if rate == max(results.values()) else ""
        print(f"  {name:20s}: {rate:6.2f}%{marker}")

    # Phase analysis
    analyze_phases(algorithms)

    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    print("""
The stress test has 3 phases:
1. Corda (936K accesses, 936K unique keys) - low locality noise
2. Loop x5 (2.5M accesses, 1011 unique keys) - tight loop stress
3. Corda again (936K accesses) - recovery after loop

With cache_size=512 and loop cycling 1011 items:
- LRU/ARC: 0% on loop (classic thrashing)
- TinyLFU: Frequency filter helps but window can hurt
- Chameleon: Basin of Leniency should detect the loop phase
""")

if __name__ == "__main__":
    main()
