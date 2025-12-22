#!/usr/bin/env python3
"""
STRESS TEST VISUALIZATION
==========================
Visualizes Chameleon's adaptive behavior during phase transitions.
Tracks: hit rate, window size, mode, ghost utility over time.

This helps diagnose why Chameleon fails on corda→loop→corda transitions.
"""

import gzip
import sys
from pathlib import Path
from collections import OrderedDict

sys.path.insert(0, str(Path(__file__).parent.parent))
from chameleon.core import ChameleonCache

CACHE_SIZE = 512
SAMPLE_INTERVAL = 1000  # Sample every N operations
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

def run_with_telemetry(trace, cache, phase_boundaries):
    """Run trace and collect telemetry data at regular intervals."""
    telemetry = {
        'ops': [],
        'hit_rate': [],
        'window_pct': [],
        'mode': [],
        'ghost_utility': [],
        'phase': [],
    }

    window_hits = 0
    window_total = 0

    for i, key in enumerate(trace):
        hit = cache.access(key)
        window_hits += 1 if hit else 0
        window_total += 1

        # Sample at intervals
        if (i + 1) % SAMPLE_INTERVAL == 0:
            # Determine current phase
            phase = "corda1"
            for boundary, name in phase_boundaries:
                if i >= boundary:
                    phase = name

            telemetry['ops'].append(i + 1)
            telemetry['hit_rate'].append(window_hits / window_total * 100 if window_total > 0 else 0)
            telemetry['window_pct'].append(cache.win_cap / cache.cap * 100)
            telemetry['mode'].append(cache.mode)
            telemetry['ghost_utility'].append(cache.ghost_utility * 100)
            telemetry['phase'].append(phase)

            # Reset window
            window_hits = 0
            window_total = 0

    return telemetry

def print_ascii_chart(telemetry, metric, title, max_val=100):
    """Print a simple ASCII chart of a metric over time."""
    values = telemetry[metric]
    phases = telemetry['phase']
    width = 60

    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")

    # Find phase boundaries for markers
    phase_changes = []
    prev_phase = None
    for i, phase in enumerate(phases):
        if phase != prev_phase:
            phase_changes.append((i, phase))
            prev_phase = phase

    # Print chart
    for i, val in enumerate(values):
        bar_len = int(val / max_val * width)
        bar = '█' * bar_len + '░' * (width - bar_len)

        # Mark phase boundaries
        marker = ""
        for idx, name in phase_changes:
            if i == idx:
                marker = f" <-- {name.upper()}"
                break

        print(f"{telemetry['ops'][i]:>8,} | {bar} | {val:6.2f}%{marker}")

    print()

def print_mode_timeline(telemetry):
    """Show mode changes over time."""
    print(f"\n{'='*70}")
    print("MODE TIMELINE")
    print(f"{'='*70}")

    prev_mode = None
    for i, mode in enumerate(telemetry['mode']):
        if mode != prev_mode:
            phase = telemetry['phase'][i]
            hit_rate = telemetry['hit_rate'][i]
            ghost_util = telemetry['ghost_utility'][i]
            win_pct = telemetry['window_pct'][i]
            print(f"  ops {telemetry['ops'][i]:>10,}: {mode:8s} (phase={phase:7s}, HR={hit_rate:5.2f}%, GU={ghost_util:5.2f}%, win={win_pct:.1f}%)")
            prev_phase = mode
        prev_mode = mode

def print_summary_table(telemetry, phase_boundaries):
    """Print per-phase statistics."""
    print(f"\n{'='*70}")
    print("PER-PHASE SUMMARY")
    print(f"{'='*70}")
    print(f"{'Phase':<10} | {'Avg HR':>8} | {'Avg GU':>8} | {'Avg Win%':>8} | {'Dominant Mode':<10}")
    print("-" * 70)

    # Group by phase
    phases = {}
    for i, phase in enumerate(telemetry['phase']):
        if phase not in phases:
            phases[phase] = {'hr': [], 'gu': [], 'win': [], 'modes': []}
        phases[phase]['hr'].append(telemetry['hit_rate'][i])
        phases[phase]['gu'].append(telemetry['ghost_utility'][i])
        phases[phase]['win'].append(telemetry['window_pct'][i])
        phases[phase]['modes'].append(telemetry['mode'][i])

    for phase_name in ['corda1', 'loop', 'corda2']:
        if phase_name not in phases:
            continue
        p = phases[phase_name]
        avg_hr = sum(p['hr']) / len(p['hr'])
        avg_gu = sum(p['gu']) / len(p['gu'])
        avg_win = sum(p['win']) / len(p['win'])

        # Find dominant mode
        from collections import Counter
        mode_counts = Counter(p['modes'])
        dominant = mode_counts.most_common(1)[0][0]

        print(f"{phase_name:<10} | {avg_hr:>7.2f}% | {avg_gu:>7.2f}% | {avg_win:>7.2f}% | {dominant:<10}")

def main():
    print("="*70)
    print("CHAMELEON STRESS TEST VISUALIZATION")
    print("="*70)
    print(f"Cache size: {CACHE_SIZE}")
    print(f"Sample interval: {SAMPLE_INTERVAL} ops")
    print()

    # Load traces
    print("Loading traces...")
    corda = load_corda_trace()
    loop = load_loop_trace()

    print(f"  Corda: {len(corda):,} accesses")
    print(f"  Loop:  {len(loop):,} accesses x5 = {len(loop)*5:,}")

    # Build chained trace
    trace = []
    trace.extend(corda)  # Phase 1: corda1
    loop5_start = len(trace)
    for _ in range(5):
        trace.extend(loop)  # Phase 2: loop
    corda2_start = len(trace)
    trace.extend(corda)  # Phase 3: corda2

    print(f"  Total: {len(trace):,} accesses")

    # Phase boundaries
    phase_boundaries = [
        (0, 'corda1'),
        (loop5_start, 'loop'),
        (corda2_start, 'corda2'),
    ]

    print(f"\nPhase boundaries:")
    print(f"  corda1: 0 - {loop5_start:,}")
    print(f"  loop:   {loop5_start:,} - {corda2_start:,}")
    print(f"  corda2: {corda2_start:,} - {len(trace):,}")

    # Run with telemetry
    print("\nRunning Chameleon with telemetry...")
    cache = ChameleonCache(CACHE_SIZE)
    telemetry = run_with_telemetry(trace, cache, phase_boundaries)

    # Calculate overall hit rate
    total_samples = len(telemetry['hit_rate'])
    overall_hr = sum(telemetry['hit_rate']) / total_samples
    print(f"\nOverall average hit rate: {overall_hr:.2f}%")

    # Print visualizations
    print_mode_timeline(telemetry)
    print_summary_table(telemetry, phase_boundaries)

    # ASCII charts (sampled to fit screen)
    sample_step = max(1, len(telemetry['ops']) // 40)  # Show ~40 rows
    sampled_telemetry = {
        k: [v[i] for i in range(0, len(v), sample_step)]
        for k, v in telemetry.items()
    }

    print_ascii_chart(sampled_telemetry, 'hit_rate', 'HIT RATE OVER TIME (%)', max_val=100)
    print_ascii_chart(sampled_telemetry, 'ghost_utility', 'GHOST UTILITY OVER TIME (%)', max_val=50)
    print_ascii_chart(sampled_telemetry, 'window_pct', 'WINDOW SIZE OVER TIME (% of capacity)', max_val=25)

    # Print key insights
    print("="*70)
    print("KEY DIAGNOSTIC QUESTIONS")
    print("="*70)
    print("""
1. Does mode switch correctly when entering loop phase?
2. Does ghost utility rise during loop phase?
3. Does window size adjust appropriately?
4. What happens at the transition points?
5. Why does hit rate stay so low during loop?
""")

if __name__ == "__main__":
    main()
