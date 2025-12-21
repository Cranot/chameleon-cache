# Caffeine Stress Test Analysis

## Configuration
- Cache size: 512
- Trace chain: `corda (936K)` -> `loop x5 (2.5M)` -> `corda (936K)`
- Total accesses: 4.4M

## Results (Original Chameleon)

```
Algorithm            Hit Rate    Notes
---------------------------------------
tinylfu              26.26%      WINNER
tinylfu-adaptive     24.55%
chameleon             0.01%      CATASTROPHIC FAILURE
arc                   0.03%
sieve                 0.03%
lru                   0.02%
lirs                  0.01%
```

## Phase Analysis (fresh cache per phase)

```
Algorithm            Corda    Loop x1    Loop x5
-------------------------------------------------
tinylfu              0.00%    46.13%     45.75%
chameleon            0.02%    45.74%     45.94%
lirs                 0.00%    49.95%     50.03%
arc/lru/sieve        0.04%    0.00%      0.00%
```

**Key insight:** In isolation, Chameleon matches TinyLFU on the loop (45-46%). The failure happens on the **phase transition** from Corda to loop.

## Root Cause Investigation

### Finding 1: Ghost Boost Creates Frequency Arms Race

With 1011 loop items and 512 cache slots:
- ~499 items are always "homeless"
- They cycle: window -> ghost -> window -> ghost
- Each ghost return boosts their frequency
- This creates constant churn as items evict each other with ever-increasing frequencies

**Measured impact:**
```
TinyLFU (no ghost):     34.24%
+ ghost tracking:       28.45%  (-6pp)
+ ghost boost:          10.17%  (-24pp)
```

### Finding 2: Corda Pollution Dilutes Ghost Utility

After 936K Corda accesses:
- ghost_lookups = 936K, ghost_hits ≈ 4K
- ghost_utility = 4K / 936K ≈ 0%

This prevents the "Basin of Leniency" from detecting the loop pattern.

### Finding 3 (KEY): Mode Switching Causes Catastrophic Oscillation

When Chameleon's mode detection sees high hit rate during the loop phase:
1. SCAN mode (win_cap=5) -> hit rate climbs to 60%
2. Triggers MIXED mode -> win_cap jumps to 51
3. Large window causes churn -> hit rate drops to 0%
4. Triggers SCAN mode -> back to win_cap=5
5. Repeat...

This oscillation between window sizes (5 <-> 51) prevents any stable equilibrium.

### Finding 4 (CRITICAL): Lenient Admission is Fatal

```
Strict admission (k > v):   26.26%
Lenient admission (k >= v):  0.02%
```

Any mode that uses lenient admission (MIXED, RECENCY) causes the cache to fail.
In the loop phase, all items have similar frequencies, so `>=` admits everyone, causing constant churn.

## The Fix

To match TinyLFU on this stress test, Chameleon must:
1. Keep window size constant at 1% (no oscillation)
2. Always use strict admission (`k_freq > v_freq`)
3. Never use lenient tie-breaking

**Result after fix:**
```
chameleon           :  26.26%  <-- MATCHES TINYLFU
tinylfu             :  26.26%
```

## The Trade-off

The fix works, but it eliminates Chameleon's adaptive advantage:

| Workload | Original Chameleon | Fixed Chameleon | TinyLFU |
|----------|-------------------|-----------------|---------|
| LOOP-N+10 | 99.04% (+10.15pp) | 88.89% | 88.89% |
| TEMPORAL | 48.45% (+7.50pp) | 40.93% | 40.93% |
| Stress Test | 0.01% | 26.26% | 26.26% |

**The fixed version is essentially TinyLFU.** The adaptive behavior that made Chameleon special (winning on LOOP-N+10 and TEMPORAL) is gone.

## Conclusion

This stress test reveals a fundamental tension:
- Adaptive behavior (lenient admission, mode switching) helps on some workloads
- But it catastrophically fails on noise-to-loop phase transitions
- The safe option (strict admission always) works everywhere but provides no advantage

**TinyLFU's simplicity wins here.** No ghost buffer, no mode switching, no lenient admission - just strict frequency comparison. This design is robust to phase transitions.

## Next Steps

Fixing this properly requires:
1. Better phase transition detection (EMA for ghost utility, variance tracking)
2. Conditional lenient admission only when safe
3. Faster response to workload changes

This is a research problem, not a simple bug fix. We acknowledge the weakness.
