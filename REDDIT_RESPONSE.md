# Response to u/NovaX

---

Thank you for pointing us to this stress test. We ran it and **TinyLFU wins decisively (26.26% vs 0.01%)**.

## Root Cause

The failure is a fundamental design tension, not a bug:

1. **Lenient admission is fatal** - strict (`>`) gets 26.26%, lenient (`>=`) gets 0.02%
2. **Mode switching causes oscillation** - window size bounces between 5↔51 slots, preventing equilibrium
3. **Ghost boost creates arms race** - homeless items evict each other with inflating frequencies

## The Trade-off

We can fix it by using strict admission everywhere - but then Chameleon becomes TinyLFU and loses its advantage on other workloads (LOOP-N+10: +10pp, TEMPORAL: +7pp).

**TinyLFU's simplicity wins here.** No ghost buffer, no mode switching - just strict frequency comparison. Robust to phase transitions.

We acknowledge this as a legitimate weakness. Thanks for the rigorous testing.

---

*Full analysis: [STRESS_TEST_ANALYSIS.md](https://github.com/Cranot/chameleon-cache/blob/main/STRESS_TEST_ANALYSIS.md)*
