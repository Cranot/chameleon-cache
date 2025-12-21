# Social Media Launch Posts

---

## Hacker News

**Title:**
Chameleon: A variance-aware cache that beats TinyLFU

**URL:**
https://github.com/Cranot/chameleon-cache

**Comment to post immediately after submission:**

The key insight: the optimal response to ghost buffer utility isn't linear.

Everyone assumes "more evidence of return = more likely to admit." This is wrong.

The Basin of Leniency:
- Ghost utility <2%: STRICT (random noise, don't trust)
- Ghost utility 2-12%: LENIENT (working set shift, trust the ghost)
- Ghost utility >12%: STRICT again (strong loop, items WILL return anyway)

That last part is counter-intuitive. In a tight loop, every item returns eventually. Rushing to admit causes churn. Being patient maintains stability.

Results: +1.42pp over TinyLFU overall. +10.15pp on loop workloads. Never more than 0.51pp behind anywhere.

390 lines of Python. Happy to answer questions about the algorithm or benchmarks.

---

## r/compsci

**Title:**
The Basin of Leniency: Why non-linear cache admission beats frequency-only policies

**Body:**

I've been researching cache replacement policies and discovered something counter-intuitive about admission control.

**The conventional wisdom:** More evidence that an item will return → more aggressively admit it.

**The problem:** This breaks down in loop/scan workloads. TinyLFU, the current state-of-the-art, struggles here because its frequency-only admission doesn't adapt to workload phase changes.

**The discovery:** The optimal admission response is non-linear. I call it the "Basin of Leniency":

| Ghost Utility | Behavior | Reasoning |
|---------------|----------|-----------|
| <2% | STRICT | Random noise - ghost hits are coincidental |
| 2-12% | LENIENT | Working set shift - trust the ghost buffer |
| >12% | STRICT | Strong loop - items WILL return, prevent churn |

The third zone is the key insight. When ghost utility is very high (>12%), you're in a tight loop. Every evicted item will return eventually. Rushing to admit them causes cache churn. Being patient and requiring stronger frequency evidence maintains stability.

**The mechanism:** Track ghost buffer utility (ghost_hits / ghost_lookups). Use this to modulate admission strictness. Combine with variance detection (max_freq / avg_freq) for Zipf vs loop classification.

**Results against TinyLFU:**
- Overall: +1.42pp (61.16% vs 59.74%)
- LOOP-N+10: +10.15pp
- TEMPORAL: +7.50pp
- Worst regression: -0.51pp (Hill-Cache trace)

**Complexity:** O(1) amortized access, O(capacity) space.

The 12% threshold was auto-tuned across 9 workloads. It represents the "thrashing point" where loop behavior dominates.

Paper-length writeup with benchmarks: https://github.com/Cranot/chameleon-cache

Curious what the community thinks about this non-linear approach. Has anyone seen similar patterns in other admission control domains?

---

## r/Python (Showcase format)

**Title:**
Chameleon Cache - A variance-aware cache replacement policy that adapts to your workload

**Flair:** Showcase

**Body:**

# What My Project Does

Chameleon is a cache replacement algorithm that automatically detects workload patterns (Zipf vs loops vs mixed) and adapts its admission policy accordingly. It beats TinyLFU by +1.42pp overall through a novel "Basin of Leniency" admission strategy.

```python
from chameleon import ChameleonCache

cache = ChameleonCache(capacity=1000)
hit = cache.access("user:123")  # Returns True on hit, False on miss
```

Key features:
- Variance-based mode detection (Zipf vs loop patterns)
- Adaptive window sizing (1-20% of capacity)
- Ghost buffer utility tracking with non-linear response
- O(1) amortized access time

# Target Audience

This is for developers building caching layers who need adaptive behavior without manual tuning. Production-ready but also useful for learning about modern cache algorithms.

**Use cases:**
- Application-level caches with mixed access patterns
- Research/benchmarking against other algorithms
- Learning about cache replacement theory

**Not for:**
- Memory-constrained environments (uses more memory than Bloom filter approaches)
- Pure sequential scan workloads (TinyLFU with doorkeeper is better there)

# Comparison

| Algorithm | Zipf (Power Law) | Loops (Scans) | Adaptive |
|-----------|------------------|---------------|----------|
| LRU | Poor | Good | No |
| TinyLFU | Excellent | Poor | No |
| Chameleon | Excellent | Excellent | Yes |

Benchmarked on 3 real-world traces (Twitter, CloudPhysics, Hill-Cache) + 6 synthetic workloads.

# Links

- **Source:** https://github.com/Cranot/chameleon-cache
- **Install:** `pip install chameleon-cache`
- **Tests:** 24 passing, Python 3.8-3.12
- **License:** MIT

---

## Images to Use

| Platform | Image |
|----------|-------|
| All | `assets/basin-of-leniency.png` (theory diagram) |
| All | `assets/benchmark-results.png` (bar chart) |

**Note:** Reddit allows image uploads in posts. HN is link-only but images are in the README.
