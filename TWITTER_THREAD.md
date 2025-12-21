# Twitter/X Launch Thread

## Tweet 1 (Hook)

🦎 I built a cache that beats TinyLFU.

Not by being smarter about frequency.
By discovering that the optimal admission response is *non-linear*.

Introducing Chameleon Cache.

https://github.com/Cranot/chameleon-cache

---

## Tweet 2 (Problem)

The problem with TinyLFU:

It's optimized for Zipf (power-law) workloads.
But real systems have LOOPS. Scans. Phase shifts.

TinyLFU chokes on these. Hard.

---

## Tweet 3 (Insight Setup)

The insight everyone missed:

"More evidence of return = more likely to admit"

Sounds obvious, right?

It's wrong.

---

## Tweet 4 (Basin of Leniency)

**[ATTACH: assets/basin-of-leniency.png]**

Introducing the Basin of Leniency:

• Low ghost utility (<2%): STRICT → random noise
• Medium (2-12%): LENIENT → trust the ghost
• High (>12%): STRICT again → items WILL return, prevent churn

The U-curve nobody expected.

---

## Tweet 5 (Why)

Why strict at HIGH utility?

Counter-intuitive but critical:

In a tight loop, EVERY item will return eventually.
Rushing to admit them = cache churn.
Being patient = stability.

The "Thrashing Point" is 12%. Auto-tuned.

---

## Tweet 6 (Results)

**[ATTACH: assets/benchmark-results.png]**

Results:

Overall:     +1.42pp vs TinyLFU
LOOP-N+10:   +10.15pp
TEMPORAL:    +7.50pp
Twitter:     +1.10pp

Never more than 0.51pp behind. Anywhere.

---

## Tweet 7 (Simplicity)

The algorithm in 4 lines:

```python
if ghost_utility < 2%:    strict()  # noise
elif ghost_utility < 12%: lenient() # trust
else:                     strict()  # loop
```

That's it. That's the whole insight.

---

## Tweet 8 (CTA)

390 lines of Python. MIT licensed. 24 tests passing.

```
pip install chameleon-cache
```

```python
from chameleon import ChameleonCache
cache = ChameleonCache(1000)
hit = cache.access(key)
```

🦎 https://github.com/Cranot/chameleon-cache

---

## Tweet 9 (Future + Engagement)

What's next:

• Bloom filter variant for memory-constrained envs
• Concurrent/thread-safe version
• Maybe a paper (HotStorage?)

If you work on caching, I'd love feedback.

RT if you found this interesting. 🦎

---

## Images to Attach

| Tweet | Image |
|-------|-------|
| 4 | `assets/basin-of-leniency.png` |
| 6 | `assets/benchmark-results.png` |
