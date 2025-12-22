Thanks for the detailed questions!

**Basin of Leniency vs SLRU Probation**

Not quite the same. SLRU's probation segment is a physical queue where items must prove themselves before promotion. The Basin of Leniency is an **admission policy** - it controls how strict the frequency comparison is when deciding whether a new item can evict a cached one.

The "basin" shape comes from ghost utility (how often evicted items return):
- **Low ghost utility (<2%)**: Strict admission - returning items are noise
- **Medium ghost utility (2-12%)**: Lenient admission - working set is shifting, trust the ghost
- **High ghost utility (>12%)**: Strict again - strong loop pattern, items will return anyway, prevent churn

So it's more about the admission decision than a separate queue structure.

**Memory Overhead**

For 1 million items, here's the breakdown:

- **Ghost buffer**: 2x cache size (so 2M entries if cache holds 1M). Each entry stores key + frequency (1 byte) + timestamp (4 bytes). For 64-bit keys, that's ~26MB for the ghost.

- **Frequency sketch**: Same as TinyLFU - 4-bit counters, so ~500KB for 1M items.

- **Variance tracking**: Fixed size window of 500 keys + a set for uniques in current detection window. Negligible compared to ghost.

Total overhead is roughly **2.5x the key storage** for the ghost buffer. If your keys are large objects, the ghost only stores hashes, so it's more like +26MB fixed overhead regardless of key size.

You're not doubling your footprint for the cached data itself - the overhead scales with cache capacity, not with the size of cached values. For memory-constrained environments where even the ghost buffer is too much, you could shrink it to 1x or 0.5x cache size at the cost of reduced loop detection accuracy.

**Update**: Just pushed v1.1.0 with a "skip-decay" enhancement that improved performance on stress tests to 28.72% (98.8% of theoretical optimal). The memory overhead stays the same.
