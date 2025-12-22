**Update: Your suggestion worked!**

I took your advice about the hill climber and dug deeper into what was actually happening. The breakthrough came from an unexpected direction - I discovered that **frequency decay** was the real culprit, not the admission policy.

The key insight: decay helps during phase transitions (flushes stale frequencies) but **hurts during stable phases** by causing unnecessary churn. I added "skip-decay" - when hit rate is above 40%, I skip the frequency halving entirely.

Results on your stress test:
- **Chameleon: 28.72%** (up from 0.01%)
- TinyLFU: 26.26%
- Loop phase: **50.01%** (now matching LIRS at 50.03%)

That's **98.8% of theoretical optimal** (29.08%). I also validated across 8 different workload types to make sure I wasn't overfitting - wins 7, ties 1, loses 0.

Your point about heuristics vs direct optimization was spot on. While I didn't end up using the hill climber for window sizing (skip-decay alone got me there), your explanation of how Caffeine approaches this problem helped me think about decay as a tunable parameter rather than a fixed operation.

Code is updated in the repo. Thanks again for pushing me to look harder at this - wouldn't have found it without your stress test and insights.
