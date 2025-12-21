"""
Chameleon Cache - A Variance-Adaptive Cache Replacement Policy

Usage:
    from chameleon import ChameleonCache

    cache = ChameleonCache(capacity=1000)
    hit = cache.access("key")  # Returns True on hit, False on miss
"""

from .core import ChameleonCache

__version__ = "1.0.0"
__all__ = ["ChameleonCache"]
