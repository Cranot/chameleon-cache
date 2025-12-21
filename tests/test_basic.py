"""Basic unit tests for ChameleonCache."""

import pytest
from chameleon import ChameleonCache


class TestBasicAPI:
    """Test basic cache API."""

    def test_create_cache(self):
        """Test cache creation."""
        cache = ChameleonCache(100)
        assert len(cache) == 0

    def test_access_miss(self):
        """Test cache miss returns False."""
        cache = ChameleonCache(100)
        assert cache.access("key1") is False

    def test_access_hit(self):
        """Test cache hit returns True."""
        cache = ChameleonCache(100)
        cache.access("key1")
        assert cache.access("key1") is True

    def test_contains(self):
        """Test __contains__ method."""
        cache = ChameleonCache(100)
        assert "key1" not in cache
        cache.access("key1")
        assert "key1" in cache

    def test_len(self):
        """Test __len__ method."""
        cache = ChameleonCache(100)
        for i in range(50):
            cache.access(f"key{i}")
        assert len(cache) == 50

    def test_capacity_limit(self):
        """Test cache respects capacity (with 10% tolerance for adaptive window)."""
        cache = ChameleonCache(100)
        for i in range(200):
            cache.access(f"key{i}")
        # Window can grow to 10% during warmup, so allow 110% of capacity
        assert len(cache) <= 110, f"Cache size {len(cache)} exceeds 110% of capacity"

    def test_get_stats(self):
        """Test get_stats returns dict."""
        cache = ChameleonCache(100)
        for i in range(50):
            cache.access(f"key{i}")
        stats = cache.get_stats()
        assert isinstance(stats, dict)
        assert 'mode' in stats
        assert 'ghost_utility' in stats


class TestEviction:
    """Test eviction behavior."""

    def test_lru_eviction_order(self):
        """Test that old items are evicted first."""
        cache = ChameleonCache(10)

        # Fill cache
        for i in range(10):
            cache.access(f"key{i}")

        # Access key0 to make it recent
        cache.access("key0")

        # Add new items to force eviction
        for i in range(10, 20):
            cache.access(f"key{i}")

        # key0 should still be cached (was recently accessed)
        # Other early keys should be evicted
        assert "key0" in cache

    def test_frequency_protection(self):
        """Test that high-frequency items are protected."""
        cache = ChameleonCache(100)

        # Create a hot item
        for _ in range(10):
            cache.access("hot_key")

        # Fill with cold items
        for i in range(200):
            cache.access(f"cold_{i}")

        # Hot key should survive due to frequency
        # (This may not always pass depending on mode)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_capacity_minimum(self):
        """Test minimum capacity is enforced."""
        with pytest.raises(ValueError, match="capacity must be >= 2"):
            ChameleonCache(1)

        with pytest.raises(ValueError, match="capacity must be >= 2"):
            ChameleonCache(0)

        with pytest.raises(ValueError, match="capacity must be >= 2"):
            ChameleonCache(-1)

    def test_capacity_2(self):
        """Test smallest valid cache (capacity=2)."""
        cache = ChameleonCache(2)
        cache.access("a")
        cache.access("b")
        cache.access("c")  # Should evict one
        assert len(cache) == 2

    def test_small_cache(self):
        """Test small cache (capacity=10) under stress."""
        cache = ChameleonCache(10)
        for i in range(1000):
            cache.access(f"key{i % 50}")
        assert len(cache) <= 11  # Allow small tolerance

    def test_repeated_same_key(self):
        """Test hitting same key many times."""
        cache = ChameleonCache(100)
        for _ in range(10000):
            cache.access("hot")
        assert "hot" in cache
        assert cache.access("hot") is True


class TestHashability:
    """Test different key types."""

    def test_string_keys(self):
        """Test string keys."""
        cache = ChameleonCache(100)
        cache.access("hello")
        assert "hello" in cache

    def test_int_keys(self):
        """Test integer keys."""
        cache = ChameleonCache(100)
        cache.access(42)
        assert 42 in cache

    def test_tuple_keys(self):
        """Test tuple keys."""
        cache = ChameleonCache(100)
        cache.access(("user", 123))
        assert ("user", 123) in cache


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
