from vmem.cache import ValueCache
from vmem.config import CacheConfig


def test_cache_roundtrip(tmp_path):
    config = CacheConfig(path=str(tmp_path / "cache.sqlite"))
    cache = ValueCache(config)
    payload = {
        "value_score": 0.9,
        "subject": "Alice",
        "predicate": "met",
        "object": "Bob",
    }
    cache.set("alice met bob", payload)
    loaded = cache.get("alice met bob")
    assert loaded["value_score"] == 0.9
    assert loaded["subject"] == "Alice"
    cache.close()
