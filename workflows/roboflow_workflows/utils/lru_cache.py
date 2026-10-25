import collections


class LRUCache:
    """Copied verbatim from ``inference.core.cache.lru_cache``.

    Do not "improve" it. Its eviction is deliberately reproduced: ``set``
    enforces the size bound only when inserting a NEW key, and does so BEFORE
    inserting, so the cache transiently holds ``capacity + 1`` entries. A
    reimplementation that enforces after insert diverges on the very first
    eviction - verified against the original with a differential test.
    """

    def __init__(self, capacity=128):
        self.capacity = capacity
        self.cache = collections.OrderedDict()

    def set_max_size(self, capacity):
        self.capacity = capacity
        self.enforce_size()

    def enforce_size(self):
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def get(self, key):
        try:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        except KeyError:
            return None

    def set(self, key, value):
        try:
            self.cache.pop(key)
        except KeyError:
            self.enforce_size()
        self.cache[key] = value
