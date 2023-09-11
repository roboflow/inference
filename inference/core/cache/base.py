class BaseCache:
    def set(self, key, value, expire=None):
        raise NotImplementedError()

    def get(self, key):
        raise NotImplementedError()
