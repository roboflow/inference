from threading import Lock


class WorkflowMemoryCache:
    """Process-global, refcounted namespace store for Cache Set / Cache Get.

    Workflows execute steps in a ``ThreadPoolExecutor`` (``max_concurrent_steps``
    may be > 1), so different block instances can touch the shared cache
    concurrently. All mutations are guarded by ``_lock``. Reads of a namespace
    dict by an instance that has already retained it are safe without the lock
    because that instance's retain prevents the count from reaching zero while
    it holds the reference.
    """

    cache = {}
    _retain_counts = {}
    _lock = Lock()

    @classmethod
    def get_dict(cls, namespace):
        """Return the shared dict for `namespace`, retaining it.

        Each retain must be paired with a later `release_namespace`. The dict
        is deleted only when the retain count reaches zero.
        """
        with cls._lock:
            if namespace not in cls.cache:
                cls.cache[namespace] = {}
            cls._retain_counts[namespace] = cls._retain_counts.get(namespace, 0) + 1
            return cls.cache[namespace]

    @classmethod
    def release_namespace(cls, namespace):
        with cls._lock:
            count = cls._retain_counts.get(namespace)
            if not count:
                return
            count -= 1
            if count > 0:
                cls._retain_counts[namespace] = count
                return
            cls._retain_counts.pop(namespace, None)
            cls.cache.pop(namespace, None)
