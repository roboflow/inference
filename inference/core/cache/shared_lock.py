import uuid

from inference.core.cache import cache


class SharedLock:
    def __init__(self, key, share_key=None, expire=None):
        self.key = key
        self.share_key = share_key
        self.expire = expire
        self.id = str(uuid.uuid4())

    def __enter__(self):
        self.locked = acquire_lock(
            self.key, share_key=self.share_key, share_id=self.id, expire=self.expire
        )
        return self.locked

    def __exit__(self, type, value, traceback):
        release_lock(self.key, share_key=self.share_key, share_id=self.id)


def acquire_lock(
    key: str, share_key: str = None, share_id: str = None, expire: float = None
) -> bool:
    """Acquires a lock on a given key.

    Args:
        key (str): The key to lock.
        share_key (str, optional): The key to share the lock with. Defaults to None.
        expire (float, optional): The time, in seconds, after which the lock will expire. Defaults to None.

    Returns:
        Any: The lock object.
    """
    locked = True
    shared_lock = None
    while locked:
        locked = cache.get(key)
        if locked:
            shared_lock = cache.get(share_key)
            if shared_lock:
                locked = False

    cache.set(key, True, expire=expire)
    if share_key is not None:
        if shared_lock is None:
            shared_lock = []
        shared_lock.append(share_id)
        cache.set(share_key, shared_lock, expire=expire)
    return True


def release_lock(key: str, share_key: str = None, share_id: str = None):
    """Releases a lock on a given key.

    Args:
        key (str): The key to unlock.
        share_key (str, optional): The key to share the lock with. Defaults to None.
    """
    if share_key is not None:
        shared_lock = cache.get(share_key)
        if shared_lock:
            shared_lock.remove(share_id)
            if not shared_lock:
                cache.delete(share_key)
                cache.delete(key)
            else:
                cache.set(share_key, shared_lock)
    else:
        cache.delete(key)
