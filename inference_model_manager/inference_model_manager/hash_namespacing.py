"""Tenant namespacing for client-provided embedding hash ids.

Client-generated ids are cache keys in the worker-side embedding caches.
Prefixing them with a digest of the caller's api key makes the namespaces
disjoint per tenant: the worker caches stay untouched and simply see longer
keys. Server-generated content hashes are never prefixed; responses strip
the caller's own prefix so clients always see their raw ids.
"""

from __future__ import annotations

import hashlib
from typing import List, Optional, Union


def tenant_namespace(api_key: Optional[str]) -> str:
    return hashlib.sha256((api_key or "anonymous").encode()).hexdigest()[:16]


def namespace_client_hash_id(hash_id: str, api_key: Optional[str]) -> str:
    return f"{tenant_namespace(api_key)}:{hash_id}"


def namespace_client_hash_ids(
    hash_ids: Union[str, List[str]], api_key: Optional[str]
) -> Union[str, List[str]]:
    if isinstance(hash_ids, str):
        return namespace_client_hash_id(hash_ids, api_key)
    return [namespace_client_hash_id(hash_id, api_key) for hash_id in hash_ids]


def strip_tenant_namespace(hash_id: str, api_key: Optional[str]) -> str:
    prefix = f"{tenant_namespace(api_key)}:"
    if hash_id.startswith(prefix):
        return hash_id[len(prefix) :]
    return hash_id
