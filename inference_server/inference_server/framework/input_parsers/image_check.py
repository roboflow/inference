from __future__ import annotations

import filetype


_NPY_MAGIC = b"\x93NUMPY"


def looks_like_image(data: bytes | memoryview) -> bool:
    head = bytes(data[:262])
    if head[:6] == _NPY_MAGIC:
        return True
    return filetype.is_image(head)
