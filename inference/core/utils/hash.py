import hashlib


def get_text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()
