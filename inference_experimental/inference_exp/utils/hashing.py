import hashlib
import json


def hash_dict_content(content: dict) -> str:
    content_string = json.dumps(content, sort_keys=True)
    return hashlib.sha256(content_string.encode()).hexdigest()
