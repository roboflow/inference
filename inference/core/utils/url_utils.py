import urllib

from inference.core.env import SECURE_GATEWAY


def wrap_url(url: str) -> str:
    if not SECURE_GATEWAY:
        return url
    return f"http://{SECURE_GATEWAY}/proxy?url=" + urllib.parse.quote(
        url, safe="~()*!'"
    )
