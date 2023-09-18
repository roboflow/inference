import urllib

from inference.core.env import LICENSE_SERVER


def ProxyUrl(url):
    """Returns a proxied URL if according to LICENSE_SERVER settings"""
    return f"http://{LICENSE_SERVER}/proxy?url=" + urllib.parse.quote(
        url, safe="~()*!'"
    )


def RawUrl(url):
    """Returns a raw URL"""
    return url


if LICENSE_SERVER:
    ApiUrl = ProxyUrl
else:
    ApiUrl = RawUrl
