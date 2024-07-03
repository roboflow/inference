# Input formats accepted by `inference` server

## Why should I care?

Roboflow team has designed the `inference` server to be as easy to integrate as possible. That is why
we did not forbid to use potentially unsafe data loading methods - having in mind that some amount of
users would simply like to have easy serving mechanism, without full-pledged security rigor applied.

At the same time we want the server to be production-redy solution - that's why we provide configuration
options making it possible to disable unsafe behaviours.

In this document we explain how to configure the server to mitigate security risks or enable unsafe
behaviours if needed.


## Deserialization of pickled `numpy` objects

One of the way to make request to inference server is to send serialised numpy object:

```python
import cv2
import pickle
import requests

image = cv2.imread("...")
img_str = pickle.dumps(image)

infer_payload = {
    "model_id": "{project_id}/{model_version}",
    "image": {
        "type": "numpy",
        "value": img_str,
    },
    "api_key": "YOUR-API-KEY",
}

res = requests.post(
    "http://localhost:9001/infer/{task}",
    json=infer_payload,
)
```

Deserializing of this type of payload got disabled by default in `v0.14.0`, but it is possible to enable it
by setting `ALLOW_NUMPY_INPUT=True`. Check [inference cli docs](../inference_helpers/inference_cli.md) to
see how to run server with that flag. This option is **not available at Roboflow hosted inference**.

!!! warning

    Roboflow advices all users hosting inference server in PRODUCTION environments not to enable that
    option if there is any chance for malicious requests reaching the server.

## Sending URLs to inference images

Making GET requests to obtain images from URLs make it vulnerable to 
[server-side request forgery (SSRF) attacks](https://en.wikipedia.org/wiki/Server-side_request_forgery). At the same
time it is extremely convenient to run requests only specifying image URL:
```python
import requests


infer_payload = {
    "model_id": "{project_id}/{model_version}",
    "image": {
        "type": "numpy",
        "value": "https://some.com/image.jpg",
    },
    "api_key": "YOUR-API-KEY",
}

res = requests.post(
    "http://localhost:9001/infer/{task}",
    json=infer_payload,
)
```

This option is **enabled by default**, but we advise configuring the server to make it more secure, using one or many of the
following environmental variables:
* `ALLOW_URL_INPUT` - boolean flag to disable URL inputs - set that into `True` whenever you do not need pulling images from
URLs server-side
* `ALLOW_NON_HTTPS_URL_INPUT` - set to `False` to only allow https protocol in URLs (useful to make sure domain names are
not maliciously resolved)
* `ALLOW_URL_INPUT_WITHOUT_FQDN` - set to `False` to enforce URLs with fully qualified domain names only - and reject
URLs based on IPs
* `WHITELISTED_DESTINATIONS_FOR_URL_INPUT` - can be optionally set with comma separated destination for requests that are 
allowed, example: `WHITELISTED_DESTINATIONS_FOR_URL_INPUT=192.168.0.15,some.site.com` - setting the value makes URLs pointing
to different targets being rejected
* `BLACKLISTED_DESTINATIONS_FOR_URL_INPUT` - can be optionally set with comma separated destination for requests that are 
forbidden, example: `BLACKLISTED_DESTINATIONS_FOR_URL_INPUT=192.168.0.15,some.site.com` - setting the value makes URLs pointing
to selected targets being rejected


Check [inference cli docs](../inference_helpers/inference_cli.md) to see how to run server with specific flags.