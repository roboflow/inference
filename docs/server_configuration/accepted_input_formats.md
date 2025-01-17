# Input formats accepted by `inference` server

## Why should I care?

The Roboflow team has designed the inference server to be as user-friendly and straightforward to integrate as 
possible. We understand that some users prioritize ease of use, which is why we did not restrict the use of 
potentially less secure data loading methods. This approach caters to those who prefer a simple and accessible 
serving mechanism without the need for rigorous security measures.

However, we also recognize the importance of having a production-ready solution. Therefore, we offer configuration 
options that allow users to disable potentially unsafe behaviors.

In this document, we explain how to configure the server to either enhance security or enable more 
flexible behaviors, depending on your needs.


## Deserialization of pickled `numpy` objects

One of the ways to send requests to the inference server is via serialized numpy objects:

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

Starting from version `v0.14.0`, deserialization of this type of payload is disabled by default. However, you can 
enable it by setting an environmental variable, `ALLOW_NUMPY_INPUT=True`. Check [inference cli docs](../inference_helpers/inference_cli.md) to
see how to run the server with that flag. This option is **not available in Roboflow's Hosted Inference API**.

!!! warning

    Roboflow advises all users hosting the inference server in production environments not to enable this option if 
    the server is open to requests from the open Internet or is not locked down to accept only authenticated requests from your workspace's API key.

## Sending URLs to inference images

Making GET requests to obtain images from URLs can expose the server to 
[server-side request forgery (SSRF) attacks](https://en.wikipedia.org/wiki/Server-side_request_forgery). However, it is also very convenient to simply provide an image URL 
for requests:
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

This option is **enabled by default**, but we recommend configuring the server to enhance security using one or more of
the following environment variables:
* `ALLOW_URL_INPUT` - Set to `False` disable image URLs of any kind to be accepted by server - default: `True`.
* `ALLOW_NON_HTTPS_URL_INPUT` - set to `False` to only allow https protocol in URLs (useful to make sure domain names are
not maliciously resolved) - default: `False`
* `ALLOW_URL_INPUT_WITHOUT_FQDN` - set to `False` to enforce URLs with fully qualified domain names only - and reject
URLs based on IPs - default: `False`
* `WHITELISTED_DESTINATIONS_FOR_URL_INPUT` - Optionally, you can specify a comma-separated list of allowed destinations 
for URL requests. For example: `WHITELISTED_DESTINATIONS_FOR_URL_INPUT=192.168.0.15,some.site.com`. URLs pointing to 
other targets will be rejected.
* `BLACKLISTED_DESTINATIONS_FOR_URL_INPUT` - Optionally, you can specify a comma-separated list of forbidden 
destinations for URL requests. For example:  `BLACKLISTED_DESTINATIONS_FOR_URL_INPUT=192.168.0.15,some.site.com`.
URLs pointing to these targets will be rejected.
* `ALLOW_ACCESS_TO_LOCAL_FILESYSTEM` - Set to `False` to disable local filesystem access to images - default: `True`.

Check [inference cli docs](../inference_helpers/inference_cli.md) to see how to run server with specific flags.