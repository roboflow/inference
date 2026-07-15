import hashlib
import os.path
import re
import urllib.parse
from typing import List, Optional, Union

import backoff
import cv2
import numpy as np
import pybase64
import requests
import torch
from requests import Timeout
from tldextract import tldextract
from tldextract.tldextract import ExtractResult

from inference_models.configuration import (
    API_CALLS_MAX_TRIES,
    IDEMPOTENT_API_REQUEST_CODES_TO_RETRY,
)
from inference_models.errors import ModelInputError, ModelRuntimeError, RetryError

BASE64_DATA_TYPE_PATTERN = re.compile(r"^data:image\/[a-z]+;base64,")

SIGNED_URL_MARKER_PARAMS = {
    "x-goog-signature",  # GCS V4
    "x-amz-signature",  # S3 V4
    "awsaccesskeyid",  # S3 V2
    "googleaccessid",  # GCS V2
    "sig",  # Azure SAS
    "signature",  # GCS/S3 V2, CloudFront
}

# Auth-only query parameters, removed from the cache key when a signature
# marker is present. Content-selecting parameters that can ride alongside a
# signature (GCS `generation`, S3 `versionId`, Azure `versionid`/`snapshot`)
# are deliberately NOT listed: they select different bytes for the same path
# and must stay part of the cache identity.
SIGNED_URL_AUTH_PARAMS = SIGNED_URL_MARKER_PARAMS | {
    # GCS V4
    "x-goog-algorithm",
    "x-goog-credential",
    "x-goog-date",
    "x-goog-expires",
    "x-goog-signedheaders",
    # S3 V4
    "x-amz-algorithm",
    "x-amz-credential",
    "x-amz-date",
    "x-amz-expires",
    "x-amz-signedheaders",
    "x-amz-security-token",
    # GCS/S3 V2, CloudFront
    "expires",
    "policy",
    "key-pair-id",
    # Azure SAS
    "sv",
    "ss",
    "srt",
    "sp",
    "se",
    "st",
    "spr",
    "sr",
    "sip",
    "ses",
    "sdd",
    "skoid",
    "sktid",
    "skt",
    "ske",
    "sks",
    "skv",
    "saoid",
    "suoid",
    "scid",
}


def canonicalize_url_for_hashing(reference: str) -> str:
    """Signed URLs (GCS/S3/Azure/CloudFront) carry rotating auth parameters
    (signature, timestamp, expiry) in the query string, so the same object
    yields a different URL string on every signing. Hashing the raw URL would
    defeat the image-embeddings cache (and the class-embeddings cache keyed on
    top of it) for every caller that re-signs URLs per request. When the query
    string contains a recognized signature parameter, the recognized auth
    parameters are removed from the cache key; any remaining parameters (e.g.
    GCS `generation`, S3 `versionId`) still select content and are kept,
    sorted for a signing-order-independent key. URLs without signature
    parameters keep their full form: a bare query string (e.g. ?v=2) may
    legitimately select different content.
    """
    parsed = urllib.parse.urlparse(reference)
    if not parsed.query:
        return reference
    query_params = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    query_param_names = {key.lower() for key, _ in query_params}
    if query_param_names.isdisjoint(SIGNED_URL_MARKER_PARAMS):
        return reference
    content_params = sorted(
        (key, value)
        for key, value in query_params
        if key.lower() not in SIGNED_URL_AUTH_PARAMS
    )
    canonical_query = urllib.parse.urlencode(content_params)
    return urllib.parse.urlunparse(
        parsed._replace(query=canonical_query, fragment="")
    )


class LazyImageWrapper:

    @classmethod
    def init(
        cls,
        image: Union[np.ndarray, torch.Tensor, str, bytes],
        allow_url_input: bool,
        allow_non_https_url: bool,
        allow_url_without_fqdn: bool,
        whitelisted_domains: Optional[List[str]],
        blacklisted_domains: Optional[List[str]],
        allow_local_storage_access: bool,
    ):
        image_in_memory, image_reference = None, None
        if isinstance(image, (torch.Tensor, np.ndarray)):
            image_in_memory = image
        else:
            image_reference = image
        return cls(
            allow_url_input=allow_url_input,
            allow_non_https_url=allow_non_https_url,
            allow_url_without_fqdn=allow_url_without_fqdn,
            whitelisted_domains=whitelisted_domains,
            blacklisted_domains=blacklisted_domains,
            allow_local_storage_access=allow_local_storage_access,
            image_in_memory=image_in_memory,
            image_reference=image_reference,
        )

    def __init__(
        self,
        allow_url_input: bool,
        allow_non_https_url: bool,
        allow_url_without_fqdn: bool,
        whitelisted_domains: Optional[List[str]],
        blacklisted_domains: Optional[List[str]],
        allow_local_storage_access: bool,
        image_in_memory: Optional[Union[np.ndarray, torch.Tensor]] = None,
        image_reference: Optional[Union[str, bytes]] = None,
        image_hash: Optional[str] = None,
    ):
        self._allow_url_input = allow_url_input
        self._allow_non_https_url = allow_non_https_url
        self._allow_url_without_fqdn = allow_url_without_fqdn
        self._whitelisted_domains = whitelisted_domains
        self._blacklisted_domains = blacklisted_domains
        self._allow_local_storage_access = allow_local_storage_access
        if image_in_memory is None and image_reference is None:
            raise ModelRuntimeError(
                message="Attempted to use OWLv2 image lazy loading not providing neither image "
                "location nor image instance - this is invalid input. Contact Roboflow to get help.",
                help_url="https://inference-models.roboflow.com/errors/models-runtime/#modelruntimeerror",
            )
        self._image_in_memory = image_in_memory
        self._image_reference = image_reference
        self._image_hash = image_hash

    def as_numpy(self) -> np.ndarray:
        if self._image_in_memory is not None:
            if isinstance(self._image_in_memory, torch.Tensor):
                self._image_in_memory = self._image_in_memory.cpu().numpy()
            return self._image_in_memory
        image = load_image_reference(
            image_reference=self._image_reference,
            allow_url_input=self._allow_url_input,
            allow_non_https_url=self._allow_non_https_url,
            allow_url_without_fqdn=self._allow_url_without_fqdn,
            whitelisted_domains=self._whitelisted_domains,
            blacklisted_domains=self._blacklisted_domains,
            allow_local_storage_access=self._allow_local_storage_access,
        )
        self._image_in_memory = image
        return image

    def get_hash(self) -> str:
        if self._image_hash is not None:
            return self._image_hash
        if self._image_reference is not None:
            reference = self._image_reference
            if isinstance(reference, str) and is_url(reference=reference):
                reference = canonicalize_url_for_hashing(reference=reference)
            self._image_hash = hash_function(value=reference)
        else:
            self._image_hash = hash_function(value=self.as_numpy().tobytes())
        return self._image_hash

    def unload_image(self) -> None:
        if self._image_in_memory is not None and self._image_reference is not None:
            self._image_in_memory = None


def load_image_reference(
    image_reference: Union[str, bytes],
    allow_url_input: bool,
    allow_non_https_url: bool,
    allow_url_without_fqdn: bool,
    whitelisted_domains: Optional[List[str]],
    blacklisted_domains: Optional[List[str]],
    allow_local_storage_access: bool,
) -> np.ndarray:
    if isinstance(image_reference, bytes):
        return decode_image_from_bytes(image_bytes=image_reference)
    if is_url(reference=image_reference):
        return decode_image_from_url(
            url=image_reference,
            allow_url_input=allow_url_input,
            allow_non_https_url=allow_non_https_url,
            allow_url_without_fqdn=allow_url_without_fqdn,
            whitelisted_domains=whitelisted_domains,
            blacklisted_domains=blacklisted_domains,
        )
    if not allow_local_storage_access:
        return decode_image_from_base64(value=image_reference)
    elif os.path.isfile(image_reference):
        return cv2.imread(image_reference)
    else:
        return decode_image_from_base64(value=image_reference)


def decode_image_from_url(
    url: str,
    allow_url_input: bool,
    allow_non_https_url: bool,
    allow_url_without_fqdn: bool,
    whitelisted_domains: Optional[List[str]],
    blacklisted_domains: Optional[List[str]],
):
    if not allow_url_input:
        raise ModelInputError(
            message="Providing images via URL is not supported in this configuration of `inference-models`.",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        )
    try:
        parsed_url = urllib.parse.urlparse(url)
    except ValueError as error:
        raise ModelInputError(
            message="Provided image URL is invalid.",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        ) from error
    if parsed_url.scheme != "https" and not allow_non_https_url:
        raise ModelInputError(
            message="Providing images via non https:// URL is not supported in this configuration of `inference-models`.",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        )
    domain_extraction_result = tldextract.TLDExtract(suffix_list_urls=())(
        parsed_url.netloc
    )  # we get rid of potential ports and parse FQDNs
    _ensure_resource_fqdn_allowed(
        fqdn=domain_extraction_result.fqdn,
        allow_url_without_fqdn=allow_url_without_fqdn,
    )
    address_parts_concatenated = _concatenate_chunks_of_network_location(
        extraction_result=domain_extraction_result
    )  # concatenation of chunks - even if there is no FQDN, but address
    # it allows white-/black-list verification
    _ensure_location_matches_destination_whitelist(
        destination=address_parts_concatenated,
        whitelisted_domains=whitelisted_domains,
    )
    _ensure_location_matches_destination_blacklist(
        destination=address_parts_concatenated,
        blacklisted_domains=blacklisted_domains,
    )
    image_content = _get_from_url(url=url)
    return decode_image_from_bytes(image_bytes=image_content)


def decode_image_from_base64(value: str) -> np.ndarray:
    try:
        value = BASE64_DATA_TYPE_PATTERN.sub("", value)
        decoded = pybase64.b64decode(value, validate=True)
        return decode_image_from_bytes(image_bytes=decoded)
    except Exception as error:
        value_prefix = value[:16]
        raise ModelInputError(
            message=f"Could not decode bas64 image fro reference {value_prefix}.",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        ) from error


def decode_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    byte_array = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(byte_array, cv2.IMREAD_COLOR)


def is_url(reference: str) -> bool:
    return reference.startswith("http://") or reference.startswith("https://")


def _ensure_resource_fqdn_allowed(fqdn: str, allow_url_without_fqdn: bool) -> None:
    if not fqdn and not allow_url_without_fqdn:
        raise ModelInputError(
            message="Providing images via URL without FQDN is not supported in this configuration of  `inference-models`.",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        )
    return None


def _concatenate_chunks_of_network_location(extraction_result: ExtractResult) -> str:
    chunks = [
        extraction_result.subdomain,
        extraction_result.domain,
        extraction_result.suffix,
    ]
    non_empty_chunks = [chunk for chunk in chunks if chunk]
    result = ".".join(non_empty_chunks)
    if result.startswith("[") and result.endswith("]"):
        # dropping brackets for IPv6
        return result[1:-1]
    return result


def _ensure_location_matches_destination_whitelist(
    destination: str, whitelisted_domains: Optional[List[str]]
) -> None:
    if whitelisted_domains is None:
        return None
    if destination not in whitelisted_domains:
        raise ModelInputError(
            message="It is not allowed to reach image URL - prohibited by whitelisted destinations",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        )
    return None


def _ensure_location_matches_destination_blacklist(
    destination: str,
    blacklisted_domains: Optional[List[str]],
) -> None:
    if blacklisted_domains is None:
        return None
    if destination in blacklisted_domains:
        raise ModelInputError(
            message="It is not allowed to reach image URL - prohibited by blacklisted destinations.",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        )
    return None


@backoff.on_exception(
    backoff.constant,
    exception=RetryError,
    max_tries=API_CALLS_MAX_TRIES,
    interval=1,
)
def _get_from_url(url: str, timeout: int = 5) -> bytes:
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            if response.status_code in IDEMPOTENT_API_REQUEST_CODES_TO_RETRY:
                raise RetryError(
                    message=f"File hosting returned {response.status_code}",
                    help_url="https://inference-models.roboflow.com/errors/file-download/#retryerror",
                )
            response.raise_for_status()
            return response.content
    except (ConnectionError, Timeout, requests.exceptions.ConnectionError) as error:
        raise RetryError(
            message=f"Connectivity error",
            help_url="https://inference-models.roboflow.com/errors/file-download/#retryerror",
        ) from error


def compute_image_hash(image: Union[torch.Tensor, np.ndarray]) -> str:
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    return hash_function(value=image.tobytes())


def hash_function(value: Union[str, bytes]) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha1(value).hexdigest()
