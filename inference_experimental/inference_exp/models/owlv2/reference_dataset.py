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
from inference_exp.configuration import (
    API_CALLS_MAX_TRIES,
    IDEMPOTENT_API_REQUEST_CODES_TO_RETRY,
)
from inference_exp.errors import ModelInputError, ModelRuntimeError, RetryError
from requests import Timeout
from tldextract import tldextract
from tldextract.tldextract import ExtractResult

BASE64_DATA_TYPE_PATTERN = re.compile(r"^data:image\/[a-z]+;base64,")


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
                help_url="https://todo",
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

    def get_hash(self, unload_image_if_loaded: bool = True) -> str:
        if self._image_hash is not None:
            return self._image_hash
        if self._image_reference is not None:
            self._image_hash = hash_function(value=self._image_reference)
        else:
            self._image_hash = hash_function(value=self.as_numpy().tobytes())
            if unload_image_if_loaded:
                self.unload_image()
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
            message="Providing images via URL is not supported in this configuration of `inference-exp`.",
            help_url="https://todo",
        )
    try:
        parsed_url = urllib.parse.urlparse(url)
    except ValueError as error:
        raise ModelInputError(
            message="Provided image URL is invalid.", help_url="https://todo"
        ) from error
    if parsed_url.scheme != "https" and not allow_non_https_url:
        raise ModelInputError(
            message="Providing images via non https:// URL is not supported in this configuration of `inference-exp`.",
            help_url="https://todo",
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
            help_url="https://todo",
        ) from error


def decode_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    byte_array = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(byte_array, cv2.IMREAD_COLOR)


def is_url(reference: str) -> bool:
    return reference.startswith("http://") or reference.startswith("https://")


def _ensure_resource_fqdn_allowed(fqdn: str, allow_url_without_fqdn: bool) -> None:
    if not fqdn and not allow_url_without_fqdn:
        raise ModelInputError(
            message="Providing images via URL without FQDN is not supported in this configuration of  `inference-exp`.",
            help_url="https://todo",
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
            help_url="https://todo",
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
            help_url="https://todo",
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
                    help_url="https://todo",
                )
            response.raise_for_status()
            return response.content
    except (ConnectionError, Timeout, requests.exceptions.ConnectionError):
        raise RetryError(
            message=f"Connectivity error",
            help_url="https://todo",
        )


def compute_image_hash(image: Union[torch.Tensor, np.ndarray]) -> str:
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    return hash_function(value=image.tobytes())


def hash_function(value: Union[str, bytes]) -> str:
    return hashlib.sha1(value).hexdigest()
