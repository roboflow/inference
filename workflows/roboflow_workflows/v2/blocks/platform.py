"""The Roboflow platform port of V2 blocks, offline by default.

Blocks that route requests through the Roboflow API (managed email/SMS, LLM
``rf_key:*`` keys) take two constructor resources, the names V1 composition
roots already inject::

    platform_client: PlatformClient = OFFLINE_PLATFORM_CLIENT
    api_key: Optional[Union[str, Secret]] = None

``PlatformClient`` has the method shapes of V1's ``RoboflowPlatformClient``,
so a host can pass its existing client object unchanged. The default client
opens no connection: ``post`` refuses, header builders return empty values and
``wrap_url`` returns the URL unchanged. Tests inject a fake client.
"""

from typing import Callable, Dict, List, Optional, Protocol, Tuple, Union

HttpErrorHandlers = Dict[int, Callable[[Exception], None]]
"""HTTP status code to handler, as V1 ``post`` accepts."""

Headers = Dict[str, Union[str, List[str]]]


class PlatformUnavailableError(RuntimeError):
    """The host provided no Roboflow platform client."""


class PlatformClient(Protocol):
    """Structural port to the Roboflow API; the shape of V1's client.

    Not ``runtime_checkable``: blocks call it, nothing isinstance-checks it.
    """

    def post(
        self,
        endpoint: str,
        api_key: Optional[str],
        payload: Optional[dict] = None,
        params: Optional[List[Tuple[str, str]]] = None,
        http_errors_handlers: Optional[HttpErrorHandlers] = None,
    ) -> dict:
        """Send a request to a Roboflow API endpoint and return its JSON."""

    def build_api_headers(self, explicit_headers: Optional[Headers] = None) -> Headers:
        """Return headers for a request, merged with ``explicit_headers``."""

    def build_weights_provider_headers(
        self,
        countinference: Optional[bool] = None,
        service_secret: Optional[str] = None,
    ) -> Optional[Dict[str, str]]:
        """Return headers for a model weights provider, or ``None``."""

    def wrap_url(self, url: str) -> str:
        """Return ``url`` routed through the host's gateway, if it has one."""


class OfflinePlatformClient:
    """Default client: no platform, and it says so.

    Only ``post``, which can only target Roboflow, refuses. Header builders and
    ``wrap_url`` stay usable so a block can still call a non-Roboflow endpoint.
    """

    def post(
        self,
        endpoint: str,
        api_key: Optional[str],
        payload: Optional[dict] = None,
        params: Optional[List[Tuple[str, str]]] = None,
        http_errors_handlers: Optional[HttpErrorHandlers] = None,
    ) -> dict:
        """Refuse: this host has no Roboflow platform client.

        Args:
            endpoint: Roboflow API endpoint the step wanted to call.
            api_key: Ignored.
            payload: Ignored.
            params: Ignored.
            http_errors_handlers: Ignored.

        Raises:
            PlatformUnavailableError: Always.
        """
        raise PlatformUnavailableError(
            f"this step calls the Roboflow API ({endpoint}), but the host "
            "provided no platform client. Inject a 'platform_client' resource, "
            "or configure the step to call its provider directly with your own "
            "API key."
        )

    def build_api_headers(self, explicit_headers: Optional[Headers] = None) -> Headers:
        """Return a copy of ``explicit_headers``.

        Args:
            explicit_headers: Headers the caller set.

        Returns:
            The same headers; an empty dict for ``None``.
        """
        headers = dict(explicit_headers or {})

        return headers

    def build_weights_provider_headers(
        self,
        countinference: Optional[bool] = None,
        service_secret: Optional[str] = None,
    ) -> Optional[Dict[str, str]]:
        """Return ``None``: there is no weights provider.

        Args:
            countinference: Ignored.
            service_secret: Ignored.

        Returns:
            ``None``.
        """
        return None

    def wrap_url(self, url: str) -> str:
        """Return ``url`` unchanged: there is no gateway.

        Args:
            url: Request URL.

        Returns:
            ``url``.
        """
        return url

    def __repr__(self) -> str:
        return "OfflinePlatformClient()"


OFFLINE_PLATFORM_CLIENT = OfflinePlatformClient()
"""Shared stateless default of every ``platform_client`` resource."""
