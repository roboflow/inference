"""Secure gateway transport policy, safe to import during configuration."""

import ipaddress
import urllib.parse
import warnings


def validate_secure_gateway_url(gateway: str) -> str:
    """Require TLS except for explicitly configured loopback HTTP gateways."""
    if any(character.isspace() or ord(character) < 32 for character in gateway):
        raise ValueError(
            "SECURE_GATEWAY must not contain whitespace or control characters"
        )
    candidate = gateway.rstrip("/")
    if "://" not in candidate:
        candidate = f"https://{candidate}"
    parsed = urllib.parse.urlsplit(candidate)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "SECURE_GATEWAY must be an HTTPS URL without credentials, query or fragment"
        )
    _ = parsed.port
    if parsed.scheme == "http":
        try:
            loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            loopback = parsed.hostname == "localhost"
        if not loopback:
            raise ValueError(
                "Plaintext SECURE_GATEWAY is allowed only for explicit loopback HTTP; configure HTTPS"
            )
    return urllib.parse.urlunsplit(parsed)


def normalize_secure_gateway_configuration(gateway: str) -> str:
    normalized = validate_secure_gateway_url(gateway)
    if "://" not in gateway:
        warnings.warn(
            "SECURE_GATEWAY bare host configuration now uses HTTPS. Configure an explicit "
            "https:// URL and trusted server certificate; explicit http:// is supported "
            "only for loopback tunnels.",
            RuntimeWarning,
            stacklevel=2,
        )
    return normalized
