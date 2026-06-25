"""Shared HTTP client helpers for the PLC Relay reader/writer blocks.

Both blocks talk to the on-device **PLC Relay** service over HTTP rather than
opening a direct PLC connection. The relay owns the protocol (Allen-Bradley,
Modbus, or Siemens S7), the device IP, and the tag schema.

These helpers use the relay's **batch** endpoints (`/read_batch`, `/write_batch`)
so that reading or writing N tags costs a single HTTP round-trip and a single PLC
transaction per frame, which matters at high FPS. The relay rejects the whole batch
with an HTTP error only for structural problems (an unknown tag, a non-writable tag,
or an empty/duplicate list); in that case every tag in the batch is reported as a
failure and the relay's error detail is logged. A per-tag value problem (wrong type
or out of range) does NOT fail the batch: the relay returns HTTP 200 and marks just
that tag with ``success=false``, so individual tag errors are reported and logged per
tag.
"""

from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import requests

from inference.core.logger import logger

DEFAULT_RELAY_URL = "http://localhost:8007"
DEFAULT_RELAY_PORT = 8007

# Connecting to the on-device relay is near-instant, so cap the connect phase tightly to
# fail fast when the relay process is down. The caller-supplied ``timeout`` is the *read*
# budget: it must cover the relay's synchronous PLC batch transaction, which can run for
# seconds against a slow or disconnected PLC. Splitting them means a down relay still
# fails fast even when the read budget is large.
RELAY_CONNECT_TIMEOUT = 3

READ_FAILURE = "ReadFailure"
WRITE_FAILURE = "WriteFailure"
WRITE_SUCCESS = "WriteSuccess"


def _request_timeout(read_timeout: int) -> Tuple[int, int]:
    """Build the (connect, read) timeout tuple passed to ``requests``."""
    return (RELAY_CONNECT_TIMEOUT, read_timeout)


def relay_base_url(ip_address: str, relay_port: int) -> str:
    """Build the relay base URL from a host/IP and port.

    Accepts a bare host/IP (``192.168.1.10`` -> ``http://192.168.1.10:<port>``) or a
    full URL (``http://host:8007``), which is used as-is.
    """
    addr = str(ip_address).strip()
    if addr.startswith("http://") or addr.startswith("https://"):
        return addr.rstrip("/")
    return f"http://{addr}:{relay_port}"


def read_tags(
    session: requests.Session,
    base_url: str,
    tags: Iterable[str],
    timeout: int,
) -> Tuple[Dict[str, Any], bool]:
    """Read a batch of tags through the relay's ``/read_batch`` endpoint.

    Returns ``(tag_values, had_failure)``. ``tag_values`` maps each requested tag to
    its value, or to the ``READ_FAILURE`` sentinel when that tag could not be read.
    ``had_failure`` is ``True`` if any tag failed.
    """
    tags = list(tags)
    if not tags:
        return {}, False

    try:
        response = session.post(
            f"{base_url}/read_batch", json={"tags": tags}, timeout=_request_timeout(timeout)
        )
    except requests.exceptions.RequestException as e:
        logger.error("Failed to reach PLC Relay while reading tags %s: %s", tags, e)
        return _all_failed(tags, READ_FAILURE), True

    if response.status_code != 200:
        # Whole-batch rejection (e.g. an unknown tag yields 404 for the entire batch).
        logger.error(
            "Error reading tags %s from PLC Relay: HTTP %s: %s",
            tags,
            response.status_code,
            _extract_detail(response),
        )
        return _all_failed(tags, READ_FAILURE), True

    data = _parse_json_object(response)
    if data is None:
        logger.error(
            "Malformed success response reading tags %s from PLC Relay: %s",
            tags,
            response.text,
        )
        return _all_failed(tags, READ_FAILURE), True

    by_name = _index_by_name(data.get("tags"))
    results: Dict[str, Any] = {}
    had_failure = False
    for tag in tags:
        entry = by_name.get(tag)
        if entry is None:
            logger.error("PLC Relay did not return a value for tag '%s'", tag)
            results[tag] = READ_FAILURE
            had_failure = True
        elif entry.get("error"):
            logger.error("Error reading tag '%s' from PLC: %s", tag, entry["error"])
            results[tag] = READ_FAILURE
            had_failure = True
        else:
            results[tag] = entry.get("value")
    return results, had_failure


def write_tags(
    session: requests.Session,
    base_url: str,
    tags_to_write: Mapping[str, Any],
    timeout: int,
) -> Tuple[Dict[str, str], bool]:
    """Write a batch of tags through the relay's ``/write_batch`` endpoint.

    Returns ``(write_results, had_failure)``. ``write_results`` maps each tag to
    ``WRITE_SUCCESS`` or the ``WRITE_FAILURE`` sentinel. ``had_failure`` is ``True``
    if any tag failed.
    """
    if not tags_to_write:
        return {}, False

    names = list(tags_to_write)
    writes = [{"name": name, "value": value} for name, value in tags_to_write.items()]

    try:
        response = session.post(
            f"{base_url}/write_batch",
            json={"writes": writes},
            timeout=_request_timeout(timeout),
        )
    except requests.exceptions.RequestException as e:
        logger.error("Failed to reach PLC Relay while writing tags %s: %s", names, e)
        return _all_failed(names, WRITE_FAILURE), True

    if response.status_code != 200:
        # Whole-batch rejection: unknown tag (404), non-writable tag (403), or an
        # empty/duplicate write list (400). The detail names the offending tag.
        logger.error(
            "Error writing tags %s to PLC Relay: HTTP %s: %s",
            names,
            response.status_code,
            _extract_detail(response),
        )
        return _all_failed(names, WRITE_FAILURE), True

    data = _parse_json_object(response)
    if data is None:
        logger.error(
            "Malformed success response writing tags %s to PLC Relay: %s",
            names,
            response.text,
        )
        return _all_failed(names, WRITE_FAILURE), True

    by_name = _index_by_name(data.get("results"))
    results: Dict[str, str] = {}
    had_failure = False
    for name, value in tags_to_write.items():
        entry = by_name.get(name)
        if entry is None:
            logger.error("PLC Relay did not return a result for tag '%s'", name)
            results[name] = WRITE_FAILURE
            had_failure = True
        elif not entry.get("success", False):
            # The relay accepted the request but rejected the value (wrong type or
            # out of range for the tag's data type).
            logger.error(
                "Error writing tag '%s' with value '%s': %s",
                name,
                value,
                entry.get("error", WRITE_FAILURE),
            )
            results[name] = WRITE_FAILURE
            had_failure = True
        else:
            results[name] = WRITE_SUCCESS
    return results, had_failure


def _all_failed(tags: Iterable[str], sentinel: str) -> Dict[str, str]:
    """Map every tag to the failure sentinel (used when the whole batch fails)."""
    return {tag: sentinel for tag in tags}


def _index_by_name(entries: Any) -> Dict[str, dict]:
    """Index a relay batch result array by each entry's 'name'."""
    if not isinstance(entries, list):
        return {}
    return {
        entry["name"]: entry
        for entry in entries
        if isinstance(entry, dict) and "name" in entry
    }


def _parse_json_object(response: requests.Response) -> Optional[dict]:
    """Return the response body as a dict, or None if it is missing/malformed/non-object."""
    try:
        data = response.json()
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _extract_detail(response: requests.Response) -> str:
    """Try to extract 'detail' from a JSON error response, falling back to raw text."""
    data = _parse_json_object(response)
    if data is not None and "detail" in data:
        return str(data["detail"])
    return response.text
