# Changelog

## Unreleased

### Added

- MQTT Reader enterprise block (`roboflow_enterprise/mqtt_reader@v1`): subscribes to an MQTT topic and returns one message per run as raw text and parsed JSON, with `read_mode` selecting the newest message (`latest`) or the next unread one (`sequential`). Retained messages are returned on the first run; the subscription is restored after a reconnect. Live subscriptions need an `InferencePipeline`; over the HTTP API each request only sees retained messages.
- Operator policy for the MQTT Reader and MQTT Writer broker address: `MQTT_WORKFLOWS_BLOCKS_WHITELISTED_HOSTS` (comma-separated `host[:port]` allowlist, no DNS) and `MQTT_WORKFLOWS_BLOCKS_ALLOW_USER_PROVIDED_HOST` (default `True`; when `False` the workflow's host and port are ignored and the first allowlist entry is used). Defaults preserve existing behaviour.

---

## `0.1.2`

### Fixed

- Webhook sink: SSRF hardening (destination validation, DNS pinning, redirect and proxy refusal) now applies only when `ALLOW_WEBHOOK_WORKFLOWS_SINK_TO_NON_GLOBAL_ADDRESSES=false`. The default restores the previous plain `requests` transport, so environment `HTTP_PROXY` / `HTTPS_PROXY` work again.

---

## `0.1.1`

### Fixed

- Accept Python string class names from NumPy object arrays in detection-property expressions, avoiding `.item()` errors after custom Python blocks.

### Added

- Kafka Consumer and Kafka Producer enterprise blocks; `confluent-kafka` and `aws-msk-iam-sasl-signer-python` join the `enterprise` extra.

### Changed

- Dropped unused dependencies: `opencv-contrib-python`, `requests-toolbelt`, and test extras `pytest-asyncio`, `pytest-timeout`, `aioresponses`.

---

## `0.1.0`

- Initial release: Workflows execution engine and block library extracted from `inference`.
