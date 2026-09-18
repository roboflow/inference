# Changelog

## Unreleased

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
