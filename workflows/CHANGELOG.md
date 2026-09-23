# Changelog

## Unreleased

### Added

- MQTT Reader enterprise block (`roboflow_enterprise/mqtt_reader@v1`): subscribes to an MQTT topic and returns one message per run as raw text and parsed JSON, with `read_mode` selecting the newest message (`latest`) or the next unread one (`sequential`). Retained messages are returned on the first run; the subscription is restored after a reconnect, and a 15 s keepalive reports a silently lost broker within about 25 s. Live subscriptions need an `InferencePipeline`; over the HTTP API each request only sees retained messages.
- Persistent sessions for the MQTT Reader: an optional `client_id` (unique per broker, for example the pipeline name from a workflow input) connects with that id and a non-clean session, so the broker keeps the subscription and queues QoS 1/2 messages while the block is away and delivers the backlog when the same id reconnects, after a pipeline or process restart included. Requires `qos` 1 or 2 (a run with `client_id` and `qos` 0 reports an error); the publisher must also publish at QoS 1 or higher. Meant for `InferencePipeline`s; `sequential` works through the backlog one message per run. Left empty, the behaviour is unchanged.
- TLS for the MQTT Reader and MQTT Writer: an `encryption` dropdown (`none`, default, or `tls`) encrypts the connection and verifies the broker's certificate against the system trust store; `ca_certificate_path` (shown for TLS only) points at a PEM bundle for a private CA and requires `ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE=True`. The port is not switched automatically and verification cannot be disabled.
- Operator policy for the MQTT Reader and MQTT Writer broker address: `MQTT_WORKFLOWS_BLOCKS_WHITELISTED_HOSTS` (comma-separated `host[:port]` allowlist, no DNS) and `MQTT_WORKFLOWS_BLOCKS_ALLOW_USER_PROVIDED_HOST` (default `True`; when `False` the workflow's host and port are ignored and the first allowlist entry is used). Defaults preserve existing behaviour.

---

## `0.2.1`

### Fixed

- Expose native Qwen 3.8 VL 27B in Qwen VLM blocks v2–v4, including model discovery and thinking support.
- SpaceXAI object detection encodes PNG at compression level 9. OpenCV's default compression put large frames over xAI's 25MB upload limit.

### Added

- SpaceXAI block (`spacexai@v3`): `grok-4.7` model option with `low`/`medium`/`high`/`xhigh` reasoning effort. Detection reuses the Grok 4.5/4.6 prompt.

---

## `0.2.0`

### Added

- Python 3.13 support (`requires-python` is now `>=3.10,<3.14`).
- OpenAI v7 detection and instance-segmentation accept optional `output_classes`, keeping visual
  prompts in `classes` while constraining and decoding model output with stable labels.

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
