# Changelog

## Unreleased

### Execution engine

- Reject cyclic saved inner-workflow references during resolution with a composition
  error instead of `RecursionError`. Enforce nesting depth and total inner-workflow
  count limits during reference expansion, before fetching or expanding children
  beyond those limits. Valid repeated references and remote dispatch are unchanged;
  no migration is required.

---

## `0.2.2`

### Added
- OpenAI block (`open_ai@v7`): `gpt-6-sol` and `gpt-6-luna` model options.
- Anthropic Claude block (`anthropic_claude@v5`): `claude-opus-5-5` model option.

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
