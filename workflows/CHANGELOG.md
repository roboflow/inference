# Changelog

This is the canonical changelog for the `roboflow-workflows` package, including
blocks, dependencies, and execution-engine behavior. Add engine behavior changes
under `## Unreleased` → `### Execution engine`; one entry is sufficient.

Package releases record their bundled execution-engine compatibility version,
which can remain unchanged across package releases. The package version and
engine version are separate; workflow and block compatibility use the engine
version. The mapping starts with package `0.2.2` below.

Earlier engine changes and migration guidance remain in the
[historical execution-engine changelog](https://docs.roboflow.com/workflows/developer-guide/developer-guide/execution-engine-changelog).
See the [engine release rule](../.cursor/rules/execution-engine-version-changelog.mdc)
for contributor and maintainer responsibilities.

## Unreleased

### Fixed

- Blur Visualization: instance segmentation predictions are now blurred in the shape of each mask, as the block's description always said, instead of as a rectangle covering the bounding box. The blur also covers mask pixels that reach past the box. Predictions without masks (object detection, keypoints) are blurred exactly as before. Masks can leave thin edges such as hair or fingers uncovered where the box used to hide them; set the new `padding` to widen the blur.
- Keypoint Visualization draws partial skeletons at their true joints. Keypoints below the model's confidence threshold are left out of predictions and the rest were padded at the end, so every keypoint after a missing one shifted position: bones joined the wrong joints, and a frame where nobody had all 17 COCO keypoints drew no bones at all. Keypoints are now placed by their keypoint class id, and COCO skeletons that are missing trailing joints are still recognised. Keypoints without class ids, or with ids that are negative or too large for the keypoint slot or padding limits, are drawn as before. The tensor-native path (`ENABLE_TENSOR_DATA_REPRESENTATION`) places keypoints the same way when they are rebuilt from a remotely executed model step, a rollup or a dynamic block, so both representations render alike.
- MQTT Writer: a broker that refuses the connection (bad user name or password, not authorised, unacceptable protocol version) is now reported in the outputs with the broker's reason, and the client stops reconnecting instead of retrying the same credentials about once a second, which tripped brokers' authentication rate limiting. The refusal is reported as soon as the broker answers rather than after `timeout`, and every later run repeats it without touching the broker; fix the configuration and restart the pipeline. A broker answering "unavailable" keeps retrying in the background.
- Preserve image dimensions and parent lineage when Template Matching returns no detections.

### Added

- Blur Visualization `padding`: blurs extra pixels around each detection, growing bounding boxes on every side and segmentation masks outward. Defaults to `0`, which keeps the previous coverage.
- MQTT Reader enterprise block (`roboflow_enterprise/mqtt_reader@v1`): subscribes to an MQTT topic and returns one message per run as raw text and parsed JSON, with `read_mode` selecting the newest message (`latest`) or the next unread one (`sequential`). Retained messages are returned on the first run; the subscription is restored after a reconnect, and a 15 s keepalive reports a silently lost broker within about 25 s. Live subscriptions need an `InferencePipeline`; over the HTTP API each request only sees retained messages. A broker that refuses the connection (bad user name or password, not authorised, rejected client id) is reported with the broker's reason as soon as it answers, and the client stops reconnecting; a broker answering "unavailable" keeps retrying. After the first subscription a run never waits for the broker: while the connection is down, runs return the last message with `error_status` set instead of an empty message, buffered messages first; the outage is logged once at its start and once at recovery.
- Persistent sessions for the MQTT Reader: an optional `client_id` (unique per broker, for example the pipeline name from a workflow input) connects with that id and a non-clean session, so the broker keeps the subscription and queues QoS 1/2 messages while the block is away and delivers the backlog when the same id reconnects, after a pipeline or process restart included. Requires `qos` 1 or 2 (a run with `client_id` and `qos` 0 reports an error); the publisher must also publish at QoS 1 or higher. Meant for `InferencePipeline`s; `sequential` works through the backlog one message per run. Left empty, the behaviour is unchanged.
- TLS for the MQTT Reader and MQTT Writer: an `encryption` dropdown (`none`, default, or `tls`) encrypts the connection and verifies the broker's certificate against the system trust store; `ca_certificate_path` (shown for TLS only) points at a PEM bundle for a private CA and requires `ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE=True`. The port is not switched automatically and verification cannot be disabled.
- Operator policy for the MQTT Reader and MQTT Writer broker address: `MQTT_WORKFLOWS_BLOCKS_WHITELISTED_HOSTS` (comma-separated `host[:port]` allowlist, no DNS) and `MQTT_WORKFLOWS_BLOCKS_ALLOW_USER_PROVIDED_HOST` (default `True`; when `False` the workflow's host and port are ignored and the first allowlist entry is used). Defaults preserve existing behaviour.

### Changed

- `zxing-cpp` is now pinned per Python version: `~=2.2.0` on Python < 3.13 (unchanged from before) and `==2.3.0` on Python 3.13, which is the first release with 3.13 wheels. The pin is split rather than widened because neither 2.2.0 nor 2.3.0 ships aarch64 wheels, so Jetson installs compile from source, and the 2.3.0 sources need C++20 concepts, which the JetPack 5 toolchain (GCC 9.4) does not support.

### Execution Engine Change

- **Run-scoped thread pool for executor-less runs** — runs that receive no
  host-provided executor (standalone `roboflow-workflows`, SDK and embedded
  usage) now reuse one run-scoped thread pool across step waves instead of
  constructing a `ThreadPoolExecutor` per wave, lowering latency on multi-wave
  chains; host-provided executors and scheduling are unchanged. No migration
  is needed.

- Reject cyclic saved inner-workflow references during resolution with a composition
  error instead of `RecursionError`. Enforce nesting depth and total inner-workflow
  count limits during reference expansion, before fetching or expanding children
  beyond those limits. Valid repeated references and remote dispatch are unchanged;
  no migration is required.

---

## `0.2.2`

Bundled execution engine: `1.15.2`.

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
