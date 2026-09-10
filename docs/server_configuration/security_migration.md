# Security configuration migration

These changes require explicit administrator choices for features that can load code,
interpret raw media pipelines, or transmit authentication credentials. Apply the
server, CLI and image changes together when using these features.

## Video source validation

Both standard and enterprise remote stream request schemas accept integer camera
indices, explicit absolute or relative paths (`/video.mp4`, `./video.mp4`,
`../video.mp4`, or Windows drive paths), and encoded
HTTP(S), RTSP(S), force-TCP `rtspt`/`rtspst`, RTMP(S), UDP, SRT, RTP, TCP, file
URLs, or numeric `csi://0` camera references. This is the request transport matrix;
the selected image must still provide the corresponding decoder/backend. RTMP(S)
and conventional FFmpeg network transports do not require the raw-pipeline opt-in. Raw GStreamer launch descriptions are disabled by
default. This also rejects unencoded whitespace or quotes in URLs, unsupported URL
schemes outside that matrix (for example `concat`, `subfile` or custom plugin
schemes), and file names containing `!`, `=`, or quotes. Encode video URLs and rename
ambiguous local files. Bare identifiers, including `video.mp4`, must be written
as explicit paths such as `./video.mp4`, preventing bare GStreamer element names
from reaching backend auto-detection. Administrators who deliberately need arbitrary GStreamer
pipelines can set `ALLOW_UNSAFE_GSTREAMER_PIPELINES=True`; this restores raw pipeline
interpretation and its plugin capabilities for stream requests. Only enable it for
trusted workloads.
This is not a general restriction on camera network destinations. Direct local
Python media use is unchanged. Actual plugin capabilities vary by target image.

## Model package trust

Benchmarking with `inference-models` now defaults to
`allow_untrusted_packages=False` at the CLI, adapter and implementation layers.
The Jetson images set `ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES=False`. Administrators who intentionally
trust an executable package can still explicitly use the CLI's
`--allow-untrusted-packages` flag or the corresponding Python/deployment setting.
Only enable it when the package and its origin are trusted.

The current source no longer contains a Jetson 6.0 Dockerfile. Inventory and retire
or rebuild previously published 6.0 images separately; changing these source files
does not update existing containers.

## HTTPS and gateway transport

When `ENABLE_HTTPS=True` and `SSL_CA_CERTS` is set, every server launcher now requires
a client certificate signed by that CA. Clients without a certificate or with an
untrusted certificate cannot complete the TLS handshake. Configure each client's
certificate/key and CA trust before rollout. The parallel Gunicorn launcher does not support encrypted private keys via
`SSL_KEYFILE_PASSWORD` and now rejects that configuration before launching services;
the previous Gunicorn flag was unsupported. Use the Uvicorn entrypoint when an
encrypted key is required. Python and shell Uvicorn support the password option.
HTTPS without `SSL_CA_CERTS` retains
ordinary server-authenticated TLS. All client-certificate requirements also apply
to health checks that connect to the TLS listener.

`SECURE_GATEWAY=host:port` now means `https://host:port` and emits a startup migration
warning. Both distributions validate and normalize the gateway during configuration
import, so unsupported URLs fail startup instead of failing on each request. Explicit HTTP is accepted
only for loopback IP addresses or `localhost`, allowing local tunnels such as
`http://127.0.0.1:8080`. Remote and LAN gateways must use HTTPS with a trusted server
certificate. Unsupported schemes, embedded credentials, queries, fragments and
whitespace are rejected. Update legacy plaintext gateway endpoints before rollout;
certificate verification is not disabled. The policy is shared by behavior tests
for both the core package and the independently installable `inference-models` package.
Customer-facing gateway deployment documentation maintained outside this repository
must be updated alongside endpoint migrations before rollout.

## Offline custom Python and OPC UA

`OFFLINE_MODE=True` together with `WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE=modal` now
fails startup. Values are stripped of surrounding whitespace and normalized to
lowercase; only `local` and `modal` are accepted. Unknown values fail startup. It
never silently converts remote execution into local execution.
To intentionally execute trusted custom Python locally, explicitly configure
`WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE=local` and the relevant local-execution
permission. Leave local execution disabled when that trust is not intended.

OPC UA pooled connections are isolated by the complete endpoint, username and
password tuple. An incorrect, empty or omitted password cannot reuse another
credential's authenticated session. Pool keys are process-local keyed digests;
passwords are not embedded in them. Release/invalidation callers must explicitly
provide the original password (including explicit `None` for anonymous sessions);
omitting it raises an error instead of silently targeting the wrong entry. Credential rotation establishes a separate
connection; existing connections drain under the existing pool lifecycle.
