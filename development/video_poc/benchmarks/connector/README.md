# Headless staging connector renderer

This directory renders one deterministic file-source connector for staging
benchmarks. It never imports a Kubernetes client or invokes `kubectl`; the only
outputs are a Kubernetes JSON manifest and a redacted plan containing literal
apply, observe, and cleanup command strings.

The renderer is deliberately bound to the actual Crusoe staging cell:

- context and kubeconfig cluster name: `ck8s-stg`;
- API server:
  `https://ck8s-stg-83c07ac7.us-east1-a.cmk.crusoecloudcompute.com`;
- a dedicated DNS-label namespace whose name contains `bench`;
- exactly `https://api.roboflow.one` or the canonical staging Functions base.

It refuses tags and placeholder digests. Both the connector and fixture-init
images are supplied as immutable `@sha256:` references. The API key is not
accepted in configuration or emitted in the manifest: the connector reads
`ROBOFLOW_API_KEY` exclusively from a pre-existing Kubernetes Secret key.

The checked-in connector image was built from `rf-video-connector` commit
`e39c5d865323140651e62341fa1e965b79b2ed70`; its Linux amd64 manifest digest is
recorded in `staging.example.json`. The init image is the Linux amd64 manifest
for `curlimages/curl:8.16.0`. Its real non-root, read-only-root download and
checksum path was exercised against the pinned fixture before recording it.

## Fixture provenance

`staging.example.json` names the canonical public fixture:

`https://media.roboflow.com/supervision/video-examples/vehicles.mp4`

The exact 35,345,757-byte object was independently verified as
`ac81100d9310bd4e9c02bc0b13b6492781d009742ced347766b2601be3c44ad4`.
The init container downloads only plain HTTPS URLs on the hardcoded
`media.roboflow.com` allowlist and verifies the exact digest before the
connector starts.

## Render

Copy the example and render:

```bash
python development/video_poc/benchmarks/connector/render_staging_connector.py \
  --config /path/to/staging-connector.json \
  --output-dir development/video_poc/benchmarks/results/connector
```

The plan records the exact cluster identity and references the Secret by name
and key while showing its value as `[redacted]`. Inspect the generated files.
Running any rendered command remains a separate staging cluster write requiring
explicit approval.

## Runtime boundary

The Deployment uses one replica with `Recreate`, a stable `bench-...` connector
ID, no Service, no host devices, no service-account token, and no inbound
network access. The local UI and ONVIF network discovery are disabled. Because
the pod receives no host video devices, only the verified files mounted
read-only at `/fixtures` are registered. `/state` and `/tmp` are bounded
`emptyDir` mounts; the root filesystem is read-only and all containers run
non-root with dropped capabilities.

The NetworkPolicy permits outbound DNS plus TCP 443 and RTSP/TCP 8554. This is
enough for staging API control, HTTPS fixture initialization, and relay
publishing without creating an inbound path. If the connector protocol changes,
update and review the explicit ports rather than broadening egress implicitly.

The default cleanup command removes only the Deployment, NetworkPolicy, and
ServiceAccount. It preserves the namespace and its separately managed API-key
Secret. A distinct `cleanupNamespace` command is rendered for deliberate final
teardown; deleting the namespace also deletes that Secret and every other
namespaced object, so it should not be part of routine benchmark cleanup.
