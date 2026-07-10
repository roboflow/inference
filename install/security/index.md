---
description: Securing a self-hosted Roboflow Inference Server — network isolation, authentication, TLS, and disabling custom Python execution. Security of local deployments is your responsibility.
---

# Securing a Self-Hosted Server

When you run Inference on your own hardware, **you own its security posture**.
A locally deployed server does not enforce authentication, encryption, or
network restrictions by default — it is built to be easy to start, not to be
safe to expose. Out of the box it will answer any request that reaches it,
including requests to run models and execute Workflows.

This page covers the five controls every self-hosted deployment should
review before it handles anything beyond local development traffic. They are
complementary — apply as many as your environment allows.

!!! warning "This is your responsibility"

    Roboflow secures the managed Cloud, Serverless, and Dedicated Deployment
    offerings. For a server you run yourself, securing the host, the network
    around it, and the credentials it accepts is **your responsibility**. If
    your server is reachable from an untrusted network without the controls
    below, treat it as open to the world.

## 1. Restrict network access

The single most effective control is to not expose the server in the first
place. Inference listens on port `9001` by default and has no concept of a
"trusted" network — anything that can reach the port can use it.

- **Bind to localhost** when only processes on the same host need it
  (e.g. publish the container port as `127.0.0.1:9001:9001` instead of
  `9001:9001`).
- **Keep it on a private network / VPC** and reach it through a VPN, SSH
  tunnel, or service mesh rather than a public IP.
- **Use host and cloud firewalls / security groups** to allow `9001` only
  from the specific clients that need it.
- **Put a reverse proxy in front of it** (nginx, Traefik, Caddy, a cloud
  load balancer) if you need to expose it more broadly — that gives you a
  single place to add TLS, rate limiting, and access logging.

Never publish the inference port directly to the public internet without
authentication and TLS in place.

## 2. Enforce authentication

By default a self-hosted server does **not** require an API key to make requests — as a result,
beyond the auth that happens at the Roboflow API level when fetching data from the platform, there is
no additional security on the server itself. To turn on authentication, set
`WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT` to a comma-separated list of the Roboflow workspace slug
allowed to use the server:

```bash
docker run --rm -p 9001:9001 \
  -e WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT=your-workspace-url-slug,another-workspace-url-slug \
  roboflow/roboflow-inference-server-cpu:latest
```

With this set, the server installs an authorization middleware. Every
inference and Workflow request must carry an `api_key` (as a query parameter
or in the JSON body) that resolves — via Roboflow — to one of the whitelisted
workspaces. Requests with a missing, invalid, or non-whitelisted key are
rejected with `401 Unauthorized`.

!!! note "What is *not* covered by the API-key check"

    A small set of unauthenticated endpoints stay open so the server remains
    usable and observable: `/`, `/docs`, `/redoc`, `/info`, `/healthz`,
    `/readiness`, `/metrics`, `/openapi.json`, and static assets
    (`/static/...`, `/_next/...`). Treat `/info` and `/metrics` as
    information that anyone who can reach the server can read, and rely on
    network restrictions (control #1) to limit who that is.

**Bring your own auth.** The built-in check ties authorization to Roboflow
workspaces. If you have your own identity model, you can instead place a
reverse proxy or authentication middleware in front of the server — enforcing
OAuth/OIDC, mTLS, signed headers, an API gateway, or whatever your
organization already uses — and let only authenticated traffic through to
port `9001`. The two approaches can be combined.

## 3. Enable TLS when the network requires it

The built-in API-key check sends credentials in the request. If those
requests travel over any network you do not fully control, the connection
must be encrypted, otherwise keys and payloads are exposed in plaintext.

You have two options:

- **Terminate TLS at a reverse proxy / load balancer** in front of the
  server. This is the usual choice when you already run one.
- **Serve HTTPS directly from the server** by mounting a certificate and
  key and setting `ENABLE_HTTPS=true`. See
  [Serving inference over HTTPS](/server_configuration/https.md) for the
  full guide, including mutual TLS (client certificates) via `SSL_CA_CERTS`.

For purely local, loopback-only traffic (control #1, bound to `127.0.0.1`)
TLS is optional. Any time requests leave the host over an untrusted network,
TLS is required.

## 4. Disable custom Python execution in Workflows

Workflows can contain **Custom Python blocks** — arbitrary Python that runs
inside the server process. This is a powerful feature, but it means that
anyone who can submit a Workflow to the server can run arbitrary code on your
host. On a server reachable by untrusted clients, that is remote code
execution.

This is controlled by `ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS`.

| Setting | Effect |
| --- | --- |
| `True` (current default) | Workflows may define and run custom Python blocks. |
| `False` | Custom Python blocks are rejected; all other Workflow features still work. |

**If your Workflows do not rely on custom Python, set it to `False`:**

```bash
docker run --rm -p 9001:9001 \
  -e ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS=false \
  roboflow/roboflow-inference-server-cpu:latest
```

!!! warning "The default is changing on 2026-06-19"

    Today this flag defaults to `True` for backward compatibility. On
    **2026-06-19** the default will change to `False`. If your Workflows
    depend on custom Python blocks, set
    `ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS=true` explicitly so they
    keep working after that date. Otherwise, leave it disabled — and prefer
    enabling it only on deployments where the network and authentication
    controls above are already in place.

## 5. Restrict image fetching from URLs (SSRF)

Inference can load images straight from a URL supplied in the request
(`{"image": {"type": "url", "value": "https://..."}}`). Any time a server
fetches a URL that a caller controls, the caller can try to steer it into
making requests on their behalf — a class of attack called **Server-Side
Request Forgery (SSRF)**. Someone who cannot reach your internal network
directly can ask *your server* to fetch, for example:

- `http://169.254.169.254/latest/meta-data/` — the cloud metadata service
  (AWS/GCP/Azure), which can hand back instance credentials.
- `http://127.0.0.1:9001/...` and other `localhost` services — admin panels,
  databases, or the inference server's own unauthenticated endpoints.
- `http://10.0.0.5/`, `http://192.168.1.1/`, and other private (RFC1918),
  link-local, CGNAT, or IPv6 ULA hosts that sit behind your perimeter.

A public-looking hostname is not proof of a public target: it may resolve to a
private IP, redirect to one, or use **DNS rebinding** (resolve to a public IP
for the validation check, then a private IP for the actual connection).
Inference ships controls for all of these.

### Turn URL input off if you don't need it

The strongest control is to not accept URL images at all. If your clients
always send images as base64 or file uploads, disable URL fetching outright:

```bash
docker run --rm -p 9001:9001 \
  -e ALLOW_URL_INPUT=false \
  roboflow/roboflow-inference-server-cpu:latest
```

### Harden URL input when you do need it

When URL images are required, these flags narrow what the server is allowed to
fetch. Together they reject internal targets, **pin the connection to the
validated IP** (defeating DNS rebinding), and re-check every redirect hop:

| Variable | Default | Effect |
| --- | --- | --- |
| `ALLOW_URL_INPUT` | `True` | Master switch for URL image input. `False` rejects all URL images. |
| `ALLOW_URL_TO_NON_GLOBAL_ADDRESSES` | `True` | When `False`, a URL whose host resolves to a **non-global** address (loopback, private, link-local/metadata, CGNAT, IPv6 ULA, …) is rejected, and the connection is pinned to the validated IP so a second DNS answer cannot swap the target. |
| `VALIDATE_IMAGE_URL_REDIRECTS` | `False` | When `True`, redirects are followed **one hop at a time** and each hop URL is re-validated, instead of being followed blindly. |
| `MAX_IMAGE_URL_REDIRECTS` | `30` | Hard cap on redirect hops, enforced regardless of the flag above. |
| `ALLOW_NON_HTTPS_URL_INPUT` | `False` | When `False`, only `https://` URLs are accepted. |
| `ALLOW_URL_INPUT_WITHOUT_FQDN` | `False` | When `False`, URLs whose host is a bare IP or has no public suffix are rejected — callers must use a real domain name. |
| `WHITELISTED_DESTINATIONS_FOR_URL_INPUT` | *(unset)* | Comma-separated allow-list of destinations (`subdomain.domain.suffix`). When set, **only** these are permitted. |
| `BLACKLISTED_DESTINATIONS_FOR_URL_INPUT` | *(unset)* | Comma-separated block-list of destinations that are always rejected. |

A hardened configuration that still allows public HTTPS image URLs:

```bash
docker run --rm -p 9001:9001 \
  -e ALLOW_URL_TO_NON_GLOBAL_ADDRESSES=false \
  -e VALIDATE_IMAGE_URL_REDIRECTS=true \
  roboflow/roboflow-inference-server-cpu:latest
```

For the tightest control, add an allow-list so the server can only reach the
exact hosts you serve images from:

```bash
docker run --rm -p 9001:9001 \
  -e ALLOW_URL_TO_NON_GLOBAL_ADDRESSES=false \
  -e VALIDATE_IMAGE_URL_REDIRECTS=true \
  -e WHITELISTED_DESTINATIONS_FOR_URL_INPUT=images.example.com,cdn.example.com \
  roboflow/roboflow-inference-server-cpu:latest
```

!!! warning "Two defaults are changing in Q4 2026"

    `ALLOW_URL_TO_NON_GLOBAL_ADDRESSES` (→ `False`) and
    `VALIDATE_IMAGE_URL_REDIRECTS` (→ `True`) currently default to the legacy,
    permissive behaviour for backward compatibility. Both defaults are
    scheduled to flip to the secure values in **Q4 2026**. Set them explicitly
    now — to the secure values to opt in early, or to the legacy values if a
    workflow genuinely depends on fetching internal URLs — so the change does
    not surprise you.

!!! note "Proxies bypass this protection"

    If an HTTP(S) proxy is configured for the server, the proxy — not
    Inference — resolves the destination, so non-global blocking and
    connection pinning cannot be enforced (the server emits a warning when it
    detects this). Restrict what the proxy itself can reach if you rely on
    these controls.

!!! note "Same controls in the Python SDK"

    The `inference-sdk` client applies the same URL policy and SSRF protections
    — and reads the same environment variables — when it loads images from
    URLs, so a client that hydrates URL images before sending them is covered
    too.

## Recommended baseline

For any self-hosted server that is reachable beyond `localhost`:

- [ ] Network access restricted to known clients (firewall / private network / proxy).
- [ ] `WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT` set, or your own auth in front.
- [ ] TLS terminated at the server or an upstream proxy.
- [ ] `ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS=false` unless you genuinely need it.
- [ ] URL image input disabled (`ALLOW_URL_INPUT=false`), or hardened with `ALLOW_URL_TO_NON_GLOBAL_ADDRESSES=false` and `VALIDATE_IMAGE_URL_REDIRECTS=true` (plus an allow-list where you can).
