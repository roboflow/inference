---
description: Securing a self-hosted Roboflow Inference Server — network isolation, authentication, TLS, and disabling custom Python execution. Security of local deployments is your responsibility.
---

# Securing a Self-Hosted Server

When you run Inference on your own hardware, **you own its security posture**.
A locally deployed server does not enforce authentication, encryption, or
network restrictions by default — it is built to be easy to start, not to be
safe to expose. Out of the box it will answer any request that reaches it,
including requests to run models and execute Workflows.

This page covers the four controls every self-hosted deployment should
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
`WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT` to a comma-separated list of the Roboflow workspace URLs
allowed to use the server:

```bash
docker run --rm -p 9001:9001 \
  -e WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT=your-workspace-url,another-workspace-url \
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

## Recommended baseline

For any self-hosted server that is reachable beyond `localhost`:

- [ ] Network access restricted to known clients (firewall / private network / proxy).
- [ ] `WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT` set, or your own auth in front.
- [ ] TLS terminated at the server or an upstream proxy.
- [ ] `ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS=false` unless you genuinely need it.
