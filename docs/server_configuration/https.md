# Serving inference over HTTPS

The inference server can serve HTTPS directly when you provide your own
TLS certificate and private key. This is useful for self-hosted deployments
where you cannot place a TLS-terminating reverse proxy in front of the
container.

HTTPS is configured entirely through environment variables.

## Environment variables

| Variable | Description | Default |
| --- | --- | --- |
| `ENABLE_HTTPS` | Master switch. Set to `True` / `1` / `yes` to enable HTTPS. | `False` |
| `SSL_CERTFILE` | Path inside the container to the PEM-encoded certificate. | `/etc/inference/certs/server.crt` |
| `SSL_KEYFILE` | Path inside the container to the PEM-encoded private key. | `/etc/inference/certs/server.key` |
| `SSL_KEYFILE_PASSWORD` | Passphrase for an encrypted private key. | unset |
| `SSL_CA_CERTS` | CA bundle used to verify client certificates (mTLS). | unset |

The cert and key paths default to `/etc/inference/certs/...`, so the
simplest deployment only needs to mount your cert/key at those paths and
flip `ENABLE_HTTPS=true`.

If `ENABLE_HTTPS` is set but the cert or key is missing, the server
refuses to start with an error listing the paths it tried to read.

## Quickstart with self-signed certs

The example below runs the CPU image on `https://localhost:9001` using a
self-signed certificate. Replace the cert generation step with your own
CA-issued certificate in production.

```bash
# 1. Generate a self-signed cert/key pair valid for localhost
mkdir -p /tmp/inference-certs
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout /tmp/inference-certs/server.key \
  -out /tmp/inference-certs/server.crt \
  -days 365 \
  -subj "/CN=localhost" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"

# 2. Run the inference server with HTTPS enabled
docker run --rm -p 9001:9001 \
  -e ENABLE_HTTPS=true \
  -v /tmp/inference-certs:/etc/inference/certs:ro \
  roboflow/roboflow-inference-server-cpu:latest

# 3. Validate from another shell
curl -sk https://localhost:9001/info
```

`-k` (or `--insecure`) is only required because the cert is self-signed;
clients that trust your CA do not need it.

## Custom cert paths

If your certs live somewhere other than the defaults, set the paths
explicitly:

```bash
docker run --rm -p 9001:9001 \
  -e ENABLE_HTTPS=true \
  -e SSL_CERTFILE=/run/secrets/tls/fullchain.pem \
  -e SSL_KEYFILE=/run/secrets/tls/privkey.pem \
  -v /etc/letsencrypt/live/inference.example.com:/run/secrets/tls:ro \
  roboflow/roboflow-inference-server-cpu:latest
```

## Encrypted private keys

When the key file is encrypted, supply the passphrase via
`SSL_KEYFILE_PASSWORD`. Prefer Docker secrets or another secret store
over baking the value into the image.

## Mutual TLS (client certs)

Set `SSL_CA_CERTS` to a CA bundle that should be used to verify client
certificates. Only clients presenting a certificate signed by one of the
listed CAs will be allowed.
