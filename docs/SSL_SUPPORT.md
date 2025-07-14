# SSL Support for Inference Server

This document describes the SSL/TLS support implementation for the Roboflow Inference Server.

## Overview

The Inference Server now supports serving traffic over HTTPS in addition to HTTP. This enables secure communication with the server, especially useful when:
- Communicating with browsers that require HTTPS
- Using the `roboflow.host` Dynamic DNS SSL indirection domain
- Deploying in production environments requiring encrypted traffic

## Configuration

SSL support is controlled by environment variables:

### `ENABLE_SSL`
- Type: Boolean
- Default: `False`
- Description: Enables SSL/HTTPS server alongside the HTTP server

### `SSL_PORT`
- Type: Integer
- Default: `9002`
- Description: Port for HTTPS traffic. If not set, defaults to HTTP port + 1 (except 80 → 443)

### `SSL_CERTIFICATE`
- Type: String
- Default: `"INDIRECT"`
- Description: Certificate configuration mode:
  - `"INDIRECT"` - Downloads and uses roboflow.host wildcard certificate (renewed daily)
  - `"GENERATE"` - Generates and caches a self-signed certificate
  - `/path/to/cert.pem` - Uses a custom certificate file

## Usage Examples

### Using roboflow.host wildcard certificate (default):
```bash
docker run -e ENABLE_SSL=True -p 9001:9001 -p 9002:9002 roboflow/inference-server:latest
```

### Using self-signed certificate:
```bash
docker run -e ENABLE_SSL=True -e SSL_CERTIFICATE=GENERATE -p 9001:9001 -p 9002:9002 roboflow/inference-server:latest
```

### Using custom certificate:
```bash
docker run -e ENABLE_SSL=True -e SSL_CERTIFICATE=/certs/mycert.pem -v /path/to/certs:/certs -p 9001:9001 -p 9002:9002 roboflow/inference-server:latest
```

### Custom SSL port:
```bash
docker run -e ENABLE_SSL=True -e SSL_PORT=8443 -p 9001:9001 -p 8443:8443 roboflow/inference-server:latest
```

## Certificate Management

### roboflow.host Wildcard Certificate
When using `SSL_CERTIFICATE=INDIRECT` (default), the server:
1. Downloads the wildcard certificate from `https://roboflow.host/certificates/`
2. Caches it locally in the model cache directory
3. Checks for updates once per day
4. Hot-swaps the certificate if it has been renewed (no downtime)

This certificate works for any `*.roboflow.host` domain, enabling the Dynamic DNS functionality where IP addresses are encoded in the subdomain (e.g., `192-168-1-1.roboflow.host`).

### Self-Signed Certificates
When using `SSL_CERTIFICATE=GENERATE`, the server:
1. Generates a 2048-bit RSA self-signed certificate
2. Caches it for reuse across restarts
3. Includes SANs for `localhost`, `127.0.0.1`, and `*.roboflow.host`

### Custom Certificates
Provide the path to a PEM-encoded certificate file. The server expects:
- Certificate file at the specified path
- Private key in the same directory with `.key` extension
- Or following common naming patterns (e.g., `fullchain.pem` → `privkey.pem`)

## Architecture

The SSL implementation:
1. Runs HTTP and HTTPS servers in parallel when SSL is enabled
2. Uses separate processes to allow independent scaling
3. HTTPS server runs with a single worker to simplify certificate management
4. Certificate updates happen in a background thread without server restart

## Testing

Test the SSL functionality:

```bash
# Run tests
python tests/test_ssl_functionality.py

# Start server with SSL
ENABLE_SSL=True SSL_CERTIFICATE=GENERATE python start_server.py

# Test endpoints
curl http://localhost:9001/
curl -k https://localhost:9002/  # -k accepts self-signed certs
```

## Security Considerations

1. **Private Key Protection**: Private keys are stored with 0600 permissions
2. **Certificate Validation**: When downloading roboflow.host certificates, ensure you're on a trusted network
3. **Self-Signed Certificates**: Only use for development/testing. Browsers will show security warnings
4. **Certificate Rotation**: roboflow.host certificates are checked daily for updates

## Troubleshooting

- **Port already in use**: Check if another process is using the SSL port
- **Certificate download fails**: Ensure internet connectivity and firewall allows HTTPS to roboflow.host
- **Permission denied**: Ensure the process has write access to the cache directory
- **SSL handshake errors**: Check certificate validity and that private key matches certificate

## Future Enhancements

- Support for Let's Encrypt certificates with auto-renewal
- Certificate monitoring and alerting
- Support for client certificate authentication
- HTTP/2 and HTTP/3 support
