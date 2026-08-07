#!/usr/bin/env bash
# Start inference server with MMP + uvicorn.
# Serves plain HTTP by default. Set ENABLE_HTTPS=true to serve self-signed TLS;
# a cert is generated on first run (valid 365 days) and clients must then use
# --insecure / verify=False when connecting.
#
# Usage:
#   ./start_server.sh                       # defaults (plain HTTP)
#   ENABLE_HTTPS=true ./start_server.sh     # self-signed HTTPS
#   PORT=9443 NUM_WORKERS=8 ./start_server.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

is_truthy() {
    case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
        1|true|t|yes|y) return 0 ;;
        *) return 1 ;;
    esac
}

# ── Self-signed TLS (opt-in via ENABLE_HTTPS) ───────────────────────────────
if is_truthy "${ENABLE_HTTPS:-}"; then
    CERT_DIR="${CERT_DIR:-${SCRIPT_DIR}}"
    CERT_FILE="${CERT_DIR}/c.pem"
    KEY_FILE="${CERT_DIR}/c.key"

    if [ ! -f "${CERT_FILE}" ] || [ ! -f "${KEY_FILE}" ]; then
        echo "[start_server] Generating self-signed certificate..."

        # Suppress RANDFILE warning on some Linux distros
        if [ -f /etc/ssl/openssl.cnf ]; then
            sed -i 's/\(.*RANDFILE.*\)/# \1/g' /etc/ssl/openssl.cnf 2>/dev/null || true
        fi

        # Use primary IP as CN so the cert at least matches what clients connect to.
        # Falls back to plain hostname if 'hostname -I' is unavailable (macOS).
        CN="$(hostname -I 2>/dev/null | awk '{print $1}')"
        CN="${CN:-$(hostname)}"

        openssl req -new -x509 -days 365 -nodes -newkey rsa:2048 \
            -out  "${CERT_FILE}" \
            -keyout "${KEY_FILE}" \
            -subj "/C=US/O=Roboflow/CN=${CN}"

        echo "[start_server] Certificate: ${CERT_FILE}  (CN=${CN})"
    fi

    export SSL_CERTFILE="${CERT_FILE}"
    export SSL_KEYFILE="${KEY_FILE}"
else
    echo "[start_server] Serving plain HTTP (set ENABLE_HTTPS=true for TLS)"
fi

# ── Activate venv if present ────────────────────────────────────────────────
if [ -f "${SCRIPT_DIR}/venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/venv/bin/activate"
fi

# ── Launch ──────────────────────────────────────────────────────────────────
exec python -m inference_server.server
