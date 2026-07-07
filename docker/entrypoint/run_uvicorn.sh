#!/bin/sh
# Wrapper that launches uvicorn with optional HTTPS flags driven by env vars.
#
# Usage: run_uvicorn.sh <app_module:attribute> [extra uvicorn args...]
#
# Recognised env vars:
#   HOST, PORT, NUM_WORKERS  - standard uvicorn options
#   ENABLE_HTTPS             - "true"/"1"/"yes" enables HTTPS
#   SSL_CERTFILE             - path to PEM cert (required when HTTPS enabled)
#   SSL_KEYFILE              - path to PEM key  (required when HTTPS enabled)
#   SSL_KEYFILE_PASSWORD     - optional key passphrase
#   SSL_CA_CERTS             - optional CA bundle for client cert verification
#
# Any extra arguments are forwarded verbatim to uvicorn.
set -e

APP="$1"
if [ -z "$APP" ]; then
    echo "run_uvicorn.sh: missing required app module argument" >&2
    exit 64
fi
shift

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-9001}"
NUM_WORKERS="${NUM_WORKERS:-1}"

is_truthy() {
    case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
        1|true|t|yes|y) return 0 ;;
        *) return 1 ;;
    esac
}

SSL_CERTFILE="${SSL_CERTFILE:-/etc/inference/certs/server.crt}"
SSL_KEYFILE="${SSL_KEYFILE:-/etc/inference/certs/server.key}"

SSL_ARGS=""
if is_truthy "$ENABLE_HTTPS"; then
    if [ ! -r "$SSL_CERTFILE" ] || [ ! -r "$SSL_KEYFILE" ]; then
        echo "run_uvicorn.sh: ENABLE_HTTPS is set but SSL_CERTFILE ($SSL_CERTFILE) and SSL_KEYFILE ($SSL_KEYFILE) must both be readable" >&2
        exit 78
    fi
    SSL_ARGS="--ssl-certfile $SSL_CERTFILE --ssl-keyfile $SSL_KEYFILE"
    if [ -n "$SSL_KEYFILE_PASSWORD" ]; then
        SSL_ARGS="$SSL_ARGS --ssl-keyfile-password $SSL_KEYFILE_PASSWORD"
    fi
    if [ -n "$SSL_CA_CERTS" ]; then
        SSL_ARGS="$SSL_ARGS --ssl-ca-certs $SSL_CA_CERTS"
    fi
fi

if command -v uvicorn >/dev/null 2>&1; then
    set -- uvicorn "$APP" --workers "$NUM_WORKERS" --host "$HOST" --port "$PORT" $SSL_ARGS "$@"
else
    set -- python3 -m uvicorn "$APP" --workers "$NUM_WORKERS" --host "$HOST" --port "$PORT" $SSL_ARGS "$@"
fi

if is_truthy "$LEGACY_MMP_ADAPTER_ENABLED"; then
    # Run the ModelManagerProcess supervisor next to uvicorn. Workers reach it
    # via INFERENCE_MMP_ADDR; the SHM pool name is discovered over T_SHM_INFO.
    INFERENCE_MMP_ADDR="${INFERENCE_MMP_ADDR:-ipc:///tmp/inference_mmp.sock}"
    export INFERENCE_MMP_ADDR
    python3 mmp_supervisor.py &
    MMP_PID=$!
    "$@" &
    UV_PID=$!
    trap 'kill -TERM "$MMP_PID" "$UV_PID" 2>/dev/null' TERM INT
    # Either process dying takes the container down so both restart together;
    # workers never re-handshake a new SHM pool mid-life.
    while kill -0 "$MMP_PID" 2>/dev/null && kill -0 "$UV_PID" 2>/dev/null; do
        sleep 5
    done
    kill -TERM "$MMP_PID" "$UV_PID" 2>/dev/null
    wait "$UV_PID" 2>/dev/null
    wait "$MMP_PID" 2>/dev/null
    exit 1
fi

exec "$@"
