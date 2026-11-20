#!/usr/bin/env bash
# Run a test shard with bounded attempts. Never turn exhaustion into success.
set -euo pipefail
attempts="${RETRY_ATTEMPTS:-1}"
if ! [[ "$attempts" =~ ^[1-3]$ ]]; then
    echo "::error::RETRY_ATTEMPTS must be 1, 2, or 3"
    exit 2
fi
attempt_seconds="${RETRY_TIMEOUT_SECONDS:-1800}"
if ! [[ "$attempt_seconds" =~ ^[1-9][0-9]*$ ]]; then
    echo "::error::RETRY_TIMEOUT_SECONDS must be a positive integer"
    exit 2
fi
log_dir="${RETRY_LOG_DIR:?RETRY_LOG_DIR is required}"
mkdir -p "$log_dir"
: "${RETRY_COMMAND:?RETRY_COMMAND is required}"
for ((attempt=1; attempt<=attempts; attempt++)); do
    echo "::group::Test attempt $attempt/$attempts"
    set +e
    if [ "$attempts" -gt 1 ]; then
        timeout --kill-after=10s "$attempt_seconds" bash -e -o pipefail -c "$RETRY_COMMAND" </dev/null
    else
        bash -e -o pipefail -c "$RETRY_COMMAND" </dev/null
    fi
    status=$?
    set -e
    echo "::endgroup::"
    printf '%s %s\n' "$attempt" "$status" >> "$log_dir/attempts.txt"
    if [ "$status" -eq 0 ]; then
        if [ "$attempt" -gt 1 ]; then
            echo "::warning::Test shard recovered on attempt $attempt; inspect earlier attempts in the Actions console"
        fi
        exit 0
    fi
    # Interruptions and termination are not test flakiness.
    if [ "$status" -eq 130 ] || [ "$status" -eq 143 ]; then exit "$status"; fi
    if [ "$attempt" -lt "$attempts" ]; then sleep "${RETRY_DELAY_SECONDS:-10}"; fi
done
echo "::error::Test shard failed after $attempts attempts"
exit "$status"
