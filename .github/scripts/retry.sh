#!/usr/bin/env bash
# Run a test shard with bounded attempts. Never turn exhaustion into success.
set -euo pipefail
attempts="${RETRY_ATTEMPTS:-1}"
if ! [[ "$attempts" =~ ^[1-3]$ ]]; then
    echo "::error::RETRY_ATTEMPTS must be 1, 2, or 3"
    exit 2
fi
log_dir="${RETRY_LOG_DIR:?RETRY_LOG_DIR is required}"
mkdir -p "$log_dir"
script="$log_dir/command.sh"
printf '%s\n' "${RETRY_COMMAND:?RETRY_COMMAND is required}" > "$script"
for ((attempt=1; attempt<=attempts; attempt++)); do
    echo "::group::Test attempt $attempt/$attempts"
    set +e
    bash -e -o pipefail "$script" 2>&1 | tee "$log_dir/attempt-$attempt.log"
    status=${PIPESTATUS[0]}
    set -e
    echo "::endgroup::"
    printf '%s %s\n' "$attempt" "$status" >> "$log_dir/attempts.txt"
    if [ "$status" -eq 0 ]; then
        if [ "$attempt" -gt 1 ]; then
            echo "::warning::Test shard recovered on attempt $attempt; inspect earlier attempt logs"
        fi
        exit 0
    fi
    # Interruptions and termination are not test flakiness.
    if [ "$status" -eq 130 ] || [ "$status" -eq 143 ]; then exit "$status"; fi
    if [ "$attempt" -lt "$attempts" ]; then sleep "${RETRY_DELAY_SECONDS:-10}"; fi
done
echo "::error::Test shard failed after $attempts attempts"
exit "$status"
