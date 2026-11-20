#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
mkdir -p release-state
if [ "${1:-}" = prepare ]; then
    cp "$GITHUB_EVENT_PATH" release-state/event.json
fi
image="inference-release-flow:${GITHUB_RUN_ID:-local}"
docker build --tag "$image" --file .release/pypi/Dockerfile .release/pypi
image_id="$(docker image inspect --format '{{.Id}}' "$image")"
docker run --rm \
    -e GH_TOKEN -e EXPECTED_VERSION -e EXPECTED_REVISION -e PACKAGES -e PUBLISH -e RELEASE_RESULTS \
    -e GITHUB_SHA -e GITHUB_REF -e GITHUB_EVENT_NAME -e GITHUB_REPOSITORY \
    -e GITHUB_RUN_ID -e GITHUB_RUN_ATTEMPT \
    -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0=/work \
    -v "$PWD:/work" -w /work "$image_id" python .release/pypi/flow.py "$@"
if [ -f release-state/outputs ] && [ -n "${GITHUB_OUTPUT:-}" ]; then
    cat release-state/outputs >> "$GITHUB_OUTPUT"
    rm release-state/outputs
fi
