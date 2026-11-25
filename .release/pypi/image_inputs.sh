#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
image="inference-pypi-inputs:${GITHUB_RUN_ID:-local}"
docker build --tag "$image" --file .release/pypi/Dockerfile .release/pypi
image_id="$(docker image inspect --format '{{.Id}}' "$image")"
revision="$(git rev-parse HEAD)"
docker run --rm \
    --user "$(id -u):$(id -g)" -e HOME=/tmp -e UV_PYTHON_DOWNLOADS=never \
    -e GH_TOKEN -e GITHUB_EVENT_NAME -e GITHUB_REF -e RELEASE_TAG \
    -e SOURCE_REVISION="$revision" -e RELEASE_IMAGE_ID="$image_id" \
    -e GITHUB_RUN_ID -e GITHUB_RUN_ATTEMPT -e GITHUB_REPOSITORY \
    -v "$PWD:/work" -w /work "$image_id" python /image_inputs.py "$@"
