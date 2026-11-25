#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
image="inference-pypi-publish:${GITHUB_RUN_ID:-local}"
docker build --tag "$image" --file .release/pypi/Dockerfile .release/pypi
image_id="$(docker image inspect --format '{{.Id}}' "$image")"
revision="$(git rev-parse HEAD)"
printf 'Source revision: %s\nDocker image: %s\nOperation: %s\n' "$revision" "$image_id" "$*"
docker run --rm \
    -e EXPECTED_VERSION -e EXPECTED_REVISION -e PUBLISH \
    -e SOURCE_REVISION="$revision" -e RELEASE_IMAGE_ID="$image_id" \
    -e GITHUB_RUN_ID -e GITHUB_RUN_ATTEMPT -e GITHUB_REPOSITORY \
    -v "$PWD:/work" -w /work "$image_id" python /publish.py "$@"
