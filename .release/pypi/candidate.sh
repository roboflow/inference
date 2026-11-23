#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
image="inference-pypi-candidate:${GITHUB_RUN_ID:-local}"
docker build --tag "$image" --file .release/pypi/Dockerfile .release/pypi
image_id="$(docker image inspect --format '{{.Id}}' "$image")"
revision="$(git rev-parse HEAD)"
docker run --rm \
    -e GH_TOKEN -e SOURCE_REVISION="$revision" \
    -e GITHUB_RUN_ID -e GITHUB_RUN_ATTEMPT -e GITHUB_REPOSITORY \
    -v "$PWD:/work" -w /work "$image_id" python /candidate.py "$@"
