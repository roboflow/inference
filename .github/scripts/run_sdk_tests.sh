#!/usr/bin/env bash
set -euo pipefail
mode="${1:?Choose unit, integration, e2e or install}"
python_version="${2:-3.12}"
case "$python_version" in 3.10|3.11|3.12|3.13) ;; *) exit 2 ;; esac
case "$mode" in
    unit)
        target=sdk-tests
        command=(python -m pytest tests/inference_sdk/unit_tests -v)
        ;;
    integration)
        target=sdk-tests
        command=(python -m pytest tests/inference_sdk/integration_tests -v --timeout=300)
        ;;
    e2e)
        target=sdk-e2e
        command=(python -m pytest tests/inference_sdk/e2e_tests -m slow -v --timeout=300)
        ;;
    install)
        target=sdk-install
        command=(python -c 'from pathlib import Path; import inference_sdk; from inference_sdk import InferenceHTTPClient; from importlib.metadata import version; assert not Path(inference_sdk.__file__).is_relative_to("/work"); print(version("inference-sdk"), inference_sdk.__file__)')
        ;;
    *) exit 2 ;;
esac
cd "$(git rev-parse --show-toplevel)"
image="inference-sdk-validation:${python_version}-${mode}"
docker build --file .github/docker/Dockerfile.sdk-tests \
    --build-arg "PYTHON_VERSION=$python_version" --target "$target" --tag "$image" .
image_id="$(docker image inspect --format '{{.Id}}' "$image")"
printf 'Source revision: %s\nDocker image: %s\nCommand:' "$(git rev-parse HEAD)" "$image_id"
printf ' %q' "${command[@]}"
printf '\n'
docker run --rm --shm-size=1g "$image_id" "${command[@]}"
