#!/usr/bin/env bash
# Prime third-party dependencies from the real package metadata before source COPY.
# This directory intentionally contains metadata only. Temporary wheels carry no
# application code and are uninstalled before the dependency layer is committed.
set -euo pipefail

family="${1:?Expected colab or parallel}"
case "$family" in
  colab|parallel) ;;
  *) exit 2 ;;
esac
test ! -e inference/__init__.py
mkdir -p inference_sdk inference_cli dist
python3 -m pip install --upgrade pip
python3 -m pip install setuptools wheel twine requests uv

if [ "$family" = colab ]; then
  python3 -m pip uninstall -y torchaudio
  # Keep the build-time dependencies that the existing Colab make target installs.
  python3 -m pip install -r requirements/_requirements.txt \
    -r requirements/requirements.cpu.txt -r requirements/requirements.http.txt \
    -r requirements/requirements.sdk.http.txt
fi

for package in core gpu sdk cli; do
  rm -rf build
  python3 ".release/pypi/inference.$package.setup.py" bdist_wheel
done
python3 -m pip wheel --no-deps --no-build-isolation --wheel-dir dist \
  ./workflows ./stream_vision
if [ "$family" = colab ]; then
  python3 -m pip wheel --no-deps --no-build-isolation --wheel-dir dist ./inference_models
fi

if [ "$family" = colab ]; then
  python3 -m pip install --find-links dist/ \
    "$(ls dist/roboflow_workflows*.whl)[databricks]" \
    dist/streamvision*.whl dist/inference_gpu*.whl
  python3 -m pip install -r requirements/requirements.test.colab.txt
else
  python3 -m pip install --find-links dist/ \
    "$(ls dist/roboflow_workflows*.whl)[databricks]" \
    dist/streamvision*.whl dist/inference_cli*.whl dist/inference_core*.whl \
    dist/inference_gpu*.whl dist/inference_sdk*.whl "setuptools~=84.2.1"
fi

python3 -m pip uninstall -y roboflow-workflows streamvision \
  inference-gpu inference-core inference-cli inference-sdk
if [ "$family" = colab ]; then
  python3 -m pip uninstall -y inference-models
fi

