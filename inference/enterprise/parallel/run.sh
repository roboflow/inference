#!/bin/bash
# run from root of inference like ./inference/enterpise/parallel/run.sh
PORT=${PORT:=9001}
TAG="latest"
docker run --rm --network=host --gpus=all --shm-size=8g --name inference-parallel -e PORT=$PORT -v parallel:/cache -it roboflow/roboflow-inference-server-gpu-parallel:$TAG
