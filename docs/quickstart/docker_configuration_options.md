# Docker Configuration Options
Inference servers have a number of configurable parameters which can be set using environment variables. To set an environment variable with the docker run command, use the -e flag with an argument <variable name>=<variable value>.

```bash
#Example Docker run command with environment variable
docker run -it --rm -e ENV_VAR_NAME=env_var_value -p 9001:9001 --gpus all roboflow/roboflow-inference-server-trt:latest
```

**CLASS_AGNOSTIC_NMS**: Boolean (default = False)

Sets the default non-maximal suppression (NMS) behavior for detection type models (object detection, instance segmentation, etc.).  If True, the default NMS behavior will be class be class agnostic,  meaning overlapping detections from different classes may be removed based on the IoU threshold. If False, only overlapping detections from the same class will be considered for removal by NMS.

**ALLOW_ORIGINS**: String (default = "")

Sets the allow_origins property on the CORSMiddleware used with FastAPI for HTTP interfaces. Multiple values can be provided separated by a comma (ex. ALLOW_ORIGINS=orig1.com,orig2.com).

**CLIP_VERSION_ID**: String (default = ViT-B-16)

Sets the OpenAI CLIP version for use by all /clip routes. Available model versions are: RN101, RN50, RN50x16, RN50x4, RN50x64, ViT-B-16, BiT-B-32, BiT-L-14-336px, and ViT-L-14.

**CLIP_MAX_BATCH_SIZE**: Integer (default = 8)

Sets the max batch size accepted by the clip model inference functions.

**FIX_BATCH_SIZE**: Boolean (default = False)

If true, the batch size will be fixed to the maximum batch size configured for this server.

**HOST**: String (default = 0.0.0.0)

Sets the host address used by HTTP interfaces.

**LICENSE_SERVER**: String (default = None)

Sets the address of a Roboflow license server.

**MAX_ACTIVE_MODELS**: Integer (default = 8)

Sets the maximum number of models the internal model manager will store in memory at one time. By default, the model queue will remove the least recently accessed model when making space for a new model.

**MAX_CANDIDATES**: Integer (default = 3000)

The maximum number of candidates for detection.

**MAX_DETECTIONS**: Integer (default = 300)

Sets the maximum number of detections returned by a model.

**MODEL_CACHE_DIR**: String (default = /tmp/cache)

Sets the container path for the root model cache directory.

**NUM_WORKERS**: Integer (default = 1)

Sets the number of workers used by HTTP interfaces. 

**PORT**: Integer (default = 9001)

Sets the port used by HTTP interfaces.

**TENSORRT_CACHE_PATH**: String (default = MODEL_CACHE_DIR)

Sets the container path to the TensorRT cache directory. Setting this path in conjunction with mounting a host volume can reduce the cold start time of TensorRT based servers.