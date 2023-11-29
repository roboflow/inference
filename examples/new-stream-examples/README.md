# New stream interface examples
This folder contains examples that show how to use the `InferencePipeline` abstraction which is a new way to
infer against video files and video streams.

## `simple.py`

### Overview
This example provides simplistic version of `InferencePipeline` usage.

### How to run?
```bash
reposotory_root$ python -m examples.new-stream-examples.simple
```

Parameters:
* `--model_id`: name of the model, by default: `"rock-paper-scissors-sxsw/11"` from Roboflow Universe
* `--source`: reference to inference source, by default `0` which is usually webcam device, but you are free to
  pass video file or stream URL
