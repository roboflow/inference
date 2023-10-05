# Changelog

## v0.8.5

### Summary

Contains bug fixes for configurations that use the LICENSE_SERVER setting.

## v0.8.4

### Summary

Image loading is now multi-threaded for batch requests. This should increase total FPS, especially for batch requests that include large images.

The regression test Github action now runs on a Github actions runner.

The extras require has been fixed for the various distribution packages

## v0.8.2

### Summary

Updated the Jetson images so that the default execution provider is CUDA. TensorRT is now an optional configuration via the environment variable ONNXRUNTIME_EXECUTION_PROVIDERS=TensorrtExecutionProvider. The images are also renamed to:

- roboflow/roboflow-inference-server-jetson-4.5.0
- roboflow/roboflow-inference-server-jetson-4.6.1
- roboflow/roboflow-inference-server-jetson-5.1.1

## v0.8.1

### Summary

- Optional Byte Track for UDP interface
- Updated SAM and CLIP requirements and added README quickstarts
- Bug fix for single channel numpy strings

### Breaking Changes

The output for UDP JSON messages has updated the key class_name to class to match HTTP responses.

## v0.8.0

### Summary

This release includes a bit of an overhaul for the model APIs. As this repository started as an internal tool for hosting inference logic, the APIs were tailored for an HTTP interface. With this release, we have made using inference within your python code much smoother and easier. We also updated imports to be less verbose. See the README and docs for new usage. Additionally, a new interface is provided for consuming video streams, and then broadcasting the results over UDP. This interface is tuned for low latency and is ideal for use cases that need to the most up to date information as possible from a video stream. See https://blog.roboflow.com/udp-inference/ for more details.

### Breaking Changes

The main change was creating new definitions for model infer() functions that now take many keyword arguments instead of a single request argument. To continue inferring using request objects, a new method infer_from_request() is provided.

## v0.7.2

Initial release.