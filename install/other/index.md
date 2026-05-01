# Using Other Devices

Inference is tested and supported x64 and ARM processors (optionally with a
NVIDIA/CUDA GPU). Running on other devices may be possible but is not officially
tested or supported.

## Other GPUs

We do not currently support hardware acceleration on non-NVIDIA, non-Apple GPUs but ONNX
Runtime has [additional execution providers](https://onnxruntime.ai/docs/execution-providers/)
for AMD/ROCm, Arm NN, Rockchip, and others.

If you install one of these runtimes, you can enable it via the `ONNXRUNTIME_EXECUTION_PROVIDERS`
environment variable. For example:
```
export ONNXRUNTIME_EXECUTION_PROVIDERS="[ROCMExecutionProvider,OpenVINOExecutionProvider,CPUExecutionProvider]"
```

This is untested and performance improvements are not guaranteed.
Acceleration of non-CUDA GPUs is unlikely to work inside Docker. See the [Mac GPU](mac.md)
installation guide for an example of how to run the server outside of a container.

## Other Edge Devices

Roboflow has
[SDKs for running object detection natively](https://docs.roboflow.com/deploy/supported-deployment-devices)
on other deployment targets like
[Tensorflow.js in a web browser](https://docs.roboflow.com/deploy/sdks/web-browser),
[Native Swift on iOS](https://docs.roboflow.com/deploy/sdks/ios-sdk) via CoreML, and
[Snap Lens Studio](https://docs.roboflow.com/deploy/sdks/lens-studio).

For additional functionality, like running Workflows and other types of models on another
device, connect to an Inference Server over HTTP [via its API](/api.md).
