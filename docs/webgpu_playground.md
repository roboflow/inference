# On-device WebGPU playground

The **WebGPU playground** is an opt-in page that turns a phone, tablet or laptop
browser into an on-device inference target. Open it from any device that can
reach your inference server, and it:

1. checks the device for WebGPU and reads the GPU adapter: vendor and
   architecture, fp16 shader support and buffer limits;
2. measures the GPU with two short compute shaders: memory bandwidth (a `vec4`
   copy over a large buffer) and fp32 throughput (a chain of fused
   multiply-adds);
3. estimates the decode speed of each model from the measured bandwidth;
4. downloads a small instruction-tuned language model once and chats with it,
   streaming tokens and reporting the decode speed measured on that GPU.

Everything after the page load runs in the browser, on the device's own GPU.
The server only hands out one static HTML page: prompts and answers stay on the
device, and the playground adds no inference endpoints.

## Enable it

The playground is off by default. Set `ENABLE_WEBGPU_PLAYGROUND=True` and open
`/webgpu` on the server:

```bash
docker run -p 9001:9001 \
    -e ENABLE_WEBGPU_PLAYGROUND=True \
    roboflow/roboflow-inference-server-cpu
```

Then browse to `http://<server-address>:9001/webgpu` from the phone. WebGPU
needs a secure context: `localhost` works as is; from another device, serve the
page over HTTPS (see the server's HTTPS configuration) or a tunnel.

## Supported browsers

| Device | Browser |
|---|---|
| Android | Chrome 121 or later |
| iPhone, iPad | Safari on iOS / iPadOS 26 or later (earlier versions: enable the WebGPU feature flag) |
| Desktop | Chrome, Edge, or Safari 26 or later |

On a GPU without `shader-f16`, the page loads the fp32 build of each model.

## Models

The models are prebuilt [WebLLM](https://github.com/mlc-ai/web-llm) builds,
4-bit, from the `mlc-ai` organisation on Hugging Face:

| Model | GPU memory (fp16 / fp32 build) | Notes |
|---|---|---|
| SmolLM2 360M Instruct | 376 / 580 MB | the default: smallest, quickest to download |
| Qwen2.5 0.5B Instruct | 945 / 1,060 MB | best answers for its size |
| Llama 3.2 1B Instruct | 879 / 1,129 MB | for newer phones |

A model downloads once and stays in the browser's cache; the page has a button
that clears it.

## Network access

The page itself is served by the inference server. When a visitor starts a
model, the browser fetches WebLLM (`@mlc-ai/web-llm@0.2.78`, Apache-2.0) from
jsDelivr and the model weights from Hugging Face. On an air-gapped network the
GPU check and measurement still work; loading a model needs those two hosts.
