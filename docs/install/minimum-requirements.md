# Minimum Requirements

Inference adapts to your machine's resources and will run faster on more powerful machines
but it cannot run on all devices.

In order to run Inference, we recommend the following minimum requirements:

* 64-bit Processor
* 4GB of RAM
* 20GB of free disk space

??? info "Additional Requirements for Windows"
    To run on Windows, you need Windows 10 or Windows 11 with
    Windows Subsystem for Linux (WSL 2) activated.

## GPU Recommended

Inference is capable of using hardware acceleration on NVIDIA GPUs. While not required,
to run bigger models and live streaming video we recommend
[a CUDA-capable GPU](https://developer.nvidia.com/cuda-gpus).

## Suggested Edge Devices

NVIDIA Jetson Orin and Thor devices are powerful, well-rounded machines, and our
test-suite regularly runs against them. We recommend JetPack 7.2, which supports both
Orin and Thor. JetPack 6.2 and JetPack 5.1.x are still supported, but support for both
ends in 2027 (5.1.x is deprecated), so plan to migrate to JetPack 7.2.

The [Jetson Orin Nano Super Developer Kit](https://www.seeedstudio.com/NVIDIAr-Jetson-Orintm-Nano-Developer-Kit-p-5617.html)
is a good device to start building with.