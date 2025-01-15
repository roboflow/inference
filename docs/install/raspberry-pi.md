# Install on Raspberry Pi

Inference works on Raspberry Pi 4 Model B and Raspberry Pi 5 so long as you are using
[**the 64-bit version** of the operating system](https://www.raspberrypi.com/software/operating-systems/#raspberry-pi-os-64-bit) (if your SD Card is big enough, we recommend the 64-bit
"Raspberry Pi OS with desktop and recommended software" version).

Once you've installed the 64-bit OS,
[install Docker](https://docs.docker.com/engine/install/debian/) then use the
Inference CLI to automatically select, configure, and start the correct Inference
Docker container:

```bash
pip install inference-cli
inference server start
```

## Hardware Acceleration

Inference does not yet support any hardware acceleration on the Raspberry Pi. Expect
about 1fps on Pi 4 and 4fps on Pi 5 for a "Roboflow 3.0 Fast" object detection model
(equivalent to a "nano" sized YOLO model).

Larger models like Segment Anything and VLMs like Florence 2 will struggle to run with
high performance on the Pi's compute. If you need more power for higher framerates or
bigger models consider [an NVIDIA Jetson](jetson.md).

## Manually Starting the Container

If you want more control of the container settings you can also start it
manually.

```bash
sudo docker run -d \
    --name inference-server \
    --read-only \
    -p 9001:9001 \
    --volume ~/.inference/cache:/tmp:rw \
    --security-opt="no-new-privileges" \
    --cap-drop="ALL" \
    --cap-add="NET_BIND_SERVICE" \
    roboflow/roboflow-inference-server-cpu:latest
```

## Docker Compose

If you are using Docker Compose for your application, the equivalent yaml is:

```yaml
version: "3.9"

services:
    inference-server:
    container_name: inference-server
    image: roboflow/roboflow-inference-server-cpu:latest

    read_only: true
    ports:
        - "9001:9001"

    volumes:
        - "${HOME}/.inference/cache:/tmp:rw"

    security_opt:
        - no-new-privileges
    cap_drop:
        - ALL
    cap_add:
        - NET_BIND_SERVICE
```

--8<-- "docs/install/using-your-new-server.md"

--8<-- "docs/install/enterprise-considerations.md"
