You can run Inference directly on frames using UDP.

This is ideal for real-time use cases where reducing latency is essential (i.e. sports broadcasting).

This feature only works on devices with a CUDA-enabled GPU.

Inference has been used at sports broadcasting events around the world for real-time object detection.

!!! tip "Follow our [Run a Fine-Tuned Model on Images](/quickstart/run_model_on_image) guide to learn how to find a model to run."

## Run a Vision Model on a UDP Stream

To run inference on frames from a UDP stream, you will need to:

1. Set up a listening server to receive predictions from Inference, and;
2. Run Inference, connected directly to your stream.

### Authenticate with Roboflow

To use Inference with a UDP stream, you will need a Roboflow API key. If you don't already have a Roboflow account, sign up for a free Roboflow account. Then, retrieve your API key from the Roboflow dashboard. Run the following command to set your API key in your coding environment:

```
export ROBOFLOW_API_KEY=<your api key>
```

### Configure a Listening Server

You need a server to receive predictions from Inference. This server is where you can write custom logic to process predictions.

Create a new Python file and add the following code:

```python
import socket
import json
import time


HOST = "localhost"
PORT = 8000

fps_array = []

# Create a datagram (UDP) socket
UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

# Bind to the given IP address and port
UDPClientSocket.bind((HOST, PORT))

print(f"UDP server up and listening on http://{HOST}:{PORT}")

# Listen for incoming datagrams
while True:
    t0 = time.time()

    bytesAddressPair = UDPClientSocket.recvfrom(1024)
    message = bytesAddressPair[0]
    address = bytesAddressPair[1]

    clientMsg = json.loads(message)
    clientIP = "Client IP Address:{}".format(address)

    print(clientMsg)
    print(clientIP)

    t = time.time() - t0
    fps_array.append(1 / t)
    fps_array[-150:]
    fps_average = sum(fps_array) / len(fps_array)
    print("AVERAGE FPS: " + str(fps_average))
```

Above, replace `port` with the port on which you want to run your server.

### Run a Broadcasting Server

- set up socket
- render will broadcast
- https://hub.docker.com/repository/docker/roboflow/roboflow-inference-server-udp-gpu/general
