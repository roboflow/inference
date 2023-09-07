# inference-client

<a href="https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc">
    <img src="https://app.roboflow.com/images/download-dataset-badge.svg"></img>
</a>
<a href="https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/model/">
    <img src="https://app.roboflow.com/images/try-model-badge.svg"></img>
</a>

## 👋 hello

This repository contains examples of image and video inference via [Roboflow Inference (HTTP)](https://github.com/roboflow/inference) and stream inference via [Roboflow Inference (UDP)](https://github.com/roboflow/inference).

The HTTP examples take in an image or video and run inference, whereas the UDP example listens for predictions from a UDP stream and processes them.

## 💻 install client environment

```bash
# clone repository and navigate to root directory
git clone https://github.com/roboflow/inference-client.git
cd inference-client

# setup python environment and activate it
python3 -m venv venv
source venv/bin/activate

# headless install
pip install -r requirements.txt
```

## 🐋 docker

You can learn more about Roboflow Inference Docker Image build, pull and run in our [documentation](https://roboflow.github.io/inference/quickstart/docker/).

### HTTP

- Run on x86 CPU:

  ```bash
  docker run --net=host roboflow/roboflow-inference-server-cpu:latest
  ```
    
- Run on Nvidia GPU:

  ```bash
  docker run --network=host --gpus=all roboflow/roboflow-inference-server-gpu:latest
  ```

### UDP

- Run on Nvidia GPU:

```bash
docker run --gpus=all --net=host -e STREAM_ID=0 -e MODEL_ID=<> -e API_KEY=<> roboflow/roboflow-inference-server-udp-gpu:latest
```
  
<details close>
<summary>👉 more docker run options</summary>

### HTTP

- Run on arm64 CPU:

  ```bash
  docker run -p 9001:9001 roboflow/roboflow-inference-server-arm-cpu:latest
  ```
  
- Run on Nvidia GPU with TensorRT Runtime:

  ```bash
  docker run --network=host --gpus=all roboflow/roboflow-inference-server-trt:latest
  ```
  
- Run on Nvidia Jetson with JetPack `4.x`:

  ```bash
  docker run --privileged --net=host --runtime=nvidia roboflow/roboflow-inference-server-jetson:latest
  ```
  
- Run on Nvidia Jetson with JetPack `5.x`:

  ```bash
  docker run --privileged --net=host --runtime=nvidia roboflow/roboflow-inference-server-jetson-5.1.1:latest
  ```

### UDP

We only support one UDP container at the moment. Refer to the UDP command from earlier to set up UDP.

</details>

## 🔑 keys

Before running the inference script, ensure that the `API_KEY` is set as an environment variable. This key provides access to the inference API.

- For Unix/Linux:

    ```bash
    export API_KEY=your_api_key_here
    ```

- For Windows:

    ```bash
    set API_KEY=your_api_key_here
    ```
  
Replace `your_api_key_here` with your actual API key.

## 📷 image inference example (HTTP)

To run the image inference script:

```bash
python image.py \
--image_path data/a9f16c_8_9.png \
--class_list "ball" "goalkeeper" "player" "referee" \
--dataset_id "football-players-detection-3zvbc" \
--version_id 2 \
--confidence 0.5
```

## 🎬 video inference example (HTTP)

To run the video inference script:

```bash
python video.py \
--video_path "data/40cd38_5.mp4" \
--class_list "ball" "goalkeeper" "player" "referee" \
--dataset_id "football-players-detection-3zvbc" \
--version_id 2 \
--confidence 0.5
```

## 📺 stream inference example (UDP)

To run the UDP receiver, run:

```bash
python udp.py --port=12345
```