# Gaze Detection Example

## 👋 Hello

This repository shows how to use the [Roboflow Inference](https://github.com/roboflow/inference) gaze detection API.

## 💻 Getting Started

First, clone this repository and navigate to the project folder:

```bash
git clone https://github.com/roboflow/inference
cd inference/examples/gaze-detection
```

Next, set up a Python environment and install the required project dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Next, set up a Roboflow Inference Docker container. This Docker container will manage inference for the gaze detection system. [Learn how to set up an Inference Docker container](https://inference.roboflow.com/quickstart/docker/).

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
  
Replace `your_api_key_here` with your Roboflow API key. [Learn how to retrieve your Roboflow API key](https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key).

## 🎬 Run Inference

To use the gaze detection script, run the following command:

```bash
python gaze.py
```