# CLIP Example

## 👋 Hello

This repository shows how to use the [Roboflow Inference](https://github.com/roboflow/inference) CLIP API to build a semantic search engine.

With this project, you can search for related images in a given folder. For example, if you upload a photo of a cat, the dataset you have provided will be searched for photos similar to the cat photo.

## 💻 Getting Started

First, clone this repository and navigate to the project folder:

```bash
git clone https://github.com/roboflow/inference
cd inference/examples/clip-client
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

To use the CLIP search engine, run the following command:

```bash
python app.py --dataset_path=./images
```

Replace `./images` with a folder that you want to search.

The first time you run the script, an image database will be created and saved to your system. The amount of time this takes will depend on how many images you want to be able to search. This database will be saved in two files:

- `index.bin`: The database that stores the image embeddings.
- `database.json`: The database that stores the image paths.

Both of these files are saved in the root folder of the project.

When the database has been built, it will be opened in subsequent runs of the script. If you want to search a new folder of images, delete the `index.bin` and `database.json` files and run the script again.

A web application will be available at `http://localhost:9001` with which you can search for images.