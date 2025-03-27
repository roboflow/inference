import argparse
import base64
import json
import os
from io import BytesIO

import faiss
import numpy as np
import requests
from flask import Flask, render_template, request, send_from_directory
from PIL import Image

parser = argparse.ArgumentParser(description="Build a search engine with CLIP")

parser.add_argument(
    "--dataset_path", type=str, required=True, help="Path to dataset", default="images"
)
parser.add_argument(
    "--inference_endpoint",
    type=str,
    required=True,
    help="Roboflow Inference endpoint URL",
    default="http://localhost:9001",
)
parser.add_argument(
    "--api_key",
    type=str,
    required=True,
    help="Roboflow API key",
    default=os.environ.get("ROBOFLOW_API_KEY")
)

args = parser.parse_args()

app = Flask(__name__)


def get_image_embedding(image: str) -> dict:
    image = image.convert("RGB")

    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    payload = {
        "image": {"type": "base64", "value": image},
    }

    data = requests.post(
        args.inference_endpoint + "/clip/embed_image?api_key=" + args.api_key,
        json=payload,
    )

    response = data.json()

    embedding = response["embeddings"]

    return embedding

if os.path.exists("index.bin"):
    index = faiss.read_index("index.bin")
    with open("database.json", "r") as f:
        file_names = json.load(f)
else:
    index = faiss.IndexFlatL2(512)
    file_names = []

    for frame_name in os.listdir(args.dataset_path):
        try:
            frame = Image.open(os.path.join(args.dataset_path, frame_name))
        except IOError:
            print("error computing embedding for", frame_name)
            continue

        embedding = get_image_embedding(frame)

        index.add(np.array(embedding).astype(np.float32))

        file_names.append(frame_name)

    faiss.write_index(index, "index.bin")

    with open("database.json", "w") as f:
        json.dump(file_names, f)


@app.route("/", methods=["GET", "POST"])
def search():
    if request.method == "POST":
        file = request.files["image"]
        query = get_image_embedding(Image.open(file))

        _, I = index.search(np.array(query).astype(np.float32), 3)

        images = [file_names[i] for i in I[0]]

        print(images)

        return render_template("index.html", images=images)

    return render_template("index.html")


@app.route("/images/<path:path>")
def send_image(path):
    return send_from_directory(args.dataset_path, path)


if __name__ == "__main__":
    app.run()
