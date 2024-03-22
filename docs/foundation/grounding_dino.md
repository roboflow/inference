<a href="https://github.com/IDEA-Research/GroundingDINO" target="_blank">Grounding DINO</a> is a zero-shot object detection model.

You can use Grounding DINO to identify objects in images and videos using arbitrary text prompts.

To use Grounding DINO effectively, we recommend experimenting with the model to understand which text prompts help achieve the desired results.

!!! note

    Grounding DINO is most effective at identifying common objects (i.e. cars, people, dogs, etc.). It is less effective at identifying uncommon objects (i.e. a specific type of car, a specific person, a specific dog, etc.).

### How to Use Grounding DINO

First, install the Inference Grounding DINO extension:

```
pip install "inference[grounding-dino]"
```

Create a new Python file called `app.py` and add the following code:

```python
from inference.models.grounding_dino import GroundingDINO

model = GroundingDINO(api_key="")

results = model.infer(
    {
        "image": {
            "type": "url",
            "value": "https://media.roboflow.com/fruit.png",
        },
        "text": ["apple"],

        # Optional params
        "box_threshold": 0.5
        "text_threshold": 0.5
    }
)

print(results.json())
```

In this code, we load Grounding DINO, run Grounding DINO on an image, and annotate the image with the predictions from the model.

Above, replace:

1. `apple` with the object you want to detect.
2. `fruit.png` with the path to the image in which you want to detect objects.

Additionally, you can tweak the optional `box_threshold` and `class_threshold` params for your specific use case. Both values default to 0.5 if not set. See the <a href="https://github.com/IDEA-Research/GroundingDINO?tab=readme-ov-file#star-explanationstips-for-grounding-dino-inputs-and-outputs">Grounding DINO README</a> for an explanation of the model's thresholds.

To use Grounding DINO with Inference, you will need a Roboflow API key. If you don't already have a Roboflow account, <a href="https://app.roboflow.com" target="_blank">sign up for a free Roboflow account</a>. Then, retrieve your API key from the Roboflow dashboard. Run the following command to set your API key in your coding environment:

```
export ROBOFLOW_API_KEY=<your api key>
```

Then, run the Python script you have created:

```
python app.py
```

The predictions from your model will be printed to the console.
