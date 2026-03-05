<a href="https://blog.roboflow.com/florence-2/" target="_blank">Florence-2</a> is a multimodal model developed by Microsoft Research.

You can use Florence-2 for:

1. Object detection: Identify the location of all objects in an image. (`<OD>`)
2. Dense region captioning: Generate dense captions for all identified regions in an image. (`<DENSE_REGION_CAPTION>`)
3. Image captioning: Generate a caption for a whole image. (`<CAPTION>` for a short caption, `<DETAILED_CAPTION>` for a more detailed caption, and `<MORE_DETAILED_CAPTION>` for an even more detailed caption)
4. Region proposal: Identify regions where there are likely to be objects in an image. (`<REGION_PROPOSAL>`)
5. Phrase grounding: Identify the location of objects that match a text description. (`<CAPTION_TO_PHRASE_GROUNDING>`)
6. Referring expression segmentation: Identify a segmentation mask that corresponds with a text input. (`<REFERRING_EXPRESSION_SEGMENTATION>`)
7. Region to segmentation: Calculate a segmentation mask for an object from a bounding box region. (`<REGION_TO_SEGMENTATION>`)
8. Open vocabulary detection: Identify the location of objects that match a text prompt. (`<OPEN_VOCABULARY_DETECTION>`)
9. Region to description: Generate a description for a region in an image. (`<REGION_TO_DESCRIPTION>`)
10. Optical Character Recognition (OCR): Read the text in an image. (`<OCR>`)
11. OCR with region: Read the text in a specific region in an image. (`<OCR_WITH_REGION>`)

You can use Inference for all the Florence-2 tasks above.

The text in the parentheses are the task prompts you will need to use each task.

### Execution Modes

Florence-2 supports both local and remote execution modes when used in workflows:

- **Local execution**: The model runs directly on your inference server (GPU recommended)
- **Remote execution**: The model can be invoked via HTTP API on a remote inference server

When using Florence-2 in a workflow, you can specify the execution mode to control where inference happens.

### How to Use Florence-2

??? Note "Install `inference`"

    To install `inference` with Florence 2 support use the following command on CPU machine:

    ```bash
    pip install inference[transformers]
    ```

    or the following one for GPU machine:

    ```bash
    pip install inference-gpu[transformers]
    ```

Create a new Python file called `app.py` and add the following code:

```python
from inference import get_model

model = get_model("florence-2-base", api_key="API_KEY")

result = model.infer(
    "https://media.roboflow.com/inference/seawithdock.jpeg", 
    prompt="<CAPTION>",
)

print(result[0].response)
```

Above, replace `<CAPTION>` with the name of the task you want to use.

Replace `API_KEY` with your Roboflow API key. [Learn how to retrieve your Roboflow API key](https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key)

To use PaliGemma with Inference, you will need a Roboflow API key. If you don't already have a Roboflow account, <a href="https://app.roboflow.com" target="_blank">sign up for a free Roboflow account</a>.

Then, run the Python script you have created:

```
python app.py
```

The result from your model will be printed to the console.
