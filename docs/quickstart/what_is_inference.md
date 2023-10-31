Roboflow Inference enables you to deploy computer vision models faster than ever.

Here is an example of a model running on a video using Inference ([See the code](https://github.com/roboflow/inference/blob/main/examples/inference-client/video.py)):

<video width="100%" autoplay loop muted>
  <source src="https://media.roboflow.com/football-video.mp4" type="video/mp4">
</video>

Before Inference, deploying models on device involved:

1. Writing custom inference logic, which often requires machine learning knowledge.
2. Managing dependencies.
3. Optimizing for performance and memory usage.
4. Writing tests to ensure your inference logic worked.
5. Writing custom interfaces to run your model over webcams and streams, if you were deploying live.

Inference handles all of this, out of the box.

With a single pip install and one command to start Inference, you can set up a server that runs a fine-tuned model on any image or video stream.

Inference supports running object detection, classification, instance segmentation, and even foundation models (like CLIP and SAM). You can train and deploy your own custom model or use one of the [50,000+ fine-tuned models shared by the community](https://universe.roboflow.com).