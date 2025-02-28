# Detect Small Objects

The best way to improve the performance of a machine learning system is often
peripheral to improving the model. Collecting more data, labeling datasets
more accurately, adding redundancy to detections, or breaking down the problem
into smaller pieces are usually higher leverage endeavors than iterating on
model architecture.

One common shortcoming of model performance occurs when the objects of interest
are very small relative to the size of the image. One approach is tiling the image
into smaller chunks and feeding them through the model in parallel. This method 
is called SAHI (Slicing Aided Hyper Inference). A large image is divided into
overlapping slices, each slice is run through a model, and a single prediction
is reconstructed as the output.

Using SAHI in Workflows is easy using the Image Slicer and Detections Stitch blocks.

**Difficulty:** Easy<br />
**Time to Complete:** 10 minutes

## Starting Point

We will start with an output similar to the [Hello World](hello-world.md) tutorial.
Follow that guide to create a simple Workflow that visualizes predictions from
an object detection model:

![Starting Point](https://media.roboflow.com/workflows/guides/sahi/01-starting-point.webp)

But when the objects are small, this model fails to find them.

![Testing Without SAHI](https://media.roboflow.com/workflows/guides/sahi/02-testing-without-sahi.webp)

Even though cars are one of the most widely represented classes in the MS COCO
dataset that this model was trained on, it still only finds one of them because
when the image is scaled down to the model's input resolution the cars become
only a few pixels wide.

<!-- ![Output Without SAHI](https://media.roboflow.com/workflows/guides/sahi/03-output-without-sahi.webp) -->

![Original Predictions](https://media.roboflow.com/workflows/guides/sahi/original-predictions.jpeg)

## Adding SAHI

We will fix this by slicing the image into smaller tiles so that the model gets
more pixels of information to work from.

![Start SAHI Path](https://media.roboflow.com/workflows/guides/sahi/04-start-sahi-path.webp)

The first step is to add the Image Slicer block to the Workflow. We will do this
in a parallel execution path so that we can compare the outputs to the original
non-SAHI predictions at the end.

<!-- ![Add Image Slicer](https://media.roboflow.com/workflows/guides/sahi/05-add-image-slicer.webp) -->

![Add Image Slicer Block](https://media.roboflow.com/workflows/guides/sahi/06-add-block.webp)

Then we will add an Object Detection model that will run on each frame. Everything
between an Image Slicer and an Image Stitch block operates element-wise on the
set of slices. Workflows will automatically parallelize work (such as batching
predictions through the GPU) where it can.

![Add Model](https://media.roboflow.com/workflows/guides/sahi/07-add-model.webp)

Next, we'll add the Detections Stitch block to aggregate the predictions back into
the dimensionality of the original image.

![Add Detections Stitch](https://media.roboflow.com/workflows/guides/sahi/08-add-detections-stitch.webp)

Now, we can visualize these aggregated predictions in the same way as we
visualized outputs of a regular model.

![Add Visualizations](https://media.roboflow.com/workflows/guides/sahi/09-add-visualizations.webp)

We will add a Background Color Visualization to dim regions of the image that
were not predicted by the model to make the predicted areas stand out.

![Add Background Color](https://media.roboflow.com/workflows/guides/sahi/10-add-background-color.webp)

And a bounding box around predicted objects of interest.

![Add Bounding Box](https://media.roboflow.com/workflows/guides/sahi/11-add-bounding-box.webp)

## Testing SAHI

Now we're ready to see how our revised approach is working. We'll simplify the
output to predict only the two visualizations we're interested in.

![Adjust Outputs](https://media.roboflow.com/workflows/guides/sahi/12-adjust-outputs.webp)

When we run the Workflow we see that now all of the cars are detected!

![Test SAHI](https://media.roboflow.com/workflows/guides/sahi/13-test-sahi.webp)

<!-- ![Output With SAHI](https://media.roboflow.com/workflows/guides/sahi/14-output-with-sahi.webp) -->

![Output with SAHI](https://media.roboflow.com/workflows/guides/sahi/with-sahi.jpeg)

## Filter Unwanted Predictions

But the SAHI approach worked a little bit too well. The model is now picking up
some objects from the background that we weren't interested in. Let's fix this
by modifying the predictions from the model.

![Add Filter](https://media.roboflow.com/workflows/guides/sahi/15-add-filter.webp)

Add a Detections Filter block. This block lets us exclude predictions that don't
match a set of criteria that we define.

![Select Detections Filter](https://media.roboflow.com/workflows/guides/sahi/16-select-detections-filter.webp)

To set our criteria, click the Operations configuration in the block's sidebar.

![Choose Operations](https://media.roboflow.com/workflows/guides/sahi/17-choose-operations.webp)

Then choose "Filter By Class & Confidence".

![Select Class Confidence](https://media.roboflow.com/workflows/guides/sahi/18-select-class-confidence.webp)

And include only the `car` class.

![Filter By Class](https://media.roboflow.com/workflows/guides/sahi/19-filter-by-class.webp)

Now, when we test our model again we see boxes around the cars and no extraneous
detections in the background.

![Test With Filter](https://media.roboflow.com/workflows/guides/sahi/20-test-with-filter.webp)

<!-- ![Output With Filter](https://media.roboflow.com/workflows/guides/sahi/21-output-with-filter.webp) -->

![Output with SAHI and Filter](https://media.roboflow.com/workflows/guides/sahi/with-sahi-and-filter.jpeg)

## Next Steps

In this tutorial we learned how to use SAHI to detect small objects, how to perform
operations on a batch of images, and how to transform detections.

Next, we will [run a Workflow on a live video stream](/workflows/video_processing/overview.md).
