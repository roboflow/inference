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

## Starting Point

We will start with an output similar to the [Hello World](hello-world.md) tutorial.
Follow that guide to create a simple Workflow that visualizes predictions from
an object detection model:

![Starting Point](https://media.roboflow.com/workflows/guides/sahi/01-starting-point.webp)

![Testing Without SAHI](https://media.roboflow.com/workflows/guides/sahi/02-testing-without-sahi.webp)

<!-- ![Output Without SAHI](https://media.roboflow.com/workflows/guides/sahi/03-output-without-sahi.webp) -->

![Original Predictions](https://media.roboflow.com/workflows/guides/sahi/original-predictions.jpeg)

![Start SAHI Path](https://media.roboflow.com/workflows/guides/sahi/04-start-sahi-path.webp)

![Add Image Slicer](https://media.roboflow.com/workflows/guides/sahi/05-add-image-slicer.webp)

![Add Block](https://media.roboflow.com/workflows/guides/sahi/06-add-block.webp)

![Add Model](https://media.roboflow.com/workflows/guides/sahi/07-add-model.webp)

![Add Detections Stitch](https://media.roboflow.com/workflows/guides/sahi/08-add-detections-stitch.webp)

![Add Visualizations](https://media.roboflow.com/workflows/guides/sahi/09-add-visualizations.webp)

![Add Background Color](https://media.roboflow.com/workflows/guides/sahi/10-add-background-color.webp)

![Add Bounding Box](https://media.roboflow.com/workflows/guides/sahi/11-add-bounding-box.webp)

![Adjust Outputs](https://media.roboflow.com/workflows/guides/sahi/12-adjust-outputs.webp)

![Test SAHI](https://media.roboflow.com/workflows/guides/sahi/13-test-sahi.webp)

<!-- ![Output With SAHI](https://media.roboflow.com/workflows/guides/sahi/14-output-with-sahi.webp) -->

![Output with SAHI](https://media.roboflow.com/workflows/guides/sahi/with-sahi.jpeg)

![Add Filter](https://media.roboflow.com/workflows/guides/sahi/15-add-filter.webp)

![Select Detections Filter](https://media.roboflow.com/workflows/guides/sahi/16-select-detections-filter.webp)

![Choose Operations](https://media.roboflow.com/workflows/guides/sahi/17-choose-operations.webp)

![Select Class Confidence](https://media.roboflow.com/workflows/guides/sahi/18-select-class-confidence.webp)

![Filter By Class](https://media.roboflow.com/workflows/guides/sahi/19-filter-by-class.webp)

![Test With Filter](https://media.roboflow.com/workflows/guides/sahi/20-test-with-filter.webp)

<!-- ![Output With Filter](https://media.roboflow.com/workflows/guides/sahi/21-output-with-filter.webp) -->

![Output with SAHI and Filter](https://media.roboflow.com/workflows/guides/sahi/with-sahi-and-filter.jpeg)
