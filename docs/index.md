---
hide:
  - path
  - navigation
  - toc
---

<video width="100%" autoplay loop muted>
  <source src="https://github.com/user-attachments/assets/743233d9-3460-442d-83f8-20e29e76b346" type="video/mp4">
</video>

## Make Any Camera an AI Camera

Inference turns any computer or edge device into a command center for your computer vision projects.

* 🛠️ Self-host [your own fine-tuned models](/quickstart/explore_models.md)
* 🧠 Access the latest and greatest foundation models (like [Florence-2](https://blog.roboflow.com/florence-2/), [CLIP](https://blog.roboflow.com/openai-clip/), and [SAM2](https://blog.roboflow.com/what-is-segment-anything-2/))
* 🤝 Use Workflows to track, count, time, measure, and visualize
* 👁️ Combine ML with traditional CV methods (like OCR, Barcode Reading, QR, and template matching)
* 📈 Monitor, record, and analyze predictions
* 🎥 Manage cameras and video streams
* 📬 Send notifications when events happen
* 🛜 Connect with external systems and APIs
* 🔗 [Extend](/workflows/create_workflow_block.md) with your own code and models
* 🚀 Deploy production systems at scale

See [Example Workflows](/workflows/gallery/index.md) for common use-cases like detecting small objects with SAHI, multi-model consensus, active learning, reading license plates, blurring faces, background removal, and more.

<a href="/quickstart/run_a_model/" class="button">Get started with our "Run your first model" guide</a>
<div class="button-holder">
  <a href="/quickstart/inference_101/" class="button half-button">Learn about the various ways you can use Inference</a>
  <a href="/workflows/about/" class="button half-button">Build a visual agent with Workflows</a>
</div>

## Video Tutorials

<div class="tutorial-list">
  <!-- Tutorial 1 -->
  <div class="tutorial-item">
    <a href="https://youtu.be/tZa-QgFn7jg">
      <img src="https://img.youtube.com/vi/tZa-QgFn7jg/0.jpg" alt="Smart Parking with AI" />
    </a>
    <div class="tutorial-content">
      <a href="https://youtu.be/tZa-QgFn7jg">
        <strong>Tutorial: Build a Smart Parking System</strong>
      </a>
      <div><strong>Created: 27 Nov 2024</strong></div>
      <p>
        Build a smart parking lot management system using Roboflow Workflows!
        This tutorial covers license plate detection with YOLOv8, object tracking
        with ByteTrack, and real-time notifications with a Telegram bot.
      </p>
    </div>
  </div>

  <!-- Tutorial 2 -->
  <div class="tutorial-item">
    <a href="https://youtu.be/VCbcC5OEGRU">
      <img src="https://img.youtube.com/vi/VCbcC5OEGRU/0.jpg" alt="Workflows Tutorial" />
    </a>
    <div class="tutorial-content">
      <a href="https://youtu.be/VCbcC5OEGRU">
        <strong>Tutorial: Build a Traffic Monitoring Application with Workflows</strong>
      </a>
      <div><strong>Created: 22 Oct 2024</strong></div>
      <p>
        Learn how to build and deploy Workflows for common use-cases like detecting 
        vehicles, filtering detections, visualizing results, and calculating dwell 
        time on a live video stream.
      </p>
    </div>
  </div>
  
  <!-- Add more .tutorial-item blocks as needed -->
</div>

<style>
.button-holder {
  margin-bottom: 1.5rem;
}

.button {
  background-color: var(--md-primary-fg-color);
  display: flex;
  padding: 10px;
  color: white !important;
  border-radius: 5px;
  text-align: center;
  align-items: center;
  justify-content: center;
}

.md-typeset h2 {
  margin-top: 1rem;
}

ul {
  line-height: 1rem;
}

/* Hide <h1> on homepage */
.md-typeset h1 {
  display: none;
}
.md-main__inner {
  margin-top: -1rem;
}

/* constrain to same width even w/o sidebar */
.md-content {
  max-width: 50rem;
  margin: auto;
}

/* hide edit button */
article > a.md-content__button.md-icon:first-child {
    display: none;
}

/* tutorial list styling */
.tutorial-list {
  display: flex;
  flex-direction: column;
  margin-bottom: 2rem;
}

.tutorial-item {
  display: flex;
  flex-direction: row;
  align-items: center;
  gap: 1rem;

  border-bottom: 1px solid #ccc;
  padding-bottom: 0.5rem;
  margin-bottom: 1rem;
}

.tutorial-item:last-of-type {
  border-bottom: none;
  margin-bottom: 0;
}

.tutorial-item img {
  max-width: 300px;
  height: auto;
  flex-shrink: 0;
  border: 1px solid #ccc;
}

.tutorial-content {
  flex: 1 1 auto;
}

/* On mobile: stack content and center the thumbnail */
@media (max-width: 768px) {
  .tutorial-item {
    flex-direction: column;
    align-items: center;
  }
  .tutorial-item img {
    max-width: 100%;
    margin: 0 auto;
    display: block;
  }
}
</style>