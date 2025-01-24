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