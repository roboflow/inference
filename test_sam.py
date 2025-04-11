import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])  # Light blue, semi-transparent
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0, x1, y1 = box
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def show_points(points, labels, ax):
    pos_points = points[labels == 1]
    neg_points = points[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=100, label='Foreground')
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=100, label='Background')

def main():
    # Load the model (make sure you have the model checkpoint downloaded)
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    # Load and preprocess a test image
    image = cv2.imread("test_image.jpg")
    if image is None:
        raise ValueError("Could not load test_image.jpg. Please ensure it exists in the current directory.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Set image
    predictor.set_image(image)
    
    # Define input box (x0, y0, x1, y1)
    input_box = np.array([100, 100, 800, 600])  # Box around the truck
    
    # Define input points (both foreground and background)
    input_points = np.array([
        [400, 300],  # Point on the truck body
        [600, 400],  # Another point on the truck
        [200, 500],  # Background point (sky)
        [700, 200]   # Background point (sky)
    ])
    input_labels = np.array([1, 1, 0, 0])  # 1 for foreground, 0 for background
    
    # Get masks using both box and points
    masks, scores, logits = predictor.predict(
        box=input_box,
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True
    )
    
    # Visualize the results
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks[0], plt.gca())
    show_box(input_box, plt.gca())
    show_points(input_points, input_labels, plt.gca())
    plt.axis('off')
    plt.savefig('segmentation_result_combined.png')
    print("Results saved as 'segmentation_result_combined.png'")

if __name__ == "__main__":
    main() 