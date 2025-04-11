import torch
import torchvision
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import matplotlib
import matplotlib.pyplot as plt
from skimage import io

print("All imports successful!")
print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Matplotlib version: {matplotlib.__version__}") 