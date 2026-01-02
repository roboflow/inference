from inference import get_model
import numpy as np

model = get_model("resent18", api_key="Em3CODkJT1vbZ52wlWwk")

img_in, metadata = model.preprocess("new-york-style-pizza2.jpg")
output = model.predict(img_in)

raw = np.array(output[0][0][0])  # Shape (1000,)
print(f"Raw range: [{raw.min():.6f}, {raw.max():.6f}]")
print(f"Raw sum: {raw.sum():.4f}")  # If ~1.0, already probabilities

# Top 5 WITHOUT extra softmax
top5_idx = np.argsort(raw)[-5:][::-1]
print(f"\nTop 5 (no extra softmax):")
for idx in top5_idx:
    print(f"  {idx}: {model.class_names[idx]} = {raw[idx]:.4f}")

# What happens WITH extra softmax (what the code does)
softmaxed = np.exp(raw - np.max(raw)) / np.exp(raw - np.max(raw)).sum()
print(f"\nAfter extra softmax, max value: {softmaxed.max():.6f}")