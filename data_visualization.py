import os
import matplotlib.pyplot as plt
from PIL import Image

# Define dataset path
dataset_path = "Dataset_BUSI_with_GT"

# Define categories
categories = ["benign", "malignant", "normal"]

# Plot sample images from each category
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for i, category in enumerate(categories):
    category_path = os.path.join(dataset_path, category)
    sample_image = os.listdir(category_path)[0]  # Take the first image
    image = Image.open(os.path.join(category_path, sample_image))
    
    axes[i].imshow(image, cmap="gray")
    axes[i].set_title(category)
    axes[i].axis("off")

plt.show()