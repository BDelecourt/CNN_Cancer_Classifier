import os
import torch.utils.data
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# Define dataset path
DATASET_PATH = "Dataset_BUSI_with_GT"

# Custom Dataset Class
class BreastCancerDataset(Dataset):
    def __init__(self, root_dir, transform=None, segmentation=False):
        self.root_dir = root_dir
        self.transform = transform
        self.segmentation = segmentation  # New flag for segmentation task
        self.data = []

        if segmentation:
            # Load images and corresponding masks
            for label_name in ["benign", "malignant", "normal"]:
                category_path = os.path.join(root_dir, label_name)
                for img_name in os.listdir(category_path):
                    if "mask" in img_name:  # Skip mask images here, they are loaded separately
                        continue
                    img_path = os.path.join(category_path, img_name)
                    mask_path = img_path.replace(".png", "_mask.png")  # masks follow "image_name"_mask.png
                    self.data.append((img_path, mask_path))  # Store image-mask pairs
        else:
            # Load images with labels for classification
            self.class_map = {"benign": 0, "malignant": 1, "normal": 2}
            for label_name in self.class_map.keys():
                category_path = os.path.join(root_dir, label_name)
                for img_name in os.listdir(category_path):
                    if "mask" in img_name:  # Skip mask images
                        continue
                    img_path = os.path.join(category_path, img_name)
                    self.data.append((img_path, self.class_map[label_name]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.segmentation:
            img_path, mask_path = self.data[idx]
            image = Image.open(img_path).convert("L")  # Convert to grayscale
            mask = Image.open(mask_path).convert("L")  # Convert to grayscale (ensures single channel)

            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)
                
            mask = (mask > 0).float()  # Ensure binary mask (0 or 1 only)
            return image, mask  # Return image-mask pair
        else:
            img_path, label = self.data[idx]
            image = Image.open(img_path).convert("L")  # Convert to grayscale

            if self.transform:
                image = self.transform(image)

            return image, label  # Return image-label pair

def split_dataset(dataset, train=0.8, val=0.1, split_path="dataset_splits.pth"):
    """Split dataset into train, validation, and test sets & save the splits."""
    
    if train + val > 1:
        raise ValueError("Train + Val proportions cannot exceed 1.0")
    
    # Check if splits already exist
    try:
        splits = torch.load(split_path)
        train_indices, val_indices, test_indices = splits["train"], splits["val"], splits["test"]
        print("Loaded existing dataset splits.")
    except FileNotFoundError:
        print("Generating new dataset splits...")
        dataset_size = len(dataset)
        train_size = int(train * dataset_size)
        val_size = int(val * dataset_size)
        test_size = dataset_size - train_size - val_size

        # Get shuffled indices
        indices = torch.randperm(dataset_size).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        # Save splits
        torch.save({"train": train_indices, "val": val_indices, "test": test_indices}, split_path)
        print("Dataset splits saved.")

    return train_indices, val_indices, test_indices