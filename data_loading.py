import os
import torch.utils.data
from torch.utils.data import Dataset
from PIL import Image

# Define dataset path
DATASET_PATH = "Dataset_BUSI_with_GT"

# Custom Dataset Class
class BreastCancerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        # Define categories
        self.class_map = {"benign": 0, "malignant": 1, "normal": 2}

        # Label all data
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
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("L")  # Convert to grayscale

        if self.transform:
            image = self.transform(image)
        
        return image, label

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