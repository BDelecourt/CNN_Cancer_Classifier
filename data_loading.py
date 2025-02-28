import os
from torchvision import transforms
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Define dataset path
dataset_path = "Dataset_BUSI_with_GT"

# Image transformations: Resize to 224x224, normalize, and convert to tensor
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure 1-channel grayscale
    transforms.Resize((224, 224)),  # Resize images
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
])

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
                self.data.append((img_path, label_name))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("L")  # Convert to grayscale

        if self.transform:
            image = self.transform(image)
        
        return image, label

def split_dataset(dataset,train=0.8,val=0.1):
    "Split dataset into train (80%), validation (10%), test (10%)"
    if train+val>=1:
        raise Exception("Incorrect dataset repartition") # does not allow test dataset
    train_size = int(train * len(dataset))
    val_size = int(val * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    return train_dataset, val_dataset, test_dataset


# Create dataset instance
dataset = BreastCancerDataset(root_dir=dataset_path, transform=transform)
print(f"Dataset size: {len(dataset)} items")

train_dataset, val_dataset, test_dataset=split_dataset(dataset)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Check dataset size
print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

# Get a batch from train_loader
images, labels = next(iter(train_loader))

print(images.shape)  # Should be (16, 1, 224, 224) for grayscale images
print(labels)  # labels ('normal', 'malignant', 'benign')
print(images.min(), images.max())  # Should be around -1 to 1 (normalized)
