import torch
import os
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from model_initialisation_seg import UNet
from data_loading import BreastCancerDataset, split_dataset, DATASET_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create Model
model = UNet(in_channels=1, out_channels=1).to(device)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load segmentation dataset
dataset = BreastCancerDataset(root_dir=DATASET_PATH, transform=transform, segmentation=True)
train_indices, val_indices, test_indices = split_dataset(dataset)

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# Define loss (BCE + Dice Loss)
def dice_loss(pred, target, smooth=1e-5):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)
    return 1 - dice.mean()

criterion = lambda pred, target: 0.5 * nn.BCELoss()(pred, target) + 0.5 * dice_loss(pred, target)

optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs=int(input("Choose Epoch for this training:"))

sample_image, _ = next(iter(train_loader))
sample_image = sample_image.to(device)

with torch.no_grad():
    sample_output = model(sample_image)

print(f"Model output shape: {sample_output.shape}")


for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0  # Track loss over the epoch

    for batch_idx, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        # Check for NaNs in outputs
        if torch.isnan(outputs).any():
            print(f"NaN detected in model outputs at batch {batch_idx}")
            break

        loss = criterion(outputs, masks)

        # Check for NaNs in loss
        if torch.isnan(loss):
            print(f"NaN detected in loss at batch {batch_idx}")
            break

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.6f}")

    print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {running_loss / len(train_loader):.6f}")


# Save the trained model
torch.save(model.state_dict(), "tumor_segmentation_unet.pth")
print("Model saved successfully!")
