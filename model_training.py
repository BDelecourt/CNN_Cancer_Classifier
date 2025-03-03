import torch
import os
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from model_initialisation import BreastCancerCNN
from data_loading import BreastCancerDataset,split_dataset,DATASET_PATH
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create Model
model=BreastCancerCNN(num_classes=3)



#Load Dataset
dataset_path=DATASET_PATH
print(f"dataset path: {dataset_path}")

# Image transformations: Resize to 224x224, normalize, and convert to tensor
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure 1-channel grayscale
    transforms.Resize((224, 224)),  # Resize images
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
])

# Create dataset instance
dataset = BreastCancerDataset(root_dir=dataset_path, transform=transform)
print(f"Dataset size: {len(dataset)} items")

train_indices, val_indices, test_indices=split_dataset(dataset)

# Create subsets
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Check dataset size
print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

# Define Loss Function (CrossEntropy for multi-class classification)
criterion = nn.CrossEntropyLoss()

# Define Optimizer (Adam is a popular choice for CNNs)
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs=int(input("Choose Epoch for this training:"))

if os.path.exists("breast_cancer_cnn.pth"): #Load already trained model
    checkpoint = torch.load("breast_cancer_cnn.pth")  # Load the full checkpoint
    model.load_state_dict(checkpoint["model_state_dict"])  # Extract only the model state
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])  # Load optimizer state
    epoch_start = checkpoint["epoch"] +1  # Get last saved epoch
    num_epochs=epoch_start+num_epochs
    
else : 
    epoch_start = 0
    

for epoch in range(epoch_start,num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  # Reset gradients to avoid accumulation
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagation (compute gradients)
        optimizer.step()  # Update weights

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# Model Evaluation
model.eval()  # Set model to evaluation mode (disables dropout, batch norm)

correct = 0
total = 0

with torch.no_grad():  # Disable gradient calculation (we don’t update weights here)
    for images, labels in val_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # Get class with highest probability
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

val_acc = correct / total
print(f"Validation Accuracy: {val_acc:.4f}")

# Save model
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, "breast_cancer_cnn.pth")