import torch
import os
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torchinfo import summary
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
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Flip images
    transforms.RandomRotation(10),  # Rotate randomly
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),  # Distort image slightly  # Resize images
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  
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

if os.path.exists("breast_cancer_cnn.pth"): #Load already trained model
    checkpoint = torch.load("breast_cancer_cnn.pth")  # Load the full checkpoint
    model.load_state_dict(checkpoint["model_state_dict"])  # Extract only the model state
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])  # Load optimizer state

summary(model, input_size=(1, 1, 224, 224))

num_epochs=int(input("Choose Epoch for this training:"))

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_acc = correct_train / total_train

        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct_val / total_val

        # Store Loss and Accuracy
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

    return train_losses, val_losses, train_accs, val_accs

def plot_metrics(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)

    # Plot Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train Accuracy')
    plt.plot(epochs, val_accs, label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()

    plt.show()

train_losses, val_losses, train_accs, val_accs= train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)

# Save model
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, "breast_cancer_cnn.pth")

plot_metrics(train_losses, val_losses, train_accs, val_accs)