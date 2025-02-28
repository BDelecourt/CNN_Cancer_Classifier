import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define CNN Architecture
class BreastCancerCNN(nn.Module):
    def __init__(self, num_classes=3):  # 3 classes: Normal, Benign, Malignant
        super(BreastCancerCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)  # Flatten image to vector
        self.fc2 = nn.Linear(512, num_classes)  # Output layer

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv3(x)))  # Conv3 -> ReLU -> Pool
        
        x = torch.flatten(x, start_dim=1)  # Flatten tensor
        x = F.relu(self.fc1(x))  # Fully connected 1
        x = self.fc2(x)  # Fully connected 2 (output)
        
        return x

# Create the model
model = BreastCancerCNN(num_classes=3)

# Check model output shape
sample_input = torch.randn(1, 1, 224, 224)  # 1 random grayscale image, size 224x224
output = model(sample_input)
print(output.shape)  # Expected output: torch.Size([1, 3])