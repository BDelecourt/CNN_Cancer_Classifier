import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define CNN Architecture
class BreastCancerCNN(nn.Module):
    def __init__(self, num_classes=3):  # 3 classes: Normal, Benign, Malignant
        super(BreastCancerCNN, self).__init__()

        # Convolutional layers
        nb_conv2d=5
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        # Pooling layer

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 7 * 7 , 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)  # Output layer

        self.dropout = nn.Dropout(0.1)  # Drops 10% of neurons randomly

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv3(x)))  # Conv3 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv4(x)))  # Conv4 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv5(x)))  # Conv5 -> ReLU -> Pool
        
        x = torch.flatten(x, start_dim=1)  # Flatten tensor
        x = F.relu(self.fc1(x)) # Fully connected 1
        x = self.dropout(x)  # Drop neurons to prevent overfitting
        x = F.relu(self.fc2(x)) # Fully connected 2
        x = self.dropout(x)  # Drop neurons to prevent overfitting
        x = F.relu(self.fc3(x)) # Fully connected 3
        x = self.fc4(x)  # Fully connected 4 (output)
        
        return x
