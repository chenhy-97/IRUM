import torch
import torch.nn as nn


class CustomNetwork(nn.Module):
    def __init__(self):
        super(CustomNetwork, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(1024, 512)  # First fully connected layer from 512 to 256
        self.fc2 = nn.Linear(512, 256)  # Second fully connected layer from 256 to 64
        self.fc3 = nn.Linear(256, 64)  # Second fully connected layer from 256 to 64
        self.fc4 = nn.Linear(64, 256)  # Third fully connected layer from 64 to 256
        self.fc5 = nn.Linear(256, 512)  # Fourth fully connected layer from 256 to 512
        self.fc6 = nn.Linear(512, 1024)  # First fully connected layer from 512 to 256

        # Define activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Forward pass through the network
        x1 = x
        x = self.relu(self.fc1(x))
        x_residual = x  # Save the residual (skip connection before the non-linearity)

        x = self.relu(self.fc2(x))
        x_residual1 = x
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x_residual1 = x + x_residual1
        x = self.relu(self.fc5(x))

        x = x + x_residual  # Add the residual (skip connection)

        x = self.fc6(x)
        x = x + x1
        return x