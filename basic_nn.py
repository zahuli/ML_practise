# Here’s a basic neural network with PyTorch:
# This code builds and trains a simple neural network in just a few lines!

import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple model


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Create model, loss function, and optimizer
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Fake training loop
for epoch in range(100):
    inputs = torch.tensor([[1.0, 2.0]])
    target = torch.tensor([[0.0]])

    optimizer.zero_grad()
    output = model(inputs)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

print("Final loss:", loss.item())
