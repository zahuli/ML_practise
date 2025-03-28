import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# Load FashionMNIST dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Class labels for FashionMNIST
class_labels = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]


# Define the model

class FashionMNISTModel(nn.Module):
    def __init__(self):
        super(FashionMNISTModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Load model and weights
model = FashionMNISTModel()
model.load_state_dict(torch.load("fashion_mnist_model.pth"))
model.eval()  # Set the model to evaluation mode


# Select an image from the test dataset
sample_idx = 13  # Change this index to test different images
image, label = test_dataset[sample_idx]

# Prepare the image for the model
image_tensor = image.unsqueeze(0)  # Add batch dimension


# Get the model prediction
with torch.no_grad():
    output = model(image_tensor)
    predicted_label = torch.argmax(output, dim=1).item()

# Display the image with the predicted label
plt.imshow(image.squeeze(), cmap='gray')
plt.title(
    f'Predicted: {class_labels[predicted_label]}, Actual: {class_labels[label]}')
plt.axis('off')
plt.show()
