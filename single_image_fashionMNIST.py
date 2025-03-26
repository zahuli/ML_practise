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


# Instantiate model, loss function, and optimizer
model = FashionMNISTModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 3
for epoch in range(epochs):
    total_loss = 0.0
    num_batches = len(train_loader)
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / num_batches  # Average loss per batch
    print(f'Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}')

# Evaluate the model
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Test Accuracy: {accuracy:.4f}')


# Predict on a single image from the test set

# Select an image from the test dataset
sample_idx = 10  # Change this index to test different images
image, label = test_dataset[sample_idx]

# Prepare the image for the model
image_tensor = image.unsqueeze(0)  # Add batch dimension

# Get the model prediction
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    output = model(image_tensor)
    predicted_label = torch.argmax(output, dim=1).item()

# Display the image with the predicted label
plt.imshow(image.squeeze(), cmap='gray')
plt.title(
    f'Predicted: {class_labels[predicted_label]}, Actual: {class_labels[label]}')
plt.axis('off')
plt.show()


sample_idx = 15  # Change this index to test different images
image, label = test_dataset[sample_idx]

# Prepare the image for the model
image_tensor = image.unsqueeze(0)  # Add batch dimension

# Get the model prediction
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    output = model(image_tensor)
    predicted_label = torch.argmax(output, dim=1).item()

# Display the image with the predicted label
plt.imshow(image.squeeze(), cmap='gray')
plt.title(
    f'Predicted: {class_labels[predicted_label]}, Actual: {class_labels[label]}')
plt.axis('off')
plt.show()
