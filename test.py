
import torch
grayscale_image = torch.tensor([[0, 128, 255], [64, 192, 32], [255, 0, 128]])
print(grayscale_image.shape)  # Output: torch.Size([3, 3])

# Example: 28x28 grayscale image with random pixel values
image = torch.randn(28, 28)  # Shape: (28, 28)
print(image.shape)  # Output: torch.Size([28, 28])
