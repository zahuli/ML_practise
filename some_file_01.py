
# Example: Displaying a Handwritten Digit (MNIST)
# If train_images comes from a dataset like MNIST (grayscale 28x28 images of handwritten digits),
#  the image will be displayed with a color scale:

import matplotlib.pyplot as plt
import numpy as np

# Fake example image (28x28 grayscale)
train_images = np.random.rand(60000, 28, 28)  # Simulating a dataset

plt.figure()
plt.imshow(train_images[2], cmap="gray")  # Set colormap for grayscale
plt.colorbar()
plt.grid(False)
plt.show()
