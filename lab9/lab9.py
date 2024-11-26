import cv2
import numpy as np 
from matplotlib import pyplot as plt

# Load the input image
img = cv2.imread('ATU1.jpg',)

# Convert the image to greyscale
gray_image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Perform Harris corner detection
blockSize = 2
aperture_size = 3
k = 0.04
dst = cv2.cornerHarris(gray_image, blockSize, aperture_size, k)

# Create a deep copy of the original image
imgHarris = img.copy()

# Plot the results
plt.figure(figsize=(12, 8))

# Plot using Matplotlib
nrows, ncols = 2, 1

# Plot the original image
plt.subplot(3, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
plt.title('Original Image')
plt.xticks([]), plt.yticks([])

# Plot the grayscale image
plt.subplot(3, 3, 2)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')
plt.xticks([]), plt.yticks([])

# Show the plots
plt.tight_layout()
plt.show()
