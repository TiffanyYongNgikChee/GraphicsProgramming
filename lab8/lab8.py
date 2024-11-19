import cv2
import numpy as np 
from matplotlib import pyplot as plt

# Load the input image
img = cv2.imread('ATU.jpg',)

# Convert the image to greyscale
gray_image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur with different kernel sizes
blur_3x3 = cv2.GaussianBlur(gray_image, (3, 3), 0)
blur_5x5 = cv2.GaussianBlur(gray_image, (5, 5), 0)
blur_9x9 = cv2.GaussianBlur(gray_image, (9, 9), 0)
blur_13x13 = cv2.GaussianBlur(gray_image, (13, 13), 0)

# Plot the results
plt.figure(figsize=(12, 8))

# Plot using Matplotlib
nrows, ncols = 2, 1

# Plot the original image
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
plt.title('Original Image')
plt.xticks([]), plt.yticks([])

# Plot the grayscale image
plt.subplot(2, 2, 2)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')
plt.xticks([]), plt.yticks([])

# 3x3 Gaussian Blur
plt.subplot(2, 2, 3)
plt.imshow(blur_3x3, cmap='gray')
plt.title('3x3 Blur')
plt.xticks([]), plt.yticks([])

# 13x13 Gaussian Blur
plt.subplot(2, 2, 4)
plt.imshow(blur_13x13, cmap='gray')
plt.title('13x13 Blur')
plt.xticks([]), plt.yticks([])

# Show the plots
plt.tight_layout()
plt.show()
