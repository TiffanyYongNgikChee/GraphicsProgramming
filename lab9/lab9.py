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

# Threshold the Harris corner response
dst = cv2.dilate(dst, None)  # Dilate to enhance corner points
imgHarris[dst > 0.01 * dst.max()] = [0, 0, 255]  # Mark corners in red

# Define a threshold value for corner detection
threshold = 0.01  # You can experiment with this threshold value

# Loop through every element in the Harris corner response matrix
for i in range(len(dst)):
    for j in range(len(dst[i])):
        if dst[i][j] > threshold * dst.max():
            # Draw a circle at the corner location (j, i)
            cv2.circle(imgHarris, (j, i), 3, (0, 0, 255), -1)  # Red color for the circle

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

# Plot the image with detected corners
plt.subplot(3, 3, 3)
plt.imshow(cv2.cvtColor(imgHarris, cv2.COLOR_BGR2RGB))  # Convert to RGB for display
plt.title('Harris Corners')
plt.xticks([]), plt.yticks([])

# Show the plots
plt.tight_layout()
plt.show()
