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

# Apply the Sobel operator
sobel_horizontal = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)  # x direction
sobel_vertical = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)    # y direction

# Combine Sobel outputs
sobel_sum = cv2.add(np.absolute(sobel_horizontal), np.absolute(sobel_vertical))

# Perform Canny edge detection
cannyThreshold = 100
cannyParam2 = 200  # Typically 2-3 times the lower threshold
canny_edges = cv2.Canny(blur_5x5, cannyThreshold, cannyParam2)

# Plot the results
plt.figure(figsize=(12, 8))

# Plot using Matplotlib
nrows, ncols = 2, 1

# Plot the original image
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
plt.title('Original Image')
plt.xticks([]), plt.yticks([])

# Plot the grayscale image
plt.subplot(2, 3, 2)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')
plt.xticks([]), plt.yticks([])

# 3x3 Gaussian Blur
#plt.subplot(2, 3, 3)
#plt.imshow(blur_3x3, cmap='gray')
#plt.title('3x3 Blur')
#plt.xticks([]), plt.yticks([])

# 13x13 Gaussian Blur
#plt.subplot(2, 3, 4)
#plt.imshow(blur_13x13, cmap='gray')
#plt.title('13x13 Blur')
#plt.xticks([]), plt.yticks([])

# Plot horizontal Sobel output
plt.subplot(2, 3, 3)
plt.imshow(sobel_horizontal, cmap='gray')
plt.title('Sobel X')
plt.xticks([]), plt.yticks([])

# Plot vertical Sobel output
plt.subplot(2, 3, 4)
plt.imshow(sobel_vertical, cmap='gray')
plt.title('Sobel Y')
plt.xticks([]), plt.yticks([])

# Plot Combined Sobel
plt.subplot(2, 3, 5)
plt.imshow(sobel_sum, cmap='gray')
plt.title('Sobel Sum')
plt.xticks([]), plt.yticks([])

# Canny edges
plt.subplot(2, 3, 6)
plt.imshow(canny_edges, cmap='gray')
plt.title('Canny Edge Image')
plt.xticks([]), plt.yticks([])

# Show the plots
plt.tight_layout()
plt.show()
