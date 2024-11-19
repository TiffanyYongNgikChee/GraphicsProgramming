import cv2
import numpy as np 
from matplotlib import pyplot as plt

# Load the input image
img = cv2.imread('ATU.jpg',)

# Convert the image to greyscale
gray_image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Display the images
cv2.imshow('Original Image', img)
cv2.imshow('Grayscale Image', gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Plot using Matplotlib
nrows, ncols = 2, 1

# Plot the original image
plt.subplot(nrows, ncols, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
plt.title('Original Image')
plt.xticks([]), plt.yticks([])

# Plot the grayscale image
plt.subplot(nrows, ncols, 2)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')
plt.xticks([]), plt.yticks([])

# Show the plots
plt.tight_layout()
plt.show()
