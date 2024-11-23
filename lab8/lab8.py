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

# Combine the two Sobel gradients (you can either sum or compute the magnitude)
sobel_sum = cv2.add(sobel_horizontal, sobel_vertical)

# Perform Canny edge detection
cannyThreshold = 100
cannyParam2 = 200  # Typically 2-3 times the lower threshold
canny_edges = cv2.Canny(blur_5x5, cannyThreshold, cannyParam2)

# Function to apply a threshold using a for loop
def threshold_sobel(sobel_sum, threshold):
    thresholded_image = np.zeros_like(sobel_sum)  # Create an empty image of same size
    for i in range(sobel_sum.shape[0]): #iterates over all the rows of the sobel sum img
        for j in range(sobel_sum.shape[1]): #iterates over all the columns of each row
            if sobel_sum[i, j] > threshold:
                thresholded_image[i, j] = 1
    return thresholded_image #it will return the pixels(1 = if they greater than the given threshold value, and 0 = not greater)

# Threshold values to test
thresholds = [50, 100, 150, 200, 250]

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

#visualize the thresholded Sobel sum images for different thresholds
plt.figure(figsize=(12, 8))

# Plot original Sobel sum image
plt.subplot(2, 3, 1)
plt.imshow(sobel_sum, cmap='gray')
plt.title('Sobel Sum')
plt.xticks([]), plt.yticks([])

# Plot thresholded results for different thresholds
for idx, threshold in enumerate(thresholds): #loops over each threshold in the list. (enumerate gives both the index and the threshold value for each iteration)
    thresholded_image = threshold_sobel(sobel_sum, threshold) #call the function i wrote earlier
    plt.subplot(2, 3, idx + 2) #creates a new plot for each thresholded img
    plt.imshow(thresholded_image, cmap='gray')
    plt.title(f'Threshold {threshold}') #adds a title and showing its value of the threshold
    plt.xticks([]), plt.yticks([])

# Show the plots for thresholded images
plt.tight_layout()
plt.show()