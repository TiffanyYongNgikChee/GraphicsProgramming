import cv2
import numpy as np 
from matplotlib import pyplot as plt

# Load the input image
img = cv2.imread('ATU1.jpg',)
img2 = cv2.imread('rome.jpg',) # Second Image

# Convert the image to greyscale
gray_image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray_image2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

# Perform Harris corner detection
blockSize = 2
aperture_size = 3
k = 0.04
dst = cv2.cornerHarris(gray_image, blockSize, aperture_size, k)
dst2 = cv2.cornerHarris(gray_image2, blockSize, aperture_size, k)

# Create a deep copy of the original image
imgHarris = img.copy()
imgHarris2 = img2.copy()

# Define a threshold value for corner detection
threshold = 0.01  # You can experiment with this threshold value

# Threshold the Harris corner response
dst = cv2.dilate(dst, None)  # Dilate to enhance corner points
dst2 = cv2.dilate(dst2, None)
imgHarris[dst > 0.01 * dst.max()] = [0, 0, 255]  # Mark corners in red
imgHarris2[dst2 > threshold * dst2.max()] = [0, 0, 255]


# Loop through every element in the Harris corner response matrix
for i in range(len(dst)):
    for j in range(len(dst[i])):
        if dst[i][j] > threshold * dst.max():
            # Draw a circle at the corner location (j, i)
            cv2.circle(imgHarris, (j, i), 3, (0, 0, 255), -1)  # Red color for the circle

# Perform Shi-Tomasi corner detection (Good Features to Track)
maxCorners = 100 
qualityLevel = 0.01
minDistance = 10
corners = cv2.goodFeaturesToTrack(gray_image, maxCorners=maxCorners, qualityLevel=qualityLevel, minDistance=minDistance)
corners2 = cv2.goodFeaturesToTrack(gray_image2, maxCorners, qualityLevel, minDistance)

# Convert the corners array from 3D to 2D if necessary
corners = np.int0(corners)  # Convert float coordinates to integers
corners2 = np.int0(corners2)

# Create a deep copy of the original image
imgShiTomasi = img.copy()
imgShiTomasi2 = img2.copy()

# Loop through each corner detected by GFTT
for i in corners:
    x, y = i.ravel()  # Flatten the corner coordinates
    cv2.circle(imgShiTomasi, (x, y), 3, (0, 255, 0), -1)  # Draw green circles for the corners

for i in corners2:
    x, y = i.ravel()
    cv2.circle(imgShiTomasi2, (x, y), 3, (0, 255, 0), -1)

# Create an ORB object with the function - Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints with ORB
kp = orb.detect(img,None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)
kp2, des2 = orb.detectAndCompute(img2, None)
 
# draw only keypoints location,not size and orientation
imgORB = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
imgORB2 = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0), flags=0)

# Plot the results
plt.figure(figsize=(15, 10))

# ATU1.jpg plots
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('ATU1 - Original')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(gray_image, cmap='gray')
plt.title('ATU1 - Grayscale')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(imgHarris, cv2.COLOR_BGR2RGB))
plt.title('ATU1 - Harris Corners')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(cv2.cvtColor(imgShiTomasi, cv2.COLOR_BGR2RGB))
plt.title('ATU1 - Shi-Tomasi')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(imgORB, cv2.COLOR_BGR2RGB))
plt.title('ATU1 - ORB Keypoints')
plt.axis('off')

# Show the plots
plt.tight_layout()
plt.show()

# Plot the results
plt.figure(figsize=(15, 10))

# Nature.jpg plots
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.title('Nature - Original')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(gray_image2, cmap='gray')
plt.title('Nature - Grayscale')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(imgHarris2, cv2.COLOR_BGR2RGB))
plt.title('Nature - Harris Corners')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(cv2.cvtColor(imgShiTomasi2, cv2.COLOR_BGR2RGB))
plt.title('Nature - Shi-Tomasi')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(imgORB2, cv2.COLOR_BGR2RGB))
plt.title('Nature - ORB Keypoints')
plt.axis('off')

# Show the plots
plt.tight_layout()
plt.show()
