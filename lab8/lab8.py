import cv2
import numpy as np 
from matplotlib import pyplot as plt

# Load the input image
img = cv2.imread('ATU.jpg',)
#cv2.imshow('Original', img)
#cv2.waitKey(0)

# Convert the image to greyscale
gray_image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Display the grayscale image
cv2.imshow('Grayscale Image',gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
