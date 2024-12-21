import cv2
import matplotlib.pyplot as plt
import numpy as np

# Define the coordinates for the location (Latitude, Longitude)
lat = 28.646088
longs = 77.214157
location = (lat, longs)

# Load the image in grayscale
img = cv2.imread('Figure_1.png', 0)

# Check if the image is loaded successfully
if img is None:
    print("Error loading image!")
    exit()

# Threshold the image to prepare for contour detection
ret, thresh = cv2.threshold(img, 127, 255, 0)

# Find the external contours of the thresholded image
cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Detect corners in the image using the Shi-Tomasi method (goodFeaturesToTrack)
corners = cv2.goodFeaturesToTrack(img, 100, 0.01, 10)
corners = np.int32(corners)  # Convert to int32 as np.int0 is deprecated

# Initialize lists to store corner coordinates
x = []
y = []

# Loop over the detected corners and draw circles at their locations
for corner in corners:
    x1, y1 = corner.ravel()
    x.append(x1)
    y.append(y1)
    cv2.circle(img, (x1, y1), 3, 0, -1)  # Draw circles with radius 3

# Draw the contour on the image
cv2.drawContours(img, [cnts[0]], -1, (0, 255, 0), 2)  # Draw green contour

# Convert the image from BGR to RGB for displaying with matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the image with matplotlib
plt.imshow(img_rgb, cmap='gray')
plt.title("Contours and Corner Detection")
plt.axis('off')  # Hide axis
plt.show()
