from __future__ import print_function
import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
from skimage.morphology import disk, opening
import matplotlib.pyplot as plt

# Load and preprocess the image
im = cv2.imread('static/images/mvit.png')
img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
img = cv2.bitwise_not(img)  # Invert the grayscale image for better edge detection

# Initialize the image dimensions
rows, cols = img.shape

# Create a white image for preprocessing and another for polygon filling
white_img = cv2.bitwise_not(np.zeros(im.shape, np.uint8))  # Inverted black image
white_polygon = cv2.bitwise_not(np.zeros(im.shape, np.uint8))  # Polygon for filling
white_gray = cv2.cvtColor(white_img, cv2.COLOR_BGR2GRAY)

# Compute the median intensity value of the image for adaptive thresholding
v = np.median(img)
sigma = 0.33
lower_thresh = int(max(0, (1.0 - sigma) * v))  # Lower threshold
upper_thresh = int(min(255, (1.0 + sigma) * v))  # Upper threshold

# Perform Canny edge detection
edges = cv2.Canny(img, lower_thresh, upper_thresh)

# Detect lines using the Hough transform
lines = cv2.HoughLines(edges, 1, np.pi / 180, 30)
lines = np.reshape(lines, (lines.shape[0], lines.shape[2]))

# Apply KMeans clustering to group similar lines
kmeans = KMeans(n_clusters=20).fit(lines)

# Draw the clustered lines on the white image
for line in kmeans.cluster_centers_:
    rho, theta = line
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * a)
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * a)
    cv2.line(white_gray, (x1, y1), (x2, y2), 0, 2)

# Find contours in the white image (from the detected lines)
contours = cv2.findContours(white_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

# Iterate through the contours and process each one
for cnt in contours:
    cv2.drawContours(white_polygon, cnt, 0, 0, -1)  # Fill the contour
    man = []  # List to store pixel locations inside the contour
    intense = []  # List to store pixel intensities inside the contour
    
    # Iterate through every pixel in the image and check if it lies inside the contour
    for col in range(cols):
        for row in range(rows):
            if cv2.pointPolygonTest(cnt, (col, row), False) == 1:
                man.append((row, col))  # Store the pixel locations

    # Extract intensity values of the pixels inside the contour
    for k in man:
        intense.append(im[k])  # Append the pixel intensities
    
    # Calculate the average intensity of the contour region
    intensity = np.mean(intense)
    
    # If the intensity is high, fill the contour with black color
    if intensity > 170:
        cv2.drawContours(white_polygon, [cnt], 0, 0, -1)

# Convert the resulting image to grayscale
white_gray1 = cv2.cvtColor(white_polygon, cv2.COLOR_BGR2GRAY)

# Perform morphological opening to clean up the image
opened = opening(white_gray1, selem=disk(4))

# Convert the result into a PIL image and save it
opened = Image.fromarray(opened)
opened.save('result.png')

# Display the final result using matplotlib
plt.imshow(opened, cmap='gray')
plt.title("Processed Image with Detected Contours and Lines")
plt.axis('off')  # Hide the axis for cleaner visualization
plt.show()
