import numpy as np
import cv2
import imutils
from imutils import perspective
from imutils import contours
from scipy.spatial import distance as dist
import argparse

# Define the midpoint function
def midpoint(ptA, ptB):
    """Calculate the midpoint between two points."""
    return ((ptA[0] + ptB[0]) / 2, (ptA[1] + ptB[1]) / 2)

# Parsing arguments for image path and object width
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-w", "--width", type=float, required=True, help="width of the object in your measurement units")
args = vars(ap.parse_args())

# Read the image and convert to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise and detail
gray = cv2.GaussianBlur(gray, (7, 7), 0)

# Perform edge detection using Canny algorithm
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

# Find contours in the edge-detected image
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# Sort the contours from left to right
(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None

# Iterate over each contour
for c in cnts:
    if cv2.contourArea(c) < 10000:  # Skip small contours (noise)
        continue
    
    orig = image.copy()  # Make a copy of the original image to draw on
    
    # Get the bounding box for each contour
    box = cv2.minAreaRect(c)
    box = cv2.boxPoints(box)  # Get the corner points of the bounding box
    box = np.array(box, dtype="int")
    
    # Order points (top-left, top-right, bottom-right, bottom-left)
    box = perspective.order_points(box)
    
    # Draw the bounding box on the image
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 255), 2)

    # Draw circles at the corner points of the bounding box
    for (x, y) in box:
        cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

    # Get the four corners of the bounding box
    (tl, tr, br, bl) = box
    
    # Calculate midpoints for the edges
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    # Draw the midpoints on the image
    cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    # Draw lines connecting the midpoints to form the width and height of the object
    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 255, 255), 2)
    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 255, 255), 2)

    # Compute the Euclidean distance between the midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))  # Height (vertical distance)
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))  # Width (horizontal distance)

    # If pixelsPerMetric is not yet calculated, use the known width to set it
    if pixelsPerMetric is None:
        pixelsPerMetric = dB / args["width"]

    # Calculate the real-world dimensions of the object in measurement units (e.g., feet, meters)
    dimA = dA / pixelsPerMetric  # Height
    dimB = dB / pixelsPerMetric  # Width

    # Annotate the image with the measured dimensions
    cv2.putText(orig, "{:.1f} feet".format(dimA * 10), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
    cv2.putText(orig, "{:.1f} feet".format(dimB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)

    # Compute and print the area of the object
    area = dimA * dimB
    print(f"Area: {area * 10:.1f} square feet")

# Show the final output image with bounding boxes, midpoints, and measurements
cv2.imshow("Measured Image", orig)
cv2.waitKey(0)
cv2.destroyAllWindows()
