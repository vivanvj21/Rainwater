from __future__ import print_function
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, flash
from scipy.spatial import distance as dist
import cv2
import os
from werkzeug.utils import secure_filename
from PIL import Image
import imutils
from imutils import contours
from imutils import perspective

# Configuration settings
UPLOAD_FOLDER = './static/images'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask("_app_")
app.config['SECRET_KEY'] = 'a551d32359baf371b9095f28d45347c8b8621830'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file type check
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Flask route for uploading files
@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file', filename=filename))
    return render_template('calculator.html', val=1)

# Function to equalize image histogram
def equalize(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img

def Watershed(location_det):
    print(location_det)
    
    # Check if the file exists
    if not os.path.exists(location_det):
        print(f"Error: File {location_det} not found.")
        return None  # Ensure it returns None if the file doesn't exist

    # Read and process the image
    image = cv2.imread(location_det)
    if image is None:
        print(f"Error: Failed to read image {location_det}.")
        return None

    im = equalize(image)
    shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)

    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, min_distance=20, labels=thresh)

    # Create markers array with the same shape as the input image
    markers = np.zeros_like(thresh)  # Create an empty array of the same shape as the mask (thresh)
    for i, coord in enumerate(localMax):
        markers[coord[0], coord[1]] = i + 1  # Assign a unique marker for each local max

    labels = watershed(-D, markers, mask=thresh)

    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

    # Loop through the unique labels (excluding the background)
    for label in np.unique(labels):
        if label == 0:
            continue  # Skip the background label

        # Create a mask for the current label
        mask = np.zeros(image.shape[:2], dtype="uint8")  # Use original image shape for mask
        mask[labels == label] = 255  # Create mask for each region
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        for i, c in enumerate(cnts):
            (x, y), _ = cv2.minEnclosingCircle(c)
            cv2.putText(image, "#{}".format(i + 1), (int(x) - 10, int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.drawContours(image, [c], -1, (0, 255, 0), 2)

    # Save the processed image
    processed_image_path = 'static/images/fig1.png'
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.savefig(processed_image_path)

    return processed_image_path  # Return the path to the saved image




# Function to compute midpoints between two points
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# Function to measure dimensions of objects in the image
def measure_dim(loc):
    image = cv2.imread(loc)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None

    for c in cnts:
        if cv2.contourArea(c) < 100:
            continue

        orig = image.copy()
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 255), 2)

        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # Draw the midpoints on the image
        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 255, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 255, 255), 2)

        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        if pixelsPerMetric is None:
            pixelsPerMetric = dB / 750

        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric

        cv2.putText(orig, "{:.1f} feet".format(dimA * 10), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
        cv2.putText(orig, "{:.1f} feet".format(dimB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)

        area = dimA * dimB
        dims = area
        print(f'The dims: {dims}')

    return dims



@app.route('/uploads/<filename>')
def uploaded_file(filename):
    file_path = './static/images/' + filename
    if not os.path.exists(file_path):
        return f"Error: File {filename} does not exist!"
    
    # Process the image
    image_path = Watershed(file_path)
    if image_path is None:
        return "Error: Image processing failed."
    
    # Measure the dimensions
    dims = measure_dim(file_path)
    dims /= 10000  # Convert dimensions if necessary (e.g., from pixels to meters/feet)
    
    return render_template('calculator.html', val=1, dims=dims, image_path=image_path)





# Home route
@app.route('/')
def home():
    return render_template('index.html', title='SIH 2019')

# Other routes for different pages
@app.route('/rooftop')
def rooftop():
    return render_template('roof.html', title='Rooftop Detection')

@app.route('/references')
def references():
    return render_template('references.html', title='References')

@app.route('/calculator', methods=['GET', 'POST'])
def calculator():
    return render_template('calculator.html', title='Calculator')

@app.route('/trial', methods=['GET'])
def trial():
    return render_template('trial.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/contour')
def contour():
    return render_template('contour_map.html', title='Contour Map')

if __name__ == "__main__":
    app.run(debug=True)
