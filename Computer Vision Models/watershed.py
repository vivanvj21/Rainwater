import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

def Watershed(image_path):
    # Load image (ensure it's grayscale)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found.")
        return None

    # Step 1: Threshold the image
    thresh = image > 0
    print("Thresh shape:", thresh.shape)
    print("Thresh unique values:", np.unique(thresh))

    # Step 2: Distance transform
    D = ndimage.distance_transform_edt(thresh)
    print("Distance transform shape:", D.shape)

    # Step 3: Local maxima detection
    localMax = peak_local_max(D, min_distance=20, labels=thresh)
    print("LocalMax shape:", localMax.shape)
    print("LocalMax unique values:", np.unique(localMax))

    # Step 4: Label the regions
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    print("Markers shape:", markers.shape)
    print("Markers unique values:", np.unique(markers))

    # Step 5: Perform watershed segmentation
    labels = watershed(-D, markers, mask=thresh)
    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

    # Step 6: Process the labels (optional)
    for label in np.unique(labels):
        if label == 0:
            continue
        print(f"Processing label {label}")

    return labels
