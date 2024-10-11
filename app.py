import os
from flask import Flask, render_template, request, send_from_directory
import cv2
import numpy as np
from skimage.morphology import skeletonize
from sklearn.neighbors import KDTree

app = Flask(__name__)


# Function to preprocess fingerprint (convert to binary and skeletonize)
def preprocess_fingerprint(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_not(binary)
    skeleton = skeletonize(binary // 255).astype(np.uint8)
    return skeleton


# Function to extract key points (minutiae) from skeletonized image
def extract_minutiae(skeleton):
    harris_corners = cv2.cornerHarris(np.float32(skeleton), 2, 3, 0.04)
    harris_corners = cv2.dilate(harris_corners, None)
    threshold = 0.01 * harris_corners.max()
    keypoints = np.argwhere(harris_corners > threshold)
    keypoints = keypoints[(keypoints[:, 0] > 10) & (keypoints[:, 1] > 10)]
    return keypoints


# Function to match two fingerprints based on minutiae points using KDTree
def match_fingerprints(keypoints1, keypoints2, threshold=50):
    keypoints1 = np.array(keypoints1)
    keypoints2 = np.array(keypoints2)
    tree = KDTree(keypoints2)
    distances, indices = tree.query(keypoints1, k=1)
    matches = np.sum(distances < threshold)
    total_minutiae = max(len(keypoints1), len(keypoints2))
    match_percentage = (matches / total_minutiae) * 100
    return match_percentage


# Function to draw minutiae points on fingerprint images
def draw_minutiae(image, keypoints):
    for kp in keypoints:
        cv2.circle(image, (kp[1], kp[0]), 5, (0, 255, 0), -1)  # Draw green circles on keypoints
    return image


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file1' not in request.files or 'file2' not in request.files:
            return "Please upload both images!"

        file1 = request.files['file1']
        file2 = request.files['file2']

        # Create static directory if it doesn't exist
        if not os.path.exists('static'):
            os.makedirs('static')

        # Save the uploaded files
        file1_path = os.path.join('static', 'fingerprint1.jpg')
        file2_path = os.path.join('static', 'fingerprint2.jpg')
        file1.save(file1_path)
        file2.save(file2_path)

        # Read and process the images
        image1 = cv2.imread(file1_path)
        image2 = cv2.imread(file2_path)

        if image1 is None or image2 is None:
            return "Error loading one or both images. Please upload valid images."

        # Preprocess fingerprints and extract minutiae
        skeleton1 = preprocess_fingerprint(image1)
        skeleton2 = preprocess_fingerprint(image2)
        keypoints1 = extract_minutiae(skeleton1)
        keypoints2 = extract_minutiae(skeleton2)

        # Draw minutiae on original images
        image1_with_minutiae = draw_minutiae(image1.copy(), keypoints1)
        image2_with_minutiae = draw_minutiae(image2.copy(), keypoints2)

        # Save the images with minutiae points
        minutiae_image1_path = os.path.join('static', 'fingerprint1_minutiae.jpg')
        minutiae_image2_path = os.path.join('static', 'fingerprint2_minutiae.jpg')
        cv2.imwrite(minutiae_image1_path, image1_with_minutiae)
        cv2.imwrite(minutiae_image2_path, image2_with_minutiae)

        # Calculate match percentage
        match_percentage = match_fingerprints(keypoints1, keypoints2)

        # Pass the minutiae images and match result to the template
        return render_template('index.html',
                               match_percentage=match_percentage,
                               fingerprint1='fingerprint1_minutiae.jpg',
                               fingerprint2='fingerprint2_minutiae.jpg')

    return render_template('index.html')


# Serve static files for uploaded images
@app.route('/static/<path:filename>')
def send_static(filename):
    return send_from_directory('static', filename)


if __name__ == '__main__':
    app.run(debug=True)
