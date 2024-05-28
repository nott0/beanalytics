import cv2
import numpy as np
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Function to convert OpenCV image to base64 encoded string
def cv2_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

# Function to check if a contour is within the ROI
def is_contour_in_roi(contour, roi):
    x, y, w, h = cv2.boundingRect(contour)
    roi_x1, roi_y1, roi_x2, roi_y2 = roi
    return roi_x1 < x < roi_x2 and roi_y1 < y < roi_y2

def process_image(image_data):
    # Assuming images are named 'image1.jpg' and 'image2.jpg'
    image1 = cv2.cvtColor(cv2.imread('./Target.png'), cv2.COLOR_BGR2RGB)
    nparr = np.frombuffer(image_data, np.uint8)
    image2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    original_image = cv2_to_base64(image2)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    # Initiate SIFT detector
    sift = cv2.SIFT_create(nfeatures=5000)

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # Using FLANN for matching descriptors
    matches = flann.knnMatch(des2, des1, k=2)  # Notice the order: des2 (source) vs des1 (target)

    # Filter matches using the Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_matches.append(m)

    # Draw matches (optional, for verification)
    img_matches = cv2.drawMatches(image2, kp2, image1, kp1, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img_matches_64 = cv2_to_base64(cv2.cvtColor(img_matches, cv2.COLOR_RGB2BGR))

    # Extract location of good matches
    points2 = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points1 = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find homography
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0, maxIters=2000)

    # Use homography to warp image2 to match the perspective of image1
    height, width, channels = image1.shape
    im2Reg = cv2.warpPerspective(image2, h, (width, height))

    # Define the width of the edge region
    edge_width = 120  # Adjust this value as needed

    # Create a mask for the edges
    mask = np.zeros(image1.shape[:2], dtype=np.uint8)
    mask[:edge_width-40, :] = 1  # Top edge
    mask[-edge_width+40:, :] = 1  # Bottom edge
    mask[:, :edge_width] = 1  # Left edge
    mask[:, -edge_width:] = 1  # Right edge

    # Calculate the average color of the target image edges
    avg_color_edge_target = cv2.mean(image1, mask=mask)[:3]

    # Calculate the average color of the registered source image edges
    avg_color_edge_source = cv2.mean(im2Reg, mask=mask)[:3]

    # Calculate the scaling factors for each channel
    scaling_factors_edge = np.array(avg_color_edge_target) / np.array(avg_color_edge_source)

    # Apply the scaling factors to the registered source image
    im2Reg_balanced_edges = im2Reg * scaling_factors_edge
    im2Reg_balanced_edges = np.clip(im2Reg_balanced_edges, 0, 255).astype(np.uint8)

    # Convert the balanced image back to RGB for display
    im2Reg_balanced_edges = cv2.cvtColor(im2Reg_balanced_edges, cv2.COLOR_BGR2RGB)

    # Further adjust the image to make the near-white regions completely white
    # Calculate the maximum color value in the image
    max_color = im2Reg_balanced_edges.max(axis=(0, 1))

    # Calculate the scaling factor to make the maximum color value 255
    scaling_factor = 255.0 / max_color

    # Apply the scaling factor to the image
    im2Reg_balanced_pure_white = im2Reg_balanced_edges * scaling_factor
    im2Reg_balanced_pure_white = np.clip(im2Reg_balanced_pure_white, 0, 255).astype(np.uint8)

    # Set near-white regions to pure white
    threshold = 200  # Adjust this threshold as needed
    im2Reg_balanced_pure_white[np.all(im2Reg_balanced_pure_white > threshold, axis=-1)] = 255

    # Display the white balanced registered image with ROI
    im2Reg_64 = cv2_to_base64(im2Reg_balanced_pure_white)

    image = im2Reg_balanced_pure_white.copy()

    # Get image dimensions
    img_height, img_width = image.shape[:2]

    # Convert the size from pixel to cm assuming the image covers the full A4 size
    cm_per_pixel_w = 21.0 / img_width
    cm_per_pixel_h = 29.7 / img_height
    cm_per_pixel = (cm_per_pixel_w + cm_per_pixel_h) / 2

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve thresholding
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply binary thresholding using Otsu's method
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove small white noise using morphological opening
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Remove small holes using morphological closing
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Sure background area
    sure_bg = cv2.dilate(closing, kernel, iterations=3)

    # Finding sure foreground area using distance transform and thresholding
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)  # Adjust this threshold value

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Mark the region of unknown with zero
    markers[unknown == 255] = 0

    # Apply watershed
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]

    # Define the central region of interest (ROI)
    edge_width = 120
    roi = (edge_width, edge_width - 40, image.shape[1] - edge_width, image.shape[0] - edge_width + 40)

    # Initialize lists to store sizes and average colors
    bean_sizes = []
    bean_sizes_cm = []
    bean_colors = []

    # Minimum and maximum bean area thresholds
    min_bean_area = 300  # Adjust these values as necessary
    max_bean_area = 5000

    # Process each marker
    for marker in range(2, markers.max() + 1):  # Skip background and unknown markers
        # Create a mask for the current marker
        bean_mask = np.zeros_like(gray)
        bean_mask[markers == marker] = 255

        # Find contours for the current bean
        contours, _ = cv2.findContours(bean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process only if contour is within ROI
        if contours and is_contour_in_roi(contours[0], roi):
            # Fit and draw ellipse
          if len(contours[0]) >= 5:  # Fit ellipse requires at least 5 points
            # Calculate the area of the contour
            area = cv2.contourArea(contours[0])

            if min_bean_area <= area <= max_bean_area:
              bean_sizes.append(area)
              area_cm = area * cm_per_pixel ** 2
              bean_sizes_cm.append(area_cm)

              # Calculate the average color inside the contour
              mean_color = cv2.mean(image, mask=bean_mask)[:3]
              bean_colors.append(mean_color)
              # Insert oval and number on the oval
              ellipse = cv2.fitEllipse(contours[0])
              cv2.ellipse(image, ellipse, (0, 255, 0), 2)
              center = (int(ellipse[0][0]), int(ellipse[0][1]))
              cv2.putText(image, str(len(bean_sizes)), center, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Draw the ROI rectangle
    cv2.rectangle(image, (roi[0], roi[1]), (roi[2], roi[3]), (255, 0, 0), 3)

    # Display the result
    result_64 = cv2_to_base64(image)

    # Process each marker and extract information
    result_text = []
    for i, (size_pixels, size_cm, color) in enumerate(zip(bean_sizes, bean_sizes_cm, bean_colors)):
        result_text.append(f"Cacao Bean {i+1}: Size = {size_pixels} pixels ({size_cm:.2f} cm^2), Average Color = {color}")

    return original_image, img_matches_64, im2Reg_64, result_64, result_text

# Define a route to handle the image processing
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get image data from the POST request
        file = request.files['image']
        image_data = file.read()

        # Process the image and get results
        original_image, matching_image, processed_image, result_image, result_text = process_image(image_data)

        return render_template('index.html', original_image=original_image, matching_image=matching_image, processed_image=processed_image, result_image=result_image, result_text=result_text)
    else:
        return render_template('upload.html')  # Render a form to upload the image

if __name__ == '__main__':
    app.run()
