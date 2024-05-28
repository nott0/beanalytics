import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect
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
    # Read the target image once
    image1 = cv2.imread('./Target.png')
    if image1 is None:
        raise ValueError("Could not load the target image")
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

    # Decode the uploaded image
    nparr = np.frombuffer(image_data, np.uint8)
    image2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image2 is None:
        raise ValueError("Could not decode image")
    original_image = cv2_to_base64(image2)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    # Initiate SIFT detector with reduced features
    sift = cv2.SIFT_create(nfeatures=100)

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)

    # Use FLANN for matching descriptors
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=3)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des2, des1, k=2)

    # Filter matches using the Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < 0.6 * n.distance]

    # Draw matches (optional, for verification)
    img_matches = cv2.drawMatches(image2, kp2, image1, kp1, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img_matches_64 = cv2_to_base64(cv2.cvtColor(img_matches, cv2.COLOR_RGB2BGR))

    # Extract location of good matches
    points2 = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points1 = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find homography with a reduced number of iterations
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0, maxIters=300)

    # Use homography to warp image2 to match the perspective of image1
    height, width, channels = image1.shape
    im2Reg = cv2.warpPerspective(image2, h, (width, height))

    # Define the width of the edge region
    edge_width = 120

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
    max_color = im2Reg_balanced_edges.max(axis=(0, 1))
    scaling_factor = 255.0 / max_color
    im2Reg_balanced_pure_white = im2Reg_balanced_edges * scaling_factor
    im2Reg_balanced_pure_white = np.clip(im2Reg_balanced_pure_white, 0, 255).astype(np.uint8)
    threshold = 200
    im2Reg_balanced_pure_white[np.all(im2Reg_balanced_pure_white > threshold, axis=-1)] = 255
    im2Reg_64 = cv2_to_base64(im2Reg_balanced_pure_white)

    image = im2Reg_balanced_pure_white.copy()
    img_height, img_width = image.shape[:2]
    cm_per_pixel_w = 21.0 / img_width
    cm_per_pixel_h = 29.7 / img_height
    cm_per_pixel = (cm_per_pixel_w + cm_per_pixel_h) / 2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    sure_bg = cv2.dilate(closing, kernel, iterations=2)
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]

    edge_width = 120
    roi = (edge_width, edge_width - 40, image.shape[1] - edge_width, image.shape[0] - edge_width + 40)
    bean_sizes = []
    bean_sizes_cm = []
    bean_colors = []

    min_bean_area = 300
    max_bean_area = 5000

    for marker in range(2, markers.max() + 1):
        bean_mask = np.zeros_like(gray)
        bean_mask[markers == marker] = 255
        contours, _ = cv2.findContours(bean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours and is_contour_in_roi(contours[0], roi):
            if len(contours[0]) >= 5:
                area = cv2.contourArea(contours[0])
                if min_bean_area <= area <= max_bean_area:
                    bean_sizes.append(area)
                    area_cm = area * cm_per_pixel ** 2
                    bean_sizes_cm.append(area_cm)
                    mean_color = cv2.mean(image, mask=bean_mask)[:3]
                    bean_colors.append(mean_color)
                    ellipse = cv2.fitEllipse(contours[0])
                    cv2.ellipse(image, ellipse, (0, 255, 0), 2)
                    center = (int(ellipse[0][0]), int(ellipse[0][1]))
                    cv2.putText(image, str(len(bean_sizes)), center, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.rectangle(image, (roi[0], roi[1]), (roi[2], roi[3]), (255, 0, 0), 3)
    result_64 = cv2_to_base64(image)

    result_text = []
    for i, (size_pixels, size_cm, color) in enumerate(zip(bean_sizes, bean_sizes_cm, bean_colors)):
        result_text.append(f"Cacao Bean {i+1}: Size = {size_pixels} pixels ({size_cm:.2f} cm^2), Average Color = {color}")

    return original_image, img_matches_64, im2Reg_64, result_64, result_text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']

        if file.filename == '':
            return redirect(request.url)

        if file:
            image_data = file.read()
            if not image_data:
                return "No image data received", 400

            try:
                original_image, matching_image, processed_image, result_image, result_text = process_image(image_data)
                return render_template('index.html', original_image=original_image, matching_image=matching_image, processed_image=processed_image, result_image=result_image, result_text=result_text)
            except ValueError as e:
                return str(e), 400
            except Exception as e:
                return str(e), 500

    return render_template('upload.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
