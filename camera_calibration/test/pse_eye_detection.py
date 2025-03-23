import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from scipy.spatial import distance
import json

def bundle_adjustment(pts1, pts2, K, R_init, t_init, points_3d_init):
    """
    Refine camera parameters and 3D points to minimize reprojection error.
    
    Args:
        pts1: 2D points in the first image
        pts2: 2D points in the second image
        K: Camera intrinsic matrix
        R_init: Initial rotation matrix of second camera
        t_init: Initial translation vector of second camera
        points_3d_init: Initial 3D point estimates
        
    Returns:
        Refined camera parameters and 3D points
    """
    from scipy import optimize
    from scipy.spatial.transform import Rotation
    
    print("\n=== PERFORMING BUNDLE ADJUSTMENT ===")
    
    # Convert points to the right format
    pts1 = np.array(pts1, dtype=np.float64)
    pts2 = np.array(pts2, dtype=np.float64)
    points_3d = np.array(points_3d_init, dtype=np.float64)
    
    # Number of points
    n_points = len(points_3d)
    
    # Initial camera parameters
    # First camera is at origin with identity rotation
    # Second camera has the given R and t
    
    # Convert rotation matrix to rotation vector for optimization
    r_vec_init, _ = cv.Rodrigues(R_init)
    r_vec_init = r_vec_init.flatten()
    t_init = t_init.flatten()
    
    # We'll optimize:
    # 1. Focal length (same for both cameras)
    # 2. Rotation of second camera (3 parameters)
    # 3. Translation of second camera (3 parameters)
    # 4. 3D coordinates of each point (3*n_points parameters)
    
    # Initial parameters vector
    initial_focal = K[0, 0]
    params = np.hstack((
        [initial_focal],           # Focal length
        r_vec_init,                # Rotation vector of camera 2
        t_init,                    # Translation of camera 2
        points_3d.flatten()        # 3D points
    ))
    
    # Define the residual function for optimization
    def compute_residuals(params):
        # Extract parameters
        focal = params[0]
        r_vec = params[1:4]
        t_vec = params[4:7]
        points_3d = params[7:].reshape(-1, 3)
        
        # Update camera matrix with optimized focal length
        K_opt = K.copy()
        K_opt[0, 0] = K_opt[1, 1] = focal
        
        # Convert rotation vector to matrix
        R_opt, _ = cv.Rodrigues(r_vec)
        
        # Compute projection matrices
        P1 = K_opt @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = K_opt @ np.hstack((R_opt, t_vec.reshape(3, 1)))
        
        # Compute reprojection errors
        errors = []
        
        for i, point_3d in enumerate(points_3d):
            # Convert to homogeneous coordinates
            X = np.append(point_3d, 1)
            
            # Project to image 1 (first camera)
            x1_proj = P1 @ X
            x1_proj = x1_proj[:2] / x1_proj[2]  # Normalize
            error1 = pts1[i] - x1_proj
            
            # Project to image 2 (second camera)
            x2_proj = P2 @ X
            x2_proj = x2_proj[:2] / x2_proj[2]  # Normalize
            error2 = pts2[i] - x2_proj
            
            # Add all errors to the residual vector
            errors.extend(error1)
            errors.extend(error2)
            
        return np.array(errors)
    
    # Print initial reprojection error
    initial_error = np.sum(compute_residuals(params)**2)
    print(f"Initial total squared reprojection error: {initial_error:.2f}")
    
    # Run the optimization
    print("Optimizing camera parameters and 3D points...")
    result = optimize.least_squares(
        compute_residuals, 
        params, 
        method='trf',
        ftol=1e-5,
        xtol=1e-5,
        verbose=0
    )
    
    # Extract optimized parameters
    opt_params = result.x
    opt_focal = opt_params[0]
    opt_r_vec = opt_params[1:4]
    opt_t_vec = opt_params[4:7]
    opt_points_3d = opt_params[7:].reshape(-1, 3)
    
    # Convert rotation vector back to matrix
    opt_R, _ = cv.Rodrigues(opt_r_vec)
    opt_t = opt_t_vec.reshape(3, 1)
    
    # Update camera matrix
    opt_K = K.copy()
    opt_K[0, 0] = opt_K[1, 1] = opt_focal
    
    # Print final reprojection error
    final_error = np.sum(compute_residuals(opt_params)**2)
    print(f"Final total squared reprojection error: {final_error:.2f}")
    print(f"Improvement: {(initial_error - final_error) / initial_error * 100:.2f}%")
    
    # Print optimized parameters
    print(f"\nOptimized focal length: {opt_focal:.2f} (initial: {initial_focal:.2f})")
    print("\nOptimized camera 2 rotation:")
    print(opt_R)
    print("\nOptimized camera 2 translation:")
    print(opt_t)
    
    return opt_K, opt_R, opt_t, opt_points_3d

# Load calibration data from JSON file
def load_calibration(json_path):
    try:
        with open(json_path, 'r') as f:
            calib_data = json.load(f)
        
        # Convert lists back to numpy arrays
        camera_matrix = np.array(calib_data["camera_matrix"])
        dist_coeffs = np.array(calib_data["distortion_coefficients"])
        
        print("Loaded camera calibration parameters:")
        print(f"Camera Matrix:\n{camera_matrix}")
        print(f"Distortion Coefficients:\n{dist_coeffs}")
        
        return camera_matrix, dist_coeffs
    
    except Exception as e:
        print(f"Error loading calibration data: {e}")
        print("Using default placeholder camera parameters instead.")
        # Default parameters (focal length, principal point)
        focal_length = 700
        principal_point = (320, 240)
        K = np.array([
            [focal_length, 0, principal_point[0]],
            [0, focal_length, principal_point[1]],
            [0, 0, 1]
        ])
        dist = np.zeros(5)
        return K, dist
    
# Path to your calibration file
calibration_path = "camera_calibration.json"

# Load the intrinsic camera parameters
K, dist_coeffs = load_calibration(calibration_path)


# Read the images in color
img1_color = cv.imread('captures/capture_000.png')  # left image
img2_color = cv.imread('captures/capture_001.png')  # right image

# Check if images are loaded successfully
if img1_color is None or img2_color is None:
    print("Error loading images. Please check the file paths.")
    sys.exit(1)

# Undistort the images using the calibration parameters
#img1_color = cv.undistort(img1_color, K, dist_coeffs)
#img2_color = cv.undistort(img2_color, K, dist_coeffs)

# Create grayscale versions for some operations
img1 = cv.cvtColor(img1_color, cv.COLOR_BGR2GRAY)
img2 = cv.cvtColor(img2_color, cv.COLOR_BGR2GRAY)

# Function to detect white dots with blob detection
def detect_white_dots_blob(img):
    # Convert BGR to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Apply threshold to isolate white/bright areas
    # Adjust the threshold value as needed (higher for brighter white)
    _, binary = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)
    
    # Apply morphological operations to clean up the mask
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
    
    # Set up blob detector parameters
    params = cv.SimpleBlobDetector_Params()
    
    # Change thresholds
    params.minThreshold = 200
    params.maxThreshold = 255
    
    # Filter by color (255 = light, 0 = dark)
    params.filterByColor = True
    params.blobColor = 255  # White blobs on black background
    
    # Filter by area
    params.filterByArea = True
    params.minArea = 10
    params.maxArea = 100000
    
    # Filter by circularity
    params.filterByCircularity = True
    params.minCircularity = 0.3
    
    # Filter by convexity
    params.filterByConvexity = True
    params.minConvexity = 0.1
    
    # Filter by inertia (how circular vs. elliptical)
    params.filterByInertia = True
    params.minInertiaRatio = 0.2
    
    # Create detector
    detector = cv.SimpleBlobDetector_create(params)
    
    # Detect blobs
    keypoints = detector.detect(binary)
    
    # Extract coordinates
    centers = [(int(k.pt[0]), int(k.pt[1])) for k in keypoints]
    sizes = [k.size for k in keypoints]
    
    # Visualize detection
    img_with_keypoints = cv.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255), 
                                          cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return centers, sizes, img_with_keypoints, binary

# Detect white dots in both images
centers1, sizes1, img1_keypoints, mask1 = detect_white_dots_blob(img1_color)
centers2, sizes2, img2_keypoints, mask2 = detect_white_dots_blob(img2_color)

print(f"Detected {len(centers1)} white dots in left image")
print(f"Detected {len(centers2)} white dots in right image")

# Create labeled images for easy reference
img1_labeled = img1_color.copy()
img2_labeled = img2_color.copy()

for i, (center, size) in enumerate(zip(centers1, sizes1)):
    # Draw circle and label
    cv.circle(img1_labeled, center, int(size/2), (0, 0, 255), 2)
    cv.putText(img1_labeled, str(i), (center[0]-10, center[1]-10), 
              cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

for i, (center, size) in enumerate(zip(centers2, sizes2)):
    # Draw circle and label
    cv.circle(img2_labeled, center, int(size/2), (0, 0, 255), 2)
    cv.putText(img2_labeled, str(i), (center[0]-10, center[1]-10), 
              cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display labeled images
plt.figure(figsize=(15, 8))
plt.subplot(121), plt.imshow(cv.cvtColor(img1_labeled, cv.COLOR_BGR2RGB))
plt.title('Left Image - Numbered White Dots')
plt.subplot(122), plt.imshow(cv.cvtColor(img2_labeled, cv.COLOR_BGR2RGB))
plt.title('Right Image - Numbered White Dots')
plt.tight_layout()
plt.savefig('numbered_white_dots.png')
plt.show()

# Alternative detection method in case the blob detector has issues with white dots
def detect_white_dots_alternative(img):
    # Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Apply threshold
    _, thresh = cv.threshold(gray, 230, 255, cv.THRESH_BINARY)
    
    # Apply morphological operations
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and shape
    centers = []
    sizes = []
    for contour in contours:
        area = cv.contourArea(contour)
        if area < 50 or area > 5000:  # Filter by area
            continue
            
        # Calculate circularity
        perimeter = cv.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        if circularity < 0.5:  # Filter by circularity (circle = 1)
            continue
            
        # Calculate center
        M = cv.moments(contour)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY))
            
            # Estimate dot size (radius)
            radius = int(np.sqrt(area / np.pi))
            sizes.append(radius * 2)  # Diameter
    
    # Create visualization image
    img_with_keypoints = img.copy()
    for i, (center, size) in enumerate(zip(centers, sizes)):
        cv.circle(img_with_keypoints, center, int(size/2), (0, 0, 255), 2)
        cv.putText(img_with_keypoints, str(i), (center[0]-10, center[1]-10), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return centers, sizes, img_with_keypoints, thresh

# If the blob detector didn't work well, try the alternative approach
if len(centers1) != 12 or len(centers2) != 12:
    print("Blob detector may not have found all dots. Trying alternative method...")
    centers1, sizes1, img1_keypoints, mask1 = detect_white_dots_alternative(img1_color)
    centers2, sizes2, img2_keypoints, mask2 = detect_white_dots_alternative(img2_color)
    
    print(f"Alternative method detected {len(centers1)} white dots in left image")
    print(f"Alternative method detected {len(centers2)} white dots in right image")
    
    # Update labeled images
    img1_labeled = img1_keypoints
    img2_labeled = img2_keypoints
    
    # Display updated labeled images
    plt.figure(figsize=(15, 8))
    plt.subplot(121), plt.imshow(cv.cvtColor(img1_labeled, cv.COLOR_BGR2RGB))
    plt.title('Left Image - Numbered White Dots (Alternative)')
    plt.subplot(122), plt.imshow(cv.cvtColor(img2_labeled, cv.COLOR_BGR2RGB))
    plt.title('Right Image - Numbered White Dots (Alternative)')
    plt.tight_layout()
    plt.savefig('numbered_white_dots_alternative.png')
    plt.show()

# ==========================================================================
# FIXED MANUAL PATTERN MATCHING APPROACH
# ==========================================================================

def fixed_manual_pattern_matching(points1, points2):
    """
    Explicitly match points based on the pattern described:
    - Split exactly 6 points for first row, 6 points for second row in each image
    - In first row: first 5 points + last 1 point
    - In second row: first 4 points + last 2 points
    """
    print("\n=== FIXED MANUAL PATTERN MATCHING ===")
    
    # Convert to numpy arrays and ensure we have the right number of points
    pts1 = np.array(points1)
    pts2 = np.array(points2)
    
    if len(pts1) != 12 or len(pts2) != 12:
        print(f"WARNING: Expected 12 points in each image, but found {len(pts1)} and {len(pts2)}")
        # If not exactly 12 points, proceed with what we have but warn
    
    # Sort all points by y-coordinate (top to bottom)
    y_sorted_indices1 = np.argsort([p[1] for p in pts1])
    y_sorted_indices2 = np.argsort([p[1] for p in pts2])
    
    # Check if we have enough points for proper splitting
    if len(y_sorted_indices1) < 12 or len(y_sorted_indices2) < 12:
        print("WARNING: Not enough points to split into proper rows! Attempting to continue...")
        # Try to split evenly if possible
        half1 = len(y_sorted_indices1) // 2
        half2 = len(y_sorted_indices2) // 2
        top_row1_indices = y_sorted_indices1[:half1]
        bottom_row1_indices = y_sorted_indices1[half1:]
        top_row2_indices = y_sorted_indices2[:half2]
        bottom_row2_indices = y_sorted_indices2[half2:]
    else:
        # Exactly split into top and bottom rows (6 points in each)
        top_row1_indices = y_sorted_indices1[:6]
        bottom_row1_indices = y_sorted_indices1[6:]
        top_row2_indices = y_sorted_indices2[:6]
        bottom_row2_indices = y_sorted_indices2[6:]
    
    print(f"Image 1 - Top row indices: {top_row1_indices}")
    print(f"Image 1 - Bottom row indices: {bottom_row1_indices}")
    print(f"Image 2 - Top row indices: {top_row2_indices}")
    print(f"Image 2 - Bottom row indices: {bottom_row2_indices}")
    
    # For each row, sort by x-coordinate (left to right)
    def sort_indices_by_x(indices, points):
        return sorted(indices, key=lambda i: points[i][0])
    
    # Sort each row by x-coordinate
    top_row1_x_sorted = sort_indices_by_x(top_row1_indices, pts1)
    bottom_row1_x_sorted = sort_indices_by_x(bottom_row1_indices, pts1)
    top_row2_x_sorted = sort_indices_by_x(top_row2_indices, pts2)
    bottom_row2_x_sorted = sort_indices_by_x(bottom_row2_indices, pts2)
    
    print(f"Image 1 - Top row x-sorted: {top_row1_x_sorted}")
    print(f"Image 1 - Bottom row x-sorted: {bottom_row1_x_sorted}")
    print(f"Image 2 - Top row x-sorted: {top_row2_x_sorted}")
    print(f"Image 2 - Bottom row x-sorted: {bottom_row2_x_sorted}")
    
    # Apply the exact matching rules
    matches = []
    
    # Top row: first 5 points + last point
    print("\nMatching top row:")
    # First 5 points
    for i in range(min(5, len(top_row1_x_sorted), len(top_row2_x_sorted))):
        idx1 = top_row1_x_sorted[i]
        idx2 = top_row2_x_sorted[i]
        matches.append((idx1, idx2))
        print(f"  Matched: Point {idx1} in left image to Point {idx2} in right image")
    
    # Last point
    if len(top_row1_x_sorted) >= 6 and len(top_row2_x_sorted) >= 6:
        idx1 = top_row1_x_sorted[-1]
        idx2 = top_row2_x_sorted[-1]
        matches.append((idx1, idx2))
        print(f"  Matched: Point {idx1} in left image to Point {idx2} in right image")
    
    # Bottom row: first 4 points + last 2 points
    print("\nMatching bottom row:")
    # First 4 points
    for i in range(min(4, len(bottom_row1_x_sorted), len(bottom_row2_x_sorted))):
        idx1 = bottom_row1_x_sorted[i]
        idx2 = bottom_row2_x_sorted[i]
        matches.append((idx1, idx2))
        print(f"  Matched: Point {idx1} in left image to Point {idx2} in right image")
    
    # Last 2 points
    if len(bottom_row1_x_sorted) >= 6 and len(bottom_row2_x_sorted) >= 6:
        # Second to last point
        idx1 = bottom_row1_x_sorted[-2]
        idx2 = bottom_row2_x_sorted[-2]
        matches.append((idx1, idx2))
        print(f"  Matched: Point {idx1} in left image to Point {idx2} in right image")
        
        # Last point
        idx1 = bottom_row1_x_sorted[-1]
        idx2 = bottom_row2_x_sorted[-1]
        matches.append((idx1, idx2))
        print(f"  Matched: Point {idx1} in left image to Point {idx2} in right image")
    
    print(f"\nTotal matches: {len(matches)}")
    return matches

# Use the fixed manual pattern matching approach
matches = fixed_manual_pattern_matching(centers1, centers2)

# Ensure we have enough points for calculating the fundamental matrix
if len(matches) < 8:
    print(f"ERROR: Need at least 8 point matches, but only found {len(matches)}.")
    print("The matching failed to find enough correspondences.")
    sys.exit(1)

# Visualize matches
match_img = np.hstack((img1_labeled, img2_labeled))
for i, j in matches:
    pt1 = centers1[i]
    # Adjust x-coordinate for the right image in the combined image
    pt2 = (centers2[j][0] + img1_color.shape[1], centers2[j][1])
    color = tuple(np.random.randint(0, 255, 3).tolist())
    cv.line(match_img, pt1, pt2, color, 1)
    
plt.figure(figsize=(15, 8))
plt.imshow(cv.cvtColor(match_img, cv.COLOR_BGR2RGB))
plt.title(f'Fixed Manual Pattern Matching - {len(matches)} matches')
plt.savefig('fixed_manual_pattern_matches.png')
plt.show()

# Extract matched points
pts1_matched = np.array([centers1[i] for i, _ in matches], dtype=np.float32)
pts2_matched = np.array([centers2[j] for _, j in matches], dtype=np.float32)

# Find the Fundamental Matrix
F, mask = cv.findFundamentalMat(pts1_matched, pts2_matched, cv.FM_RANSAC, 1.0, 0.99)

if F is None or F.shape[0] != 3:
    print("Failed to estimate fundamental matrix. Check your matches.")
    sys.exit(1)

# Select only inlier points
inlier_mask = mask.ravel() == 1
if np.sum(inlier_mask) < 8:
    print("Warning: Not enough inliers after RANSAC. Results may be unreliable.")
    # Use all matches as a fallback
    pts1_inliers = pts1_matched
    pts2_inliers = pts2_matched
else:
    pts1_inliers = pts1_matched[inlier_mask]
    pts2_inliers = pts2_matched[inlier_mask]

print(f"Using {len(pts1_inliers)} inlier points for epipolar geometry")

# Function to draw epilines with better error handling
def draw_epilines(img1, img2, lines, pts1, pts2):
    """Draw epilines and matched points with numbers"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    img1_lines = img1.copy()
    img2_points = img2.copy()
    
    for i, (line, pt1, pt2) in enumerate(zip(lines, pts1, pts2)):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        
        # Skip invalid lines
        if abs(line[0]) < 1e-8 and abs(line[1]) < 1e-8:
            continue
            
        # Draw the line ax + by + c = 0
        try:
            # Handle vertical and horizontal lines specially
            if abs(line[1]) < 1e-8:  # Vertical line
                x = -line[2] / line[0] if abs(line[0]) > 1e-8 else 0
                img1_lines = cv.line(img1_lines, (int(x), 0), (int(x), h1-1), color, 1)
            else:
                # General case
                x0, y0 = 0, int(-line[2] / line[1])
                x1, y1 = w1-1, int(-(line[2] + line[0]*(w1-1)) / line[1])
                
                # Clip line to image boundaries
                if y0 < 0:
                    # Line crosses x-axis above the image, recalculate x0
                    x0 = int(-line[2] / line[0]) if abs(line[0]) > 1e-8 else 0
                    y0 = 0
                elif y0 >= h1:
                    # Line crosses x-axis below the image, recalculate x0
                    x0 = int(-(line[2] - line[1]*(h1-1)) / line[0]) if abs(line[0]) > 1e-8 else 0
                    y0 = h1-1
                    
                if y1 < 0:
                    # Line crosses right edge above the image, recalculate x1, y1
                    x1 = int(-line[2] / line[0]) if abs(line[0]) > 1e-8 else w1-1
                    y1 = 0
                elif y1 >= h1:
                    # Line crosses right edge below the image, recalculate x1, y1
                    x1 = int(-(line[2] - line[1]*(h1-1)) / line[0]) if abs(line[0]) > 1e-8 else w1-1
                    y1 = h1-1
                
                img1_lines = cv.line(img1_lines, (x0, y0), (x1, y1), color, 1)
        except Exception as e:
            print(f"Error drawing line {i}: {line} - {str(e)}")
            continue
            
        pt1 = tuple(map(int, pt1))
        pt2 = tuple(map(int, pt2))
        
        img1_lines = cv.circle(img1_lines, pt1, 5, color, -1)
        img2_points = cv.circle(img2_points, pt2, 5, color, -1)
        
        # Add labels to points
        cv.putText(img1_lines, str(i), (pt1[0]+5, pt1[1]-5), cv.FONT_HERSHEY_SIMPLEX, 
                  0.5, color, 2)
        cv.putText(img2_points, str(i), (pt2[0]+5, pt2[1]-5), cv.FONT_HERSHEY_SIMPLEX, 
                  0.5, color, 2)
    
    return img1_lines, img2_points

# Calculate epilines for right image points and draw them on left image
lines1 = cv.computeCorrespondEpilines(pts2_inliers.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img5, img6 = draw_epilines(img1_color, img2_color, lines1, pts1_inliers, pts2_inliers)

# Calculate epilines for left image points and draw them on right image
lines2 = cv.computeCorrespondEpilines(pts1_inliers.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = draw_epilines(img2_color, img1_color, lines2, pts2_inliers, pts1_inliers)

# Display results
plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(cv.cvtColor(img5, cv.COLOR_BGR2RGB))
plt.title('Left Image with Epilines from Right')
plt.subplot(122), plt.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB))
plt.title('Right Image with Epilines from Left')
plt.tight_layout()
plt.savefig('epipolar_result_white.png')  # Save the result as an image file
plt.show()

# Calculate Essential Matrix from Fundamental Matrix
E = K.T @ F @ K

# Recover the rotation and translation from the essential matrix
_, R, t, _ = cv.recoverPose(E, pts1_inliers, pts2_inliers, K)

# Print camera poses
print(f"Found {len(pts1_inliers)} good matches after filtering")
print(f"Fundamental Matrix:\n{F}")
print(f"Essential Matrix:\n{E}")
print("\nCamera 2 pose relative to Camera 1:")
print(f"Rotation Matrix:\n{R}")
print(f"Translation Vector:\n{t}")

# Triangulate 3D points
try:
    # Create projection matrices
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # First camera projection matrix
    P2 = K @ np.hstack((R, t))  # Second camera projection matrix
    
    # Convert points to the right format (float32)
    pts1_float = np.float32(pts1_inliers)
    pts2_float = np.float32(pts2_inliers)
    
    # Triangulate directly using OpenCV
    points_4d = cv.triangulatePoints(P1, P2, pts1_float.T, pts2_float.T)
    
    # Convert to 3D points
    points_3d_raw = points_4d[:3, :] / points_4d[3, :]
    points_3d = points_3d_raw.T
    
    # Check for invalid points (NaN or infinite values)
    valid_points = ~np.isnan(points_3d).any(axis=1) & ~np.isinf(points_3d).any(axis=1)
    points_3d = points_3d[valid_points]
    
    # Remove outliers - points that are extremely far away
    if len(points_3d) > 0:
        # Calculate distances from centroid
        centroid = np.mean(points_3d, axis=0)
        distances = np.linalg.norm(points_3d - centroid, axis=1)
        
        # Keep points within 3 standard deviations
        inliers = distances < np.mean(distances) + 3 * np.std(distances)
        points_3d = points_3d[inliers]
        
        print(f"Reconstructed {len(points_3d)} valid 3D points after filtering")
    else:
        print("No valid 3D points could be reconstructed")
except Exception as e:
    print(f"Error during triangulation: {e}")
    print("3D reconstruction failed. This could be due to:")
    print("- Poor feature matches between images")
    print("- Inaccurate camera parameters")
    print("- Nearly coplanar point configuration")
    points_3d = np.array([[0, 0, 0]])  # Create dummy point for visualization

# Perform bundle adjustment to refine camera parameters and 3D points
opt_K, opt_R, opt_t, opt_points_3d = bundle_adjustment(
    pts1_inliers, pts2_inliers, K, R, t, points_3d)

# Update parameters with optimized values
K = opt_K
R = opt_R
t = opt_t
points_3d = opt_points_3d

print("\nBundle adjustment complete!")

# Plot 3D points and camera positions
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot triangulated 3D points
for i, point in enumerate(points_3d):
    ax.scatter(point[0], point[1], point[2], c='g', s=100, alpha=0.8)
    ax.text(point[0], point[1], point[2], str(i), size=10, color='black')

# Define camera visualization function
def plot_camera(ax, R, t, scale=0.5, label=None, color='red'):
    # Camera center
    center = -R.T @ t
    
    # Camera axes
    axes = scale * R.T  # Columns are the axes
    
    # Plot camera center
    ax.scatter(center[0], center[1], center[2], c=color, marker='o', s=100, label=label)
    
    # Plot camera axes
    ax.quiver(center[0], center[1], center[2], axes[0, 0], axes[1, 0], axes[2, 0], color='r', length=scale)
    ax.quiver(center[0], center[1], center[2], axes[0, 1], axes[1, 1], axes[2, 1], color='g', length=scale)
    ax.quiver(center[0], center[1], center[2], axes[0, 2], axes[1, 2], axes[2, 2], color='b', length=scale)
    
    return center

# Plot cameras
cam1_center = plot_camera(ax, np.eye(3), np.zeros((3, 1)), label='Camera 1', color='red')
cam2_center = plot_camera(ax, R, t, label='Camera 2', color='blue')

# Add a line connecting the cameras (baseline)
ax.plot([cam1_center[0], cam2_center[0]], 
        [cam1_center[1], cam2_center[1]], 
        [cam1_center[2], cam2_center[2]], 'k--')

# Set equal aspect ratio
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# Try to set equal aspect ratio for a better 3D visualization
if len(points_3d) > 1:  # Only if we have multiple points
    max_range = np.max([
        np.max(points_3d[:, 0]) - np.min(points_3d[:, 0]),
        np.max(points_3d[:, 1]) - np.min(points_3d[:, 1]),
        np.max(points_3d[:, 2]) - np.min(points_3d[:, 2])
    ]) / 2.0

    mid_x = (np.max(points_3d[:, 0]) + np.min(points_3d[:, 0])) / 2
    mid_y = (np.max(points_3d[:, 1]) + np.min(points_3d[:, 1])) / 2
    mid_z = (np.max(points_3d[:, 2]) + np.min(points_3d[:, 2])) / 2

    # Ensure cameras are in view by expanding the range
    cam_points = np.vstack([cam1_center, cam2_center])
    max_range = max(max_range, 
                   np.max(np.abs(cam_points - np.array([mid_x, mid_y, mid_z]))))
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.title('3D Reconstruction with Camera Positions')
plt.savefig('3d_reconstruction_white.png')
plt.show()

print("Note: Epipoles are points where all epilines converge")
print("In the result image, you should see epilines converging at the epipole")
print("\nNote on 3D reconstruction:")
print("1. The scale of reconstruction is arbitrary without metric calibration")
print("2. The reconstruction quality depends on the accuracy of the camera intrinsics")
print("3. For accurate results, you should use camera calibration parameters")

print("\nDone! The script has completed successfully.")
print(f"Detected {len(centers1)} dots in left image, {len(centers2)} dots in right image")
print(f"Found {len(matches)} point correspondences")
print(f"Used {len(pts1_inliers)} inlier points for 3D reconstruction")
print(f"Reconstructed {len(points_3d)} valid 3D points")