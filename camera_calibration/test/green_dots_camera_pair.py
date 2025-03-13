import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

# Read the images in color
img1_color = cv.imread('/home/mariobo/Downloads/img_sx.jpg')  # left image
img2_color = cv.imread('/home/mariobo/Downloads/img_dx.jpg')  # right image

# Check if images are loaded successfully
if img1_color is None or img2_color is None:
    print("Error loading images. Please check the file paths.")
    sys.exit(1)

# Create grayscale versions for some operations
img1 = cv.cvtColor(img1_color, cv.COLOR_BGR2GRAY)
img2 = cv.cvtColor(img2_color, cv.COLOR_BGR2GRAY)

# Function to detect green dots with blob detection
def detect_green_dots_blob(img):
    # Convert BGR to HSV
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    # Define range for green color in HSV
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    
    # Threshold the HSV image to get only green colors
    mask = cv.inRange(hsv, lower_green, upper_green)
    
    # Apply morphological operations to clean up the mask
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    
    # Set up blob detector parameters
    params = cv.SimpleBlobDetector_Params()
    
    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200
    
    # Filter by color (255 = light, 0 = dark)
    params.filterByColor = True
    params.blobColor = 255  # White blobs (our mask has white blobs on black background)
    
    # Filter by area
    params.filterByArea = True
    params.minArea = 50
    params.maxArea = 10000
    
    # Filter by circularity
    params.filterByCircularity = True
    params.minCircularity = 0.5
    
    # Filter by convexity
    params.filterByConvexity = True
    params.minConvexity = 0.5
    
    # Filter by inertia (how circular vs. elliptical)
    params.filterByInertia = True
    params.minInertiaRatio = 0.5
    
    # Create detector
    detector = cv.SimpleBlobDetector_create(params)
    
    # Detect blobs
    keypoints = detector.detect(mask)
    
    # Extract coordinates
    centers = [(int(k.pt[0]), int(k.pt[1])) for k in keypoints]
    sizes = [k.size for k in keypoints]
    
    # Visualize detection
    img_with_keypoints = cv.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255), 
                                          cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return centers, sizes, img_with_keypoints, mask

# Detect green dots in both images
centers1, sizes1, img1_keypoints, mask1 = detect_green_dots_blob(img1_color)
centers2, sizes2, img2_keypoints, mask2 = detect_green_dots_blob(img2_color)

print(f"Detected {len(centers1)} green dots in left image")
print(f"Detected {len(centers2)} green dots in right image")

# Create labeled images for easy reference during manual matching
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
plt.title('Left Image - Numbered Green Dots')
plt.subplot(122), plt.imshow(cv.cvtColor(img2_labeled, cv.COLOR_BGR2RGB))
plt.title('Right Image - Numbered Green Dots')
plt.tight_layout()
plt.savefig('numbered_green_dots.png')
plt.show()

# Define manual matches here
# The format is: [(left_idx, right_idx), (left_idx, right_idx), ...]
# For example, if dot 0 in left image corresponds to dot 2 in right image,
# and dot 1 in left image corresponds to dot 3 in right image, etc.
# Example: manual_matches = [(0, 2), (1, 3), (2, 0), ...]

# Please fill in the correct matches after seeing the numbered dots
manual_matches = [
    # Uncomment and fill in the matches
    (0, 1),  # Left image dot 0 matches right image dot 0
    (1, 2),  # Left image dot 1 matches right image dot 1
    (2, 0),  # Left image dot 1 matches right image dot 1
    (3, 5),  # Left image dot 1 matches right image dot 1
    (4, 4),  # Left image dot 1 matches right image dot 1
    (5, 3),  # Left image dot 1 matches right image dot 1
    (6, 7),  # Left image dot 1 matches right image dot 1
    (7, 6),  # Left image dot 1 matches right image dot 1
    (8, 8),  # Left image dot 1 matches right image dot 1
    (9, 9),  # Left image dot 1 matches right image dot 1
    (10, 10),  # Left image dot 1 matches right image dot 1

]

print("Please update the manual_matches list in the code before continuing.")
print("You need to match the numbered green dots between the left and right images.")

# Comment out the next line after you've filled in the manual_matches list
#sys.exit(0)  # Remove this line after filling in the matches

# Extract matched points
pts1_matched = np.array([centers1[i] for i, _ in manual_matches], dtype=np.float32)
pts2_matched = np.array([centers2[j] for _, j in manual_matches], dtype=np.float32)

# Visualize matches
match_img = np.hstack((img1_labeled, img2_labeled))
for i, j in manual_matches:
    pt1 = centers1[i]
    # Adjust x-coordinate for the right image in the combined image
    pt2 = (centers2[j][0] + img1_color.shape[1], centers2[j][1])
    color = tuple(np.random.randint(0, 255, 3).tolist())
    cv.line(match_img, pt1, pt2, color, 1)
    
plt.figure(figsize=(15, 8))
plt.imshow(cv.cvtColor(match_img, cv.COLOR_BGR2RGB))
plt.title(f'Manual Matches Between Green Dots')
plt.savefig('manual_matches.png')
plt.show()

# Ensure we have enough points for calculating the fundamental matrix
if len(pts1_matched) < 8:
    print(f"ERROR: Need at least 8 point matches, but only found {len(pts1_matched)}.")
    print("Please add more manual matches.")
    sys.exit(1)

# Find the Fundamental Matrix
F, mask = cv.findFundamentalMat(pts1_matched, pts2_matched, cv.FM_RANSAC, 1.0, 0.99)

if F is None or F.shape[0] != 3:
    print("Failed to estimate fundamental matrix. Check your manual matches.")
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
plt.savefig('epipolar_result_manual.png')  # Save the result as an image file
plt.show()

# Define camera intrinsic parameters (these are example values - you should replace with your actual camera calibration)
# For more accurate results, you should calibrate your camera and use the actual intrinsic parameters
focal_length = 1000  # approximate focal length in pixels
principal_point = (img1.shape[1]//2, img1.shape[0]//2)  # approximate principal point at image center
K = np.array([
    [focal_length, 0, principal_point[0]],
    [0, focal_length, principal_point[1]],
    [0, 0, 1]
])

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
plt.savefig('3d_reconstruction_manual.png')
plt.show()

print("Note: Epipoles are points where all epilines converge")
print("In the result image, you should see epilines converging at the epipole")
print("\nNote on 3D reconstruction:")
print("1. The scale of reconstruction is arbitrary without metric calibration")
print("2. The reconstruction quality depends on the accuracy of the camera intrinsics")
print("3. For accurate results, you should use camera calibration parameters")