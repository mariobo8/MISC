import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read the images
img1 = cv.imread('/home/mariobo/Downloads/img_sx.jpg', cv.IMREAD_GRAYSCALE)  # left image
img2 = cv.imread('/home/mariobo/Downloads/img_dx.jpg', cv.IMREAD_GRAYSCALE)  # right image

# Check if images are loaded successfully
if img1 is None or img2 is None:
    print("Error loading images. Please check the file paths.")
    exit()

# Initialize SIFT detector
sift = cv.SIFT_create()

# Find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# FLANN parameters for matching
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

# Use FLANN matcher
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Prepare lists for storing matched points
pts1 = []
pts2 = []

# Apply ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.8 * n.distance:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

# Convert points to numpy arrays
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

# Find the Fundamental Matrix
F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)

# Select only inlier points
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

# Function to draw epilines and matched points
def drawlines(img1, img2, lines, pts1, pts2):
    """
    Draw epilines on images
    img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines
    """
    r, c = img1.shape
    img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        
        # Draw the line ax + by + c = 0
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv.circle(img2, tuple(pt2), 5, color, -1)
    
    return img1, img2

# Calculate epilines for right image points and draw them on left image
lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

# Calculate epilines for left image points and draw them on right image
lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

# Display results
plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(img5)
plt.title('Left Image with Epilines from Right')
plt.subplot(122), plt.imshow(img3)
plt.title('Right Image with Epilines from Left')
plt.tight_layout()
plt.savefig('epipolar_result.png')  # Save the result as an image file
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
_, R, t, _ = cv.recoverPose(E, pts1, pts2, K)

# Print camera poses
print(f"Found {len(pts1)} good matches after filtering")
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
    pts1_float = np.float32(pts1)
    pts2_float = np.float32(pts2)
    
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
ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='blue', s=1, alpha=0.5, label='3D Points')

# Define camera visualization function
def plot_camera(ax, R, t, scale=0.5, label=None, color='red'):
    # Camera center
    center = -R.T @ t
    
    # Camera axes
    axes = scale * R.T  # Columns are the axes
    
    # Plot camera center
    ax.scatter(center[0], center[1], center[2], c=color, marker='o', s=50, label=label)
    
    # Plot camera axes
    ax.quiver(center[0], center[1], center[2], axes[0, 0], axes[1, 0], axes[2, 0], color='r', length=scale)
    ax.quiver(center[0], center[1], center[2], axes[0, 1], axes[1, 1], axes[2, 1], color='g', length=scale)
    ax.quiver(center[0], center[1], center[2], axes[0, 2], axes[1, 2], axes[2, 2], color='b', length=scale)

# Plot cameras
plot_camera(ax, np.eye(3), np.zeros((3, 1)), label='Camera 1', color='red')
plot_camera(ax, R, t, label='Camera 2', color='green')

# Set equal aspect ratio
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# Try to set equal aspect ratio for a better 3D visualization
max_range = np.max([
    np.max(points_3d[:, 0]) - np.min(points_3d[:, 0]),
    np.max(points_3d[:, 1]) - np.min(points_3d[:, 1]),
    np.max(points_3d[:, 2]) - np.min(points_3d[:, 2])
]) / 2.0

mid_x = (np.max(points_3d[:, 0]) + np.min(points_3d[:, 0])) / 2
mid_y = (np.max(points_3d[:, 1]) + np.min(points_3d[:, 1])) / 2
mid_z = (np.max(points_3d[:, 2]) + np.min(points_3d[:, 2])) / 2

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.title('3D Reconstruction with Camera Positions')
plt.savefig('3d_reconstruction.png')
plt.show()

print("Note: Epipoles are points where all epilines converge")
print("In the result image, you should see epilines converging at the epipole")
print("\nNote on 3D reconstruction:")
print("1. The scale of reconstruction is arbitrary without metric calibration")
print("2. The reconstruction quality depends on the accuracy of the camera intrinsics")
print("3. For accurate results, you should use camera calibration parameters")