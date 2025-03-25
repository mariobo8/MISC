import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def display_labeled_dots(img1_color, img2_color, centers1, centers2, sizes1, sizes2, save_path=None):
    """Display and label detected dots in both images"""
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
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
    return img1_labeled, img2_labeled

def display_matches(img1_labeled, img2_labeled, centers1, centers2, matches, save_path=None):
    """Display matches between two images"""
    # Visualize matches
    match_img = np.hstack((img1_labeled, img2_labeled))
    for i, j in matches:
        pt1 = centers1[i]
        # Adjust x-coordinate for the right image in the combined image
        pt2 = (centers2[j][0] + img1_labeled.shape[1], centers2[j][1])
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv.line(match_img, pt1, pt2, color, 1)
    
    plt.figure(figsize=(15, 8))
    plt.imshow(cv.cvtColor(match_img, cv.COLOR_BGR2RGB))
    plt.title(f'Fixed Manual Pattern Matching - {len(matches)} matches')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
    return match_img

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

def display_epipolar_geometry(img1_color, img2_color, pts1_inliers, pts2_inliers, F, save_path=None):
    """Display epipolar lines for both images"""
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
    
    if save_path:
        plt.savefig(save_path)  # Save the result as an image file
    plt.show()
    
    return img5, img3

def plot_camera(ax, R, t, scale=0.5, label=None, color='red'):
    """Plot camera in 3D space"""
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

def plot_3d_reconstruction(points_3d, R, t, save_path=None):
    """Plot 3D reconstruction with camera positions"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot triangulated 3D points
    for i, point in enumerate(points_3d):
        ax.scatter(point[0], point[1], point[2], c='g', s=100, alpha=0.8)
        ax.text(point[0], point[1], point[2], str(i), size=10, color='black')

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
    
    if save_path:
        plt.savefig(save_path)
    plt.show()