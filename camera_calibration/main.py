#!/usr/bin/env python3
"""
Multi-Camera Stereo Reconstruction Main Script
This script handles the 3D reconstruction from multiple stereo images using white dot markers.
It supports calibration of 3 cameras, performing pairwise calibrations between cameras 1-2 and 2-3.
"""

import numpy as np
import cv2 as cv
import sys
import os
import argparse
from calibration import (
    load_calibration,
    detect_white_dots_blob,
    detect_white_dots_alternative,
    fixed_manual_pattern_matching,
    bundle_adjustment
)
from visualization_utils import (
    display_labeled_dots,
    display_matches,
    display_epipolar_geometry,
    plot_3d_reconstruction
)

def process_image_pair(img1_color, img2_color, K, dist_coeffs, output_prefix, skip_bundle=False):
    """
    Process a pair of images to perform stereo reconstruction
    
    Args:
        img1_color: The first (reference) image
        img2_color: The second image
        K: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        output_prefix: Prefix for output files
        skip_bundle: If True, skip bundle adjustment
        
    Returns:
        Dict containing camera parameters and 3D points
    """
    print(f"\n=== Processing image pair with output prefix: {output_prefix} ===")
    
    # Create grayscale versions for some operations
    img1 = cv.cvtColor(img1_color, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(img2_color, cv.COLOR_BGR2GRAY)

    # Detect white dots in both images
    centers1, sizes1, img1_keypoints, mask1 = detect_white_dots_blob(img1_color)
    centers2, sizes2, img2_keypoints, mask2 = detect_white_dots_blob(img2_color)

    print(f"Detected {len(centers1)} white dots in reference image")
    print(f"Detected {len(centers2)} white dots in secondary image")

    # Create labeled images for easy reference
    img1_labeled, img2_labeled = display_labeled_dots(
        img1_color, img2_color, centers1, centers2, sizes1, sizes2, 
        f'{output_prefix}_numbered_white_dots.png'
    )

    # If the blob detector didn't work well, try the alternative approach
    if len(centers1) != 12 or len(centers2) != 12:
        print("Blob detector may not have found all dots. Trying alternative method...")
        centers1, sizes1, img1_keypoints, mask1 = detect_white_dots_alternative(img1_color)
        centers2, sizes2, img2_keypoints, mask2 = detect_white_dots_alternative(img2_color)
        
        print(f"Alternative method detected {len(centers1)} white dots in reference image")
        print(f"Alternative method detected {len(centers2)} white dots in secondary image")
        
        # Update labeled images
        img1_labeled, img2_labeled = display_labeled_dots(
            img1_color, img2_color, centers1, centers2, sizes1, sizes2, 
            f'{output_prefix}_numbered_white_dots_alternative.png'
        )

    # Use the fixed manual pattern matching approach
    matches = fixed_manual_pattern_matching(centers1, centers2)

    # Ensure we have enough points for calculating the fundamental matrix
    if len(matches) < 8:
        print(f"ERROR: Need at least 8 point matches, but only found {len(matches)}.")
        print("The matching failed to find enough correspondences.")
        return None

    # Visualize matches
    match_img = display_matches(
        img1_labeled, img2_labeled, centers1, centers2, matches, 
        f'{output_prefix}_matches.png'
    )

    # Extract matched points
    pts1_matched = np.array([centers1[i] for i, _ in matches], dtype=np.float32)
    pts2_matched = np.array([centers2[j] for _, j in matches], dtype=np.float32)

    # Find the Fundamental Matrix
    F, mask = cv.findFundamentalMat(pts1_matched, pts2_matched, cv.FM_RANSAC, 1.0, 0.99)

    if F is None or F.shape[0] != 3:
        print("Failed to estimate fundamental matrix. Check your matches.")
        return None

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

    # Display epipolar geometry
    img_left_epilines, img_right_epilines = display_epipolar_geometry(
        img1_color, img2_color, pts1_inliers, pts2_inliers, F, 
        f'{output_prefix}_epipolar.png'
    )

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
    if not skip_bundle:
        opt_K, opt_R, opt_t, opt_points_3d = bundle_adjustment(
            pts1_inliers, pts2_inliers, K, R, t, points_3d)

        # Update parameters with optimized values
        K_opt = opt_K
        R_opt = opt_R
        t_opt = opt_t
        points_3d_opt = opt_points_3d

        print("\nBundle adjustment complete!")
    else:
        print("\nSkipping bundle adjustment as requested")
        K_opt = K
        R_opt = R
        t_opt = t
        points_3d_opt = points_3d

    # Plot 3D points and camera positions
    plot_3d_reconstruction(points_3d_opt, R_opt, t_opt, f'{output_prefix}_3d_reconstruction.png')

    # Return the results
    result = {
        'camera_matrix': K_opt,
        'rotation': R_opt,
        'translation': t_opt,
        'points_3d': points_3d_opt,
        'points2D_img1': pts1_inliers,
        'points2D_img2': pts2_inliers,
        'F': F,
        'E': E
    }
    
    return result


def merge_reconstructions(results_1_2, results_2_3):
    """
    Merge two stereo reconstruction results with camera 2 as the bridge between 1 and 3
    
    Args:
        results_1_2: Results from cameras 1-2 calibration
        results_2_3: Results from cameras 2-3 calibration
        
    Returns:
        Merged 3D points and camera parameters
    """
    print("\n=== Merging Reconstructions ===")
    
    # Get camera parameters
    K = results_1_2['camera_matrix']  # Use optimized K from first pair
    
    # Camera 1-2 parameters (camera 2 relative to camera 1)
    R_1_2 = results_1_2['rotation']
    t_1_2 = results_1_2['translation']
    
    # Camera 2-3 parameters (camera 3 relative to camera 2)
    R_2_3 = results_2_3['rotation']
    t_2_3 = results_2_3['translation']
    
    # Calculate camera 3's position in camera 1's coordinate system
    # R_1_3 and t_1_3 define the transformation from camera 3 to camera 1
    R_1_3 = R_1_2 @ R_2_3
    t_1_3 = t_1_2 + R_1_2 @ t_2_3
    
    # Get 3D points from the first calibration - these are already in camera 1's coordinate system
    points_3d_1_2 = results_1_2['points_3d']
    
    # Get 3D points from the second calibration - these are in camera 2's coordinate system
    points_3d_2_3 = results_2_3['points_3d']
    
    # Transform points from camera 2's coordinate system to camera 1's coordinate system
    # We need to invert the transformation from camera 1 to camera 2
    R_2_1 = R_1_2.T  # Transpose for rotation inversion
    t_2_1 = -R_2_1 @ t_1_2.ravel()  # Translation inversion
    
    points_3d_2_3_transformed = []
    for point in points_3d_2_3:
        # First transform from camera 3's system to camera 2's system (already done)
        # Then transform from camera 2's system to camera 1's system
        transformed_point = R_2_1 @ point + t_2_1
        points_3d_2_3_transformed.append(transformed_point)
    
    points_3d_2_3_transformed = np.array(points_3d_2_3_transformed)
    
    # Combine the point sets in camera 1's coordinate system
    all_points_3d = np.vstack([points_3d_1_2, points_3d_2_3_transformed])
    
    print(f"Combined {len(points_3d_1_2)} points from cameras 1-2 with {len(points_3d_2_3)} points from cameras 2-3")
    print(f"Total of {len(all_points_3d)} 3D points")
    
    return {
        'camera_matrix': K,
        'rotation_2': R_1_2,      # Camera 2 relative to camera 1
        'translation_2': t_1_2,   # Camera 2 relative to camera 1
        'rotation_3': R_1_3,      # Camera 3 relative to camera 1
        'translation_3': t_1_3,   # Camera 3 relative to camera 1
        'points_3d': all_points_3d
    }


def plot_multi_camera_reconstruction(points_3d, R2, t2, R3, t3, save_path=None):
    """
    Plot 3D reconstruction with three camera positions
    
    Args:
        points_3d: 3D points in the scene
        R2: Rotation matrix of camera 2
        t2: Translation vector of camera 2
        R3: Rotation matrix of camera 3
        t3: Translation vector of camera 3
        save_path: Path to save the plot image
    """
    from visualization_utils import plot_camera
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot 3D points
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='g', s=50, alpha=0.5)

    # Plot cameras
    cam1_center = plot_camera(ax, np.eye(3), np.zeros((3, 1)), label='Camera 1', color='red')
    cam2_center = plot_camera(ax, R2, t2, label='Camera 2', color='blue')
    cam3_center = plot_camera(ax, R3, t3, label='Camera 3', color='green')

    # Add lines connecting the cameras to show the camera network
    ax.plot([cam1_center[0], cam2_center[0]], 
            [cam1_center[1], cam2_center[1]], 
            [cam1_center[2], cam2_center[2]], 'k--')
    
    ax.plot([cam2_center[0], cam3_center[0]], 
            [cam2_center[1], cam3_center[1]], 
            [cam2_center[2], cam3_center[2]], 'k--')

    # Set equal aspect ratio
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # Set view limits based on points and cameras
    if len(points_3d) > 1:
        # Include cameras in bounds calculations
        all_points = np.vstack([
            points_3d,
            cam1_center.reshape(1, 3),
            cam2_center.reshape(1, 3),
            cam3_center.reshape(1, 3)
        ])
        
        # Calculate center and range
        mid_x = (np.max(all_points[:, 0]) + np.min(all_points[:, 0])) / 2
        mid_y = (np.max(all_points[:, 1]) + np.min(all_points[:, 1])) / 2
        mid_z = (np.max(all_points[:, 2]) + np.min(all_points[:, 2])) / 2
        
        max_range = np.max([
            np.max(all_points[:, 0]) - np.min(all_points[:, 0]),
            np.max(all_points[:, 1]) - np.min(all_points[:, 1]),
            np.max(all_points[:, 2]) - np.min(all_points[:, 2])
        ]) / 2.0
        
        # Add a bit of margin
        max_range *= 1.2
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.title('3D Reconstruction with Three Cameras')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    """Main function to process images and perform multi-camera 3D reconstruction"""
    print("Multi-Camera Stereo Reconstruction - White Dot Pattern")
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(
        description='Stereo reconstruction from three images with white dot markers.'
    )
    parser.add_argument('--img1', default='test/captures/capture_000.png', help='Path to image from camera 1')
    parser.add_argument('--img2', default='test/captures/capture_001.png', help='Path to image from camera 2 (bridge camera)')
    parser.add_argument('--img3', default='test/captures/capture_002.png', help='Path to image from camera 3')
    parser.add_argument('--calib', default='camera_calibration.json', help='Path to camera calibration file')
    parser.add_argument('--output', default='results', help='Output directory for results')
    parser.add_argument('--no-bundle', action='store_true', help='Skip bundle adjustment')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Load the intrinsic camera parameters
    K, dist_coeffs = load_calibration(args.calib)

    # Read the images in color
    img1_color = cv.imread(args.img1)  # camera 1 image
    img2_color = cv.imread(args.img2)  # camera 2 image (bridge camera)
    img3_color = cv.imread(args.img3)  # camera 3 image

    # Check if images are loaded successfully
    if img1_color is None or img2_color is None or img3_color is None:
        print(f"Error loading one or more images:")
        print(f"Camera 1: {args.img1} - {'Loaded' if img1_color is not None else 'Failed'}")
        print(f"Camera 2: {args.img2} - {'Loaded' if img2_color is not None else 'Failed'}")
        print(f"Camera 3: {args.img3} - {'Loaded' if img3_color is not None else 'Failed'}")
        print("Please check that the file paths are correct.")
        sys.exit(1)

    # Process image pair 1-2
    output_prefix_1_2 = f"{args.output}/camera_1_2"
    results_1_2 = process_image_pair(
        img1_color, img2_color, K, dist_coeffs, output_prefix_1_2, args.no_bundle
    )
    
    if results_1_2 is None:
        print("Failed to process image pair 1-2. Exiting.")
        sys.exit(1)
    
    # Process image pair 2-3
    output_prefix_2_3 = f"{args.output}/camera_2_3"
    results_2_3 = process_image_pair(
        img2_color, img3_color, K, dist_coeffs, output_prefix_2_3, args.no_bundle
    )
    
    if results_2_3 is None:
        print("Failed to process image pair 2-3. Exiting.")
        sys.exit(1)
    
    # Merge reconstructions
    merged_results = merge_reconstructions(results_1_2, results_2_3)
    
    # Save merged results
    np.savez(
        f'{args.output}/merged_reconstruction.npz',
        camera_matrix=merged_results['camera_matrix'],
        rotation_camera2=merged_results['rotation_2'],
        translation_camera2=merged_results['translation_2'],
        rotation_camera3=merged_results['rotation_3'],
        translation_camera3=merged_results['translation_3'],
        points_3d=merged_results['points_3d']
    )
    
    # Create a visualization of all three cameras and the combined 3D points
    plot_multi_camera_reconstruction(
        merged_results['points_3d'],
        merged_results['rotation_2'], 
        merged_results['translation_2'],
        merged_results['rotation_3'],
        merged_results['translation_3'],
        f'{args.output}/multi_camera_reconstruction.png'
    )

    print("\nSummary:")
    print(f"Camera 1-2 Reconstruction: {len(results_1_2['points_3d'])} points")
    print(f"Camera 2-3 Reconstruction: {len(results_2_3['points_3d'])} points")
    print(f"Combined Reconstruction: {len(merged_results['points_3d'])} points")
    print(f"\nResults saved to {args.output}/ directory")
    print("\nDone! The script has completed successfully.")


if __name__ == "__main__":
    main()