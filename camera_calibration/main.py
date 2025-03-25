#!/usr/bin/env python3
"""
Stereo Reconstruction Main Script
This script handles the 3D reconstruction from stereo images using white dot markers.
It imports the core functionality from stereo_reconstruction.py which contains
detection, matching, and triangulation logic, while visualization functions are in
visualization_utils.py.
"""

import numpy as np
import cv2 as cv
import sys
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

def main():
    """Main function to process images and perform 3D reconstruction"""
    print("Stereo Reconstruction - White Dot Pattern")
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Stereo reconstruction from two images with white dot markers.')
    parser.add_argument('--left', default='test/captures/capture_000.png', help='Path to left image')
    parser.add_argument('--right', default='test/captures/capture_001.png', help='Path to right image')
    parser.add_argument('--calib', default='camera_calibration.json', help='Path to camera calibration file')
    parser.add_argument('--output', default='results', help='Output directory for results')
    parser.add_argument('--no-bundle', action='store_true', help='Skip bundle adjustment')
    args = parser.parse_args()
    
    # Load the intrinsic camera parameters
    K, dist_coeffs = load_calibration(args.calib)

    # Read the images in color
    img1_color = cv.imread(args.left)  # left image
    img2_color = cv.imread(args.right)  # right image

    # Check if images are loaded successfully
    if img1_color is None or img2_color is None:
        print(f"Error loading images: {args.left} or {args.right}")
        print("Please check that the file paths are correct.")
        sys.exit(1)

    # Create grayscale versions for some operations
    img1 = cv.cvtColor(img1_color, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(img2_color, cv.COLOR_BGR2GRAY)

    # Detect white dots in both images
    centers1, sizes1, img1_keypoints, mask1 = detect_white_dots_blob(img1_color)
    centers2, sizes2, img2_keypoints, mask2 = detect_white_dots_blob(img2_color)

    print(f"Detected {len(centers1)} white dots in left image")
    print(f"Detected {len(centers2)} white dots in right image")

    # Create labeled images for easy reference
    img1_labeled, img2_labeled = display_labeled_dots(
        img1_color, img2_color, centers1, centers2, sizes1, sizes2, 
        f'{args.output}/numbered_white_dots.png'
    )

    # If the blob detector didn't work well, try the alternative approach
    if len(centers1) != 12 or len(centers2) != 12:
        print("Blob detector may not have found all dots. Trying alternative method...")
        centers1, sizes1, img1_keypoints, mask1 = detect_white_dots_alternative(img1_color)
        centers2, sizes2, img2_keypoints, mask2 = detect_white_dots_alternative(img2_color)
        
        print(f"Alternative method detected {len(centers1)} white dots in left image")
        print(f"Alternative method detected {len(centers2)} white dots in right image")
        
        # Update labeled images
        img1_labeled, img2_labeled = display_labeled_dots(
            img1_color, img2_color, centers1, centers2, sizes1, sizes2, 
            f'{args.output}/numbered_white_dots_alternative.png'
        )

    # Use the fixed manual pattern matching approach
    matches = fixed_manual_pattern_matching(centers1, centers2)

    # Ensure we have enough points for calculating the fundamental matrix
    if len(matches) < 8:
        print(f"ERROR: Need at least 8 point matches, but only found {len(matches)}.")
        print("The matching failed to find enough correspondences.")
        sys.exit(1)

    # Visualize matches
    match_img = display_matches(img1_labeled, img2_labeled, centers1, centers2, matches, 
                              f'{args.output}/fixed_manual_pattern_matches.png')

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

    # Display epipolar geometry
    img_left_epilines, img_right_epilines = display_epipolar_geometry(
        img1_color, img2_color, pts1_inliers, pts2_inliers, F, 
        f'{args.output}/epipolar_result.png'
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
    if not args.no_bundle:
        opt_K, opt_R, opt_t, opt_points_3d = bundle_adjustment(
            pts1_inliers, pts2_inliers, K, R, t, points_3d)

        # Update parameters with optimized values
        K = opt_K
        R = opt_R
        t = opt_t
        points_3d = opt_points_3d

        print("\nBundle adjustment complete!")
    else:
        print("\nSkipping bundle adjustment as requested")

    # Plot 3D points and camera positions
    plot_3d_reconstruction(points_3d, R, t, f'{args.output}/3d_reconstruction.png')

    # Save reconstructed parameters and 3D points
    np.savez(
        f'{args.output}/reconstruction_results.npz',
        camera_matrix=K,
        rotation=R,
        translation=t,
        points_3d=points_3d,
        points2D_img1=pts1_inliers,
        points2D_img2=pts2_inliers
    )

    print("\nSummary:")
    print(f"Detected {len(centers1)} dots in left image, {len(centers2)} dots in right image")
    print(f"Found {len(matches)} point correspondences")
    print(f"Used {len(pts1_inliers)} inlier points for 3D reconstruction")
    print(f"Reconstructed {len(points_3d)} valid 3D points")
    print(f"\nResults saved to {args.output}/ directory")
    print("\nDone! The script has completed successfully.")

if __name__ == "__main__":
    main()