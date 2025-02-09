from calibration import TwoCameraCalibrator
import numpy as np

def main():
    calibrator = TwoCameraCalibrator()
    
    print("Displaying camera views. Press 'q' to continue to calibration...")
    dots1, dots2 = calibrator.show_camera_views()
    
    print("\nPerforming initial calibration...")
    R, t, F = calibrator.calibrate_pair(dots1, dots2)
    R_est, t_est, R_gt, t_gt = calibrator.analyze_transformations(dots1, dots2)
    
    if F is not None:
        print("\nShowing epipolar lines...")
        calibrator.show_epipolar_lines(dots1, dots2, F)
        
        while True:
            print("\nShowing 3D epipolar geometry...")
            do_bundle = calibrator.visualize_epipolar_geometry_3d(dots1, dots2, R_est, t_est)
            
            if do_bundle:
                print("\nPerforming bundle adjustment...")
                R_bundle, t_bundle, _ = calibrator.bundle_adjustment(dots1, dots2, R_est, t_est)
                R_est, t_est = R_bundle, t_bundle  # Update the current estimates
                
                # Analyze final results
                print("\nAnalyzing bundle adjustment results:")
                params = calibrator.cameras.get_camera_parameters()
                R_gt = params['R2']
                t_gt = params['t2']
                
                from scipy.spatial.transform import Rotation
                r_gt = Rotation.from_matrix(R_gt)
                r_bundle = Rotation.from_matrix(R_bundle)
                relative_rot = r_gt.inv() * r_bundle
                angle_error = relative_rot.magnitude() * 180 / np.pi
                
                scale = np.linalg.norm(t_gt) / np.linalg.norm(t_bundle)
                t_bundle_scaled = t_bundle * scale
                translation_error = np.linalg.norm(t_gt - t_bundle_scaled)
                
                print(f"Final rotation error: {angle_error:.2f} degrees")
                print(f"Final translation error: {translation_error:.2f} units")
            else:
                break

if __name__ == "__main__":
    main()