import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize
from scipy.spatial.transform import Rotation
from cameras import MockTwoCameras

class TwoCameraCalibrator:
    def __init__(self):
        self.cameras = MockTwoCameras()
        self.F = None
        self.K = self.cameras.K  # We assume known intrinsics for simplicity

    def calibrate_pair(self, dots1, dots2):
        print(f"\nCalibrating with detected points:")
        print(f"Camera 1: {len(dots1)} points")
        print(f"Camera 2: {len(dots2)} points")

        if len(dots1) != len(dots2) or len(dots1) < 8:
            print("Error: Need at least 8 corresponding points")
            return None, None, None

        pts1 = np.float32(dots1)
        pts2 = np.float32(dots2)

        pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), self.K, None)
        pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), self.K, None)
        pts1_norm = pts1_norm.reshape(-1, 2)
        pts2_norm = pts2_norm.reshape(-1, 2)

        E, mask = cv2.findEssentialMat(pts1_norm, pts2_norm, focal=1.0, pp=(0., 0.), 
                                     method=cv2.RANSAC, prob=0.999, threshold=1e-3)

        _, R, t, mask = cv2.recoverPose(E, pts1_norm, pts2_norm)
        self.F = np.linalg.inv(self.K.T) @ E @ np.linalg.inv(self.K)

        return R, t, self.F

    def analyze_transformations(self, dots1, dots2):
        params = self.cameras.get_camera_parameters()
        R_gt = params['R2']
        t_gt = params['t2']

        R_est, t_est, _ = self.calibrate_pair(dots1, dots2)
        
        r_gt = Rotation.from_matrix(R_gt)
        r_est = Rotation.from_matrix(R_est)
        
        angle_gt = r_gt.as_rotvec()
        angle_est = r_est.as_rotvec()
        
        angle_gt_deg = np.degrees(np.linalg.norm(angle_gt))
        angle_est_deg = np.degrees(np.linalg.norm(angle_est))
        
        axis_gt = angle_gt / np.linalg.norm(angle_gt)
        axis_est = angle_est / np.linalg.norm(angle_est)

        print("\nRotation Analysis:")
        print(f"Ground Truth:")
        print(f"- Angle: {angle_gt_deg:.2f} degrees")
        print(f"- Rotation Matrix:\n{R_gt}")
        print(f"- Rotation Axis: {axis_gt}")
        
        print(f"\nEstimated:")
        print(f"- Angle: {angle_est_deg:.2f} degrees")
        print(f"- Rotation Matrix:\n{R_est}")
        print(f"- Rotation Axis: {axis_est}")
        
        scale = np.linalg.norm(t_gt) / np.linalg.norm(t_est)
        t_est_scaled = t_est * scale
        
        print("\nTranslation Analysis:")
        print(f"Ground Truth: {t_gt.flatten()}")
        print(f"Estimated (scaled): {t_est_scaled.flatten()}")
        
        angle_error = np.abs(angle_gt_deg - angle_est_deg)
        axis_error = np.arccos(np.clip(np.abs(np.dot(axis_gt, axis_est)), -1.0, 1.0))
        translation_error = np.linalg.norm(t_gt - t_est_scaled)
        
        print("\nErrors:")
        print(f"Rotation Angle Error: {angle_error:.2f} degrees")
        print(f"Rotation Axis Error: {np.degrees(axis_error):.2f} degrees")
        print(f"Translation Error (after scaling): {translation_error:.2f} units")

        return R_est, t_est, R_gt, t_gt

    def detect_dots(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        dots = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                dots.append([cx, cy])
        
        return np.array(sorted(dots, key=lambda p: (p[1], p[0])))

    def draw_epipolar_lines(self, frame1, frame2, pts1, pts2, F):
        def draw_line(img, line, color):
            height, width = img.shape[:2]
            x0, y0 = 0, int((-line[2]) / line[1])
            x1, y1 = width, int((-line[2] - line[0] * width) / line[1])
            cv2.line(img, (x0, y0), (x1, y1), color, 1)

        frame1_lines = frame1.copy()
        frame2_lines = frame2.copy()

        for i in range(len(pts1)):
            pt1 = np.append(pts1[i], 1)
            pt2 = np.append(pts2[i], 1)

            line2 = F @ pt1
            line1 = F.T @ pt2

            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            cv2.circle(frame1_lines, tuple(pts1[i].astype(int)), 3, color, -1)
            cv2.circle(frame2_lines, tuple(pts2[i].astype(int)), 3, color, -1)

            draw_line(frame2_lines, line2, color)

        return frame1_lines, frame2_lines

    def show_camera_views(self):
        while True:
            frame1, _ = self.cameras.read(0)
            frame2, _ = self.cameras.read(1)

            dots1 = self.detect_dots(frame1)
            dots2 = self.detect_dots(frame2)

            frame1_marked = frame1.copy()
            frame2_marked = frame2.copy()
            
            for i, (dot1, dot2) in enumerate(zip(dots1, dots2)):
                x1, y1 = dot1.astype(int)
                x2, y2 = dot2.astype(int)
                cv2.drawMarker(frame1_marked, (x1, y1), (0, 0, 255), cv2.MARKER_CROSS, 10, 2)
                cv2.drawMarker(frame2_marked, (x2, y2), (0, 0, 255), cv2.MARKER_CROSS, 10, 2)
                cv2.putText(frame1_marked, f"D{i}", (x1-20, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.putText(frame2_marked, f"D{i}", (x2-20, y2-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            cv2.putText(frame1_marked, f"Camera 1 - {len(dots1)} points", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame2_marked, f"Camera 2 - {len(dots2)} points", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            combined_frame = np.hstack((frame1_marked, frame2_marked))
            cv2.imshow('Two Camera Views', combined_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        return dots1, dots2

    def show_epipolar_lines(self, dots1, dots2, F):
        frame1, _ = self.cameras.read(0)
        frame2, _ = self.cameras.read(1)

        frame1_lines, frame2_lines = self.draw_epipolar_lines(frame1, frame2, dots1, dots2, F)

        combined_frame = np.hstack((frame1_lines, frame2_lines))
        cv2.imshow('Epipolar Lines', combined_frame)
        print("\nShowing epipolar lines. Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def triangulate_points(self, dots1, dots2, R, t):
        pts1 = np.float32(dots1).reshape(-1, 1, 2)
        pts2 = np.float32(dots2).reshape(-1, 1, 2)
        
        pts1_norm = cv2.undistortPoints(pts1, self.K, None)
        pts2_norm = cv2.undistortPoints(pts2, self.K, None)
        
        pts1_norm = pts1_norm.reshape(-1, 2)
        pts2_norm = pts2_norm.reshape(-1, 2)
        
        P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = self.K @ np.hstack((R, t))
        
        points_4d = cv2.triangulatePoints(P1, P2, 
                                        pts1.reshape(-1, 2).T,
                                        pts2.reshape(-1, 2).T)
        
        points_3d = points_4d[:3] / points_4d[3]
        return points_3d.T

    def bundle_adjustment(self, dots1, dots2, R_init, t_init):
        pts1_norm = cv2.undistortPoints(np.float32(dots1).reshape(-1, 1, 2), 
                                    self.K, None).reshape(-1, 2)
        pts2_norm = cv2.undistortPoints(np.float32(dots2).reshape(-1, 1, 2), 
                                    self.K, None).reshape(-1, 2)
        
        points_3d_init = self.triangulate_points(dots1, dots2, R_init, t_init)
        r_init = Rotation.from_matrix(R_init).as_rotvec()
        
        def project_points_norm(points_3d, rvec, tvec):
            R = Rotation.from_rotvec(rvec).as_matrix()
            
            points_c1 = points_3d
            points_norm1 = points_c1[:, :2] / points_c1[:, 2:]
            
            points_c2 = (R @ points_3d.T).T + tvec
            points_norm2 = points_c2[:, :2] / points_c2[:, 2:]
            
            return points_norm1, points_norm2

        def compute_cost(params):
            rvec = params[:3]
            tvec = params[3:6]
            points_3d = params[6:].reshape(-1, 3)
            
            proj1, proj2 = project_points_norm(points_3d, np.zeros(3), np.zeros(3))
            proj1_2, proj2_2 = project_points_norm(points_3d, rvec, tvec)
            
            reproj_error = np.sum((pts1_norm - proj1)**2) + np.sum((pts2_norm - proj2_2)**2)
            
            R_init_vec = Rotation.from_matrix(R_init).as_rotvec()
            rot_reg = 1.0 * np.sum((rvec - R_init_vec)**2)
            
            t_init_dir = t_init.flatten() / np.linalg.norm(t_init.flatten())
            t_cur_dir = tvec / np.linalg.norm(tvec)
            trans_reg = 1.0 * np.sum((t_cur_dir - t_init_dir)**2)
            
            dist_reg = 0.1 * np.sum(np.maximum(0, np.linalg.norm(points_3d, axis=1) - 15)**2)
            depth_reg = 1.0 * np.sum(np.maximum(0, -points_3d[:, 2])**2)
            
            total_cost = reproj_error + rot_reg + trans_reg + dist_reg + depth_reg
            return total_cost
        
        initial_params = np.concatenate([
            r_init,
            t_init.flatten(),
            points_3d_init.flatten()
        ])
        
        print("\nPerforming bundle adjustment with strong regularization...")
        result = optimize.minimize(
            compute_cost,
            initial_params,
            method='BFGS',
            options={'maxiter': 100, 'disp': True}
        )
        
        rvec_opt = result.x[:3]
        tvec_opt = result.x[3:6]
        points_3d_opt = result.x[6:].reshape(-1, 3)
        
        R_opt = Rotation.from_rotvec(rvec_opt).as_matrix()
        
        scale = np.linalg.norm(t_init) / np.linalg.norm(tvec_opt)
        t_opt = tvec_opt * scale
        
        return R_opt, t_opt.reshape(3, 1), points_3d_opt

    def visualize_epipolar_geometry_3d(self, dots1, dots2, R_est, t_est, wait_for_bundle=True):
        """
        Visualize epipolar geometry in 3D space with interactive key handling
        """
        class KeyHandler:
            def __init__(self):
                self.do_bundle = False
                
            def on_key(self, event):
                if event.key == 'b':
                    plt.close()
                    self.do_bundle = True
                elif event.key == 'q':
                    plt.close()
                    self.do_bundle = False

        key_handler = KeyHandler()
        fig = plt.figure(figsize=(12, 8))
        fig.canvas.mpl_connect('key_press_event', key_handler.on_key)
        ax = fig.add_subplot(111, projection='3d')
        
        # Get ground truth parameters
        params = self.cameras.get_camera_parameters()
        R_gt = params['R2']
        t_gt = params['t2']
        
        # Camera 1 is always at origin
        C1 = np.array([0, 0, 0])
        
        # Plot camera 1 (common to both configurations)
        ax.scatter(*C1, color='blue', s=100, label='Camera 1')
        
        # Draw camera 1 coordinate frame (common)
        length = 1.0
        colors = ['r', 'g', 'b']
        for i in range(3):
            axis = np.zeros((2, 3))
            axis[1, i] = length
            ax.plot(axis[:, 0], axis[:, 1], axis[:, 2], color=colors[i])
        
        # Plot ground truth configuration (grey, dashed)
        C2_gt = t_gt.flatten()
        ax.scatter(*C2_gt, color='gray', s=100, label='Camera 2 (Ground Truth)')
        
        # Ground truth camera 2 axes
        for i in range(3):
            axis = np.zeros((2, 3))
            axis[1, i] = length
            axis_rotated = (R_gt @ axis.T).T + C2_gt
            ax.plot(axis_rotated[:, 0], axis_rotated[:, 1], axis_rotated[:, 2], 
                color='gray', linestyle='--')
        
        # Ground truth 3D points
        points_3d_gt = self.triangulate_points(dots1, dots2, R_gt, t_gt)
        ax.scatter(points_3d_gt[:, 0], points_3d_gt[:, 1], points_3d_gt[:, 2],
                color='gray', s=50, alpha=0.5, label='3D Points (Ground Truth)')
        
        # Ground truth camera rays
        for i in range(len(dots1)):
            ax.plot([C1[0], points_3d_gt[i, 0]], 
                [C1[1], points_3d_gt[i, 1]], 
                [C1[2], points_3d_gt[i, 2]], 
                'gray', linestyle='--', alpha=0.3)
            ax.plot([C2_gt[0], points_3d_gt[i, 0]],
                [C2_gt[1], points_3d_gt[i, 1]],
                [C2_gt[2], points_3d_gt[i, 2]],
                'gray', linestyle='--', alpha=0.3)
        
        # Plot estimated configuration (colored, solid)
        # Scale estimated translation to match ground truth scale
        scale = np.linalg.norm(t_gt) / np.linalg.norm(t_est)
        t_est_scaled = t_est * scale
        C2_est = t_est_scaled.flatten()
        ax.scatter(*C2_est, color='red', s=100, label='Camera 2 (Estimated)')
        
        # Estimated camera 2 axes
        for i in range(3):
            axis = np.zeros((2, 3))
            axis[1, i] = length
            axis_rotated = (R_est @ axis.T).T + C2_est
            ax.plot(axis_rotated[:, 0], axis_rotated[:, 1], axis_rotated[:, 2], 
                color=colors[i], linestyle='-')
        
        # Estimated 3D points
        points_3d_est = self.triangulate_points(dots1, dots2, R_est, t_est_scaled)
        ax.scatter(points_3d_est[:, 0], points_3d_est[:, 1], points_3d_est[:, 2],
                color='green', s=50, alpha=0.5, label='3D Points (Estimated)')
        
        # Estimated camera rays
        for i in range(len(dots1)):
            ax.plot([C1[0], points_3d_est[i, 0]], 
                [C1[1], points_3d_est[i, 1]], 
                [C1[2], points_3d_est[i, 2]], 
                'b-', alpha=0.3)
            ax.plot([C2_est[0], points_3d_est[i, 0]],
                [C2_est[1], points_3d_est[i, 1]],
                [C2_est[2], points_3d_est[i, 2]],
                'r-', alpha=0.3)
        
        # Set axis properties
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Epipolar Geometry Visualization\n(Ground Truth in Grey Dashed, Estimated in Color)')
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim([-2, 6])
        ax.set_ylim([-2, 2])
        ax.set_zlim([0, 10])
        ax.legend()
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        
        if wait_for_bundle:
            print("\nPress 'b' to perform bundle adjustment, 'q' to exit...")
            plt.show()
            return key_handler.do_bundle
        else:
            plt.show()
            return False