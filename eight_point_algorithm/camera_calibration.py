import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MockTwoCameras:
    def __init__(self, fps=30, resolution=(640, 480)):
        self.fps = fps
        self.width, self.height = resolution
        self.num_cameras = 2
        
        # Create asymmetric 3D-like pattern
        self.pattern_points = np.array([
            [150, 100],   # Top left region
            [200, 120],
            [180, 150],
            [350, 150],   # Top right region
            [400, 180],
            [380, 200],
            [150, 300],   # Bottom left region
            [200, 350],
            [450, 250],   # Bottom right region
            [500, 300],


        ], dtype=np.float32)
        
        # Define transformation for second camera view
        angle = np.radians(20)
        self.view2_matrix = np.array([
            [np.cos(angle), 0, -10],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=np.float32)

    def generate_frame(self, camera_index):
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        if camera_index == 0:
            points = self.pattern_points
        else:
            points_homog = np.column_stack([self.pattern_points, np.ones(len(self.pattern_points))])
            transformed_points = (self.view2_matrix @ points_homog.T).T
            points = transformed_points[:, :2]
        
        # Draw points
        for i, (x, y) in enumerate(points.astype(int)):
            if 0 <= x < self.width and 0 <= y < self.height:
                cv2.circle(frame, (int(x), int(y)), 4, (255, 255, 255), -1)
                cv2.putText(frame, str(i), (int(x)+5, int(y)+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame
    
    def read(self, camera_index):
        frame = self.generate_frame(camera_index)
        return frame, 0

class TwoCameraCalibrator:
    def __init__(self):
        self.cameras = MockTwoCameras()
        self.F = None  # Store fundamental matrix

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
        """Draw epipolar lines and points."""
        def draw_line(img, line, color):
            height, width = img.shape[:2]
            # Get points for line ax + by + c = 0
            x0, y0 = 0, int((-line[2]) / line[1])
            x1, y1 = width, int((-line[2] - line[0] * width) / line[1])
            cv2.line(img, (x0, y0), (x1, y1), color, 1)  # Made line thinner

        frame1_lines = frame1.copy()
        frame2_lines = frame2.copy()

        # Draw points and epipolar lines
        for i in range(len(pts1)):
            # Get epipolar lines
            pt1 = np.append(pts1[i], 1)  # Homogeneous coordinates
            pt2 = np.append(pts2[i], 1)

            # Line in second image for point in first image
            line2 = F @ pt1
            # Line in first image for point in second image
            line1 = F.T @ pt2

            # Draw points
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            cv2.circle(frame1_lines, tuple(pts1[i].astype(int)), 3, color, -1)  # Made points smaller
            cv2.circle(frame2_lines, tuple(pts2[i].astype(int)), 3, color, -1)  # Made points smaller

            # Draw epipolar lines
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
            
            # Draw detected points
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

            # Show combined view
            combined_frame = np.hstack((frame1_marked, frame2_marked))
            cv2.imshow('Two Camera Views', combined_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        return dots1, dots2

    def show_epipolar_lines(self, dots1, dots2, F):
        """Show frames with epipolar lines after calibration."""
        frame1, _ = self.cameras.read(0)
        frame2, _ = self.cameras.read(1)

        # Draw epipolar lines
        frame1_lines, frame2_lines = self.draw_epipolar_lines(frame1, frame2, dots1, dots2, F)

        # Show the result
        combined_frame = np.hstack((frame1_lines, frame2_lines))
        cv2.imshow('Epipolar Lines', combined_frame)
        print("\nShowing epipolar lines. Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def calibrate_pair(self, dots1, dots2):
        print(f"\nCalibrating with detected points:")
        print(f"Camera 1: {len(dots1)} points")
        print(f"Camera 2: {len(dots2)} points")

        if len(dots1) != len(dots2) or len(dots1) < 8:
            print("Error: Need at least 8 corresponding points")
            return None, None, None

        # Convert points to float32
        pts1 = np.float32(dots1)
        pts2 = np.float32(dots2)

        # Normalize points to [-1, 1] range
        height, width = self.cameras.height, self.cameras.width
        pts1_norm = pts1.copy()
        pts2_norm = pts2.copy()
        
        pts1_norm[:, 0] = (pts1[:, 0] - width/2) / (width/2)
        pts2_norm[:, 0] = (pts2[:, 0] - width/2) / (width/2)
        pts1_norm[:, 1] = (pts1[:, 1] - height/2) / (height/2)
        pts2_norm[:, 1] = (pts2[:, 1] - height/2) / (height/2)

        # Calculate fundamental matrix
        F, mask = cv2.findFundamentalMat(pts1_norm, pts2_norm, cv2.FM_8POINT, 1e-8, 0.99)
        print("\nFundamental Matrix:")
        print(F)

        # Recover rotation and translation
        points1 = pts1_norm.reshape(-1, 1, 2)
        points2 = pts2_norm.reshape(-1, 1, 2)
        K = np.eye(3)  # Identity for normalized coordinates
        retval, R, t, mask = cv2.recoverPose(F, points1, points2, K)

        print("\nRecovered Rotation Matrix:")
        print(R)
        print("\nRecovered Translation Vector:")
        print(t)

        angle_rad = np.arccos((np.trace(R) - 1) / 2)
        print(f"\nRecovered rotation angle: {np.degrees(angle_rad):.2f} degrees")
        print(f"Expected rotation angle: 20.00 degrees")

        # Store F for epipolar line visualization
        self.F = F

        return R, t, F

    def analyze_transformations(self, dots1, dots2):
        """Analyze and compare ground truth vs estimated camera transformations."""
        # Ground truth transformation (from MockTwoCameras)
        angle_gt = np.radians(20)  # 20 degrees
        R_gt = np.array([
            [np.cos(angle_gt), 0, -np.sin(angle_gt)],
            [0, 1, 0],
            [np.sin(angle_gt), 0, np.cos(angle_gt)]
        ])
        t_gt = np.array([-50, 0, 0]).reshape(3, 1)  # -50 pixels in X

        # Get estimated transformation
        pts1 = np.float32(dots1)
        pts2 = np.float32(dots2)
        
        # Normalize points to [-1, 1] range
        height, width = self.cameras.height, self.cameras.width
        pts1_norm = pts1.copy()
        pts2_norm = pts2.copy()
        
        pts1_norm[:, 0] = (pts1[:, 0] - width/2) / (width/2)
        pts2_norm[:, 0] = (pts2[:, 0] - width/2) / (width/2)
        pts1_norm[:, 1] = (pts1[:, 1] - height/2) / (height/2)
        pts2_norm[:, 1] = (pts2[:, 1] - height/2) / (height/2)

        # Calculate fundamental matrix and recover pose
        F, mask = cv2.findFundamentalMat(pts1_norm, pts2_norm, cv2.FM_8POINT, 1e-8, 0.99)
        points1 = pts1_norm.reshape(-1, 1, 2)
        points2 = pts2_norm.reshape(-1, 1, 2)
        K = np.eye(3)  # Identity for normalized coordinates
        retval, R_est, t_est, mask = cv2.recoverPose(F, points1, points2, K)

        # Compare rotations
        angle_est_rad = np.arccos((np.trace(R_est) - 1) / 2)
        angle_est_deg = np.degrees(angle_est_rad)
        
        # Extract rotation axes
        from scipy.spatial.transform import Rotation
        r_gt = Rotation.from_matrix(R_gt)
        r_est = Rotation.from_matrix(R_est)
        
        axis_gt = r_gt.as_rotvec() / np.linalg.norm(r_gt.as_rotvec())
        axis_est = r_est.as_rotvec() / np.linalg.norm(r_est.as_rotvec())

        print("\nRotation Analysis:")
        print(f"Ground Truth:")
        print(f"- Angle: 20.00 degrees")
        print(f"- Rotation Matrix:\n{R_gt}")
        print(f"- Rotation Axis: {axis_gt}")
        
        print(f"\nEstimated:")
        print(f"- Angle: {angle_est_deg:.2f} degrees")
        print(f"- Rotation Matrix:\n{R_est}")
        print(f"- Rotation Axis: {axis_est}")
        
        # Compare translations
        # Normalize estimated translation to match scale
        t_est_norm = t_est * 50 / np.abs(t_est[0])  # Scale to match ground truth X translation
        
        print("\nTranslation Analysis:")
        print(f"Ground Truth: {t_gt.flatten()}")
        print(f"Estimated (scaled): {t_est_norm.flatten()}")
        
        # Calculate errors
        angle_error = np.abs(20 - angle_est_deg)
        translation_error = np.linalg.norm(t_gt - t_est_norm)
        axis_error = np.arccos(np.clip(np.dot(axis_gt, axis_est), -1.0, 1.0))
        
        print("\nErrors:")
        print(f"Rotation Angle Error: {angle_error:.2f} degrees")
        print(f"Rotation Axis Error: {np.degrees(axis_error):.2f} degrees")
        print(f"Translation Error (after scaling): {translation_error:.2f} units")

        return R_est, t_est, R_gt, t_gt



    def visualize_epipolar_geometry_3d(self, dots1, dots2, R, t):
        """
        Visualize epipolar geometry in 3D space, showing:
        - Camera positions and orientations
        - 3D points
        - Epipolar lines
        - Camera rays
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set up first camera at origin
        C1 = np.array([0, 0, 0])
        R1 = np.eye(3)
        
        # Set up second camera using R and t
        C2 = t.flatten()
        R2 = R
        
        # Plot camera positions
        ax.scatter(*C1, color='blue', s=100, label='Camera 1')
        ax.scatter(*C2, color='red', s=100, label='Camera 2')
        
        # Draw camera coordinate frames
        length = 0.5
        colors = ['r', 'g', 'b']
        
        # Camera 1 axes
        for i in range(3):
            axis = np.zeros((2, 3))
            axis[1, i] = length
            ax.plot(axis[:, 0], axis[:, 1], axis[:, 2], color=colors[i])
            
        # Camera 2 axes
        for i in range(3):
            axis = np.zeros((2, 3))
            axis[1, i] = length
            axis_rotated = (R2 @ axis.T).T + C2
            ax.plot(axis_rotated[:, 0], axis_rotated[:, 1], axis_rotated[:, 2], 
                   color=colors[i], linestyle='--')
        
        # Calculate and plot epipolar lines for a few sample points
        for i in range(len(dots1)):
            # Convert image points to normalized coordinates
            x1 = (dots1[i][0] - self.cameras.width/2) / self.cameras.width
            y1 = (dots1[i][1] - self.cameras.height/2) / self.cameras.height
            x2 = (dots2[i][0] - self.cameras.width/2) / self.cameras.width
            y2 = (dots2[i][1] - self.cameras.height/2) / self.cameras.height
            
            # Create rays from each camera
            ray1 = np.array([x1, y1, 1.0])
            ray2 = np.array([x2, y2, 1.0])
            
            # Transform ray2 to world coordinates
            ray2_world = (R2 @ ray2) + t.flatten()
            
            # Plot rays
            ax.plot([C1[0], ray1[0]], [C1[1], ray1[1]], [C1[2], ray1[2]], 
                   'b-', alpha=0.3)
            ax.plot([C2[0], ray2_world[0]], [C2[1], ray2_world[1]], 
                   [C2[2], ray2_world[2]], 'r-', alpha=0.3)
            
            # Calculate and plot epipolar line (plane intersection)
            points = []
            for depth in np.linspace(0, 2, 20):
                point1 = ray1 * depth
                point2 = ray2_world * depth
                points.append((point1 + point2) / 2)
            
            points = np.array(points)
            ax.plot(points[:, 0], points[:, 1], points[:, 2], 'g-', alpha=0.5)
        
        # Set axis labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Epipolar Geometry Visualization')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Add legend
        ax.legend()
        
        # Show plot
        plt.show()


def main():
    calibrator = TwoCameraCalibrator()
    
    print("Displaying camera views. Press 'q' to continue to calibration...")
    dots1, dots2 = calibrator.show_camera_views()
    
    print("\nPerforming calibration...")
    R, t, F = calibrator.calibrate_pair(dots1, dots2)

    R_est, t_est, R_gt, t_gt = calibrator.analyze_transformations(dots1, dots2)
    
    if F is not None:
        print("\nShowing epipolar lines...")
        calibrator.show_epipolar_lines(dots1, dots2, F)
        
        print("\nShowing 3D epipolar geometry...")
        calibrator.visualize_epipolar_geometry_3d(dots1, dots2, R_est, t_est)

if __name__ == "__main__":
    main()