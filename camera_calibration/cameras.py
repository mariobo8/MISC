import numpy as np
import cv2

class MockTwoCameras:
    def __init__(self, fps=30, resolution=(640, 480)):
        self.fps = fps
        self.width, self.height = resolution
        self.num_cameras = 2
        
        # Create 3D points in world coordinates - adjusted to be more visible
        self.world_points = np.array([
            [-1, -1, 8],    # Front top left
            [1, -1, 8],     # Front top right
            [-1, 1, 8],     # Front bottom left
            [1, 1, 8],      # Front bottom right
            [-0.5, -0.5, 10],    # Back top left
            [0.5, -0.5, 10],     # Back top right
            [-0.5, 0.5, 10],     # Back bottom left
            [0.5, 0.5, 10],      # Back bottom right
            [0, 0, 9],      # Center point
            [-0.75, 0, 9]    # Extra point for asymmetry
        ], dtype=np.float32)

        # Camera 1 parameters (at origin, looking along Z axis)
        focal_length = self.width / (2 * np.tan(np.radians(30)))
        self.K = np.array([
            [focal_length, 0, self.width/2],
            [0, focal_length, self.height/2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.R1 = np.eye(3)
        self.t1 = np.zeros((3, 1))
        
        # Camera 2 parameters (rotated and translated)
        angle = np.radians(45)  # 45-degree rotation around Y axis
        self.R2 = np.array([
            [np.cos(angle), 0, -np.sin(angle)],
            [0, 1, 0],
            [np.sin(angle), 0, np.cos(angle)]
        ], dtype=np.float32)
        
        self.t2 = np.array([5, 0, 0], dtype=np.float32).reshape(3, 1)  # Translation

    def project_points(self, points_3d, R, t):
        """Project 3D points onto camera image plane."""
        # Convert to homogeneous coordinates
        points_3d_homog = np.hstack((points_3d, np.ones((len(points_3d), 1))))
        
        # Create projection matrix
        P = self.K @ np.hstack((R, t))
        
        # Project points
        points_proj = points_3d_homog @ P.T
        
        # Convert from homogeneous to image coordinates
        points_2d = points_proj[:, :2] / points_proj[:, 2:]
        return points_2d

    def generate_frame(self, camera_index):
        """Generate a frame for the specified camera."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Choose camera parameters based on index
        R = self.R1 if camera_index == 0 else self.R2
        t = self.t1 if camera_index == 0 else self.t2
        
        # Project 3D points to 2D
        points_2d = self.project_points(self.world_points, R, t)
        
        # Draw points
        for i, (x, y) in enumerate(points_2d.astype(int)):
            if 0 <= x < self.width and 0 <= y < self.height:
                cv2.circle(frame, (int(x), int(y)), 4, (255, 255, 255), -1)
                cv2.putText(frame, str(i), (int(x)+5, int(y)+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame
    
    def read(self, camera_index):
        """Read a frame from the specified camera."""
        frame = self.generate_frame(camera_index)
        return frame, 0

    def get_camera_parameters(self):
        """Return the true camera parameters."""
        return {
            'K': self.K,
            'R1': self.R1,
            't1': self.t1,
            'R2': self.R2,
            't2': self.t2
        }