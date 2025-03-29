import numpy as np
import cv2
import math
import os

class MultiViewCalibrationEnvironment:
    def __init__(self, views):
        """
        Initialize the calibration environment with predefined views.
        
        Args:
            views: List of dictionaries containing camera parameters for each view
                  Each dict should have: 'x', 'y', 'z', 'yaw', 'pitch'
        """
        # Window settings
        self.width, self.height = 800, 600
        
        # Camera settings
        self.focal_length = 300
        self.center_x = self.width / 2
        self.center_y = self.height / 2
        
        # Store the predefined views
        self.views = views
        
        # Create the 3D pattern
        self.create_pattern()
        
    def create_pattern(self):
        """Create the 3D calibration pattern."""
        self.pattern_3d = []
        
        # Parameters for pattern positioning
        x_spacing = 60
        
        # First row, first group (5 dots)
        for i in range(5):
            self.pattern_3d.append({"X": -120 + i*x_spacing, "Y": -40, "Z": 0})
        
        # First row, second group (1 dot)
        self.pattern_3d.append({"X": 200, "Y": -40, "Z": 0})
        
        # Second row, first group (4 dots)
        for i in range(4):
            self.pattern_3d.append({"X": -120 + i*x_spacing, "Y": 40, "Z": 0})
        
        # Second row, second group (2 dots)
        self.pattern_3d.append({"X": 180, "Y": 40, "Z": 0})
        self.pattern_3d.append({"X": 240, "Y": 40, "Z": 0})
    
    def render_view(self, camera_x, camera_y, camera_z, camera_yaw, camera_pitch):
        """
        Render the view from the given camera position and orientation.
        
        Returns:
            The rendered image
        """
        # Create black background
        image = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Create transformation matrices
        sin_yaw = math.sin(math.radians(camera_yaw))
        cos_yaw = math.cos(math.radians(camera_yaw))
        sin_pitch = math.sin(math.radians(camera_pitch))
        cos_pitch = math.cos(math.radians(camera_pitch))
        
        # Project all 3D points to 2D
        for point in self.pattern_3d:
            # Translate point relative to camera
            x = point["X"] - camera_x
            y = point["Y"] - camera_y
            z = point["Z"] - camera_z
            
            # Apply yaw rotation (around Y axis)
            x_rot = x * cos_yaw + z * sin_yaw
            z_rot = -x * sin_yaw + z * cos_yaw
            
            # Apply pitch rotation (around X axis)
            y_rot = y * cos_pitch - z_rot * sin_pitch
            z_rot_final = y * sin_pitch + z_rot * cos_pitch
            
            # Perspective projection
            if z_rot_final > 0:  # Only render points in front of camera
                u = int(self.focal_length * x_rot / z_rot_final + self.center_x)
                v = int(self.focal_length * y_rot / z_rot_final + self.center_y)
                
                if 0 <= u < self.width and 0 <= v < self.height:
                    # Size based on distance
                    size = max(2, int(8 * self.focal_length / z_rot_final))
                    cv2.circle(image, (u, v), size, 255, -1)
        
        return image
    
    def render_view_with_info(self, camera_x, camera_y, camera_z, camera_yaw, camera_pitch):
        """Render the view with camera information overlay."""
        # Get the base image
        image = self.render_view(camera_x, camera_y, camera_z, camera_yaw, camera_pitch)
        
        # Add camera position and orientation info
        info_text = f"Pos: ({camera_x:.1f}, {camera_y:.1f}, {camera_z:.1f}) | Yaw: {camera_yaw:.1f} | Pitch: {camera_pitch:.1f}"
        cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        
        return image
    
    def display_all_views(self):
        """
        Display all predefined views in separate windows.
        """
        # Create a window for each view
        for i, view in enumerate(self.views):
            window_name = f"View {i}: Camera Position {i+1}"
            cv2.namedWindow(window_name)
            
            # Get camera parameters for this view
            camera_x = view['x']
            camera_y = view['y']
            camera_z = view['z']
            camera_yaw = view['yaw']
            camera_pitch = view['pitch']
            
            # Render the view with information overlay
            display_image = self.render_view_with_info(camera_x, camera_y, camera_z, camera_yaw, camera_pitch)
            
            # Display the view in its window
            cv2.imshow(window_name, display_image)
            
            # Position the windows on screen
            x_pos = (i % 3) * 850  # Horizontally space windows
            y_pos = (i // 3) * 650  # Vertically space windows after 3 in a row
            cv2.moveWindow(window_name, x_pos, y_pos)
        
        print(f"Displaying {len(self.views)} views in separate windows.")
        print("Press any key to exit.")
        
        # Wait for a key press to exit
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    # Define your camera views here
    # Each view is a dictionary with x, y, z coordinates and yaw, pitch angles
    views = [
        # Example views from different positions and angles
        {'x': -223.6, 'y': -180, 'z': -359.4, 'yaw': -25, 'pitch': 25},  # View 1
        {'x': 278, 'y': -140, 'z': -319.44, 'yaw': 30, 'pitch': 25},     # View 2
    ]
    
    print("Starting Multiple View Display...")
    print(f"Setting up {len(views)} predefined views")
    
    # Create environment and display all views
    env = MultiViewCalibrationEnvironment(views)
    env.display_all_views()

if __name__ == "__main__":
    main()