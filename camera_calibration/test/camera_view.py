import numpy as np
import cv2
import math
import os
import threading
import time

class InteractiveMultiViewEnvironment:
    def __init__(self, views):
        """
        Initialize the calibration environment with predefined views.
        
        Args:
            views: List of dictionaries containing camera parameters for each view
                  Each dict should have: 'x', 'y', 'z', 'yaw', 'pitch'
        """
        # Window settings
        self.width, self.height = 800, 600
        self.main_window = "Pattern Control (Press Arrow Keys to Move Pattern)"
        
        # Camera settings
        self.focal_length = 300
        self.center_x = self.width / 2
        self.center_y = self.height / 2
        
        # Store the predefined views
        self.views = views
        
        # Pattern position (can be moved with arrow keys)
        self.pattern_x = 0
        self.pattern_y = 0
        self.pattern_z = 0
        
        # Movement speed for pattern
        self.pattern_move_speed = 10
        
        # Create the 3D pattern
        self.create_pattern()
        
        # Flag to control the rendering thread
        self.running = True
        
    def create_pattern(self):
        """Create the 3D calibration pattern."""
        self.base_pattern_3d = []
        
        # Parameters for pattern positioning
        x_spacing = 60
        
        # First row, first group (5 dots)
        for i in range(5):
            self.base_pattern_3d.append({"X": -120 + i*x_spacing, "Y": -40, "Z": 0})
        
        # First row, second group (1 dot)
        self.base_pattern_3d.append({"X": 200, "Y": -40, "Z": 0})
        
        # Second row, first group (4 dots)
        for i in range(4):
            self.base_pattern_3d.append({"X": -120 + i*x_spacing, "Y": 40, "Z": 0})
        
        # Second row, second group (2 dots)
        self.base_pattern_3d.append({"X": 180, "Y": 40, "Z": 0})
        self.base_pattern_3d.append({"X": 240, "Y": 40, "Z": 0})
    
    def get_pattern_3d(self):
        """Get the pattern with the current offset applied."""
        pattern_3d = []
        for point in self.base_pattern_3d:
            pattern_3d.append({
                "X": point["X"] + self.pattern_x,
                "Y": point["Y"] + self.pattern_y,
                "Z": point["Z"] + self.pattern_z
            })
        return pattern_3d
    
    def render_view(self, camera_x, camera_y, camera_z, camera_yaw, camera_pitch):
        """
        Render the view from the given camera position and orientation.
        
        Returns:
            The rendered image
        """
        # Create black background
        image = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Get current pattern position
        pattern_3d = self.get_pattern_3d()
        
        # Create transformation matrices
        sin_yaw = math.sin(math.radians(camera_yaw))
        cos_yaw = math.cos(math.radians(camera_yaw))
        sin_pitch = math.sin(math.radians(camera_pitch))
        cos_pitch = math.cos(math.radians(camera_pitch))
        
        # Project all 3D points to 2D
        for point in pattern_3d:
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
    
    def render_view_with_info(self, camera_x, camera_y, camera_z, camera_yaw, camera_pitch, window_title=""):
        """Render the view with camera information overlay."""
        # Get the base image
        image = self.render_view(camera_x, camera_y, camera_z, camera_yaw, camera_pitch)
        
        # Add camera position and orientation info
        camera_info = f"Camera: ({camera_x:.1f}, {camera_y:.1f}, {camera_z:.1f}) | Yaw: {camera_yaw:.1f} | Pitch: {camera_pitch:.1f}"
        cv2.putText(image, camera_info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
        
        # Add pattern position info
        pattern_info = f"Pattern: ({self.pattern_x:.1f}, {self.pattern_y:.1f}, {self.pattern_z:.1f})"
        cv2.putText(image, pattern_info, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
        
        # Add window title if provided
        if window_title:
            cv2.putText(image, window_title, (10, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
        
        return image
    
    def update_windows(self):
        """Update all windows with the current pattern position."""
        # Update the main control window
        control_image = np.zeros((self.height, self.width), dtype=np.uint8)
        pattern_info = f"Pattern Position: ({self.pattern_x:.1f}, {self.pattern_y:.1f}, {self.pattern_z:.1f})"
        cv2.putText(control_image, pattern_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        cv2.putText(control_image, "Controls:", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        cv2.putText(control_image, "Left/Right Arrow: Move X", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
        cv2.putText(control_image, "Up/Down Arrow: Move Y", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
        cv2.putText(control_image, "Page Up/Down: Move Z", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
        cv2.putText(control_image, "ESC: Exit", (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
        cv2.imshow(self.main_window, control_image)
        
        # Update each camera view
        for i, view in enumerate(self.views):
            window_name = f"View {i}: Camera Position {i+1}"
            camera_x = view['x']
            camera_y = view['y']
            camera_z = view['z']
            camera_yaw = view['yaw']
            camera_pitch = view['pitch']
            
            # Render the view with information overlay
            display_image = self.render_view_with_info(
                camera_x, camera_y, camera_z, camera_yaw, camera_pitch,
                f"Camera {i+1}"
            )
            
            # Display the view in its window
            cv2.imshow(window_name, display_image)
    
    def run(self):
        """Run the interactive environment."""
        # Create windows
        cv2.namedWindow(self.main_window)
        
        for i, view in enumerate(self.views):
            window_name = f"View {i}: Camera Position {i+1}"
            cv2.namedWindow(window_name)
            
            # Position the windows on screen
            x_pos = (i % 3) * 850  # Horizontally space windows
            y_pos = (i // 3) * 650  # Vertically space windows after 3 in a row
            cv2.moveWindow(window_name, x_pos, y_pos)
        
        cv2.moveWindow(self.main_window, 850, 0)  # Position control window
        
        print("Interactive 3D Pattern Control")
        print("Use arrow keys to move the pattern in X and Y")
        print("Use Page Up/Down to move the pattern in Z")
        print("Press ESC to exit")
        
        # Initial render
        self.update_windows()
        
        # Event loop
        while True:
            key = cv2.waitKey(30)
            
            # Check for exit
            if key == 27:  # ESC key
                break
                
            # Pattern movement
            if key == ord('a') or key == 2490368:  # Left arrow
                self.pattern_x -= self.pattern_move_speed
                self.update_windows()
            elif key == ord('d') or key == 2555904:  # Right arrow
                self.pattern_x += self.pattern_move_speed
                self.update_windows()
            elif key == ord('w') or key == 2621440:  # Up arrow
                self.pattern_y -= self.pattern_move_speed
                self.update_windows()
            elif key == ord('s') or key == 2424832:  # Down arrow
                self.pattern_y += self.pattern_move_speed
                self.update_windows()
            elif key == 2162688:  # Page Up
                self.pattern_z += self.pattern_move_speed
                self.update_windows()
            elif key == 2228224:  # Page Down
                self.pattern_z -= self.pattern_move_speed
                self.update_windows()
        
        # Clean up
        cv2.destroyAllWindows()

def main():
    # Define your camera views here
    # Each view is a dictionary with x, y, z coordinates and yaw, pitch angles
    views = [
        # Example views from different positions and angles
        {'x': -223.6, 'y': -180, 'z': -359.4, 'yaw': -25, 'pitch': 25},  # View 1
        {'x': 278, 'y': -140, 'z': -319.44, 'yaw': 30, 'pitch': 25},     # View 2
    ]
    
    print("Starting Interactive Pattern Control...")
    print(f"Setting up {len(views)} camera views")
    
    # Create environment and run
    env = InteractiveMultiViewEnvironment(views)
    env.run()

if __name__ == "__main__":
    main()