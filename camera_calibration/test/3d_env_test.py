import numpy as np
import cv2
import math
import os
from pynput import keyboard

class CalibrationEnvironment:
    def __init__(self):
        # Window settings
        self.window_name = "3D Calibration Environment"
        self.width, self.height = 800, 600
        
        # Camera settings
        self.focal_length = 300
        self.center_x = self.width / 2
        self.center_y = self.height / 2
        
        # Camera position and orientation
        self.camera_x = 0
        self.camera_y = 0
        self.camera_z = -200  # Start behind the pattern
        self.camera_yaw = 0   # Rotation around Y axis (left/right)
        self.camera_pitch = 0 # Rotation around X axis (up/down)
        
        # Movement speed settings
        self.move_speed = 10
        self.rotate_speed = 5
        
        # Capture settings
        self.capture_count = 0
        self.capture_dir = "captures"
        os.makedirs(self.capture_dir, exist_ok=True)
        
        # Create the 3D pattern
        self.create_pattern()
        
        # Key states for smooth movement
        self.key_states = {
            'w': False,  # Forward
            's': False,  # Backward
            'a': False,  # Left
            'd': False,  # Right
            'q': False,  # Up
            'e': False,  # Down
            'left': False,  # Turn left
            'right': False,  # Turn right
            'up': False,  # Look up
            'down': False  # Look down
        }
        
        # Key controller
        self.key_listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release)
        
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
    
    def render_view(self):
        """Render the current camera view of the environment."""
        # Create black background
        image = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Create transformation matrices
        sin_yaw = math.sin(math.radians(self.camera_yaw))
        cos_yaw = math.cos(math.radians(self.camera_yaw))
        sin_pitch = math.sin(math.radians(self.camera_pitch))
        cos_pitch = math.cos(math.radians(self.camera_pitch))
        
        # Project all 3D points to 2D
        for point in self.pattern_3d:
            # Translate point relative to camera
            x = point["X"] - self.camera_x
            y = point["Y"] - self.camera_y
            z = point["Z"] - self.camera_z
            
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
        
        # Add camera position and orientation info
        info_text = f"Pos: ({self.camera_x:.1f}, {self.camera_y:.1f}, {self.camera_z:.1f}) | Yaw: {self.camera_yaw:.1f} | Pitch: {self.camera_pitch:.1f}"
        cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        
        # Add help text
        help_text = "WASD: Move | QE: Up/Down | Arrows: Look | C: Capture | ESC: Exit"
        cv2.putText(image, help_text, (10, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        
        return image
    
    def update_camera(self):
        """Update camera position and orientation based on key states."""
        # Calculate forward vector
        forward_x = math.sin(math.radians(self.camera_yaw))
        forward_z = math.cos(math.radians(self.camera_yaw))
        
        # Calculate right vector
        right_x = math.sin(math.radians(self.camera_yaw + 90))
        right_z = math.cos(math.radians(self.camera_yaw + 90))
        
        # Forward/backward
        if self.key_states['w']:
            self.camera_x += forward_x * self.move_speed
            self.camera_z += forward_z * self.move_speed
        if self.key_states['s']:
            self.camera_x -= forward_x * self.move_speed
            self.camera_z -= forward_z * self.move_speed
        
        # Left/right
        if self.key_states['a']:
            self.camera_x -= right_x * self.move_speed
            self.camera_z -= right_z * self.move_speed
        if self.key_states['d']:
            self.camera_x += right_x * self.move_speed
            self.camera_z += right_z * self.move_speed
        
        # Up/down
        if self.key_states['q']:
            self.camera_y -= self.move_speed
        if self.key_states['e']:
            self.camera_y += self.move_speed
        
        # Rotation
        if self.key_states['left']:
            self.camera_yaw -= self.rotate_speed
        if self.key_states['right']:
            self.camera_yaw += self.rotate_speed
        if self.key_states['up']:
            self.camera_pitch -= self.rotate_speed
        if self.key_states['down']:
            self.camera_pitch += self.rotate_speed
        
        # Keep pitch in reasonable range
        self.camera_pitch = max(-89, min(89, self.camera_pitch))
    
    def on_key_press(self, key):
        """Handle key press events."""
        try:
            if key.char.lower() == 'w':
                self.key_states['w'] = True
            elif key.char.lower() == 's':
                self.key_states['s'] = True
            elif key.char.lower() == 'a':
                self.key_states['a'] = True
            elif key.char.lower() == 'd':
                self.key_states['d'] = True
            elif key.char.lower() == 'q':
                self.key_states['q'] = True
            elif key.char.lower() == 'e':
                self.key_states['e'] = True
            elif key.char.lower() == 'c':
                self.capture_image()
        except AttributeError:
            if key == keyboard.Key.left:
                self.key_states['left'] = True
            elif key == keyboard.Key.right:
                self.key_states['right'] = True
            elif key == keyboard.Key.up:
                self.key_states['up'] = True
            elif key == keyboard.Key.down:
                self.key_states['down'] = True
            elif key == keyboard.Key.esc:
                return False
    
    def on_key_release(self, key):
        """Handle key release events."""
        try:
            if key.char.lower() == 'w':
                self.key_states['w'] = False
            elif key.char.lower() == 's':
                self.key_states['s'] = False
            elif key.char.lower() == 'a':
                self.key_states['a'] = False
            elif key.char.lower() == 'd':
                self.key_states['d'] = False
            elif key.char.lower() == 'q':
                self.key_states['q'] = False
            elif key.char.lower() == 'e':
                self.key_states['e'] = False
        except AttributeError:
            if key == keyboard.Key.left:
                self.key_states['left'] = False
            elif key == keyboard.Key.right:
                self.key_states['right'] = False
            elif key == keyboard.Key.up:
                self.key_states['up'] = False
            elif key == keyboard.Key.down:
                self.key_states['down'] = False
    
    def render_clean_view(self):
        """Render the current camera view without any text overlays, just dots."""
        # Create black background
        image = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Create transformation matrices
        sin_yaw = math.sin(math.radians(self.camera_yaw))
        cos_yaw = math.cos(math.radians(self.camera_yaw))
        sin_pitch = math.sin(math.radians(self.camera_pitch))
        cos_pitch = math.cos(math.radians(self.camera_pitch))
        
        # Project all 3D points to 2D
        for point in self.pattern_3d:
            # Translate point relative to camera
            x = point["X"] - self.camera_x
            y = point["Y"] - self.camera_y
            z = point["Z"] - self.camera_z
            
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
        
    def capture_image(self):
        """Capture the current view and save it to disk."""
        # Get a clean view with only the dots, no text
        clean_image = self.render_clean_view()
        filename = os.path.join(self.capture_dir, f"capture_{self.capture_count:03d}.png")
        cv2.imwrite(filename, clean_image)
        
        # Save camera parameters
        params_file = os.path.join(self.capture_dir, f"capture_{self.capture_count:03d}_params.txt")
        with open(params_file, 'w') as f:
            f.write(f"Position: {self.camera_x}, {self.camera_y}, {self.camera_z}\n")
            f.write(f"Orientation: Yaw={self.camera_yaw}, Pitch={self.camera_pitch}\n")
            f.write(f"Focal Length: {self.focal_length}\n")
        
        print(f"Captured image saved as: {filename}")
        self.capture_count += 1
    
    def run(self):
        """Run the 3D environment viewer."""
        cv2.namedWindow(self.window_name)
        
        # Start the key listener
        self.key_listener.start()
        
        try:
            while True:
                # Update camera based on key states
                self.update_camera()
                
                # Render the current view
                image = self.render_view()
                
                # Display the rendered image
                cv2.imshow(self.window_name, image)
                
                # Check for exit
                key = cv2.waitKey(30)
                if key == 27:  # ESC key
                    break
        
        finally:
            # Clean up
            cv2.destroyAllWindows()
            self.key_listener.stop()

def main():
    print("Starting 3D Calibration Environment...")
    print("\nControls:")
    print("  W/A/S/D: Move forward/left/backward/right")
    print("  Q/E: Move up/down")
    print("  Arrow keys: Look around")
    print("  C: Capture current view")
    print("  ESC: Exit")
    
    # Create and run the environment
    env = CalibrationEnvironment()
    env.run()

if __name__ == "__main__":
    main()