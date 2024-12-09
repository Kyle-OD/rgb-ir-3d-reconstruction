import cv2
import numpy as np
from pathlib import Path
import json

class CalibrationTool:
    def __init__(self):
        self.rgb_points = []
        self.ir_points = []
        self.current_image = None
        self.current_mode = None
        self.window_name = "Calibration Tool"
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for point selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_mode == "rgb":
                self.rgb_points.append((x, y))
                # Draw point and number
                cv2.circle(self.current_image, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(self.current_image, str(len(self.rgb_points)), 
                          (x+5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            elif self.current_mode == "ir":
                self.ir_points.append((x, y))
                cv2.circle(self.current_image, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(self.current_image, str(len(self.ir_points)), 
                          (x+5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Update display
            cv2.imshow(self.window_name, self.current_image)
            
            # Print current point count
            print(f"Selected point {len(self.rgb_points if self.current_mode == 'rgb' else self.ir_points)} at ({x}, {y})")
    
    def select_points(self, image, mode):
        """
        Allow user to select points on an image.
        
        Args:
            image: Input image
            mode: Either "rgb" or "ir"
        """
        self.current_image = image.copy()
        self.current_mode = mode
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print(f"\nSelecting points for {mode.upper()} image.")
        print("Click to select points. Press:")
        print("  'r' to reset all points")
        print("  'c' to clear last point")
        print("  'q' to finish selection")
        print("\nRecommended: Select 8-12 well-distributed points")
        
        while True:
            cv2.imshow(self.window_name, self.current_image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset points
                if mode == "rgb":
                    self.rgb_points = []
                else:
                    self.ir_points = []
                self.current_image = image.copy()
            elif key == ord('c'):
                # Clear last point
                if mode == "rgb" and self.rgb_points:
                    self.rgb_points.pop()
                elif mode == "ir" and self.ir_points:
                    self.ir_points.pop()
                # Redraw image
                self.current_image = image.copy()
                points = self.rgb_points if mode == "rgb" else self.ir_points
                for idx, pt in enumerate(points, 1):
                    cv2.circle(self.current_image, pt, 5, (0, 255, 0), -1)
                    cv2.putText(self.current_image, str(idx), 
                              (pt[0]+5, pt[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (0, 255, 0), 1)
        
        cv2.destroyWindow(self.window_name)

def calibrate_cameras(rgb_path, ir_path, output_path):
    """
    Perform manual calibration between RGB and IR cameras.
    
    Args:
        rgb_path: Path to RGB image
        ir_path: Path to IR image
        output_path: Path to save calibration matrix
    """
    # Read images
    rgb_img = cv2.imread(str(rgb_path))
    ir_img = cv2.imread(str(ir_path))
    
    if rgb_img is None or ir_img is None:
        raise ValueError("Could not read input images")
    
    # Convert IR image to 3 channels if it's grayscale
    if len(ir_img.shape) == 2:
        ir_img = cv2.cvtColor(ir_img, cv2.COLOR_GRAY2BGR)
    
    # Create calibration tool instance
    tool = CalibrationTool()
    
    # Select points on both images
    print("\nSelect corresponding points on RGB image:")
    tool.select_points(rgb_img, "rgb")
    
    print("\nSelect corresponding points on IR image:")
    tool.select_points(ir_img, "ir")
    
    # Check if we have enough corresponding points
    if len(tool.rgb_points) < 4 or len(tool.ir_points) < 4:
        raise ValueError("Need at least 4 corresponding points for homography")
    
    if len(tool.rgb_points) != len(tool.ir_points):
        raise ValueError("Number of points must match between RGB and IR images")
    
    # Convert points to numpy arrays
    rgb_points = np.float32(tool.rgb_points).reshape(-1, 1, 2)
    ir_points = np.float32(tool.ir_points).reshape(-1, 1, 2)
    
    # Calculate homography matrix using RANSAC
    transform_matrix, mask = cv2.findHomography(ir_points, rgb_points, cv2.RANSAC, 5.0)
    
    # Print information about inliers
    inliers = np.sum(mask)
    print(f"\nUsed {inliers} inlier points out of {len(tool.rgb_points)} total points")
    
    # Save calibration matrix
    calibration_data = {
        "transform_matrix": transform_matrix.tolist(),
        "rgb_shape": rgb_img.shape[:2],  # (height, width)
        "number_of_points": len(tool.rgb_points),
        "number_of_inliers": int(inliers)
    }
    
    with open(output_path, 'w') as f:
        json.dump(calibration_data, f, indent=2)
    
    print(f"\nCalibration matrix saved to {output_path}")
    
    # Test the transformation
    transformed_ir = cv2.warpPerspective(ir_img, transform_matrix, 
                                       (rgb_img.shape[1], rgb_img.shape[0]))
    
    # Create side-by-side comparison
    comparison = np.hstack([rgb_img, transformed_ir])
    cv2.imshow("Comparison (RGB | Transformed IR)", comparison)
    print("\nShowing comparison view. Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return transform_matrix

def apply_calibration(ir_img, calibration_path):
    """
    Apply saved calibration to an IR image.
    
    Args:
        ir_img: Input IR image
        calibration_path: Path to calibration file
        
    Returns:
        Transformed IR image
    """
    # Load calibration data
    with open(calibration_path, 'r') as f:
        calibration_data = json.load(f)
    
    transform_matrix = np.array(calibration_data["transform_matrix"])
    output_shape = calibration_data["rgb_shape"]
    
    # Apply transform
    transformed_ir = cv2.warpPerspective(ir_img, transform_matrix,
                                       (output_shape[1], output_shape[0]))
    
    return transformed_ir

def main():
    # Example usage
    rgb_path = "path/to/rgb_image.jpg"
    ir_path = "path/to/ir_image.jpg"
    calibration_path = "calibration_matrix.json"
    
    # Perform calibration
    transform_matrix = calibrate_cameras(rgb_path, ir_path, calibration_path)

if __name__ == "__main__":
    main()