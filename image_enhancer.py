import cv2
import numpy as np
from pathlib import Path
import re
import json

class ImageEnhancer:
    def __init__(self, input_dir, output_dir, calibration_path):
        """
        Initialize the image enhancer with input and output directories.
        
        Args:
            input_dir (str): Directory containing the input images
            output_dir (str): Directory where enhanced images will be saved
            calibration_path (str): Path to the calibration matrix JSON file
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load calibration data
        with open(calibration_path, 'r') as f:
            self.calibration_data = json.load(f)
        self.transform_matrix = np.array(self.calibration_data["transform_matrix"])
        self.target_shape = tuple(self.calibration_data["rgb_shape"])  # (height, width)
        
    def get_image_pairs(self):
        """Find matching pairs of RGB and IR images based on filenames."""
        # Get all files in input directory
        all_files = list(self.input_dir.glob('*'))
        
        # Create dictionaries for RGB and IR images
        rgb_images = {}
        ir_images = {}
        
        for file_path in all_files:
            # Extract base name without camera identifier
            base_name = re.sub(r'_camera_[01]', '', file_path.stem)
            
            if '_camera_0' in file_path.name:
                rgb_images[base_name] = file_path
            elif '_camera_1' in file_path.name:
                ir_images[base_name] = file_path
        
        # Find matching pairs
        pairs = []
        for base_name in rgb_images:
            if base_name in ir_images:
                pairs.append((rgb_images[base_name], ir_images[base_name]))
        
        return pairs

    def align_ir_image(self, ir_img):
        """
        Align IR image to RGB space using calibration matrix.
        
        Args:
            ir_img (ndarray): Input IR image
            
        Returns:
            ndarray: Aligned IR image
        """
        # Ensure IR image is grayscale
        if len(ir_img.shape) == 3:
            ir_img = cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY)
            
        # Apply perspective transform
        aligned_ir = cv2.warpPerspective(ir_img, self.transform_matrix,
                                       (self.target_shape[1], self.target_shape[0]))
        
        return aligned_ir

    def enhance_edges(self, rgb_img, ir_img, alpha=0.6):
        """
        Enhance edges using both RGB and IR images.
        
        Args:
            rgb_img (ndarray): RGB image
            ir_img (ndarray): Aligned IR image
            alpha (float): Blending factor for edge combination
        
        Returns:
            ndarray: Edge-enhanced RGB image
        """
        # Convert RGB to grayscale for edge detection
        gray_rgb = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        
        # Detect edges in both images
        edges_rgb = cv2.Canny(gray_rgb, 50, 150)
        edges_ir = cv2.Canny(ir_img, 30, 100)
        
        # Combine edges
        combined_edges = cv2.addWeighted(edges_rgb.astype(float), alpha, 
                                       edges_ir.astype(float), 1-alpha, 0)
        
        # Create edge mask
        edge_mask = combined_edges > 0
        
        # Enhance original RGB image along edges
        enhanced = rgb_img.copy()
        enhanced[edge_mask] = np.clip(enhanced[edge_mask] * 1.2, 0, 255)
        
        return enhanced

    def enhance_contrast(self, rgb_img, ir_img, clip_limit=3.0, tile_size=8):
        """
        Enhance contrast using IR guidance.
        
        Args:
            rgb_img (ndarray): RGB image
            ir_img (ndarray): Aligned IR image
            clip_limit (float): Contrast limit for CLAHE
            tile_size (int): Size of grid for CLAHE
        
        Returns:
            ndarray: Contrast-enhanced RGB image
        """
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, 
                               tileGridSize=(tile_size, tile_size))
        
        # Convert RGB to LAB color space
        lab = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        l_enhanced = clahe.apply(l)
        
        # Use IR image to guide enhancement
        ir_normalized = cv2.normalize(ir_img, None, 0, 255, cv2.NORM_MINMAX)
        ir_enhanced = clahe.apply(ir_normalized.astype(np.uint8))
        
        # Combine enhanced L channel with IR guidance
        l_final = cv2.addWeighted(l_enhanced, 0.7, ir_enhanced, 0.3, 0)
        
        # Merge channels and convert back to BGR
        enhanced_lab = cv2.merge([l_final, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced

    def process_images(self, save_intermediate=False):
        """
        Process all image pairs in the input directory.
        
        Args:
            save_intermediate (bool): Whether to save intermediate results
        """
        # Get image pairs
        pairs = self.get_image_pairs()
        
        for rgb_path, ir_path in pairs:
            print(f"Processing {rgb_path.name} and {ir_path.name}")
            
            # Read images
            rgb_img = cv2.imread(str(rgb_path))
            ir_img = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
            
            if rgb_img is None or ir_img is None:
                print(f"Error reading images for pair: {rgb_path.name}")
                continue
            
            # Align IR image with RGB
            aligned_ir = self.align_ir_image(ir_img)
            
            # Apply enhancements
            edge_enhanced = self.enhance_edges(rgb_img, aligned_ir)
            final_enhanced = self.enhance_contrast(edge_enhanced, aligned_ir)
            
            # Save results
            output_path = self.output_dir / f"{rgb_path.stem}_enhanced{rgb_path.suffix}"
            cv2.imwrite(str(output_path), final_enhanced)
            
            if save_intermediate:
                # Save aligned IR image
                ir_output = self.output_dir / f"{ir_path.stem}_aligned{ir_path.suffix}"
                cv2.imwrite(str(ir_output), aligned_ir)
                
                # Save edge-enhanced intermediate
                edge_output = self.output_dir / f"{rgb_path.stem}_edges{rgb_path.suffix}"
                cv2.imwrite(str(edge_output), edge_enhanced)

def main():
    # Example usage
    enhancer = ImageEnhancer(
        input_dir="path/to/input/directory",
        output_dir="path/to/output/directory",
        calibration_path="calibration_matrix.json"
    )
    
    # Process all images with intermediate results saved
    enhancer.process_images(save_intermediate=True)

if __name__ == "__main__":
    main()