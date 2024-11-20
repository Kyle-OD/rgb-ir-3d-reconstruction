import numpy as np
from plyfile import PlyData
import numpy as np
from scipy.spatial.transform import Rotation
from pointcloud_gaussian_comparison import compare_pointcloud_to_gaussians

def load_gaussian_splat_ply(filepath):
    """
    Load a Gaussian Splat .ply file and extract relevant parameters
    
    Args:
        filepath: Path to the .ply file
        
    Returns:
        positions: Nx3 array of Gaussian centers
        covariances: Nx3x3 array of covariance matrices
        scales: Nx3 array of scaling factors
        rotations: Nx4 array of rotation quaternions
        opacities: Nx1 array of opacity values
    """
    plydata = PlyData.read(filepath)
    vertex_data = plydata['vertex']
    
    # Extract positions
    positions = np.column_stack([
        vertex_data['x'],
        vertex_data['y'],
        vertex_data['z']
    ])
    
    # Extract scaling factors
    scales = np.column_stack([
        vertex_data['scale_0'],
        vertex_data['scale_1'],
        vertex_data['scale_2']
    ])
    
    # Extract rotations (stored as quaternions)
    rotations = np.column_stack([
        vertex_data['rot_0'],
        vertex_data['rot_1'],
        vertex_data['rot_2'],
        vertex_data['rot_3']
    ])
    
    # Extract opacities
    opacities = vertex_data['opacity']
    
    # Convert rotation and scale to covariance matrices
    covariances = np.zeros((len(positions), 3, 3))
    for i in range(len(positions)):
        # Convert quaternion to rotation matrix
        rot_matrix = Rotation.from_quat(rotations[i]).as_matrix()
        
        # Create scaling matrix
        scale_matrix = np.diag(scales[i])
        
        # Compute covariance matrix: R * S * S * R^T
        covariances[i] = rot_matrix @ (scale_matrix @ scale_matrix) @ rot_matrix.T
    
    return positions, covariances, scales, rotations, opacities

def load_lidar_pointcloud_ply(filepath):
    """
    Load a LiDAR point cloud .ply file
    
    Args:
        filepath: Path to the .ply file
        
    Returns:
        points: Nx3 array of point coordinates
        intensities: Nx1 array of intensity values (if available)
    """
    plydata = PlyData.read(filepath)
    vertex_data = plydata['vertex']
    
    # Extract point coordinates
    points = np.column_stack([
        vertex_data['x'],
        vertex_data['y'],
        vertex_data['z']
    ])
    
    # Try to extract intensities if available
    try:
        intensities = vertex_data['intensity']
    except ValueError:
        intensities = None
        
    return points, intensities

def compare_gaussian_splat_to_lidar(gaussian_ply_path, lidar_ply_path, method='optimal_transport'):
    """
    Load and compare a Gaussian splat model to a LiDAR point cloud
    
    Args:
        gaussian_ply_path: Path to Gaussian splat .ply file
        lidar_ply_path: Path to LiDAR point cloud .ply file
        method: Comparison method ('optimal_transport', 'chamfer', 'emd')
        
    Returns:
        distance: Computed distance metric between the two point sets
    """
    # Load Gaussian splat data
    positions, covariances, scales, rotations, opacities = load_gaussian_splat_ply(gaussian_ply_path)
    
    # Load LiDAR point cloud
    points, intensities = load_lidar_pointcloud_ply(lidar_ply_path)
    
    # Filter out low opacity Gaussians if desired
    opacity_threshold = 0.5
    valid_indices = opacities > opacity_threshold
    positions = positions[valid_indices]
    covariances = covariances[valid_indices]
    
    # Use the comparison function from the previous implementation
    distance = compare_pointcloud_to_gaussians(points, positions, covariances, method=method)
    
    return distance

# Example usage
def main():
    # Example paths - replace with your actual file paths
    gaussian_path = "gaussian_model.ply"
    lidar_path = "lidar_scan.ply"
    
    try:
        # Load and visualize the data
        positions, covariances, scales, rotations, opacities = load_gaussian_splat_ply(gaussian_path)
        points, intensities = load_lidar_pointcloud_ply(lidar_path)
        
        print(f"Loaded {len(positions)} Gaussians and {len(points)} LiDAR points")
        
        # Compare using different metrics
        metrics = ['optimal_transport', 'chamfer', 'emd']
        for metric in metrics:
            distance = compare_gaussian_splat_to_lidar(gaussian_path, lidar_path, method=metric)
            print(f"{metric} distance: {distance}")
            
    except Exception as e:
        print(f"Error processing PLY files: {str(e)}")

'''
if __name__ == "__main__":
    main()
'''