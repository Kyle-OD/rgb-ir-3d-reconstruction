import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import linear_sum_assignment
import ot
from sklearn.neighbors import KDTree
import torch

def batch_chamfer_distance(pointcloud, gaussians, covariances, batch_size=1000):
    """
    Memory-efficient implementation of Chamfer distance
    
    Args:
        pointcloud: Nx3 array of point cloud coordinates
        gaussians: Mx3 array of Gaussian centers
        covariances: Mx3x3 array of covariance matrices
        batch_size: Number of points to process at once
    """
    # Build KD-Tree for faster nearest neighbor search
    kdtree_gaussian = KDTree(gaussians)
    kdtree_pc = KDTree(pointcloud)
    
    # Calculate distances in batches
    distances_pc_to_gaussian = []
    for i in range(0, len(pointcloud), batch_size):
        batch_points = pointcloud[i:i + batch_size]
        dist, _ = kdtree_gaussian.query(batch_points, k=1)
        distances_pc_to_gaussian.extend(dist)
    
    distances_gaussian_to_pc = []
    for i in range(0, len(gaussians), batch_size):
        batch_gaussians = gaussians[i:i + batch_size]
        dist, _ = kdtree_pc.query(batch_gaussians, k=1)
        distances_gaussian_to_pc.extend(dist)
    
    distances_pc_to_gaussian = np.array(distances_pc_to_gaussian)
    distances_gaussian_to_pc = np.array(distances_gaussian_to_pc)
    
    # Incorporate covariance information by weighting distances
    weights = np.array([np.linalg.det(cov) for cov in covariances])
    weights = weights / weights.sum()
    
    return np.mean(distances_pc_to_gaussian) + np.sum(distances_gaussian_to_pc * weights)

def compute_wasserstein_distance(pc_hist, prob_field, num_points=1000):
    """
    Compute Wasserstein distance with better convergence parameters
    """
    # Subsample if necessary
    if len(pc_hist.reshape(-1)) > num_points:
        idx1 = np.random.choice(len(pc_hist.reshape(-1)), num_points, replace=False)
        idx2 = np.random.choice(len(prob_field.reshape(-1)), num_points, replace=False)
        pc_hist_sub = pc_hist.reshape(-1)[idx1]
        prob_field_sub = prob_field.reshape(-1)[idx2]
        pc_hist_sub = pc_hist_sub / pc_hist_sub.sum()
        prob_field_sub = prob_field_sub / prob_field_sub.sum()
    else:
        pc_hist_sub = pc_hist.reshape(-1)
        prob_field_sub = prob_field.reshape(-1)
    
    # Compute distance matrix
    M = ot.dist(pc_hist_sub.reshape(-1, 1), prob_field_sub.reshape(-1, 1))
    M = M / M.max()  # Normalize distances
    
    # Use Sinkhorn algorithm with better parameters
    reg = 0.01  # Smaller regularization parameter
    distance = ot.sinkhorn2(
        pc_hist_sub,
        prob_field_sub,
        M,
        reg,
        numItermax=2000,  # Increase maximum iterations
        stopThr=1e-9,     # Stricter convergence threshold
        verbose=True
    )
    
    return distance

def gaussian_to_probability_field(gaussians, covariances, grid_resolution=64):
    """
    Memory-efficient conversion of 3D Gaussians to probability field
    """
    # Create grid points
    x = np.linspace(gaussians[:, 0].min(), gaussians[:, 0].max(), grid_resolution)
    y = np.linspace(gaussians[:, 1].min(), gaussians[:, 1].max(), grid_resolution)
    z = np.linspace(gaussians[:, 2].min(), gaussians[:, 2].max(), grid_resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    
    # Initialize probability field
    probability_field = np.zeros((grid_resolution, grid_resolution, grid_resolution))
    
    # Process Gaussians in batches
    batch_size = 100
    for i in range(0, len(gaussians), batch_size):
        batch_end = min(i + batch_size, len(gaussians))
        for j in range(i, batch_end):
            rv = multivariate_normal(gaussians[j], covariances[j])
            positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
            probability_field += rv.pdf(positions.T).reshape(grid_resolution, grid_resolution, grid_resolution)
    
    return probability_field / probability_field.sum()

def compare_pointcloud_to_gaussians(pointcloud, gaussians, covariances, method='optimal_transport'):
    """
    Compare a point cloud to a set of 3D Gaussians
    
    Args:
        pointcloud: Nx3 array of point cloud coordinates
        gaussians: Mx3 array of Gaussian centers
        covariances: Mx3x3 array of covariance matrices
        method: Comparison method ('optimal_transport', 'chamfer', 'emd')
    """
    if method == 'optimal_transport':
        print('Calculating Optimal Transport')
        # Convert Gaussians to probability field
        print('\tConverting gaussian to probability field')
        grid_resolution = 64  # Reduced resolution
        prob_field = gaussian_to_probability_field(gaussians, covariances, grid_resolution)
        
        # Normalize point cloud to match probability field dimensions
        print('\tNormalizing point cloud')
        pc_normalized = (pointcloud - pointcloud.min(axis=0)) / (pointcloud.max(axis=0) - pointcloud.min(axis=0))
        pc_hist, _ = np.histogramdd(pc_normalized, bins=prob_field.shape[0])
        pc_hist = pc_hist / pc_hist.sum()
        
        # Calculate Wasserstein distance
        print('\tCalculating Wasserstein distance')
        distance = compute_wasserstein_distance(pc_hist, prob_field)
        
    elif method == 'chamfer':
        print('Calculating chamfer distance')
        # Use memory-efficient Chamfer distance implementation
        distance = batch_chamfer_distance(pointcloud, gaussians, covariances)
        
    elif method == 'emd':
        print('Earth Mover\'s Distance')
        # Earth Mover's Distance with covariance-weighted costs
        max_points = 5000
        if len(pointcloud) > max_points or len(gaussians) > max_points:
            idx_pc = np.random.choice(len(pointcloud), max_points, replace=False)
            idx_gaussian = np.random.choice(len(gaussians), max_points, replace=False)
            pointcloud_sub = pointcloud[idx_pc]
            gaussians_sub = gaussians[idx_gaussian]
            covariances_sub = covariances[idx_gaussian]
        else:
            pointcloud_sub = pointcloud
            gaussians_sub = gaussians
            covariances_sub = covariances
        print('\tCovariance weighted costs')
        cost_matrix = np.zeros((len(pointcloud_sub), len(gaussians_sub)))
        for i, p in enumerate(pointcloud_sub):
            for j, (g, cov) in enumerate(zip(gaussians_sub, covariances_sub)):
                diff = p - g
                cost_matrix[i, j] = np.sqrt(diff.T @ np.linalg.inv(cov) @ diff)
        
        # Solve optimal transport problem
        print('\tSolving transport problem')
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        distance = cost_matrix[row_ind, col_ind].sum()
    
    return distance

def visualize_comparison(pointcloud, gaussians, covariances):
    """
    Visualize point cloud and Gaussian splats for comparison
    Implement visualization using your preferred 3D plotting library
    """
    pass  # Implementation depends on visualization requirements