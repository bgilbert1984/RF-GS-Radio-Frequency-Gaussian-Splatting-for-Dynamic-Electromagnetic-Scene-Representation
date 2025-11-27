import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import math


class GaussianSplatModel(nn.Module):
    """
    Neural Gaussian Splats model for efficient rendering of RF visualization
    Represents the scene as a collection of 3D Gaussians with learned parameters
    """
    
    def __init__(
        self,
        num_gaussians: int = 10000,        # Initial number of Gaussians
        feature_dim: int = 32,             # Feature vector dimension
        color_dim: int = 3,                # RGB color dimension
        min_opacity: float = 0.005,        # Minimum opacity for pruning
        learning_rate: float = 0.005,      # Learning rate for optimization
        adaptive_density: bool = True,     # Whether to adaptively adjust Gaussian density
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        super(GaussianSplatModel, self).__init__()
        
        self.num_gaussians = num_gaussians
        self.feature_dim = feature_dim
        self.color_dim = color_dim
        self.min_opacity = min_opacity
        self.learning_rate = learning_rate
        self.adaptive_density = adaptive_density
        self.device = device
        
        # Initialize Gaussian parameters
        # We use separate tensors for better optimization
        
        # Positions (x, y, z)
        self.positions = nn.Parameter(torch.randn(num_gaussians, 3, device=device) * 0.1)
        
        # Scales (log-scale for optimization stability)
        self.scales = nn.Parameter(torch.zeros(num_gaussians, 3, device=device) - 2.0)
        
        # Rotations (quaternions: w, x, y, z)
        self.rotations = nn.Parameter(torch.zeros(num_gaussians, 4, device=device))
        # Initialize as identity quaternions
        self.rotations.data[:, 0] = 1.0
        
        # Opacity (logit for optimization stability)
        self.opacity = nn.Parameter(torch.zeros(num_gaussians, 1, device=device) - 2.0)
        
        # Feature vectors
        self.features = nn.Parameter(torch.randn(num_gaussians, feature_dim, device=device) * 0.01)
        
        # Neural shader (features -> RGB)
        self.shader = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, color_dim),
            nn.Sigmoid()  # Normalize to [0, 1]
        ).to(device)
        
        # Active mask (for pruning)
        self.active_mask = torch.ones(num_gaussians, dtype=torch.bool, device=device)
        self.num_active = num_gaussians
        
        # Optimizer
        self.optimizer = torch.optim.Adam([
            {'params': self.positions, 'lr': learning_rate},
            {'params': self.scales, 'lr': learning_rate},
            {'params': self.rotations, 'lr': learning_rate * 0.1},  # Lower LR for rotations
            {'params': self.opacity, 'lr': learning_rate},
            {'params': self.features, 'lr': learning_rate},
            {'params': self.shader.parameters(), 'lr': learning_rate * 0.1}
        ])
    
    def get_covariance_matrices(self) -> torch.Tensor:
        """
        Convert scale and rotation parameters to 3D covariance matrices
        Returns:
            Covariance matrices of shape (num_active, 3, 3)
        """
        # Get active parameters
        scales = torch.exp(self.scales[self.active_mask])  # Convert from log-space
        rots = self.rotations[self.active_mask]
        
        # Normalize quaternions
        rots = F.normalize(rots, p=2, dim=1)
        
        # Convert quaternions to rotation matrices
        # q = (w, x, y, z)
        w, x, y, z = rots[:, 0], rots[:, 1], rots[:, 2], rots[:, 3]
        
        # Build rotation matrices
        rot_matrices = torch.zeros(self.num_active, 3, 3, device=self.device)
        
        # First row
        rot_matrices[:, 0, 0] = 1 - 2 * (y**2 + z**2)
        rot_matrices[:, 0, 1] = 2 * (x * y - w * z)
        rot_matrices[:, 0, 2] = 2 * (x * z + w * y)
        
        # Second row
        rot_matrices[:, 1, 0] = 2 * (x * y + w * z)
        rot_matrices[:, 1, 1] = 1 - 2 * (x**2 + z**2)
        rot_matrices[:, 1, 2] = 2 * (y * z - w * x)
        
        # Third row
        rot_matrices[:, 2, 0] = 2 * (x * z - w * y)
        rot_matrices[:, 2, 1] = 2 * (y * z + w * x)
        rot_matrices[:, 2, 2] = 1 - 2 * (x**2 + y**2)
        
        # Create scale matrices (diagonal)
        scale_matrices = torch.zeros(self.num_active, 3, 3, device=self.device)
        scale_matrices[:, 0, 0] = scales[:, 0]
        scale_matrices[:, 1, 1] = scales[:, 1]
        scale_matrices[:, 2, 2] = scales[:, 2]
        
        # Compute covariance: R * S * S * R^T
        # First, R * S
        RS = torch.bmm(rot_matrices, scale_matrices)
        # Then, (R * S) * (R * S)^T = R * S * S * R^T
        cov_matrices = torch.bmm(RS, torch.transpose(RS, 1, 2))
        
        return cov_matrices
    
    def get_colors(self) -> torch.Tensor:
        """Get RGB colors for active Gaussians"""
        # Pass features through the shader
        features = self.features[self.active_mask]
        colors = self.shader(features)
        return colors
    
    def get_active_parameters(self) -> Dict[str, torch.Tensor]:
        """Get all parameters for active Gaussians"""
        positions = self.positions[self.active_mask]
        scales = torch.exp(self.scales[self.active_mask])  # Convert from log-space
        rotations = F.normalize(self.rotations[self.active_mask], p=2, dim=1)
        opacity = torch.sigmoid(self.opacity[self.active_mask])  # Convert from logit-space
        colors = self.get_colors()
        covariance = self.get_covariance_matrices()
        
        return {
            'positions': positions,
            'scales': scales,
            'rotations': rotations,
            'opacity': opacity,
            'colors': colors,
            'covariance': covariance
        }
    
    def prune(self) -> int:
        """
        Prune low-opacity Gaussians
        Returns:
            Number of Gaussians removed
        """
        opacity = torch.sigmoid(self.opacity)  # Convert from logit-space
        
        # Find Gaussians with opacity below threshold
        prune_mask = opacity.squeeze() < self.min_opacity
        
        # Update active mask
        prev_active = self.active_mask.clone()
        self.active_mask[prev_active] = ~prune_mask[prev_active]
        
        # Update active count
        self.num_active = self.active_mask.sum().item()
        
        return prune_mask.sum().item()
    
    def densify(
        self, 
        positions: torch.Tensor,      # Target positions (N, 3)
        features: torch.Tensor,       # RF features at positions (N, feature_dim)
        max_gaussians: int = 50000    # Maximum number of Gaussians
    ) -> int:
        """
        Add new Gaussians based on RF data
        Returns:
            Number of Gaussians added
        """
        if self.num_active >= max_gaussians or positions.shape[0] == 0:
            return 0
        
        # Determine how many new Gaussians to add
        num_to_add = min(positions.shape[0], max_gaussians - self.num_active)
        
        # Find positions furthest from existing Gaussians
        active_positions = self.positions[self.active_mask]
        
        if active_positions.shape[0] == 0:
            # No active positions, use the first num_to_add from provided positions
            idx_to_add = torch.arange(num_to_add, device=self.device)
        else:
            # Compute distances to closest existing Gaussian
            distances = torch.cdist(positions, active_positions)
            min_distances, _ = torch.min(distances, dim=1)
            
            # Select points with largest minimum distance
            _, idx_to_add = torch.topk(min_distances, num_to_add)
        
        # Find inactive Gaussians to reuse
        inactive_indices = (~self.active_mask).nonzero().squeeze()
        
        if inactive_indices.shape[0] < num_to_add:
            # Not enough inactive Gaussians, need to expand
            return 0
        
        # Select indices to reuse
        reuse_indices = inactive_indices[:num_to_add]
        
        # Update Gaussian parameters
        self.positions.data[reuse_indices] = positions[idx_to_add]
        
        # Initialize scales as small
        self.scales.data[reuse_indices] = torch.log(torch.ones_like(self.scales[reuse_indices]) * 0.01)
        
        # Initialize rotations as identity
        self.rotations.data[reuse_indices, 0] = 1.0
        self.rotations.data[reuse_indices, 1:] = 0.0
        
        # Initialize opacity as visible but low
        self.opacity.data[reuse_indices] = torch.logit(torch.ones_like(self.opacity[reuse_indices]) * 0.1)
        
        # Initialize features based on RF features
        if features is not None and features.shape[1] == self.feature_dim:
            self.features.data[reuse_indices] = features[idx_to_add]
        else:
            self.features.data[reuse_indices] = torch.randn_like(self.features[reuse_indices]) * 0.01
        
        # Update active mask
        self.active_mask[reuse_indices] = True
        self.num_active += num_to_add
        
        return num_to_add
    
    def render_image(
        self,
        camera_position: torch.Tensor,    # Camera position (3,)
        camera_matrix: torch.Tensor,      # Camera-to-world matrix (4, 4)
        width: int,                       # Image width
        height: int,                      # Image height
        focal_length: float,              # Focal length in pixels
        near_plane: float = 0.1,          # Near clipping plane
        far_plane: float = 100.0,         # Far clipping plane
        sort_points: bool = True,         # Whether to sort points by depth
        num_depth_bits: int = 16,         # Depth buffer precision
        depth_premultiplier: float = 10.0  # Depth precision multiplier
    ) -> Dict[str, torch.Tensor]:
        """
        Render an image from the Gaussian splat model
        """
        if self.num_active == 0:
            # Return empty image if no active Gaussians
            return {
                'rgb': torch.zeros(height, width, 3, device=self.device),
                'depth': torch.ones(height, width, device=self.device) * far_plane,
                'opacity': torch.zeros(height, width, device=self.device)
            }
        
        # Get active parameters
        params = self.get_active_parameters()
        positions = params['positions']
        covariance = params['covariance']
        colors = params['colors']
        opacity = params['opacity']
        
        # Transform positions to camera space
        # Extract rotation and translation from camera matrix
        rotation = camera_matrix[:3, :3]
        translation = camera_matrix[:3, 3]
        
        # Transform positions
        positions_cam = torch.matmul(positions - translation, rotation.T)
        
        # Filter out Gaussians behind the camera
        in_front = positions_cam[:, 2] > near_plane
        positions_cam = positions_cam[in_front]
        colors = colors[in_front]
        opacity = opacity[in_front]
        covariance = covariance[in_front]
        
        if positions_cam.shape[0] == 0:
            # Return empty image if all Gaussians are behind camera
            return {
                'rgb': torch.zeros(height, width, 3, device=self.device),
                'depth': torch.ones(height, width, device=self.device) * far_plane,
                'opacity': torch.zeros(height, width, device=self.device)
            }
        
        # Transform covariance to camera space
        covariance_cam = torch.matmul(torch.matmul(rotation, covariance), rotation.T)
        
        # Project to 2D
        # Perspective division
        positions_2d = positions_cam[:, :2] / positions_cam[:, 2:3]
        
        # Scale by focal length and shift to image center
        positions_2d = positions_2d * focal_length + torch.tensor([width / 2, height / 2], device=self.device)
        
        # Calculate 2D covariance matrices for screen-space Gaussians
        # For perspective projection, we need to account for depth
        depth = positions_cam[:, 2]
        
        # Extract components for 2D covariance
        cov_00 = covariance_cam[:, 0, 0]
        cov_01 = covariance_cam[:, 0, 1]
        cov_02 = covariance_cam[:, 0, 2]
        cov_11 = covariance_cam[:, 1, 1]
        cov_12 = covariance_cam[:, 1, 2]
        cov_22 = covariance_cam[:, 2, 2]
        
        # Compute 2D covariance (perspective projection of 3D covariance)
        inv_depth = 1.0 / depth
        inv_depth2 = inv_depth * inv_depth
        focal2 = focal_length * focal_length
        
        cov2d_00 = focal2 * (inv_depth2 * cov_00 - 2 * inv_depth2 * positions_cam[:, 0] * cov_02) + focal2 * inv_depth2 * positions_cam[:, 0]**2 * cov_22
        cov2d_01 = focal2 * (inv_depth2 * cov_01 - inv_depth2 * positions_cam[:, 0] * cov_12 - inv_depth2 * positions_cam[:, 1] * cov_02) + focal2 * inv_depth2 * positions_cam[:, 0] * positions_cam[:, 1] * cov_22
        cov2d_11 = focal2 * (inv_depth2 * cov_11 - 2 * inv_depth2 * positions_cam[:, 1] * cov_12) + focal2 * inv_depth2 * positions_cam[:, 1]**2 * cov_22
        
        # Construct 2D covariance matrices
        covariance_2d = torch.zeros(positions_2d.shape[0], 2, 2, device=self.device)
        covariance_2d[:, 0, 0] = cov2d_00
        covariance_2d[:, 0, 1] = cov2d_01
        covariance_2d[:, 1, 0] = cov2d_01
        covariance_2d[:, 1, 1] = cov2d_11
        
        # Ensure minimum size for numerical stability
        min_size = 0.3
        det = cov2d_00 * cov2d_11 - cov2d_01 * cov2d_01
        det = torch.clamp(det, min=min_size**4)
        
        # Sort by depth if requested
        if sort_points:
            sort_idx = torch.argsort(depth, descending=True)
            positions_2d = positions_2d[sort_idx]
            colors = colors[sort_idx]
            opacity = opacity[sort_idx]
            covariance_2d = covariance_2d[sort_idx]
            depth = depth[sort_idx]
        
        # Create point-based renderer
        renderer = GaussianPointRenderer(width, height, self.device)
        
        # Render
        render_output = renderer.render(
            positions_2d, 
            covariance_2d, 
            colors, 
            opacity, 
            depth,
            depth_premultiplier=depth_premultiplier
        )
        
        return render_output
    
    def gradient_step(
        self, 
        loss: torch.Tensor,
        max_grad_norm: float = 1.0
    ) -> float:
        """
        Perform an optimization step
        Returns:
            Loss value
        """
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
        
        # Update parameters
        self.optimizer.step()
        
        return loss.item()
    
    def fit_to_rf_data(
        self,
        positions: torch.Tensor,         # Positions (N, 3)
        rf_features: torch.Tensor,       # RF features (N, F)
        colors: Optional[torch.Tensor] = None,  # Optional colors (N, 3)
        num_iterations: int = 100,       # Number of iterations
        learning_rate: Optional[float] = None,  # Override learning rate
        regularization: float = 0.001,   # L2 regularization strength
        prune_interval: int = 10,        # Interval for pruning
        densify_interval: int = 20,      # Interval for densification
        verbose: bool = False            # Whether to print progress
    ) -> List[float]:
        """
        Fit the model to RF data
        Returns:
            List of loss values per iteration
        """
        # Update learning rate if specified
        if learning_rate is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate
        
        # Normalize positions to [-1, 1] range
        pos_min = positions.min(dim=0)[0]
        pos_max = positions.max(dim=0)[0]
        pos_range = pos_max - pos_min
        normalized_positions = (positions - pos_min) / pos_range * 2 - 1
        
        # Initialize with some points
        if self.num_active == 0:
            self.densify(normalized_positions, rf_features)
        
        # Track losses
        losses = []
        
        # Training loop
        for iteration in range(num_iterations):
            # Compute point-to-gaussian distances
            active_positions = self.positions[self.active_mask]
            distances = torch.cdist(normalized_positions, active_positions)
            
            # Find closest Gaussian for each point
            min_distances, closest_indices = torch.min(distances, dim=1)
            
            # Convert closest indices to original indices
            original_indices = torch.arange(self.num_gaussians, device=self.device)[self.active_mask][closest_indices]
            
            # Get corresponding Gaussian parameters
            gaussian_positions = self.positions[original_indices]
            gaussian_scales = torch.exp(self.scales[original_indices])
            gaussian_features = self.features[original_indices]
            
            # Compute position loss (weighted by RF feature magnitude)
            rf_magnitudes = torch.norm(rf_features, dim=1, keepdim=True)
            position_loss = torch.sum(rf_magnitudes * torch.sum((gaussian_positions - normalized_positions)**2, dim=1))
            
            # Compute feature loss
            feature_loss = F.mse_loss(gaussian_features, rf_features)
            
            # Compute scale regularization (prefer smaller Gaussians)
            scale_reg = torch.mean(torch.sum(gaussian_scales**2, dim=1))
            
            # Compute color loss if colors are provided
            if colors is not None:
                gaussian_colors = self.shader(gaussian_features)
                color_loss = F.mse_loss(gaussian_colors, colors)
            else:
                color_loss = 0.0
            
            # Total loss
            loss = position_loss + feature_loss + regularization * scale_reg
            if colors is not None:
                loss = loss + color_loss
            
            # Optimization step
            loss_value = self.gradient_step(loss)
            losses.append(loss_value)
            
            # Prune low-opacity Gaussians
            if (iteration + 1) % prune_interval == 0:
                num_pruned = self.prune()
                if verbose and num_pruned > 0:
                    print(f"Iteration {iteration+1}: Pruned {num_pruned} Gaussians")
            
            # Densify with new Gaussians where needed
            if (iteration + 1) % densify_interval == 0 and self.adaptive_density:
                # Find points that are poorly represented
                poorly_fit_mask = min_distances > torch.median(min_distances) * 2
                if poorly_fit_mask.sum() > 0:
                    num_added = self.densify(
                        normalized_positions[poorly_fit_mask], 
                        rf_features[poorly_fit_mask]
                    )
                    if verbose and num_added > 0:
                        print(f"Iteration {iteration+1}: Added {num_added} Gaussians")
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration+1}: Loss = {loss_value:.6f}, Active Gaussians = {self.num_active}")
        
        return losses
    
    def update_from_rf_nerf(
        self,
        rf_nerf_model,          # RF-NeRF model
        grid_resolution: int = 32,  # Resolution of sampling grid
        threshold: float = 0.01,    # Density threshold for creating Gaussians
        max_points: int = 10000     # Maximum number of points to sample
    ) -> int:
        """
        Update the Gaussian Splat model based on an RF-NeRF model
        Returns:
            Number of Gaussians updated
        """
        # Create a grid of points
        x = torch.linspace(-1, 1, grid_resolution, device=self.device)
        y = torch.linspace(-1, 1, grid_resolution, device=self.device)
        z = torch.linspace(-1, 1, grid_resolution, device=self.device)
        
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
        grid_points = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1)
        
        # Sample RF-NeRF at grid points
        with torch.no_grad():
            # This part depends on the specific RF-NeRF implementation
            # For example, assuming it has a sample_points method:
            try:
                densities, colors = rf_nerf_model.sample_points(grid_points)
            except:
                # Fallback: create random sampling
                densities = torch.rand(grid_points.shape[0], 1, device=self.device)
                colors = torch.rand(grid_points.shape[0], 3, device=self.device)
        
        # Filter points by density
        valid_points = densities.squeeze() > threshold
        filtered_points = grid_points[valid_points]
        filtered_colors = colors[valid_points]
        
        # Limit number of points if needed
        if filtered_points.shape[0] > max_points:
            # Randomly sample points
            idx = torch.randperm(filtered_points.shape[0], device=self.device)[:max_points]
            filtered_points = filtered_points[idx]
            filtered_colors = filtered_colors[idx]
        
        # Create RF features (simplified - in practice would come from RF-NeRF)
        rf_features = torch.randn(filtered_points.shape[0], self.feature_dim, device=self.device)
        
        # Update Gaussian model
        num_iterations = 50  # Fewer iterations for incremental update
        self.fit_to_rf_data(
            filtered_points,
            rf_features,
            filtered_colors,
            num_iterations=num_iterations,
            verbose=False
        )
        
        return self.num_active


class GaussianPointRenderer:
    """
    Renderer for Gaussian splats using point-based rendering
    """
    
    def __init__(
        self,
        width: int,
        height: int,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        background_color: torch.Tensor = None  # Default background color (black)
    ):
        self.width = width
        self.height = height
        self.device = device
        
        if background_color is None:
            self.background_color = torch.zeros(3, device=device)
        else:
            self.background_color = background_color
            
        # Create pixel grid for vectorized rendering
        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing='ij'
        )
        self.pixel_coords = torch.stack([x_coords.flatten(), y_coords.flatten()], dim=1).float()
    
    def render(
        self,
        positions: torch.Tensor,       # 2D positions (N, 2)
        covariances: torch.Tensor,     # 2D covariance matrices (N, 2, 2)
        colors: torch.Tensor,          # Colors (N, 3)
        opacities: torch.Tensor,       # Opacities (N, 1)
        depths: torch.Tensor,          # Depths (N,)
        depth_premultiplier: float = 10.0  # Depth precision multiplier
    ) -> Dict[str, torch.Tensor]:
        """
        Render Gaussian splats to an image
        """
        num_points = positions.shape[0]
        
        # Initialize accumulators
        accum_colors = torch.zeros(self.height, self.width, 3, device=self.device)
        accum_alpha = torch.zeros(self.height, self.width, device=self.device)
        accum_depth = torch.zeros(self.height, self.width, device=self.device)
        
        # Maximum splat radius in pixels (clamp for efficiency)
        max_radius = 15  # Pixel radius
        
        # Process each Gaussian
        for i in range(num_points):
            # Extract parameters for this Gaussian
            pos = positions[i]
            cov = covariances[i]
            color = colors[i]
            opacity = opacities[i].item()
            depth = depths[i].item()
            
            # Skip nearly transparent Gaussians
            if opacity < 1e-3:
                continue
                
            # Calculate bounding box for this Gaussian
            # Get eigenvalues of covariance matrix
            try:
                eigenvalues, _ = torch.linalg.eigh(cov)
                # Radius is determined by the square root of the largest eigenvalue
                radius = torch.sqrt(torch.max(eigenvalues)) * 3  # 3 standard deviations
                radius = min(radius.item(), max_radius)  # Clamp radius for efficiency
            except:
                # Fallback if eigendecomposition fails
                radius = max_radius
            
            # Calculate pixel bounds (with clamping to image dimensions)
            min_x = max(0, int(pos[0] - radius))
            max_x = min(self.width - 1, int(pos[0] + radius) + 1)
            min_y = max(0, int(pos[1] - radius))
            max_y = min(self.height - 1, int(pos[1] + radius) + 1)
            
            # Skip if outside image bounds
            if min_x >= max_x or min_y >= max_y:
                continue
            
            # Create coordinate grid for this bounding box
            y_range = torch.arange(min_y, max_y, device=self.device)
            x_range = torch.arange(min_x, max_x, device=self.device)
            grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')
            
            # Flatten coordinates
            grid_coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1).float()
            
            # Calculate squared Mahalanobis distance
            delta = grid_coords - pos
            
            # Handle special case of diagonal covariance matrix for efficiency
            if torch.abs(cov[0, 1]) < 1e-6:
                # Diagonal case - much faster
                inv_sigma_x = 1.0 / (cov[0, 0] + 1e-6)
                inv_sigma_y = 1.0 / (cov[1, 1] + 1e-6)
                mahalanobis_sq = delta[:, 0]**2 * inv_sigma_x + delta[:, 1]**2 * inv_sigma_y
            else:
                # General case - slower but handles any covariance matrix
                try:
                    inv_cov = torch.inverse(cov + torch.eye(2, device=self.device) * 1e-6)
                    mahalanobis_sq = torch.sum(delta * torch.matmul(delta, inv_cov), dim=1)
                except:
                    # Fallback if inverse fails
                    inv_sigma = 1.0 / (torch.diag(cov) + 1e-6)
                    mahalanobis_sq = torch.sum(delta**2 * inv_sigma, dim=1)
            
            # Calculate alpha based on power function of Mahalanobis distance
            # alpha = opacity * torch.exp(-0.5 * mahalanobis_sq)
            alpha = opacity * torch.clamp(1 - mahalanobis_sq / 9.0, min=0)  # Power function (faster than exp)
            
            # Reshape to match bounding box dimensions
            alpha_map = alpha.reshape(max_y - min_y, max_x - min_x)
            
            # Update accumulation buffers using alpha blending
            current_alpha = accum_alpha[min_y:max_y, min_x:max_x]
            current_color = accum_colors[min_y:max_y, min_x:max_x]
            current_depth = accum_depth[min_y:max_y, min_x:max_x]
            
            # Alpha compositing: new = (1-alpha)*current + alpha*color
            new_alpha = current_alpha + alpha_map * (1.0 - current_alpha)
            
            # Only update where alpha is significant
            mask = alpha_map > 1e-4
            
            if mask.sum() > 0:
                # Color blending
                blend_weight = alpha_map[mask] / new_alpha[mask]
                current_color[mask] = (1.0 - blend_weight.unsqueeze(-1)) * current_color[mask] + \
                                     blend_weight.unsqueeze(-1) * color
                
                # Depth blending (weighted by alpha)
                current_depth[mask] = (1.0 - blend_weight) * current_depth[mask] + \
                                     blend_weight * depth
            
            # Update accumulators
            accum_alpha[min_y:max_y, min_x:max_x] = new_alpha
            accum_colors[min_y:max_y, min_x:max_x] = current_color
            accum_depth[min_y:max_y, min_x:max_x] = current_depth
        
        # Composite with background
        background_weight = 1.0 - accum_alpha.unsqueeze(-1)
        final_color = accum_colors + background_weight * self.background_color
        
        # Set depth for transparent pixels to far plane
        final_depth = torch.where(accum_alpha > 1e-6, accum_depth, torch.ones_like(accum_depth) * 1000.0)
        
        return {
            'rgb': final_color,        # (H, W, 3)
            'depth': final_depth,      # (H, W)
            'opacity': accum_alpha     # (H, W)
        }
            