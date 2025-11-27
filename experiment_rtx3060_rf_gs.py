#!/usr/bin/env python3
"""
RTX 3060 Benchmark for RF-GS: Radio-Frequency Gaussian Splatting
==================================================================

This script validates RF-GS performance claims on RTX 3060 12GB consumer hardware.
Expected results: 20K Gaussians @ 512x512: ~45-65 FPS
                  40K Gaussians @ 512x512: ~25-35 FPS

Usage:
    python experiment_rtx3060_rf_gs.py --sweep
    python experiment_rtx3060_rf_gs.py --single --gaussians 25000 --resolution 512
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")


@dataclass
class BenchmarkConfig:
    """Configuration for RTX 3060 benchmark experiments."""
    num_gaussians: int = 25000
    resolution: Tuple[int, int] = (512, 512)
    batch_size: int = 1
    num_warmup: int = 10
    num_trials: int = 50
    mixed_precision: bool = True
    compilation: bool = True  # torch.compile for extra speed


class RFEncoder(nn.Module):
    """
    Radio-Frequency Feature Encoder for CSI/mmWave data.
    Converts complex-valued RF measurements to Gaussian features.
    """
    
    def __init__(self, input_channels: int = 64, feature_dim: int = 256):
        super().__init__()
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        
        # Complex-aware convolutions for CSI processing
        self.csi_encoder = nn.Sequential(
            nn.Conv2d(input_channels * 2, 128, 3, padding=1),  # *2 for real/imag
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(256 * 64, feature_dim)
        )
        
        # Learnable RF-specific parameters
        self.frequency_embedding = nn.Parameter(torch.randn(input_channels, 32))
        self.spatial_embedding = nn.Parameter(torch.randn(64, 32))
        
    def forward(self, csi_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            csi_data: Complex CSI tensor [B, C, H, W, 2] (real/imag)
        Returns:
            RF features: [B, feature_dim]
        """
        B, C, H, W, _ = csi_data.shape
        
        # Reshape for conv2d: [B, C*2, H, W]
        x = csi_data.view(B, C*2, H, W)
        
        # Extract features
        features = self.csi_encoder(x)
        
        # Add frequency and spatial embeddings
        freq_embed = self.frequency_embedding.mean(dim=0).expand(B, -1)
        spatial_embed = self.spatial_embedding.mean(dim=0).expand(B, -1)
        
        # Concatenate all features
        combined = torch.cat([features, freq_embed, spatial_embed], dim=-1)
        
        return F.normalize(combined, dim=-1)


class RTX3060GaussianRenderer(nn.Module):
    """
    Optimized RF Gaussian Renderer for RTX 3060.
    Uses memory-efficient implementations and mixed precision.
    """
    
    def __init__(self, config: BenchmarkConfig):
        super().__init__()
        self.config = config
        self.num_gaussians = config.num_gaussians
        self.resolution = config.resolution
        
        # Gaussian parameters (optimized for RTX 3060 memory layout)
        self.register_buffer('positions', torch.randn(self.num_gaussians, 3))
        self.register_buffer('scales', torch.ones(self.num_gaussians, 3) * 0.1)
        self.register_buffer('rotations', torch.zeros(self.num_gaussians, 4))
        self.rotations[:, 0] = 1.0  # w=1 for identity quaternions
        
        # RF-specific features
        self.register_buffer('rf_features', torch.randn(self.num_gaussians, 256))
        self.register_buffer('opacity', torch.sigmoid(torch.randn(self.num_gaussians, 1)))
        
        # Learned projection matrix
        self.projection = nn.Parameter(torch.eye(4))
        
        # RF feature decoder
        self.rf_decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)  # RGB output
        )
        
    def project_gaussians(self, viewpoint_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project 3D Gaussians to 2D screen space."""
        # Transform positions
        positions_hom = torch.cat([self.positions, torch.ones(self.num_gaussians, 1, device=self.positions.device)], dim=-1)
        projected = positions_hom @ (viewpoint_matrix @ self.projection).T
        
        # Perspective division
        screen_pos = projected[:, :2] / (projected[:, 2:3] + 1e-6)
        depths = projected[:, 2]
        
        # Convert to pixel coordinates
        H, W = self.resolution
        screen_pos[:, 0] = (screen_pos[:, 0] + 1) * W * 0.5
        screen_pos[:, 1] = (screen_pos[:, 1] + 1) * H * 0.5
        
        return screen_pos, depths
    
    def compute_2d_covariance(self, screen_pos: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
        """Compute 2D covariance matrices for screen-space Gaussians."""
        # Simplified covariance for RTX 3060 efficiency
        scale_factor = 100.0 / (depths.unsqueeze(-1) + 1.0)
        scales_2d = self.scales[:, :2] * scale_factor
        
        # Diagonal covariance matrices (faster on RTX 3060)
        cov_2d = torch.zeros(self.num_gaussians, 2, 2, device=screen_pos.device)
        cov_2d[:, 0, 0] = scales_2d[:, 0] ** 2
        cov_2d[:, 1, 1] = scales_2d[:, 1] ** 2
        
        return cov_2d
    
    def render_gaussians(self, screen_pos: torch.Tensor, depths: torch.Tensor, 
                        cov_2d: torch.Tensor) -> torch.Tensor:
        """Render Gaussians to image using optimized splatting."""
        H, W = self.resolution
        device = screen_pos.device
        
        # Sort by depth (front-to-back for efficiency)
        sorted_indices = torch.argsort(depths)
        
        # Initialize output image
        image = torch.zeros(3, H, W, device=device, dtype=torch.float16 if self.config.mixed_precision else torch.float32)
        alpha_accum = torch.zeros(H, W, device=device, dtype=image.dtype)
        
        # Create pixel grid
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device, dtype=image.dtype),
            torch.arange(W, device=device, dtype=image.dtype),
            indexing='ij'
        )
        pixel_coords = torch.stack([x_grid, y_grid], dim=-1)  # [H, W, 2]
        
        # Efficient batched rendering (RTX 3060 optimized)
        batch_size = 1000  # Process Gaussians in batches to fit memory
        
        for i in range(0, self.num_gaussians, batch_size):
            end_idx = min(i + batch_size, self.num_gaussians)
            batch_indices = sorted_indices[i:end_idx]
            
            batch_pos = screen_pos[batch_indices]  # [B, 2]
            batch_cov = cov_2d[batch_indices]      # [B, 2, 2]
            batch_opacity = self.opacity[batch_indices]  # [B, 1]
            batch_features = self.rf_features[batch_indices]  # [B, 256]
            
            # Compute Gaussian values
            diff = pixel_coords.unsqueeze(0) - batch_pos.unsqueeze(1).unsqueeze(1)  # [B, H, W, 2]
            
            # Use inverse covariance for efficiency
            inv_cov = torch.inverse(batch_cov + torch.eye(2, device=device) * 1e-6)  # [B, 2, 2]
            
            # Mahalanobis distance
            mahal = torch.einsum('bhwi,bij,bhwj->bhw', diff, inv_cov, diff)  # [B, H, W]
            gaussian_vals = torch.exp(-0.5 * mahal)  # [B, H, W]
            
            # Apply opacity
            alpha = batch_opacity.unsqueeze(-1).unsqueeze(-1) * gaussian_vals  # [B, H, W]
            
            # Decode RF features to colors
            colors = self.rf_decoder(batch_features)  # [B, 3]
            colors = torch.sigmoid(colors)  # Ensure [0,1] range
            
            # Alpha compositing
            for j in range(alpha.shape[0]):
                curr_alpha = alpha[j] * (1 - alpha_accum)
                
                # Add contribution to image
                for c in range(3):
                    image[c] += colors[j, c] * curr_alpha
                
                # Update accumulated alpha
                alpha_accum += curr_alpha
                
                # Early termination if fully opaque
                if torch.all(alpha_accum > 0.99):
                    break
        
        return image.permute(1, 2, 0)  # [H, W, 3]
    
    def forward(self, viewpoint_matrix: torch.Tensor) -> torch.Tensor:
        """Render from given viewpoint."""
        # Project Gaussians to screen space
        screen_pos, depths = self.project_gaussians(viewpoint_matrix)
        
        # Compute 2D covariance
        cov_2d = self.compute_2d_covariance(screen_pos, depths)
        
        # Render to image
        image = self.render_gaussians(screen_pos, depths, cov_2d)
        
        return image


def benchmark_render_loop(renderer: RTX3060GaussianRenderer, config: BenchmarkConfig) -> Dict[str, float]:
    """Benchmark rendering performance on RTX 3060."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate random viewpoint matrices
    viewpoints = []
    for _ in range(config.num_trials + config.num_warmup):
        # Random camera position
        theta = torch.rand(1) * 2 * np.pi
        phi = torch.rand(1) * np.pi
        radius = 3.0 + torch.rand(1) * 2.0
        
        x = radius * torch.sin(phi) * torch.cos(theta)
        y = radius * torch.sin(phi) * torch.sin(theta)
        z = radius * torch.cos(phi)
        
        # Create view matrix
        eye = torch.tensor([x, y, z], dtype=torch.float32)
        target = torch.zeros(3)
        up = torch.tensor([0., 1., 0.])
        
        # Simple view matrix construction
        w = F.normalize(eye - target, dim=0)
        u = F.normalize(torch.cross(up, w), dim=0)
        v = torch.cross(w, u)
        
        view_matrix = torch.eye(4)
        view_matrix[:3, :3] = torch.stack([u, v, w], dim=0)
        view_matrix[:3, 3] = -torch.stack([
            torch.dot(u, eye),
            torch.dot(v, eye),
            torch.dot(w, eye)
        ])
        
        viewpoints.append(view_matrix.to(device))
    
    # Warmup runs
    print(f"Warming up RTX 3060 with {config.num_warmup} frames...")
    renderer.eval()
    with torch.no_grad():
        for i in range(config.num_warmup):
            _ = renderer(viewpoints[i])
            if i == 0:
                torch.cuda.synchronize()  # Ensure first frame is loaded
    
    # Actual benchmark
    print(f"Benchmarking {config.num_trials} frames...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(config.num_warmup, config.num_warmup + config.num_trials):
            if config.mixed_precision:
                with torch.cuda.amp.autocast():
                    image = renderer(viewpoints[i])
            else:
                image = renderer(viewpoints[i])
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    fps = config.num_trials / total_time
    ms_per_frame = (total_time / config.num_trials) * 1000
    
    # Memory usage
    memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    return {
        'fps': fps,
        'ms_per_frame': ms_per_frame,
        'total_time': total_time,
        'memory_mb': memory_mb,
        'num_gaussians': config.num_gaussians,
        'resolution': f"{config.resolution[0]}x{config.resolution[1]}",
        'mixed_precision': config.mixed_precision
    }


def sweep_experiments() -> List[Dict[str, float]]:
    """Run comprehensive performance sweep on RTX 3060."""
    print("=== RTX 3060 RF-GS Performance Validation ===\n")
    
    # Test configurations
    test_configs = [
        # Standard resolution tests
        (10000, (512, 512)),
        (15000, (512, 512)),
        (20000, (512, 512)),
        (25000, (512, 512)),
        (30000, (512, 512)),
        (40000, (512, 512)),
        
        # High resolution tests
        (20000, (640, 640)),
        (25000, (640, 640)),
        (30000, (640, 640)),
        
        # Lower resolution for comparison
        (40000, (256, 256)),
        (50000, (256, 256)),
    ]
    
    results = []
    
    for num_gaussians, resolution in test_configs:
        print(f"\nTesting {num_gaussians:,} Gaussians @ {resolution[0]}x{resolution[1]}...")
        
        # Create configuration
        config = BenchmarkConfig(
            num_gaussians=num_gaussians,
            resolution=resolution,
            mixed_precision=True,
            num_trials=30,
            num_warmup=5
        )
        
        try:
            # Create renderer
            renderer = RTX3060GaussianRenderer(config)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            renderer = renderer.to(device)
            
            # Optimize for RTX 3060
            if config.compilation and hasattr(torch, 'compile'):
                renderer = torch.compile(renderer, mode='max-autotune')
            
            # Run benchmark
            result = benchmark_render_loop(renderer, config)
            results.append(result)
            
            print(f"  → {result['fps']:.1f} FPS ({result['ms_per_frame']:.1f} ms/frame)")
            print(f"  → Memory: {result['memory_mb']:.0f} MB")
            
            # Clear GPU memory
            del renderer
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print(f"  → OOM: Configuration too large for RTX 3060 12GB")
            results.append({
                'fps': 0.0,
                'ms_per_frame': float('inf'),
                'memory_mb': float('inf'),
                'num_gaussians': num_gaussians,
                'resolution': f"{resolution[0]}x{resolution[1]}",
                'error': 'OOM'
            })
            torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"  → Error: {str(e)}")
            results.append({
                'fps': 0.0,
                'ms_per_frame': float('inf'),
                'memory_mb': float('inf'),
                'num_gaussians': num_gaussians,
                'resolution': f"{resolution[0]}x{resolution[1]}",
                'error': str(e)
            })
            torch.cuda.empty_cache()
    
    return results


def save_results(results: List[Dict[str, float]], output_path: str = "rtx3060_benchmark_results.json"):
    """Save benchmark results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


def plot_performance_curve(results: List[Dict[str, float]], output_path: str = "rtx3060_performance.png"):
    """Generate performance visualization."""
    # Filter successful results
    valid_results = [r for r in results if 'error' not in r and r['fps'] > 0]
    
    if not valid_results:
        print("No valid results to plot")
        return
    
    # Group by resolution
    res_groups = {}
    for r in valid_results:
        res = r['resolution']
        if res not in res_groups:
            res_groups[res] = []
        res_groups[res].append(r)
    
    plt.figure(figsize=(12, 8))
    
    # Plot FPS vs Gaussians for each resolution
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, (res, data) in enumerate(res_groups.items()):
        gaussians = [d['num_gaussians'] for d in data]
        fps_values = [d['fps'] for d in data]
        
        plt.plot(gaussians, fps_values, 'o-', color=colors[i % len(colors)], 
                linewidth=2, markersize=8, label=f'{res}')
    
    plt.axhline(y=60, color='black', linestyle='--', alpha=0.7, label='60 FPS target')
    plt.axhline(y=30, color='gray', linestyle='--', alpha=0.7, label='30 FPS target')
    
    plt.xlabel('Number of Gaussians', fontsize=12)
    plt.ylabel('Rendering FPS', fontsize=12)
    plt.title('RTX 3060 RF-GS Performance Validation', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Performance plot saved to {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='RTX 3060 RF-GS Benchmark')
    parser.add_argument('--sweep', action='store_true', help='Run full performance sweep')
    parser.add_argument('--single', action='store_true', help='Run single configuration')
    parser.add_argument('--gaussians', type=int, default=25000, help='Number of Gaussians')
    parser.add_argument('--resolution', type=int, default=512, help='Image resolution (square)')
    parser.add_argument('--output', type=str, default='rtx3060_results', help='Output file prefix')
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. RTX 3060 benchmark requires GPU.")
        return
    
    # Check GPU type
    gpu_name = torch.cuda.get_device_name()
    print(f"GPU: {gpu_name}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")
    
    if args.sweep:
        print("Running comprehensive RTX 3060 performance sweep...")
        results = sweep_experiments()
        
        # Save and plot results
        save_results(results, f"{args.output}.json")
        plot_performance_curve(results, f"{args.output}.png")
        
        # Print summary
        print("\n=== RTX 3060 Performance Summary ===")
        valid_results = [r for r in results if 'error' not in r and r['fps'] > 0]
        
        if valid_results:
            best_512 = max([r for r in valid_results if r['resolution'] == '512x512'], 
                          key=lambda x: x['num_gaussians'], default=None)
            
            if best_512:
                print(f"Best 512x512 performance: {best_512['num_gaussians']:,} Gaussians @ {best_512['fps']:.1f} FPS")
            
            # Check paper claims
            paper_claim_20k = [r for r in valid_results if r['num_gaussians'] == 20000 and r['resolution'] == '512x512']
            paper_claim_40k = [r for r in valid_results if r['num_gaussians'] == 40000 and r['resolution'] == '512x512']
            
            if paper_claim_20k:
                fps_20k = paper_claim_20k[0]['fps']
                print(f"Paper validation - 20K Gaussians: {fps_20k:.1f} FPS (target: 45-65 FPS)")
                
            if paper_claim_40k:
                fps_40k = paper_claim_40k[0]['fps']
                print(f"Paper validation - 40K Gaussians: {fps_40k:.1f} FPS (target: 25-35 FPS)")
    
    elif args.single:
        print(f"Running single benchmark: {args.gaussians:,} Gaussians @ {args.resolution}x{args.resolution}")
        
        config = BenchmarkConfig(
            num_gaussians=args.gaussians,
            resolution=(args.resolution, args.resolution),
            mixed_precision=True
        )
        
        renderer = RTX3060GaussianRenderer(config)
        renderer = renderer.cuda()
        
        if hasattr(torch, 'compile'):
            renderer = torch.compile(renderer, mode='max-autotune')
        
        result = benchmark_render_loop(renderer, config)
        
        print(f"\nResults:")
        print(f"  FPS: {result['fps']:.1f}")
        print(f"  ms/frame: {result['ms_per_frame']:.1f}")
        print(f"  Memory: {result['memory_mb']:.0f} MB")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()