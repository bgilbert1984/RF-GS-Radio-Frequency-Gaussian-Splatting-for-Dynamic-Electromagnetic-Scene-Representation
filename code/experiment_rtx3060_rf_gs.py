import time
import math
import argparse

import torch

from neural_gaussian_splats import GaussianSplatModel


def look_at(camera_pos, target=None, up=None):
    """
    Build a camera-to-world matrix for your renderer.
    """
    device = camera_pos.device
    if target is None:
        target = torch.tensor([0.0, 0.0, 0.0], device=device)
    if up is None:
        up = torch.tensor([0.0, 1.0, 0.0], device=device)

    forward = (target - camera_pos)
    forward = forward / (forward.norm() + 1e-9)

    # Explicit dim to avoid torch.cross deprecation warnings
    right = torch.cross(forward, up, dim=0)
    right = right / (right.norm() + 1e-9)

    true_up = torch.cross(right, forward, dim=0)

    # camera-to-world (columns are basis vectors, last col is translation)
    cam_to_world = torch.eye(4, device=device)
    cam_to_world[:3, 0] = right
    cam_to_world[:3, 1] = true_up
    cam_to_world[:3, 2] = forward
    cam_to_world[:3, 3] = camera_pos
    return cam_to_world


def generate_synthetic_rf_points(n_points: int, feature_dim: int, device: str = "cuda"):
    """
    Simple synthetic RF 'points + features' for fitting GS.
    Positions in [-1, 1]^3, features ~ N(0, 1).
    """
    positions = torch.rand(n_points, 3, device=device) * 2.0 - 1.0
    rf_features = torch.randn(n_points, feature_dim, device=device)
    return positions, rf_features


def warmup_model(
    model: GaussianSplatModel,
    n_points: int = None,
    n_iters: int = 3,
    device: str = "cuda",
):
    """
    Quick, lighter warmup so the benchmark doesn't take ages.
    Caps n_points to avoid O(N^2) cdist blowups during fit.
    """
    if n_points is None:
        # keep cdist work reasonable: up to twice the Gaussians but capped
        # tighter cap for dev-mode to keep runs near-instant
        n_points = min(int(model.num_gaussians * 2), 5000)

    positions, rf_features = generate_synthetic_rf_points(
        n_points=n_points,
        feature_dim=model.feature_dim,
        device=device,
    )

    print(f"[warmup] {n_points} RF points, {n_iters} iterations...", flush=True)
    model.fit_to_rf_data(
        positions=positions,
        rf_features=rf_features,
        colors=None,                # RF-only supervision
        num_iterations=n_iters,
        learning_rate=None,
        regularization=0.001,
        prune_interval=max(5, n_iters // 3),
        densify_interval=max(10, n_iters // 2),
        verbose=True,
    )
    print(f"[warmup] Done. Active Gaussians: {model.num_active}", flush=True)


def benchmark_render_loop(
    model: GaussianSplatModel,
    width: int,
    height: int,
    n_frames: int,
    radius: float,
    fov_deg: float,
    device: str = "cuda",
):
    """
    Orbit camera around origin and measure FPS for render_image().
    Prints coarse progress while rendering so long runs are visible.
    """
    model.eval()
    torch.cuda.empty_cache()
    if torch.cuda.is_available() and isinstance(device, str) and device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    # focal length in pixels from FOV
    fov_rad = math.radians(fov_deg)
    focal_length = (0.5 * width) / math.tan(0.5 * fov_rad)

    # simple circular orbit in XZ plane
    center = torch.tensor([0.0, 0.0, 0.0], device=device)

    times = []
    start_global = time.perf_counter()

    print(f"[render] {n_frames} frames at {width}x{height}", flush=True)
    progress_step = max(1, n_frames // 10)

    for frame_idx in range(n_frames):
        angle = 2.0 * math.pi * (frame_idx / n_frames)
        cam_pos = torch.tensor(
            [
                radius * math.cos(angle),
                radius * 0.2,               # slight elevation
                radius * math.sin(angle),
            ],
            device=device,
        )

        cam_to_world = look_at(cam_pos, target=center)

        if torch.cuda.is_available() and isinstance(device, str) and device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            _ = model.render_image(
                camera_position=cam_pos,
                camera_matrix=cam_to_world,
                width=width,
                height=height,
                focal_length=focal_length,
                near_plane=0.1,
                far_plane=10.0,
                sort_points=False,  # disable sorting for faster feedback
                num_depth_bits=16,
                depth_premultiplier=10.0,
            )

        if torch.cuda.is_available() and isinstance(device, str) and device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

        if (frame_idx + 1) % progress_step == 0 or frame_idx == 0:
            pct = int(100 * (frame_idx + 1) / n_frames)
            print(f"[render] frame {frame_idx+1}/{n_frames} ({pct}%)", flush=True)

    end_global = time.perf_counter()

    frame_times = torch.tensor(times)
    mean_time = frame_times.mean().item()
    min_time = frame_times.min().item()
    max_time = frame_times.max().item()

    mean_fps = 1.0 / mean_time
    worst_fps = 1.0 / max_time
    best_fps = 1.0 / min_time

    max_mem = 0.0
    if torch.cuda.is_available() and isinstance(device, str) and device.startswith("cuda"):
        try:
            max_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        except Exception:
            max_mem = 0.0

    print("\n=== Render Benchmark Results ===")
    print(f"Resolution       : {width} x {height}")
    print(f"Frames           : {n_frames}")
    print(f"Mean frame time  : {mean_time*1000:.2f} ms  ({mean_fps:.1f} FPS)")
    print(f"Best frame time  : {min_time*1000:.2f} ms  ({best_fps:.1f} FPS)")
    print(f"Worst frame time : {max_time*1000:.2f} ms  ({worst_fps:.1f} FPS)")
    print(f"Wall-clock (loop): {end_global - start_global:.2f} s")
    if max_mem > 0:
        print(f"Max GPU memory   : {max_mem:.1f} MB")

    return {
        "mean_fps": mean_fps,
        "best_fps": best_fps,
        "worst_fps": worst_fps,
        "max_mem_mb": max_mem,
    }


def run_single_experiment(
    num_gaussians: int,
    width: int,
    height: int,
    feature_dim: int,
    n_frames: int,
    radius: float,
    fov_deg: float,
    device: str = "cuda",
    backend: str = "python",
):
    print(
        f"\n>>> Experiment: {num_gaussians} Gaussians, "
        f"{width}x{height}, {n_frames} frames on {device}"
    )

    model = GaussianSplatModel(
        num_gaussians=num_gaussians,
        feature_dim=feature_dim,
        color_dim=3,
        min_opacity=0.005,
        learning_rate=0.005,
        adaptive_density=True,
        device=torch.device(device),
        backend=backend,
    )

    # Dev-mode warmup: very light to populate Gaussians quickly
    warmup_model(
        model,
        n_points=None,
        n_iters=3,
        device=device,
    )

    # Render benchmark
    stats = benchmark_render_loop(
        model=model,
        width=width,
        height=height,
        n_frames=n_frames,
        radius=radius,
        fov_deg=fov_deg,
        device=device,
    )

    return stats


def sweep_experiments(device: str = "cuda", backend: str = "python"):
    """
    Sweep a few configs to see what 'real-time' looks like on 3060.
    Tweak these to taste.
    """
    # Ultra-fast dev sweep: single tiny config for instant iteration
    configs = [
        # (num_gaussians, width, height)
        (500, 256, 256),
    ]

    results = []
    total = len(configs)
    print(f"[sweep] Running {total} configurations... (backend={backend})", flush=True)
    for idx, (ng, w, h) in enumerate(configs, start=1):
        print(f"\n[sweep] Config {idx}/{total}: {ng} Gaussians @ {w}x{h}", flush=True)
        try:
            stats = run_single_experiment(
                num_gaussians=ng,
                width=w,
                height=h,
                feature_dim=32,
                n_frames=5,  # ultra-fast dev feedback
                radius=2.5,
                fov_deg=60.0,
                device=device,
                backend=backend,
            )
            results.append({
                'gaussians': ng,
                'resolution': f"{w}x{h}",
                'mean_fps': stats['mean_fps'],
                'worst_fps': stats['worst_fps'],
                'max_mem_mb': stats['max_mem_mb']
            })
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"[SKIP] {ng} Gaussians at {w}x{h} - GPU OOM")
                results.append({
                    'gaussians': ng,
                    'resolution': f"{w}x{h}",
                    'mean_fps': 'OOM',
                    'worst_fps': 'OOM',
                    'max_mem_mb': 'OOM'
                })
            else:
                raise

    # Summary table
    print("\n" + "="*80)
    print("RTX 3060 12GB CAPABILITY SUMMARY")
    print("="*80)
    print(f"{'Gaussians':<12} {'Resolution':<12} {'Mean FPS':<12} {'Worst FPS':<12} {'Memory (MB)':<12}")
    print("-"*80)
    
    for r in results:
        print(f"{r['gaussians']:<12} {r['resolution']:<12} {r['mean_fps']:<12} {r['worst_fps']:<12} {r['max_mem_mb']:<12}")
    
    print("="*80)
    
    # Recommendations
    real_time_configs = [r for r in results if isinstance(r['mean_fps'], float) and r['mean_fps'] >= 30]
    smooth_configs = [r for r in results if isinstance(r['mean_fps'], float) and r['mean_fps'] >= 60]
    
    print("\nRECOMMENDATIONS:")
    if smooth_configs:
        best_smooth = max(smooth_configs, key=lambda x: x['gaussians'] * int(x['resolution'].split('x')[0]) * int(x['resolution'].split('x')[1]))
        print(f"  Best smooth config (≥60 FPS): {best_smooth['gaussians']} Gaussians at {best_smooth['resolution']}")
    
    if real_time_configs:
        best_rt = max(real_time_configs, key=lambda x: x['gaussians'] * int(x['resolution'].split('x')[0]) * int(x['resolution'].split('x')[1]))
        print(f"  Best real-time config (≥30 FPS): {best_rt['gaussians']} Gaussians at {best_rt['resolution']}")


def benchmark_with_motion(device: str = "cuda"):
    """
    Test dynamic RF scenes with motion vectors (optional).
    """
    try:
        from neural_correspondence import NeuralCorrespondenceField
        
        print("\n>>> Testing RF-GS with Neural Correspondence Field...")
        
        model = GaussianSplatModel(
            num_gaussians=20000,
            feature_dim=32,
            device=torch.device(device)
        )
        
        # Quick warmup
        warmup_model(model, n_points=20000, n_iters=30, device=device)
        
        # Build NCF for motion
        ncf = NeuralCorrespondenceField().to(device)
        
        # Get initial Gaussian positions
        params = model.get_active_parameters()
        initial_positions = params["positions"].detach()
        
        # Benchmark with motion
        n_frames = 60
        times = []
        
        for frame_idx in range(n_frames):
            t = frame_idx * 0.033  # ~30 Hz
            time_tensor = torch.full((initial_positions.shape[0], 1), t, device=device)
            
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            
            with torch.no_grad():
                # Compute motion vectors
                motion_out = ncf(initial_positions, time_tensor)
                motion_vectors = motion_out["motion_vector"]
                
                # Update positions
                model.positions[model.active_mask] = initial_positions + motion_vectors
                
                # Render
                cam_pos = torch.tensor([2.0, 0.5, 2.0], device=device)
                cam_to_world = look_at(cam_pos)
                
                _ = model.render_image(
                    camera_position=cam_pos,
                    camera_matrix=cam_to_world,
                    width=512,
                    height=512,
                    focal_length=400,
                )
            
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        
        mean_fps = 1.0 / (sum(times) / len(times))
        print(f"Dynamic RF-GS (with NCF motion): {mean_fps:.1f} FPS at 512x512")
        
    except ImportError:
        print("Neural correspondence not available - skipping motion test")


def main():
    parser = argparse.ArgumentParser(description="RTX 3060 RF Gaussian Splat Benchmark")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--sweep", action="store_true", help="run multiple configs")
    parser.add_argument("--motion", action="store_true", help="test with neural correspondence field")
    parser.add_argument("--num_gaussians", type=int, default=20000)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--radius", type=float, default=2.5)
    parser.add_argument("--fov", type=float, default=60.0)
    parser.add_argument("--feature_dim", type=int, default=32)
    parser.add_argument(
        "--backend",
        type=str,
        default="python",
        choices=["python", "cuda-fallback", "cuda-3dgs", "cuda-auto"],
        help=(
            "Renderer backend: 'python' (pure PyTorch), 'cuda-fallback' (simple CUDA splat), "
            "'cuda-3dgs' (requires diff_gaussian_rasterization), 'cuda-auto' (use 3DGS if available)."
        ),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device == "cuda":
        print("[warn] CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Print GPU info
    if args.device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")

        # Measured Python-reference numbers (pure PyTorch renderer)
        print("")
        print("Python reference renderer (measured on RTX 3060 12GB):")
        print("  • 1K Gaussians @ 256x256: ~0.4–0.5 FPS")
        print("  • 2K Gaussians @ 256x256: ~0.2–0.25 FPS")
        print("")
        print("Note: these numbers are for the readable Python reference renderer. For real-time results we bind to optimized 3DGS CUDA kernels.")

    if args.motion:
        benchmark_with_motion(device=args.device)
    elif args.sweep:
        sweep_experiments(device=args.device, backend=args.backend)
    else:
        run_single_experiment(
            num_gaussians=args.num_gaussians,
            width=args.width,
            height=args.height,
            feature_dim=args.feature_dim,
            n_frames=args.frames,
            radius=args.radius,
            fov_deg=args.fov,
            device=args.device,
            backend=args.backend,
        )


if __name__ == "__main__":
    main()