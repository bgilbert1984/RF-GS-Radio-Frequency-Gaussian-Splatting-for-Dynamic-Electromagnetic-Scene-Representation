"""Adapter for a CUDA-backed 3D Gaussian Splatting renderer.

This module exposes `CUDAGaussianRenderer`, which will try to use an
external optimized renderer (if installed). If not found, it falls back to a
vectorized, GPU-friendly nearest-neighbor splat implementation as a stopgap.

The fallback is not a true Gaussian rasterizer, but it is fast and fully
vectorized on the GPU and useful for benchmarking and functional testing.

Expected `render` signature (compatible with `GaussianSplatModel._render_image_cuda`):
- positions: (N,3) world-space centers
- scales: (N,3) or (N,) approximate per-point scale
- rotations: (N,3,3) or None
- colors: (N,3) RGB in [0,1]
- opacities: (N,) in [0,1]
- camera_to_world: (4,4) transform matrix
- image_size: (H,W)
- fov: field-of-view in degrees (float)

Returns: image tensor (H,W,3) on the same device as renderer.device
"""

from typing import Optional, Tuple
import math
import torch

"""
rf_3dgs_backend.py

Adapter around a *real* 3D Gaussian Splatting CUDA backend.

Primary target: GraphDECO-style diff_gaussian_rasterization package
(https://github.com/graphdeco-inria/diff-gaussian-rasterization), either
installed directly or vendored via the official gaussian-splatting repo.

Design goals
------------
- Present a *minimal*, well-defined API to your RF Gaussian model:
    CUDAGaussianRenderer.render(...)
- Map that API 1:1 to GaussianRasterizationSettings / GaussianRasterizer.
- Fall back to a simple GPU “nearest-splat” renderer if the CUDA kernel
  is not available (so the rest of the pipeline still runs).

This file is self-contained; you just need diff_gaussian_rasterization
on PYTHONPATH for the fast path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch

# -------------------------------------------------------------------------
# 1. Try to import the *exact* 3DGS CUDA renderer we want to target
# -------------------------------------------------------------------------

_HAS_3DGS = False

try:  # Preferred: standalone pip / submodule package
    from diff_gaussian_rasterization import (  # type: ignore
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )

    _HAS_3DGS = True
except Exception:  # noqa: BLE001
    GaussianRasterizationSettings = None  # type: ignore[assignment]
    GaussianRasterizer = None  # type: ignore[assignment]


# -------------------------------------------------------------------------
# 2. Camera + render configuration
# -------------------------------------------------------------------------


@dataclass
class RF3DGSRenderConfig:
    """Camera + render configuration for a single frame."""

    width: int
    height: int
    fov_y_radians: float
    near_plane: float = 0.1
    far_plane: float = 10.0
    scale_modifier: float = 1.0
    sh_degree: int = 0  # we use pre-baked RGB, so SH degree 0 is fine
    debug: bool = False

    def tan_fovs(self) -> Tuple[float, float]:
        aspect = float(self.width) / float(self.height)
        tanfovy = math.tan(self.fov_y_radians * 0.5)
        tanfovx = tanfovy * aspect
        return tanfovx, tanfovy


# -------------------------------------------------------------------------
# 3. Utility: build view / projection matrices compatible with 3DGS
# -------------------------------------------------------------------------


def _invert_4x4(matrix: torch.Tensor) -> torch.Tensor:
    """Safe 4x4 inverse with a clear error if the input is malformed."""
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected (4,4) camera matrix, got {tuple(matrix.shape)}")
    return torch.linalg.inv(matrix)


def _build_perspective_matrix(
    fov_y_radians: float,
    aspect: float,
    z_near: float,
    z_far: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Standard right-handed perspective matrix (OpenGL-style).

    3DGS expects a 4x4 projection in clip-space; this matches the usual
    conventions and is sufficient for our RF visualization use-case.
    """
    f = 1.0 / math.tan(0.5 * fov_y_radians)
    z1 = (z_far + z_near) / (z_near - z_far)
    z2 = (2.0 * z_far * z_near) / (z_near - z_far)

    proj = torch.zeros((4, 4), device=device, dtype=dtype)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = z1
    proj[2, 3] = z2
    proj[3, 2] = -1.0
    return proj


def _build_camera_transforms(
    cam_to_world: torch.Tensor,
    cfg: RF3DGSRenderConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given a camera-to-world 4x4, build the matrices expected by 3DGS:

        world_view_transform : world -> camera
        full_proj_transform  : clip-space projection (proj @ world_view)
    """
    if cam_to_world.dim() != 2 or cam_to_world.shape != (4, 4):
        raise ValueError(
            f"cam_to_world must be (4,4), got shape {tuple(cam_to_world.shape)}"
        )

    device = cam_to_world.device
    dtype = cam_to_world.dtype

    world_view_transform = _invert_4x4(cam_to_world)
    proj = _build_perspective_matrix(
        fov_y_radians=cfg.fov_y_radians,
        aspect=float(cfg.width) / float(cfg.height),
        z_near=cfg.near_plane,
        z_far=cfg.far_plane,
        device=device,
        dtype=dtype,
    )
    full_proj_transform = proj @ world_view_transform
    return world_view_transform, full_proj_transform


# -------------------------------------------------------------------------
# 4. Main adapter class
# -------------------------------------------------------------------------


class CUDAGaussianRenderer:
    """
    Thin adapter around diff_gaussian_rasterization.GaussianRasterizer.

    High-level API (the part your RF model calls):

        renderer = CUDAGaussianRenderer(device="cuda:0")
        rgb, depth, alpha = renderer.render(
            positions_3d=positions,
            colors=colors,
            opacities=opacities,
            covariances_3d=covariance,   # or scales=..., rotations=...
            camera_matrix=cam_to_world,
            camera_position=cam_pos,
            width=W,
            height=H,
            fov_y=fov_y_radians,
        )

    All inputs are expected on the *same* device as the renderer, and in
    float32. This class will not silently move tensors back and forth.
    """

    def __init__(
        self,
        device: str | torch.device = "cuda",
        mode: str = "cuda-auto",
    ) -> None:
        self.device = torch.device(device)

        mode = (mode or "").lower()
        if mode not in ("cuda-auto", "cuda-fallback", "cuda-3dgs"):
            raise ValueError(f"Unknown CUDA renderer mode: {mode}")

        # Decide whether to use the real 3DGS kernel or the fallback splatter
        if mode == "cuda-3dgs":
            if not _HAS_3DGS:
                raise RuntimeError(
                    "[rf_3dgs_backend] Requested mode 'cuda-3dgs' but "
                    "diff_gaussian_rasterization is not importable. "
                    "Install it or use 'cuda-fallback' / 'cuda-auto'."
                )
            self._use_3dgs = True
        elif mode == "cuda-fallback":
            self._use_3dgs = False
        else:  # cuda-auto
            self._use_3dgs = bool(_HAS_3DGS)

        self._mode = mode
        self._rasterizer = None

        if not self._use_3dgs:
            print(
                f"[rf_3dgs_backend] Using fallback CUDA splatter (mode={self._mode}, 3DGS available={_HAS_3DGS})."
            )
        else:
            print(
                f"[rf_3dgs_backend] Using diff_gaussian_rasterization (mode={self._mode})."
            )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def render(self, **kwargs: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Flexible wrapper that accepts a few synonymous argument names and
        dispatches to either the true 3DGS CUDA path or a fallback splat.

        Accepted keyword arguments (canonical names in parentheses):

            positions_3d (positions_3d / means3D / xyz) : (N,3)
            colors      (colors / colors_precomp)       : (N,3)
            opacities   (opacities / opacity)           : (N,1) or (N,)
            covariances_3d (covariances_3d / cov3D_precomp) : (N,3,3) [optional]
            scales      : (N,3) [optional if covariances_3d given]
            rotations   : (N,4) quaternion [optional]

            camera_matrix (camera_matrix / cam_to_world) : (4,4)
            camera_position (camera_position / campos)    : (3,)

            width, height : ints
            fov_y         : vertical FOV in *radians* (preferred)
            focal_length  : focal length in pixels (alternative to fov_y)
            near_plane, far_plane : floats [optional]

        Returns:
            rgb   : (H,W,3) float32 image on renderer device
            depth : (H,W,1) or None
            alpha : (H,W,1) or None
        """
        (
            positions,
            colors,
            opacities,
            cov3D,
            scales,
            rotations,
            cam_to_world,
            cam_pos,
            cfg,
            bg_color,
        ) = self._normalize_inputs(**kwargs)

        if self._use_3dgs:
            return self._render_3dgs(
                positions=positions,
                colors=colors,
                opacities=opacities,
                cov3D=cov3D,
                scales=scales,
                rotations=rotations,
                cam_to_world=cam_to_world,
                cam_pos=cam_pos,
                cfg=cfg,
                bg_color=bg_color,
            )
        else:
            return self._render_fallback(
                positions=positions,
                colors=colors,
                opacities=opacities,
                cam_to_world=cam_to_world,
                cfg=cfg,
                bg_color=bg_color,
            )

    # ------------------------------------------------------------------
    # 4a. Input normalization
    # ------------------------------------------------------------------

    def _normalize_inputs(self, **kwargs: Any):
        dev = self.device

        def _pop_any(*names: str, default=None):
            for n in names:
                if n in kwargs:
                    return kwargs.pop(n)
            return default

        positions = _pop_any("positions_3d", "means3D", "xyz")
        colors = _pop_any("colors", "colors_precomp")
        opacities = _pop_any("opacities", "opacity")
        cov3D = _pop_any("covariances_3d", "cov3D_precomp")
        scales = _pop_any("scales",)
        rotations = _pop_any("rotations",)

        cam_to_world = _pop_any("camera_matrix", "cam_to_world")
        cam_pos = _pop_any("camera_position", "campos")

        width = int(_pop_any("width"))
        height = int(_pop_any("height"))

        fov_y = _pop_any("fov_y")
        focal_length = _pop_any("focal_length")

        near_plane = float(_pop_any("near_plane", default=0.1))
        far_plane = float(_pop_any("far_plane", default=10.0))

        bg_color = _pop_any("bg_color", default=torch.tensor([0.0, 0.0, 0.0], device=dev))

        if positions is None or colors is None or opacities is None:
            raise ValueError("positions_3d, colors, and opacities are required")

        # Move everything onto the renderer device and to float32
        positions = positions.to(dev, dtype=torch.float32)
        colors = colors.to(dev, dtype=torch.float32)
        opacities = opacities.to(dev, dtype=torch.float32)

        if cov3D is not None:
            cov3D = cov3D.to(dev, dtype=torch.float32)
        if scales is not None:
            scales = scales.to(dev, dtype=torch.float32)
        if rotations is not None:
            rotations = rotations.to(dev, dtype=torch.float32)

        if isinstance(bg_color, torch.Tensor):
            bg = bg_color.to(dev, dtype=torch.float32)
        else:
            # assume tuple or scalar
            if isinstance(bg_color, (tuple, list)):
                bg = torch.tensor(bg_color, device=dev, dtype=torch.float32)
            else:
                bg = torch.tensor([bg_color, bg_color, bg_color], device=dev, dtype=torch.float32)

        if cam_to_world is None:
            raise ValueError("camera_matrix / cam_to_world (4x4) is required for 3DGS backend")
        cam_to_world = cam_to_world.to(dev, dtype=torch.float32)

        if cam_pos is None:
            # Derive camera center from cam_to_world last column
            cam_pos = cam_to_world[:3, 3]
        else:
            cam_pos = cam_pos.to(dev, dtype=torch.float32)

        # FOV handling
        if fov_y is None:
            if focal_length is None:
                raise ValueError("Either fov_y (radians) or focal_length (pixels) must be provided")
            # infer FOV from focal length and image height (vertical FOV)
            fov_y = 2.0 * math.atan(0.5 * float(height) / float(focal_length))
        fov_y = float(fov_y)

        cfg = RF3DGSRenderConfig(
            width=width,
            height=height,
            fov_y_radians=fov_y,
            near_plane=near_plane,
            far_plane=far_plane,
        )

        return (
            positions,
            colors,
            opacities,
            cov3D,
            scales,
            rotations,
            cam_to_world,
            cam_pos,
            cfg,
            bg,
        )

    # ------------------------------------------------------------------
    # 4b. True 3DGS CUDA path
    # ------------------------------------------------------------------

    def _ensure_rasterizer(
        self,
        world_view_transform: torch.Tensor,
        full_proj_transform: torch.Tensor,
        cfg: RF3DGSRenderConfig,
        bg_color: torch.Tensor,
    ):
        global GaussianRasterizationSettings, GaussianRasterizer  # type: ignore[global-variable-not-assigned]

        if self._rasterizer is not None:
            return

        tanfovx, tanfovy = cfg.tan_fovs()

        raster_settings = GaussianRasterizationSettings(  # type: ignore[call-arg]
            image_height=int(cfg.height),
            image_width=int(cfg.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=cfg.scale_modifier,
            viewmatrix=world_view_transform,
            projmatrix=full_proj_transform,
            sh_degree=cfg.sh_degree,
            campos=torch.zeros(3, device=world_view_transform.device, dtype=world_view_transform.dtype),
            prefiltered=False,
            debug=cfg.debug,
        )

        self._rasterizer = GaussianRasterizer(raster_settings=raster_settings)  # type: ignore[call-arg]

    def _render_3dgs(
        self,
        *,
        positions: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        cov3D: Optional[torch.Tensor],
        scales: Optional[torch.Tensor],
        rotations: Optional[torch.Tensor],
        cam_to_world: torch.Tensor,
        cam_pos: torch.Tensor,
        cfg: RF3DGSRenderConfig,
        bg_color: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:

        world_view, full_proj = _build_camera_transforms(cam_to_world, cfg)
        self._ensure_rasterizer(world_view, full_proj, cfg, bg_color)

        # In canonical 3DGS code, screenspace_points is just a zero tensor with
        # requires_grad=True so you can back-propagate 2D positions. For our RF
        # visualization we don’t need gradients here, so we can keep it simple.
        screenspace_points = torch.zeros_like(positions, device=positions.device, dtype=positions.dtype)

        # Decide whether to use precomputed covariance or (scale, rotation)
        if cov3D is not None:
            cov3D_precomp = cov3D
            scales_arg = None
            rotations_arg = None
        else:
            cov3D_precomp = None
            scales_arg = scales
            rotations_arg = rotations

        # diff_gaussian_rasterization expects opacities shaped (N,)
        if opacities.ndim == 2 and opacities.shape[1] == 1:
            opacities_flat = opacities[:, 0]
        else:
            opacities_flat = opacities

        # Colors: shape (N,3)
        if colors.ndim != 2 or colors.shape[1] != 3:
            raise ValueError(f"colors must be (N,3), got {tuple(colors.shape)}")

        # Rasterizer returns at least an RGB image and radii; some forks also
        # return depth/alpha. We handle both gracefully.
        out = self._rasterizer(  # type: ignore[operator]
            means3D=positions,
            means2D=screenspace_points,
            shs=None,
            colors_precomp=colors,
            opacities=opacities_flat,
            scales=scales_arg,
            rotations=rotations_arg,
            cov3D_precomp=cov3D_precomp,
        )

        if isinstance(out, (list, tuple)):
            if len(out) == 2:
                rendered_image, radii = out
                depth = None
                alpha = None
            elif len(out) == 3:
                rendered_image, radii, depth = out
                alpha = None
            elif len(out) >= 4:
                rendered_image, depth, alpha, radii = out[0], out[1], out[2], out[3]
            else:
                raise RuntimeError("Unexpected 3DGS rasterizer output tuple length")
        elif isinstance(out, torch.Tensor):
            rendered_image = out
            depth = None
            alpha = None
        else:
            raise RuntimeError(f"Unexpected 3DGS rasterizer output type: {type(out)}")

        # 3DGS usually returns CHW; we want HWC to match your Python renderer.
        if rendered_image.dim() == 3 and rendered_image.shape[0] in (1, 3, 4):
            rendered_image = rendered_image.permute(1, 2, 0).contiguous()

        if depth is not None and depth.dim() == 2:
            depth = depth.unsqueeze(-1)
        if alpha is not None and alpha.dim() == 2:
            alpha = alpha.unsqueeze(-1)

        return rendered_image, depth, alpha

    # ------------------------------------------------------------------
    # 4c. Fallback CUDA path (very simple splatter, not true Gaussians)
    # ------------------------------------------------------------------

    def _render_fallback(
        self,
        *,
        positions: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        cam_to_world: torch.Tensor,
        cfg: RF3DGSRenderConfig,
        bg_color: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Very simple “nearest-splat” renderer as a safety net.

        This is *not* a proper anisotropic Gaussian renderer; it just
        projects 3D points onto the image plane and splats to the nearest
        pixel. It’s only meant to keep experiments running without the
        3DGS kernel.
        """
        H, W = cfg.height, cfg.width
        device = positions.device

        # World -> camera
        world_view, _ = _build_camera_transforms(cam_to_world, cfg)

        # Homogeneous positions (N,4)
        ones = torch.ones((positions.shape[0], 1), device=device, dtype=positions.dtype)
        pts_h = torch.cat([positions, ones], dim=-1)  # (N,4)
        pts_cam = (world_view @ pts_h.T).T  # (N,4)

        # Simple pinhole projection using vertical FOV
        tan_half = math.tan(0.5 * cfg.fov_y_radians)
        x = pts_cam[:, 0] / (-pts_cam[:, 2] * tan_half)
        y = pts_cam[:, 1] / (-pts_cam[:, 2] * tan_half)

        # Map NDC-ish [-1,1] to pixel coords
        px = (x * 0.5 + 0.5) * (W - 1)
        py = (y * 0.5 + 0.5) * (H - 1)

        # Mask out points behind camera
        valid = pts_cam[:, 2] < 0

        px = px[valid]
        py = py[valid]
        colors = colors[valid]
        opacities = opacities[valid]

        img = bg_color.view(1, 1, 3).expand(H, W, 3).clone()

        # Simple over operator in arbitrary order
        for i in range(px.numel()):
            xi = int(px[i].clamp(0, W - 1).item())
            yi = int(py[i].clamp(0, H - 1).item())
            a = float(opacities[i].item())
            c = colors[i]
            img[yi, xi] = a * c + (1.0 - a) * img[yi, xi]

        depth = None
        alpha_img = None
        return img, depth, alpha_img


if __name__ == '__main__':
    # Quick smoke test when run directly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rnd = CUDAGaussianRenderer(device=device, use_3dgs_if_available=False)
    pts = torch.tensor([[0.0, 0.0, -1.0], [0.2, 0.1, -1.2]], device=device)
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device)
    ops = torch.tensor([1.0, 0.8], device=device)
    cam = torch.eye(4, device=device)
    im, depth, alpha = rnd.render(positions_3d=pts, colors=colors, opacities=ops, camera_matrix=cam, width=64, height=64, fov_y=math.radians(60.0))
    print('Rendered image shape:', im.shape)
