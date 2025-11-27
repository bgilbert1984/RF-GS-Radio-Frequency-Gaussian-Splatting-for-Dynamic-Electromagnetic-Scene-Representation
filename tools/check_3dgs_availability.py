#!/usr/bin/env python3
"""Quick check for diff_gaussian_rasterization availability."""
import sys

try:
    import diff_gaussian_rasterization
    from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings
    print("✅ diff_gaussian_rasterization is available.")
    sys.exit(0)
except Exception as e:
    print("❌ diff_gaussian_rasterization NOT available.")
    print("Error:", repr(e))
    sys.exit(2)
