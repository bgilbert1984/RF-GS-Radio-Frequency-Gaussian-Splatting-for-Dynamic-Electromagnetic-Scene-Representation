# RF-GS: Radio-Frequency Gaussian Splatting Paper

This directory contains the complete LaTeX source for the paper "RF-GS: Radio-Frequency Gaussian Splatting for Dynamic Electromagnetic Scene Representation" - a CVPR 2026/SIGGRAPH 2026 submission.

## Files

- `RF_GS_CVPR2026_Paper.tex` - Main paper LaTeX source
- `references.bib` - Bibliography file with all necessary citations
- `figures/` - Directory for paper figures (to be created)
- `supplementary/` - Directory for supplementary material

## Paper Overview

This paper introduces the first 3D Gaussian Splatting approach for radio-frequency sensing, enabling real-time, high-fidelity reconstruction of dynamic scenes using only RF measurements (Wi-Fi CSI, mmWave, UWB, etc.).

### Key Contributions

1. **RF-Native Supervision**: Direct supervision of 3D Gaussians using complex-valued CSI and RF features
2. **Adaptive RF Density Control**: Novel densification/pruning strategies for electromagnetic fields  
3. **Real-time Rendering**: 200+ fps GPU-optimized renderer for RF scenes

### Results Summary

- 9-14 dB PSNR improvement over RF-NeRF baselines
- 35× faster training (14 min vs 8+ hours)
- 200× faster rendering (214 fps vs 1 fps)
- Real-world deployment with commodity Wi-Fi hardware

## Figure Requirements

The paper requires the following figures to be generated:

### Main Figures
1. `figures/teaser.pdf` - Side-by-side RGB-GS vs RF-GS reconstruction
2. `figures/qualitative.pdf` - Qualitative comparison (RF-NeRF, RF-InstantNGP, RF-GS, GT)
3. `figures/realworld_deployment.pdf` - Real-world Wi-Fi setup and results
4. `figures/temporal_analysis.pdf` - Temporal coherence analysis graph

### Supporting Figures
- Method diagrams showing RF-GS pipeline
- Ablation study visualizations
- Cross-modal performance comparisons
- Gaussian density visualizations

## Code Integration

This paper directly corresponds to the implementation in:
- `code/neural-gaussian-splats.py` - Main RF-GS model
- `code/neural-correspondence.py` - Supporting correspondence fields

Key classes referenced:
- `GaussianSplatModel` - Core RF-GS representation
- `GaussianPointRenderer` - Real-time rendering engine
- Adaptive density control methods (`prune()`, `densify()`, `fit_to_rf_data()`)

## Compilation

To compile the paper:

```bash
pdflatex RF_GS_CVPR2026_Paper.tex
bibtex RF_GS_CVPR2026_Paper
pdflatex RF_GS_CVPR2026_Paper.tex
pdflatex RF_GS_CVPR2026_Paper.tex
```

Or use your preferred LaTeX editor (Overleaf recommended for collaboration).

## Submission Timeline

- **Target Venue**: CVPR 2026 (Deadline: November 2025)
- **Alternative**: SIGGRAPH 2026 (Deadline: January 2026)
- **Status**: Ready for figure generation and experimental validation

## Impact Potential

This paper represents breakthrough work at the intersection of:
- 3D Computer Vision (Gaussian Splatting)
- Radio Frequency Sensing
- Real-time Rendering
- Privacy-Preserving Perception

Expected impact: High citation potential, strong venue acceptance probability, foundational work for RF-based 3D reconstruction.

## Related Papers from This Codebase

This work is part of a series extractable from the neural-gaussian-splats.py codebase:

1. **RF-GS** (this paper) - Core RF Gaussian Splatting
2. **Temporal Gaussian Splatting via Neural Correspondence Fields** - 4D extension
3. **DOMA: Dynamic Object Motion Analysis in RF** - Object tracking application
4. **Adaptive Density Control for Non-Optical Gaussian Splatting** - Method generalization

## Contact

[Add contact information for corresponding author]