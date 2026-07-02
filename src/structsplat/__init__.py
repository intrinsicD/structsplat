"""StructSplat: hierarchical, feature-aware, anisotropic blue-noise 2D Gaussian images.

Package layout keeps init-time math (structure_tensor, density, sampling) NumPy-only so it
imports without torch; autograd pieces (gaussians, render, fit, pyramid, metrics) require torch
and are imported lazily on use.
"""

__version__ = "0.1.0"
