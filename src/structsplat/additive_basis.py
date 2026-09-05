"""Bounded, fixed-geometry additive basis caching (HIER-034, ADR-0006).

The cache owns detached copies of pixel/row/weight triplets. It is a linear color operator,
not a geometry-differentiable renderer. Rebuild after any geometry, support, mask, or row change.
Torch is imported lazily. The byte limit bounds retained tensors, not construction workspace;
callers measuring peak memory must include construction and sparse conversion.
"""

from __future__ import annotations

from collections.abc import Iterable
import time


class CachedAdditiveBasis:
    """Cache an iterator of (local_row, flat_pixel, scalar_weight) tensors.

    Each (row,pixel) pair must occur once, as in the renderer's tile traversal.
    Both backends represent exactly the supplied sparse linear map. ``scatter`` preserves
    producer chunk/order; ``csr`` uses PyTorch sparse matrix products and can change reduction
    order. Exceeding the retained-storage budget raises before returning a usable cache.
    """

    def __init__(self, triplets: Iterable, *, rows: int, pixels: int,
                 reference, mode: str = "scatter", max_bytes: int = 268435456):
        import torch

        if reference.device.type == "cuda":
            torch.cuda.synchronize(reference.device)
        started = time.perf_counter()
        if mode not in ("scatter", "csr"):
            raise ValueError("cache mode must be scatter or csr")
        for name, value in (("rows", rows), ("pixels", pixels), ("max_bytes", max_bytes)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not reference.is_floating_point():
            raise ValueError("reference must have floating dtype")
        self.rows, self.pixels, self.mode = rows, pixels, mode
        self.device, self.dtype = reference.device, reference.dtype
        self._chunks = []
        self._matrix = self._transpose = None
        self.nnz = self.resident_bytes = 0
        with torch.no_grad():
            for local, flat, weight in triplets:
                if local.ndim != 1 or flat.shape != local.shape or weight.shape != local.shape:
                    raise ValueError("triplets must be equal-length one-dimensional tensors")
                if local.dtype != torch.int64 or flat.dtype != torch.int64:
                    raise ValueError("triplet indices must be int64")
                if any(t.device != self.device for t in (local, flat, weight)):
                    raise ValueError("triplets and reference must share a device")
                if weight.dtype != self.dtype or not bool(torch.isfinite(weight).all()):
                    raise ValueError("weights must be finite and match reference dtype")
                if local.numel() and (
                    int(local.min()) < 0 or int(local.max()) >= rows
                    or int(flat.min()) < 0 or int(flat.max()) >= pixels
                ):
                    raise ValueError("triplet index outside operator shape")
                keep = weight != 0
                count = int(keep.sum())
                retained = count * (16 + weight.element_size())
                if self.resident_bytes + retained > max_bytes:
                    raise MemoryError("additive basis exceeds retained cache byte budget")
                if count:
                    self._chunks.append(tuple(t[keep].detach().clone() for t in (local, flat, weight)))
                self.resident_bytes += retained
                self.nnz += count
            if mode == "csr":
                # Both CSR directions retain values and indices, plus their row pointers.
                estimate = 2 * self.nnz * (8 + reference.element_size()) + 8 * (rows + pixels + 2)
                if estimate > max_bytes:
                    raise MemoryError("CSR basis exceeds retained cache byte budget")
                if self._chunks:
                    local, flat, weight = (torch.cat(parts) for parts in zip(*self._chunks))
                else:
                    local = flat = torch.empty(0, device=self.device, dtype=torch.int64)
                    weight = reference.new_empty(0)
                matrix = torch.sparse_coo_tensor(
                    torch.stack((flat, local)), weight, (pixels, rows),
                    device=self.device, dtype=self.dtype,
                ).coalesce()
                self._matrix = matrix.to_sparse_csr()
                self._transpose = matrix.transpose(0, 1).coalesce().to_sparse_csr()
                self._chunks = []
                self.resident_bytes = sum(
                    t.numel() * t.element_size()
                    for m in (self._matrix, self._transpose)
                    for t in (m.crow_indices(), m.col_indices(), m.values())
                )
                self.nnz = self._matrix.values().numel()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        self.build_seconds = time.perf_counter() - started

    def _check_values(self, values, size):
        if values.ndim != 2 or values.shape[0] != size:
            raise ValueError("values must be a matrix with the operator's leading dimension")
        if values.device != self.device or values.dtype != self.dtype:
            raise ValueError("values must match cache device and dtype")

    def apply(self, values):
        """Apply B to an (N,C) coefficient matrix, returning (pixels,C)."""
        import torch

        self._check_values(values, self.rows)
        if self.mode == "csr":
            return torch.sparse.mm(self._matrix, values)
        result = values.new_zeros((self.pixels, values.shape[1]))
        for local, flat, weight in self._chunks:
            result.index_add_(0, flat, weight[:, None] * values[local])
        return result

    def transpose(self, image):
        """Apply B transpose to a (pixels,C) matrix, returning (N,C)."""
        import torch

        self._check_values(image, self.pixels)
        if self.mode == "csr":
            return torch.sparse.mm(self._transpose, image)
        result = image.new_zeros((self.rows, image.shape[1]))
        for local, flat, weight in self._chunks:
            result.index_add_(0, local, weight[:, None] * image[flat])
        return result

    def normal_diagonal(self):
        """Return diag(B transpose B) for the unique-incidence producer contract."""
        import torch

        if self.mode == "csr":
            squared = torch.sparse_csr_tensor(
                self._transpose.crow_indices(), self._transpose.col_indices(),
                self._transpose.values().square(), self._transpose.shape,
            )
            ones = torch.ones((self.pixels, 1), device=self.device, dtype=self.dtype)
            return torch.sparse.mm(squared, ones).flatten()
        # The producer emits each row/pixel pair once. This is the renderer's diagonal rule.
        result = torch.zeros(self.rows, device=self.device, dtype=self.dtype)
        for local, _flat, weight in self._chunks:
            result.index_add_(0, local, weight.square())
        return result
