"""
src/hessian.py — Hessian spectrum analysis (Eje C / H6).

Implements:
  - Power iteration for top-k eigenvalues of the Hessian
  - Hutchinson trace estimator
  - Convenience wrappers for evaluating sharpness at any point
"""

import torch
import torch.nn as nn
import numpy as np


def _hessian_vector_product(loss_fn, params, v):
    """
    Compute Hv (Hessian-vector product) via two backward passes.
    loss_fn: callable that returns a scalar loss (must create graph).
    params: list of tensors (model parameters).
    v: list of tensors (same shapes as params), the direction.
    """
    loss = loss_fn()
    grads = torch.autograd.grad(loss, params, create_graph=True)

    # Compute dot product grad · v
    dot = sum((g * vi).sum() for g, vi in zip(grads, v))

    # Second backward: d(dot)/d(params) = Hv
    hvp = torch.autograd.grad(dot, params)
    return [h.detach() for h in hvp]


def top_k_eigenvalues(model, loader, device, criterion=None,
                      k=5, max_iter=100, tol=1e-4):
    """
    Compute the top-k eigenvalues of the Hessian via power iteration
    with deflation.

    Returns
    -------
    eigenvalues : list of k floats (descending order)
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]

    # Build loss function over a mini-batch
    data_x, data_y = [], []
    n_samples = 0
    for x, y in loader:
        data_x.append(x.to(device))
        data_y.append(y.to(device))
        n_samples += x.size(0)
        if n_samples >= 512:
            break
    data_x = torch.cat(data_x)[:512]
    data_y = torch.cat(data_y)[:512]

    def loss_fn():
        return criterion(model(data_x), data_y)

    eigenvalues = []
    eigenvectors = []

    for ki in range(k):
        # Random init
        v = [torch.randn_like(p) for p in params]
        norm = sum((vi ** 2).sum() for vi in v).sqrt()
        v = [vi / norm for vi in v]

        eigenval = 0.0
        for it in range(max_iter):
            # Hv
            hv = _hessian_vector_product(loss_fn, params, v)

            # Deflation: remove components along previous eigenvectors
            for prev_v, prev_lam in zip(eigenvectors, eigenvalues):
                dot = sum((h * pv).sum() for h, pv in zip(hv, prev_v))
                hv = [h - dot * pv for h, pv in zip(hv, prev_v)]

            # Rayleigh quotient
            new_eigenval = sum((h * vi).sum() for h, vi in zip(hv, v)).item()

            # Normalize
            norm = sum((h ** 2).sum() for h in hv).sqrt()
            v = [h / norm.clamp(min=1e-12) for h in hv]

            if abs(new_eigenval - eigenval) < tol:
                eigenval = new_eigenval
                break
            eigenval = new_eigenval

        eigenvalues.append(eigenval)
        eigenvectors.append(v)

    return eigenvalues


def hutchinson_trace(model, loader, device, criterion=None,
                     n_probes=10):
    """
    Estimate Tr(H) using Hutchinson's stochastic trace estimator:
        Tr(H) ≈ (1/N) Σ_i v_i^T H v_i,  v_i ~ N(0, I)

    Returns
    -------
    trace_estimate : float
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]

    data_x, data_y = [], []
    n_samples = 0
    for x, y in loader:
        data_x.append(x.to(device))
        data_y.append(y.to(device))
        n_samples += x.size(0)
        if n_samples >= 512:
            break
    data_x = torch.cat(data_x)[:512]
    data_y = torch.cat(data_y)[:512]

    def loss_fn():
        return criterion(model(data_x), data_y)

    traces = []
    for _ in range(n_probes):
        v = [torch.randn_like(p) for p in params]
        hv = _hessian_vector_product(loss_fn, params, v)
        tr = sum((vi * hi).sum() for vi, hi in zip(v, hv)).item()
        traces.append(tr)

    return np.mean(traces)


def sharpness_report(model, loader, device, criterion=None, k=5):
    """
    Compute a full sharpness report: top-k eigenvalues + trace.

    Returns
    -------
    dict with 'eigenvalues' (list), 'trace' (float), 'lambda_max' (float)
    """
    eigs = top_k_eigenvalues(model, loader, device, criterion, k=k)
    tr = hutchinson_trace(model, loader, device, criterion)
    return {
        "eigenvalues": eigs,
        "trace": tr,
        "lambda_max": max(eigs),
    }
