"""
src/layerwise.py — Layer-wise sensitivity & selective merging (Eje E / H5).

Implements:
  - Per-layer barrier B^(l)
  - Selective merging (merge only resilient layers)
"""

import copy
import torch
import torch.nn as nn
import numpy as np

from .metrics import _eval_loss, _eval_acc


def compute_layerwise_barrier(model_class, model_kwargs, sd_a, sd_b,
                              loader, device):
    """
    Compute the per-layer barrier B^(l):

        B^(l) = L(θ_A^(<l), (θ_A^(l)+θ_B^(l))/2, θ_A^(>l)) - L(θ_A)

    For each layer l, we replace ONLY that layer's params with the
    midpoint, keeping all other layers from model A.

    Returns
    -------
    barriers : dict[str, float] — layer_prefix → barrier value
    base_loss : float — L(θ_A)
    """
    criterion = nn.CrossEntropyLoss()

    # Base loss
    model = model_class(**model_kwargs).to(device)
    model.load_state_dict(sd_a)
    base_loss = _eval_loss(model, loader, device, criterion)

    # Group params by layer prefix (e.g., "layer0", "conv1", "bn1")
    prefixes = set()
    for key in sd_a:
        parts = key.split(".")
        if len(parts) >= 2:
            prefixes.add(parts[0])
        else:
            prefixes.add(key)

    barriers = {}
    for prefix in sorted(prefixes):
        # Create state dict with only this layer interpolated
        sd_mixed = {}
        for key in sd_a:
            if key.startswith(prefix + ".") or key == prefix:
                # Interpolate this layer
                if sd_a[key].dtype in (torch.long, torch.int32, torch.int64):
                    sd_mixed[key] = sd_a[key]
                else:
                    sd_mixed[key] = 0.5 * sd_a[key].float() + \
                                    0.5 * sd_b[key].float()
            else:
                sd_mixed[key] = sd_a[key]

        model.load_state_dict(sd_mixed)
        mixed_loss = _eval_loss(model, loader, device, criterion)
        barriers[prefix] = mixed_loss - base_loss

    return barriers, base_loss


def selective_merge(sd_a, sd_b, barriers, threshold=None,
                    top_k_resilient=None):
    """
    Merge only the layers whose barrier B^(l) is below threshold,
    keeping the rest from model A (dominant model).

    Either threshold or top_k_resilient must be specified.

    Parameters
    ----------
    sd_a, sd_b : state dicts
    barriers : dict from compute_layerwise_barrier
    threshold : float — merge layers with B^(l) < threshold
    top_k_resilient : int — merge the top-k layers with lowest barrier

    Returns
    -------
    merged_sd : state dict
    merged_layers : list of merged layer prefixes
    """
    if threshold is not None:
        merge_prefixes = {k for k, v in barriers.items() if v < threshold}
    elif top_k_resilient is not None:
        sorted_layers = sorted(barriers.items(), key=lambda x: x[1])
        merge_prefixes = {k for k, _ in sorted_layers[:top_k_resilient]}
    else:
        raise ValueError("Specify threshold or top_k_resilient")

    merged_sd = {}
    for key in sd_a:
        prefix = key.split(".")[0]
        if prefix in merge_prefixes:
            if sd_a[key].dtype in (torch.long, torch.int32, torch.int64):
                merged_sd[key] = sd_a[key]
            else:
                merged_sd[key] = 0.5 * sd_a[key].float() + \
                                 0.5 * sd_b[key].float()
        else:
            merged_sd[key] = sd_a[key]

    return merged_sd, sorted(merge_prefixes)


def pareto_sweep(model_class, model_kwargs, sd_a, sd_b, barriers,
                 loader, device):
    """
    Sweep number of merged layers from 0 to all, measuring accuracy.
    Returns data for a Pareto curve: n_merged_layers vs accuracy.

    Returns
    -------
    n_layers_merged : list[int]
    accuracies : list[float]
    losses : list[float]
    layer_order : list[str] — layers in order of increasing barrier
    """
    criterion = nn.CrossEntropyLoss()
    sorted_layers = sorted(barriers.items(), key=lambda x: x[1])
    layer_order = [k for k, _ in sorted_layers]

    n_layers_merged = []
    accuracies = []
    losses = []

    for n in range(len(layer_order) + 1):
        if n == 0:
            sd_test = sd_a
        else:
            sd_test, _ = selective_merge(
                sd_a, sd_b, barriers, top_k_resilient=n)

        model = model_class(**model_kwargs).to(device)
        model.load_state_dict(sd_test)
        loss = _eval_loss(model, loader, device, criterion)
        acc = _eval_acc(model, loader, device)

        n_layers_merged.append(n)
        accuracies.append(acc)
        losses.append(loss)

    return n_layers_merged, accuracies, losses, layer_order


# ═════════════════════════════════════════════════════════════════════════════
# SNR proxy via Random Matrix Theory (Yu et al. / Spectrum, 2024)
# ═════════════════════════════════════════════════════════════════════════════

def snr_layerwise(model):
    """
    Compute a per-layer SNR proxy using Random Matrix Theory.

    For each weight matrix W^(l) of shape (m, n), compute eigenvalues of
    W^T W. The Marchenko-Pastur law predicts the bulk edge at:

        λ_+ = σ² (1 + √(m/n))²

    where σ² = Tr(W^T W) / (m*n) (estimated noise variance).

    Eigenvalues above λ_+ are "signal"; those below are "noise".

    SNR = (sum of signal eigenvalues) / (sum of noise eigenvalues)

    High SNR → layer has learned specialized features → harder to merge.
    Low SNR → layer is less specialized → easier to merge.

    Returns
    -------
    snr_dict : dict[str, float] — layer_name → SNR value
    mp_info : dict[str, dict] — layer_name → {n_signal, n_noise, lambda_plus}
    """
    snr_dict = {}
    mp_info = {}

    for name, param in model.named_parameters():
        if "weight" not in name or param.dim() < 2:
            continue

        W = param.detach().float()
        if W.dim() == 4:
            # Conv: reshape (C_out, C_in, kH, kW) → (C_out, C_in*kH*kW)
            W = W.reshape(W.shape[0], -1)

        m, n = W.shape
        if m > n:
            cov = W.T @ W  # (n, n)
        else:
            cov = W @ W.T  # (m, m)

        eigenvalues = torch.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues.clamp(min=0)  # numerical stability

        # Marchenko-Pastur threshold
        gamma = max(m, n) / min(m, n)
        sigma_sq = eigenvalues.sum().item() / (m * n)
        lambda_plus = sigma_sq * (1 + gamma ** 0.5) ** 2

        signal_mask = eigenvalues > lambda_plus
        n_signal = signal_mask.sum().item()
        n_noise = (~signal_mask).sum().item()

        signal_energy = eigenvalues[signal_mask].sum().item()
        noise_energy = eigenvalues[~signal_mask].sum().item()

        snr = signal_energy / max(noise_energy, 1e-12)

        # Use layer prefix as key
        layer_prefix = name.rsplit(".weight", 1)[0]
        snr_dict[layer_prefix] = snr
        mp_info[layer_prefix] = {
            "n_signal": int(n_signal),
            "n_noise": int(n_noise),
            "lambda_plus": float(lambda_plus),
            "snr": float(snr),
        }

    return snr_dict, mp_info
