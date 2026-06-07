"""
LMC Project — Package init

Este archivo centraliza y expone la API global del proyecto 'src'.
Para el correcto funcionamiento del Notebook 01, se mantienen activos los 
módulos base y comentados (#) los módulos que se programarán en las fases posteriores.
"""

# =====================================================================
# MÓDULOS ACTIVOS (Operativos para el Notebook 01)
# =====================================================================
from .utils import set_seed, get_device, get_dataloaders, get_calibration_loader
from .models import MLP3, SimpleConvBN, ConvMixer, get_model, get_mobilenetv3
from .training import (
    train_one_epoch, evaluate, train_model, train_spawned_pair,
    run_multi_seed,
)

# =====================================================================
# MÓDULOS FUTUROS (Comentados temporalmente para evitar ModuleNotFoundError)
# =====================================================================

# ── Activar para Notebook 02 y 02b (Eje A — Alignment) ──
# from .alignment import (
#     weight_matching,
#     activation_matching,
#     procrustes_alignment,
#     apply_permutation_to_state_dict,
#     apply_transform_to_state_dict,
#     cycle_consistency_error,
# )

# ── Activar para Notebook 03 (Eje B — REPAIR) ──
# from .repair import (
#     collect_activation_stats,
#     variance_collapse_ratio,
#     repair_full,
#     repair_bn_recalibration,
#     repair_reset_retrain_bn,
#     repair_layerwise,
#     variance_collapse_vs_alpha,
# )

# ── Activar para Notebook 04 (Eje C — Métricas e Interpolación) ──
# from .metrics import (
#     interpolate_state_dicts,
#     make_interpolated_model,
#     compute_barrier,
#     cka_linear,
#     compute_cka_matrix,
# )

# ── Activar para Notebook 04 y 04b (Eje C — Loss Landscapes) ──
from .landscape import (
    compute_2d_loss_landscape,
    plot_4panel_landscape,
    plot_barrier_curves,
)

# ── Activar para Notebook 05 (Eje C — Hessian Sharpness) ──
# from .hessian import (
#     top_k_eigenvalues,
#     hutchinson_trace,
#     sharpness_report,
# )

# ── Activar para Notebook 06 (Eje D — Multi-Model Merging) ──
# from .multimodel import (
#     train_n_models,
#     naive_mean_merge,
#     anchor_aligned_merge,
#     iterative_pairwise_merge,
#     ties_merge,
#     slerp_barycenter_merge,
#     compute_multimodel_barrier,
# )

# ── Activar para Notebook 07 (Eje E — Layerwise Sensitivity) ──
# from .layerwise import (
#     compute_layerwise_barrier,
#     selective_merge,
#     pareto_sweep,
#     snr_layerwise,
# )