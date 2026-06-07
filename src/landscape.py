import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from src.metrics import _eval_loss, _eval_acc

def plot_barrier_curves(curves_dict, save_path=None, figsize=(14, 5)):
    """
    Dibuja las curvas de pérdida (Loss) y precisión (Accuracy) a lo largo
    del camino de interpolación lineal para varios métodos de alineación.
    
    Args:
        curves_dict (dict): Diccionario donde las llaves son los nombres 
                            de los métodos y los valores son los diccionarios 
                            de curvas generados por compute_barrier.
                            Ej: {'Naive': {'alphas': [...], 'losses': [...], 'accs': [...]}}
        save_path (str, optional): Ruta donde guardar la figura.
        figsize (tuple, optional): Tamaño de la figura.
    """
    # Creamos una figura con dos subgráficos: Loss a la izquierda, Acc a la derecha
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Iteramos sobre cada método en el diccionario y dibujamos sus líneas
    for name, curve in curves_dict.items():
        alphas = curve['alphas']
        losses = curve['losses']
        # Convertimos la precisión a porcentaje
        accs = [a * 100 for a in curve['accs']] 
        
        ax1.plot(alphas, losses, marker='o', label=name, linewidth=2)
        ax2.plot(alphas, accs, marker='s', label=name, linewidth=2)
        
    # --- Formateo del gráfico de Pérdida (Loss) ---
    ax1.set_title("Barrera de Pérdida (Loss Landscape)", fontweight='bold')
    ax1.set_xlabel(r"Coeficiente de Interpolación ($\alpha$)")
    ax1.set_ylabel("Test Loss")
    ax1.set_xticks(alphas)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # --- Formateo del gráfico de Precisión (Accuracy) ---
    ax2.set_title("Barrera de Precisión (Accuracy Landscape)", fontweight='bold')
    ax2.set_xlabel(r"Coeficiente de Interpolación ($\alpha$)")
    ax2.set_ylabel("Test Accuracy (%)")
    ax2.set_xticks(alphas)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower center')
    
    plt.tight_layout()
    
    # Guardamos la figura si se proporciona una ruta
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def compute_2d_loss_landscape(
    model_class,
    model_kwargs,
    sd_a,
    sd_b,
    loader,
    device,
    grid_size=15,
    margin=0.3,
):
    """
    Computes the loss and accuracy over a 2D grid spanned by two models.
    
    The plane is defined by the midpoint and direction vectors:
        midpoint = (sd_a + sd_b) / 2
        u = sd_a - midpoint
        v = sd_b - midpoint
        
    Each grid point (alpha, beta) defines:
        theta(alpha, beta) = midpoint + alpha * u + beta * v
        
    Parameters
    ----------
    model_class : type
        The model class to instantiate.
    model_kwargs : dict
        Arguments for the model constructor.
    sd_a, sd_b : dict
        State dicts of the two models.
    loader : DataLoader
        DataLoader for evaluation.
    device : torch.device
        Device to run evaluation on.
    grid_size : int
        Number of points along each dimension of the grid.
    margin : float
        Margin to extend the grid beyond [0, 1] range.
        
    Returns
    -------
    alphas : np.ndarray
        2D grid of alpha coordinates.
    betas : np.ndarray
        2D grid of beta coordinates.
    loss_grid : np.ndarray
        2D grid of evaluated loss values.
    acc_grid : np.ndarray
        2D grid of evaluated accuracy values.
    """
    # 1. Define grid coordinates
    alpha_vals = np.linspace(-margin, 1.0 + margin, grid_size)
    beta_vals = np.linspace(-margin, 1.0 + margin, grid_size)
    alphas, betas = np.meshgrid(alpha_vals, beta_vals)
    
    loss_grid = np.zeros((grid_size, grid_size))
    acc_grid = np.zeros((grid_size, grid_size))
    
    # 2. Setup baseline state dicts on the target device to avoid CPU-GPU roundtrips
    sd_a_dev = {k: v.to(device) for k, v in sd_a.items()}
    sd_b_dev = {k: v.to(device) for k, v in sd_b.items()}
    
    # 3. Compute midpoint and direction vectors
    sd_mid = {}
    u = {}
    v = {}
    for k in sd_a_dev.keys():
        if sd_a_dev[k].dtype in (torch.long, torch.int32, torch.int64):
            sd_mid[k] = sd_a_dev[k]
        else:
            sd_mid[k] = 0.5 * (sd_a_dev[k].float() + sd_b_dev[k].float())
            u[k] = sd_a_dev[k].float() - sd_mid[k]
            v[k] = sd_b_dev[k].float() - sd_mid[k]
            
    # 4. Instantiate model
    model = model_class(**model_kwargs).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # 5. Evaluate over the grid
    for i in range(grid_size):
        for j in range(grid_size):
            alpha = alphas[i, j]
            beta = betas[i, j]
            
            sd_grid = {}
            for k in sd_a_dev.keys():
                if sd_a_dev[k].dtype in (torch.long, torch.int32, torch.int64):
                    sd_grid[k] = sd_mid[k]
                else:
                    sd_grid[k] = sd_mid[k] + alpha * u[k] + beta * v[k]
                    
            model.load_state_dict(sd_grid)
            loss_grid[i, j] = _eval_loss(model, loader, device, criterion)
            acc_grid[i, j] = _eval_acc(model, loader, device)
            
    return alphas, betas, loss_grid, acc_grid


def plot_4panel_landscape(panels_data, save_path=None, figsize=(14, 12)):
    """
    Plots a 2x2 grid of 2D loss landscapes for comparison.
    
    Parameters
    ----------
    panels_data : list of dict
        A list of 4 dictionaries, each representing a panel and containing:
        'title': str
        'alphas': np.ndarray (2D grid of alpha coordinates)
        'betas': np.ndarray (2D grid of beta coordinates)
        'loss_grid': np.ndarray (2D grid of loss values)
        'acc_grid': np.ndarray (2D grid of accuracy values)
    save_path : str, optional
        Path where to save the generated figure.
    figsize : tuple, optional
        Size of the output figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()
    
    # Calculate unified min/max log loss across all panels to align colors
    log_losses = []
    for data in panels_data:
        # Avoid log of non-positive numbers
        log_losses.append(np.log10(np.clip(data["loss_grid"], 1e-8, None)))
        
    vmin = min(l.min() for l in log_losses)
    vmax = max(l.max() for l in log_losses)
    if vmax - vmin < 1e-5:
        vmin = vmin - 0.5
        vmax = vmax + 0.5
    
    levels = np.linspace(vmin, vmax, 35)
    
    for idx, (ax, data) in enumerate(zip(axes, panels_data)):
        alphas = data["alphas"]
        betas = data["betas"]
        log_loss = log_losses[idx]
        
        # Draw filled contours with common levels and RdYlBu_r cmap (Li et al. 2018 standard)
        cf = ax.contourf(alphas, betas, log_loss, levels=levels, cmap="RdYlBu_r", extend="both")
        # Add black contour lines to highlight topology
        ax.contour(alphas, betas, log_loss, levels=levels[::2], colors="k", linewidths=0.5, alpha=0.3)
        
        # Plot diagonal linear interpolation path from (1,0) to (0,1)
        ax.plot([1.0, 0.0], [0.0, 1.0], color="white", linestyle="--", linewidth=2, label="Interpolation Path")
        
        # Plot markers
        # theta_A at (1, 0)
        ax.plot(1.0, 0.0, "^", color="black", markersize=10, markeredgecolor="white", label=r"$\theta_A$")
        # theta_B at (0, 1)
        ax.plot(0.0, 1.0, "s", color="black", markersize=9, markeredgecolor="white", label=r"$\theta_B$")
        # Midpoint at (0.5, 0.5)
        ax.plot(0.5, 0.5, "*", color="black", markersize=14, markeredgecolor="white", label=r"$\bar{\theta}$")
        
        ax.set_title(data["title"], fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel(r"$\alpha$ (Model A direction)", fontsize=11)
        ax.set_ylabel(r"$\beta$ (Model B direction)", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.5)
        
        if idx == 0:
            ax.legend(loc="upper right", framealpha=0.9)
            
    # Adjust spacing and add global colorbar
    fig.tight_layout()
    fig.subplots_adjust(right=0.85, top=0.92)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(axes[0].collections[0], cax=cbar_ax)
    cbar.set_label(r"$\log_{10}(\mathrm{Loss})$", fontsize=13, labelpad=10)
    
    plt.suptitle("2D Loss Landscape Comparison", fontsize=18, fontweight="bold")
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        
    return fig