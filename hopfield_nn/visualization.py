"""Visualization helpers for photonic Hopfield experiments."""

import numpy as np
import matplotlib.pyplot as plt

from .network import PhotonicHopfieldNetwork


def visualize_patterns(patterns, title: str = "Stored Patterns"):
    """Show stored patterns."""
    fig, ax = plt.subplots(figsize=(max(8, patterns.shape[1]), 
                                    max(2, patterns.shape[0] * 0.5)))
    im = ax.imshow(patterns, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
    ax.set_xlabel('Node Index')
    ax.set_ylabel('Pattern Index')
    ax.set_title(title)
    ax.set_xticks(range(patterns.shape[1]))
    ax.set_yticks(range(patterns.shape[0]))
    plt.colorbar(im, ax=ax, ticks=[-1, 0, 1], label='State')
    return fig


def visualize_recall_evolution(history, 
                               target: np.ndarray = None,
                               title = "Recall Evolution"):
    """Show recall steps."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # State evolution
    ax = axes[0]
    im = ax.imshow(history.T, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Node Index')
    ax.set_title('State Evolution')
    plt.colorbar(im, ax=ax, ticks=[-1, 0, 1])
    
    # Overlap with target
    if target is not None:
        ax = axes[1]
        overlaps = [np.mean(h == target) for h in history]
        ax.plot(overlaps, 'b-o', linewidth=2, markersize=6)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Overlap with Target')
        ax.set_title('Convergence')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig


def visualize_weight_matrix(W, title: str = "Weight Matrix"):
    """Show weight matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(W, cmap='RdBu', vmin=-np.max(np.abs(W)), vmax=np.max(np.abs(W)))
    ax.set_xlabel('Node j')
    ax.set_ylabel('Node i')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='W_ij')
    return fig


def visualize_energy_landscape(network: PhotonicHopfieldNetwork,
                               pattern_idx = 0):
    """Visualize energy landscape around a stored pattern (for N <= 8)"""
    if network.N > 8:
        print("Energy landscape visualization only for N <= 8")
        return None
    
    # Get all states at each Hamming distance from pattern
    pattern = network.patterns[pattern_idx]
    
    distances = []
    energies = []
    
    # Generate all 2^N states
    for i in range(2**network.N):
        state = np.array([1 if (i >> j) & 1 else -1 for j in range(network.N)])
        d = np.sum(state != pattern)
        E = network.compute_energy(state)
        distances.append(d)
        energies.append(E)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(distances, energies, alpha=0.5, s=50)
    ax.axhline(y=network.compute_energy(pattern), color='r', 
               linestyle='--', label='Pattern energy')
    ax.set_xlabel('Hamming Distance from Pattern')
    ax.set_ylabel('Hopfield Energy')
    ax.set_title('Energy Landscape')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig
