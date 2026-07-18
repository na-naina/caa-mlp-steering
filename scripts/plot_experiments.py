#!/usr/bin/env python3
"""Generate publication-quality plots for all experiment results."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

OUTDIR = "paper/figures/experiments"

import os
os.makedirs(OUTDIR, exist_ok=True)


def plot_bottleneck_sweep():
    """Bottleneck dimension sweep (bn=4,8,16,32,64)."""
    dims = [4, 8, 32, 64]
    truth = [79.7, 83.1, 86.5, 86.5]
    info = [96.6, 97.1, 89.7, 75.2]
    ti = [76.9, 80.6, 77.6, 65.1]
    params = [dim * 3584 * 2 + dim * 2 for dim in dims]  # rough param count
    params_k = [p / 1000 for p in params]

    fig, ax1 = plt.subplots(figsize=(6, 4))

    ax1.plot(dims, truth, 's--', color='#2196F3', linewidth=1.5, markersize=7, label='Truth%', alpha=0.7)
    ax1.plot(dims, info, '^--', color='#4CAF50', linewidth=1.5, markersize=7, label='Info%', alpha=0.7)
    ax1.plot(dims, ti, 'o-', color='#FF9800', linewidth=2.5, markersize=8, label='T*I%')

    ax1.set_xlabel('Bottleneck Dimension')
    ax1.set_ylabel('Score (%)')
    ax1.set_xticks(dims)
    ax1.set_ylim(50, 105)
    ax1.legend(loc='lower left')
    ax1.grid(True, alpha=0.3)

    # Annotate best
    best_idx = ti.index(max(ti))
    ax1.annotate(f'{ti[best_idx]:.1f}%', xy=(dims[best_idx], ti[best_idx]),
                 xytext=(5, 8), textcoords='offset points', ha='center', fontsize=9, fontweight='bold')

    ax1.set_title('Effect of Bottleneck Dimension on Performance')
    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/bottleneck_sweep.pdf')
    fig.savefig(f'{OUTDIR}/bottleneck_sweep.png')
    plt.close(fig)
    print(f'Saved bottleneck_sweep')


def plot_pool_size():
    """Pool size comparison: no-bank vs 2-fold CV."""
    pools = [1, 5, 10, 50, 100]

    # No bank
    nb_ti = [17.5, 74.8, 75.6, 78.1, 78.9]

    # 2-fold CV
    cv_ti = [32.6, 80.6, 69.7, 76.2, 80.7]

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(pools, nb_ti, 'o-', color='#2196F3', linewidth=2, markersize=7, label='No bank (single run)')
    ax.plot(pools, cv_ti, 's--', color='#FF9800', linewidth=2, markersize=7, label='2-fold CV')

    ax.set_xlabel('Steering Pool Size')
    ax.set_ylabel('T*I (%)')
    ax.set_xscale('log')
    ax.set_xticks(pools)
    ax.set_xticklabels([str(p) for p in pools])
    ax.set_ylim(10, 90)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('T*I vs Steering Pool Size')

    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/pool_size.pdf')
    fig.savefig(f'{OUTDIR}/pool_size.png')
    plt.close(fig)
    print(f'Saved pool_size')


def plot_training_size():
    """Training set size curve."""
    sizes = [50, 100, 200, 309]
    ti = [68.4, 69.4, 75.8, 80.6]
    truth = [70.5, 78.3, 83.2, 83.1]
    info = [97.0, 88.7, 91.1, 97.1]

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(sizes, ti, 'o-', color='#FF9800', linewidth=2.5, markersize=8, label='T*I%', zorder=3)
    ax.plot(sizes, truth, 's--', color='#2196F3', linewidth=1.5, markersize=6, label='Truth%', alpha=0.7)
    ax.plot(sizes, info, '^--', color='#4CAF50', linewidth=1.5, markersize=6, label='Info%', alpha=0.7)

    # Reference lines
    ax.axhline(y=77.4, color='red', linestyle=':', alpha=0.5, linewidth=1)
    ax.text(55, 78, 'RaLFiT SOTA (77.4%)', fontsize=8, color='red', alpha=0.7)

    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Score (%)')
    ax.set_ylim(55, 105)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Data Efficiency: Performance vs Training Examples')

    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/training_size.pdf')
    fig.savefig(f'{OUTDIR}/training_size.png')
    plt.close(fig)
    print(f'Saved training_size')


def plot_ablations():
    """Ablations and controls bar chart."""
    labels = [
        'Full method\n(bn=8)',
        'Noise\nablation',
        'Direct GD\n(CAA init)',
        'Direct GD\n(random)',
        'No bank\n(bn=16)',
        'Individual\nbank',
        'KL reg\n(0.01)',
    ]
    ti = [80.6, 61.8, 44.6, 41.1, 74.4, 76.8, 76.6]
    truth = [83.1, 64.0, 92.4, 84.3, 85.0, 79.9, 88.0]
    info = [97.1, 96.6, 48.3, 48.8, 87.5, 96.1, 87.0]

    colors = ['#FF9800' if i == 0 else '#90A4AE' for i in range(len(labels))]
    colors[1] = '#ef5350'  # noise = red-ish
    colors[2] = '#ef5350'  # direct GD
    colors[3] = '#ef5350'  # direct GD random

    fig, ax = plt.subplots(figsize=(8, 4.5))

    x = np.arange(len(labels))
    bars = ax.bar(x, ti, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)

    for i, (bar, val) in enumerate(zip(bars, ti)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9,
                fontweight='bold' if i == 0 else 'normal')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('T*I (%)')
    ax.set_ylim(0, 95)
    ax.axhline(y=80.6, color='#FF9800', linestyle='--', alpha=0.3, linewidth=0.8)
    ax.set_title('Ablation Study: T*I Score')
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/ablations.pdf')
    fig.savefig(f'{OUTDIR}/ablations.png')
    plt.close(fig)
    print(f'Saved ablations')


def plot_geometry():
    """Geometry findings visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel 1: The MLP correction is orthogonal to CAA
    ax = axes[0]
    categories = ['CAA vector\nvs MLP delta', 'MLP delta\npairwise\n(across inputs)', 'Individual\nvectors\npairwise']
    cosines = [0.03, 0.9997, 0.85]  # approximate values from earlier analysis
    colors_bar = ['#2196F3', '#4CAF50', '#FF9800']

    bars = ax.bar(categories, cosines, color=colors_bar, alpha=0.85, width=0.5)
    ax.set_ylabel('Cosine Similarity')
    ax.set_ylim(0, 1.1)
    ax.set_title('Geometric Relationships')
    for bar, val in zip(bars, cosines):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.axhline(y=0, color='black', linewidth=0.5)

    # Panel 2: Schematic of what MLP learns
    ax = axes[1]
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')

    # CAA vector (horizontal)
    ax.annotate('', xy=(1.2, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#2196F3', lw=2.5))
    ax.text(1.3, 0.05, 'CAA vector', fontsize=10, color='#2196F3', va='bottom')

    # MLP delta (vertical - orthogonal)
    ax.annotate('', xy=(0, 1.0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#FF9800', lw=2.5))
    ax.text(0.05, 1.05, 'MLP delta\n(orthogonal)', fontsize=10, color='#FF9800', va='bottom')

    # Combined vector
    ax.annotate('', xy=(1.2, 1.0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=2, linestyle='--'))
    ax.text(0.9, 0.7, 'MLP output\n= CAA + delta', fontsize=9, color='#4CAF50')

    ax.set_title('MLP Learns Orthogonal Correction')
    ax.set_xlabel('CAA direction')
    ax.set_ylabel('Learned correction')
    ax.grid(True, alpha=0.2)
    ax.axhline(y=0, color='gray', linewidth=0.3)
    ax.axvline(x=0, color='gray', linewidth=0.3)

    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/geometry.pdf')
    fig.savefig(f'{OUTDIR}/geometry.png')
    plt.close(fig)
    print(f'Saved geometry')


def plot_scale_sweep():
    """Scale sweep showing truth/info tradeoff."""
    scales = [0.80, 0.90, 0.95, 1.00, 1.10, 1.20]
    truth = [82.6, 85.0, 85.3, 86.0, 88.0, 89.7]
    info = [92.4, 90.9, 90.9, 90.2, 84.6, 79.9]
    ti = [76.3, 77.3, 77.6, 77.6, 74.4, 71.7]

    fig, ax1 = plt.subplots(figsize=(6, 4))

    ax1.plot(scales, truth, 's-', color='#2196F3', linewidth=2, markersize=7, label='Truth%')
    ax1.plot(scales, info, '^-', color='#4CAF50', linewidth=2, markersize=7, label='Info%')
    ax1.plot(scales, ti, 'o-', color='#FF9800', linewidth=2.5, markersize=8, label='T*I%')

    ax1.set_xlabel('Steering Scale')
    ax1.set_ylabel('Score (%)')
    ax1.set_ylim(65, 100)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Effect of Steering Scale on Performance')

    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/scale_sweep.pdf')
    fig.savefig(f'{OUTDIR}/scale_sweep.png')
    plt.close(fig)
    print(f'Saved scale_sweep')


def plot_lora_combo():
    """LoRA DPO combination results."""
    labels = ['Baseline\n(unsteered)', 'LoRA DPO\nonly', 'MLP steering\nonly (bn=8)', 'LoRA +\nMLP steering']
    ti = [56.7, 66.6, 80.6, 79.7]
    truth = [58.7, 68.6, 83.1, 82.1]
    info = [96.6, 97.1, 97.1, 97.1]

    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(len(labels))
    w = 0.25
    ax.bar(x - w, truth, w, label='Truth%', color='#2196F3', alpha=0.85)
    ax.bar(x, info, w, label='Info%', color='#4CAF50', alpha=0.85)
    ax.bar(x + w, ti, w, label='T*I%', color='#FF9800', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Score (%)')
    ax.set_ylim(40, 105)
    ax.legend(loc='lower left')
    ax.axhline(y=77.4, color='red', linestyle=':', alpha=0.5, linewidth=1)
    ax.text(0.05, 78, 'RaLFiT SOTA', fontsize=8, color='red', alpha=0.7)
    ax.set_title('Combining Parameter-space (LoRA) and Activation-space (Ours)')
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/lora_combo.pdf')
    fig.savefig(f'{OUTDIR}/lora_combo.png')
    plt.close(fig)
    print(f'Saved lora_combo')


if __name__ == '__main__':
    plot_bottleneck_sweep()
    plot_pool_size()
    plot_training_size()
    plot_ablations()
    plot_geometry()
    plot_scale_sweep()
    plot_lora_combo()
    print(f'\nAll plots saved to {OUTDIR}/')
