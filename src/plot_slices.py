"""Figure 1.3ter : quelles couches comptent ? ensembles tau={k0..10}, budget 400k constant."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parent.parent / "results"
from matplotlib.colors import LogNorm

r = json.load(open(str(RES / 'results_narma_slices.json')))
agg = r['agg']
orders = [5, 10, 15, 20, 25, 30, 35, 40]
k0s = list(range(1, 11))
X = orders
Z = np.array([[agg[f'k{k0}_x{x}']['mean'] for k0 in k0s] for x in orders])

fig = plt.figure(figsize=(13.5, 5.8), dpi=110)
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.24)
ax = fig.add_subplot(gs[0, 0])

# k0=1 (toutes les couches, avec tau=1) : vert epais
ax.plot(X, Z[:, 0], 'o-', color='#1a9850', lw=3.0, ms=6.5, zorder=11,
        label='τ = 1..10 (AVEC la couche τ=1, la référence)')
# k0=2 (on n'enleve QUE tau=1) : rouge epais
ax.plot(X, Z[:, 1], 'o-', color='#d62728', lw=3.0, ms=6.5, zorder=10,
        label='τ = 2..10 (on retire UNIQUEMENT la couche τ=1)')
# reste : gris
for j in range(2, 10):
    ax.plot(X, Z[:, j], 'o-', color='#9e9e9e', lw=1.0, ms=3, alpha=0.75, zorder=2)
ax.plot([], [], color='#9e9e9e', lw=1.2, label='τ = 3..10, 4..10, …, 10 seules')

ax.annotate("retirer la seule couche τ=1\nfait passer de ~0,37 à ~0,75\n(les 9 autres tranches ne sauvent rien)",
            xy=(27, 0.75), xytext=(3.5, 0.60), fontsize=9.5, color='#d62728',
            arrowprops=dict(arrowstyle='->', color='#d62728'))
ax.annotate("τ=1 est la récurrence : sans elle,\nle réseau ne propage plus l'information pas à pas",
            xy=(5, 0.30), xytext=(3.5, 0.10), fontsize=9.5, color='#1a9850',
            arrowprops=dict(arrowstyle='->', color='#1a9850'))
ax.set_xlabel('ordre x de la NARMA', fontsize=10)
ax.set_ylabel('NRMSE test (plus bas = meilleur)', fontsize=10)
ax.set_title('Retrait progressif des couches précoces (τ = 1..k0-1 supprimées)', fontsize=11)
ax.set_xticks(X); ax.set_ylim(0, 1.15)
ax.legend(fontsize=9, loc='upper right', framealpha=0.92)
ax.grid(alpha=0.25)

ax2 = fig.add_subplot(gs[0, 1])
im = ax2.imshow(Z.T, aspect='auto', cmap='viridis_r', norm=LogNorm(vmin=0.3, vmax=1.2), origin='lower')
ax2.set_xticks(range(len(orders))); ax2.set_xticklabels(orders)
ax2.set_yticks(range(len(k0s))); ax2.set_yticklabels([f'{k0}..10' for k0 in k0s])
ax2.set_xlabel('ordre x', fontsize=10)
ax2.set_ylabel('ensemble τ (colonne du bas = toutes les couches)', fontsize=10)
ax2.set_title('Grille complète (NRMSE, échelle log)', fontsize=11)
for i in range(len(orders)):
    for j in range(len(k0s)):
        ax2.text(j, i, f'{Z[i, j]:.2f}', ha='center', va='center', fontsize=6.8,
                 color='white' if Z[i, j] > 0.45 else '#222222')
for i in range(len(orders)):
    jbest = int(np.argmin(Z[i]))
    ax2.add_patch(plt.Rectangle((jbest - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='#ff7f0e', lw=2.2))
cb = fig.colorbar(im, ax=ax2, fraction=0.045, pad=0.02)
cb.set_label('NRMSE', fontsize=9)

fig.suptitle('Quelles couches comptent ? : budget constant N²·K = 400 000 · ρ=0,9 · 3 seeds · orange = meilleur ensemble par ligne',
             fontsize=12.5, y=1.04)
fig.savefig(str(RES / 'narma_slices.svg'), bbox_inches='tight', format='svg')
print('SVG + PNG ecrits')
