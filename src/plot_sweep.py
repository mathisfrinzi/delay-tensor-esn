"""Figure 1.3bis v2 : courbes lisibles : ESN (rouge) vs tenseur K=10 (vert) en evidence,
tenseurs partiels en gris fin + heatmap. SVG texte (fonttype none)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parent.parent / "results"
from matplotlib.colors import LogNorm

r = json.load(open(str(RES / 'results_narma_sweep.json')))
agg = r['agg']
fam = [('x5',5),('x10f',10),('x15',15),('x20',20),('x25',25),('x30',30),('x35',35),('x40',40)]
ks = list(range(1,11))
X = [x for _, x in fam]
Z = np.array([[agg[f'{n}_K{k}']['mean'] for k in ks] for n, _ in fam])

fig = plt.figure(figsize=(13.5, 5.8), dpi=110)
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.24)
ax = fig.add_subplot(gs[0, 0])

# tenseurs partiels K=2..9 : gris fins
for k in range(2, 10):
    ax.plot(X, Z[:, k-1], 'o-', color='#9e9e9e', lw=1.0, ms=3, alpha=0.75, zorder=2)
ax.plot([], [], color='#9e9e9e', lw=1.2, label='tenseurs partiels : τ=1..K, K=2..9')

# ESN : rouge epais
ax.plot(X, Z[:, 0], 'o-', color='#d62728', lw=3.0, ms=6.5, zorder=10,
        label='ESN STANDARD  (K=1, une seule tranche τ=1, N=632)')

# tenseur complet K=10 : vert epais
ax.plot(X, Z[:, 9], 'o-', color='#1a9850', lw=3.0, ms=6.5, zorder=11,
        label='TENSEUR COMPLET  (K=10, tranches τ=1..10, N=200×10)')

ax.axvline(22, color='gray', ls=':', lw=1.2, alpha=0.8)
ax.annotate("crossover x* ≈ 22\nà gauche : ESN gagne\nà droite : tenseur gagne", xy=(22, 0.63),
            xytext=(25, 0.60), fontsize=9.5, color='dimgray',
            arrowprops=dict(arrowstyle='->', color='dimgray'))
ax.annotate("l'ESN explose : sa mémoire\nfadante ne retient plus u(t-x)", xy=(38, 0.83),
            xytext=(27, 0.75), fontsize=9.5, color='#d62728',
            arrowprops=dict(arrowstyle='->', color='#d62728'))
ax.annotate("le tenseur reste plat (~0,35-0,39)\nquel que soit le besoin de mémoire", xy=(12, 0.33),
            xytext=(4.3, 0.47), fontsize=9.5, color='#1a9850',
            arrowprops=dict(arrowstyle='->', color='#1a9850'))
ax.set_xlabel('ordre x de la NARMA (famille à params uniformes : seul le besoin de mémoire varie)', fontsize=10)
ax.set_ylabel('NRMSE test (plus bas = meilleur)', fontsize=10)
ax.set_title('Lecture : rouge = ESN standard, vert = tenseur, gris = tenseurs partiels', fontsize=11)
ax.set_xticks(X); ax.set_ylim(0, 0.92)
ax.legend(fontsize=9, loc='upper left', framealpha=0.92)
ax.grid(alpha=0.25)

ax2 = fig.add_subplot(gs[0, 1])
im = ax2.imshow(Z.T, aspect='auto', cmap='viridis_r', norm=LogNorm(vmin=0.004, vmax=0.9), origin='lower')
ax2.set_xticks(range(len(X))); ax2.set_xticklabels(X)
ax2.set_yticks(range(len(ks))); ax2.set_yticklabels([f'K={k}' for k in ks])
ax2.set_xlabel('ordre x', fontsize=10)
ax2.set_ylabel('K tranches (τ = 1..K) : ligne du bas = ESN', fontsize=10)
ax2.set_title('Grille complète (NRMSE, échelle log)', fontsize=11)
for i in range(len(X)):
    for j in range(len(ks)):
        v = Z[i, j]
        ax2.text(j, i, f'{v:.2f}' if v >= 0.1 else f'{v:.3f}', ha='center', va='center', fontsize=6.8,
                 color='white' if (0.02 < v < 0.55) else ('#222222' if v <= 0.02 else 'white'))
for i in range(len(X)):
    jbest = int(np.argmin(Z[i]))
    ax2.add_patch(plt.Rectangle((jbest-0.5, i-0.5), 1, 1, fill=False, edgecolor='#ff7f0e', lw=2.2))
cb = fig.colorbar(im, ax=ax2, fraction=0.045, pad=0.02)
cb.set_label('NRMSE', fontsize=9)

fig.suptitle("Preuve de la loi 1.3 : budget de connexions constant N²·K = 400 000 · ρ=0,9 · 3 seeds · orange = meilleur K par ligne",
             fontsize=12.5, y=1.04)
fig.savefig(str(RES / 'narma_tau_sweep.svg'), bbox_inches='tight', format='svg')
print('SVG v2 + PNG ecrits')
