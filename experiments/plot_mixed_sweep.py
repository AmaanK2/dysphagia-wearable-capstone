"""Plot held-out wearable event F1 against the public-data share of the training loss.

    .venv/bin/python experiments/plot_mixed_sweep.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent / 'mixed_public_wearable'
SURFACE, INK, INK_2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, AQUA, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#8a8984'


def series(name, key):
    data = json.loads((HERE / name).read_text())['shares']
    xs = sorted(data, key=float)
    return [100 * float(x) for x in xs], [100 * data[x][key]['f1'] for x in xs]


lines = [
    ('Boosted trees, best fixed threshold', *series('evaluation_trees.json', 'oracle_fixed_settings'), BLUE, '-'),
    ('Logistic, best fixed threshold', *series('evaluation.json', 'oracle_fixed_settings'), ORANGE, '-'),
    ('Logistic, fully nested (thresholds from 2 people)', *series('evaluation.json', 'pooled'), AQUA, '-'),
    ('Unadapted public, logistic, best fixed threshold',
     *series('evaluation_unadapted_control.json', 'oracle_fixed_settings'), GRAY, '--'),
]
plt.rcParams.update({'font.size': 10, 'text.color': INK, 'axes.labelcolor': INK_2,
                     'xtick.color': INK_2, 'ytick.color': INK_2})
fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.axvspan(20, 65, color=BLUE, alpha=.06, lw=0)
ax.text(42.5, 90.6, 'plateau: 20-65% public', ha='center', color=INK_2, fontsize=9)
for label, x, y, color, style in lines:
    ax.plot(x, y, style, color=color, lw=2, marker='o', ms=6, mec=SURFACE, mew=2, label=label)
ax.annotate('86.0%', (35, 86.0), xytext=(0, 9), textcoords='offset points', ha='center', color=INK, fontsize=9)
ax.annotate('wearable only', (0, 63.9), xytext=(6, -14), textcoords='offset points', color=INK_2, fontsize=9)
ax.set_xlabel('Public share of training weight (%)  —  remainder is our wearable data')
ax.set_ylabel('Event F1 on held-out wearable participant (%)')
ax.set_ylim(50, 92)
ax.set_xlim(-3, 103)
ax.grid(axis='y', color=GRID, lw=.8)
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
for side in ('left', 'bottom'):
    ax.spines[side].set_color(GRID)
ax.legend(loc='lower left', frameon=False, fontsize=9)
ax.set_title('Adapted public data helps most at 20-65% weight', loc='left', color=INK, fontsize=12, pad=12)
fig.text(.01, .01, 'Leave-one-participant-out over 3 wearable participants (85 swallows). '
         '"Best fixed threshold" is chosen on the same pooled folds, so it is optimistic.',
         color=INK_2, fontsize=8)
fig.tight_layout(rect=(0, .03, 1, 1))
fig.savefig(HERE / 'share_sweep.png', dpi=150, facecolor=SURFACE)
print('Saved', HERE / 'share_sweep.png')
