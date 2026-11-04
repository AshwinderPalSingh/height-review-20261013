#!/usr/bin/env python3
"""Results figure for the Paper 1 lab report: success with 95% Wald intervals, from stats.json.

    python3 -I -B make_figure.py <out pdf>
"""
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / 'stats.json').read_text())['runs']
INK, MUTED, GRID, BLUE, GREY = '#0b0b0b', '#52514e', '#e4e3df', '#2a78d6', '#8a8984'
ROWS = [('HEIGHT (released)', False), ('Ours, no acoustics', False), ('Ours, beep only', True),
        ('Ours, beep + listening', True), ('… without confidence feature', True)]
PANELS = [('Nominal setting', ['height', 'nospeak', 'noreturn', 'full', 'nocal']),
          ('Denser crowd: 10–14 people', ['humans12_height', 'humans12_nospeak', 'humans12_noreturn', 'humans12_full', None]),
          ('More obstacles: 13–17', ['obstacles15_height', 'obstacles15_nospeak', 'obstacles15_noreturn', None, None])]

plt.rcParams.update({'font.family': 'serif', 'font.size': 7.5, 'axes.edgecolor': MUTED, 'axes.linewidth': 0.6,
                     'xtick.color': MUTED, 'ytick.color': INK, 'xtick.major.width': 0.6, 'ytick.major.size': 0,
                     'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.9), sharey=True)
for ax, (title, keys) in zip(axes, PANELS):
    for i, ((label, acoustic), key) in enumerate(zip(ROWS, keys)):
        y = len(ROWS) - 1 - i
        if key is None:
            ax.text(0.5, y, 'not evaluated', transform=ax.get_yaxis_transform(), ha='center', va='center',
                    color=MUTED, fontsize=6.5, style='italic')
            continue
        p = R[key]['success'] / 3500
        half = 196 * math.sqrt(p * (1 - p) / 3500)
        color = BLUE if acoustic else GREY
        ax.plot([100 * p - half, 100 * p + half], [y, y], color=color, lw=2, solid_capstyle='round')
        ax.plot(100 * p, y, 'o' if acoustic else 's', color=color, ms=4.5, mec='white', mew=0.8, zorder=3)
        ax.text(100 * p + half + 0.25, y, '%.1f' % (100 * p), va='center', fontsize=6.5, color=INK)
    lo = min(100 * R[k]['success'] / 3500 for k in keys if k) - 2.2
    ax.set_xlim(lo, lo + 7.5)
    ax.set_title(title, fontsize=7.5, color=INK, loc='left')
    ax.grid(axis='x', color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
axes[0].set_yticks(range(len(ROWS)))
axes[0].set_yticklabels([r[0] for r in reversed(ROWS)])
axes[0].set_ylim(-0.6, len(ROWS) - 0.4)
handles = [plt.Line2D([], [], color=GREY, marker='s', lw=2, ms=4.5, mec='white', label='No acoustic action'),
           plt.Line2D([], [], color=BLUE, marker='o', lw=2, ms=4.5, mec='white', label='Learned beep action')]
fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.0, -0.04), ncol=2, frameon=False, fontsize=7)
fig.text(0.99, 0.0, 'Success (%) with 95% CI; n = 3,500 episodes per point', ha='right', va='bottom', color=MUTED, fontsize=7)
fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=1.2)
fig.savefig(sys.argv[1], bbox_inches='tight')
print('written', sys.argv[1])
