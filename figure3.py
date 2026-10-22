"""Figure 3: fronts with three threshold crossings in prescribed period slots.

The coefficient has alpha = 1/4, L1 = 0.8 and L2 = 1.6, the threshold is a = 0.56,
and the crossings lie in the period slots 0, 2 and 4 (shifted by eta = 0.6 so that
each slot contains a whole coefficient interval of either kind).
Each crossing word is solved from the matching equations of Lemma 5.1, starting
from the isolated phases, and checked on every affine interval of the profile.
Panel (b) shows the pair Q_0P_2P_4 and Q_0P_2Q_4, and its inset follows both
fronts as a increases until the last crossing reaches the coefficient corner at
x = 4L, where the two fronts coincide.

Run from the repository root:  python figure3.py
"""
from pathlib import Path
import itertools
import numpy as np
from scipy.optimize import root
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import Bbox, TransformedBbox

from necklace import style
from necklace.green import GreenFront

OUT = Path(__file__).resolve().parent/'fig'
BLUE, ORANGE, GRAY = '#1768a0', '#bf6529', '#626970'
P_COLOR, Q_COLOR = '#e6eef5', '#f5ede5'  # coefficient intervals (0, L1) and (L1, L)
A, ETA = .56, .6
SLOTS = np.array([0, 2, 4])
PAIR = ('QPP', 'QPQ')


def check_profile(model, rhos, a):
    """Check the threshold values, the sign of u - a between crossings and the crossing fluxes."""
    diagnostics = model.verify(rhos, a)
    assert diagnostics['threshold_residual'] < 1e-10, diagnostics
    assert diagnostics['max_wrong_side_at_all_segment_extrema'] < 1e-10, diagnostics
    assert diagnostics['max_affine_flow_discrepancy'] < 1e-10, diagnostics
    assert np.all(np.diff(rhos) > 0)
    assert np.all(np.array(diagnostics['crossing_fluxes'])*(-1)**np.arange(len(rhos)) < 0)


def isolated_phases(model, a):
    """Crossing phases of the two decreasing fronts (key 0) and two increasing fronts (key 1)."""
    choices = {}
    for direction in (0, 1):
        phases = model.phases(a if direction == 0 else 1-a)
        assert len(phases) == 2
        choices[direction] = np.array([(phase+ETA) % model.L-ETA for phase in phases])
    return choices


def continue_to_corner(model, start, a, corner):
    """Move the last crossing from its position at threshold a to the corner, solving for the
    other crossings and the threshold at each step."""
    current = np.r_[start[:-1], a]
    path = []
    for last in np.linspace(start[-1], corner, 101):
        current = root(lambda z: model.residual(np.r_[z[:-1], last], z[-1]), current, tol=1e-11).x
        rhos = np.r_[current[:-1], last]
        check_profile(model, rhos, current[-1])
        path.append((float(current[-1]), rhos))
    assert np.all(np.diff([a for a, _ in path]) > 0)
    return path


def compute():
    model = GreenFront()
    choices = isolated_phases(model, A)
    rows = []
    for bits in itertools.product([0, 1], repeat=3):
        guess = np.array([SLOTS[i]*model.L+choices[i % 2][bit] for i, bit in enumerate(bits)])
        rhos = root(lambda r: model.residual(r, A), guess, tol=1e-11).x
        check_profile(model, rhos, A)
        phases = rhos-SLOTS*model.L
        assert np.all((-ETA < phases) & (phases < model.L-ETA))
        for phase, bit in zip(phases, bits):
            assert (phase % model.L < model.h) == (bit == 0)
        rows.append({'word': ''.join('PQ'[bit] for bit in bits), 'rhos': rhos, 'phases': phases})
    assert len({tuple(np.round(row['rhos'], 8)) for row in rows}) == 8

    # The pair Q_0P_2P_4, Q_0P_2Q_4 meets at the corner x = 4L, at the threshold a_c.
    pair = [next(row for row in rows if row['word'] == word) for word in PAIR]
    corner = SLOTS[-1]*model.L
    guess = np.r_[np.mean([row['rhos'][:2] for row in pair], axis=0), A+.02]
    z = root(lambda z: model.residual(np.r_[z[:-1], corner], z[-1]), guess, tol=1e-11).x
    a_corner, corner_rhos = float(z[-1]), np.r_[z[:-1], corner]
    check_profile(model, corner_rhos, a_corner)
    assert A < a_corner < model.b(0)
    paths = {}
    for row in pair:
        path = continue_to_corner(model, row['rhos'], A, corner)
        assert max(abs(path[-1][1]-corner_rhos)) < 1e-10 and abs(path[-1][0]-a_corner) < 1e-12
        paths[row['word']] = path
    return model, choices, rows, pair, paths, a_corner, corner_rhos


def plot(model, choices, rows, pair, paths, a_corner, corner_rhos):
    fig = plt.figure(figsize=(6.30, 3.25), layout='constrained')
    gs = fig.add_gridspec(1, 2, width_ratios=[1.02, 1.18])
    ax, prof = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    colors = [BLUE, ORANGE]

    # (a) Isolated phases (dashed) and computed crossings (dots) of all eight words.
    for i, slot in enumerate(SLOTS):
        x0 = 1.24*i

        def phase_x(phase):
            return x0+(phase+ETA)/model.L
        ax.add_patch(Rectangle((x0, -.45), 1, 7.9, facecolor=Q_COLOR, edgecolor='none', zorder=0))
        ax.add_patch(Rectangle((phase_x(0), -.45), model.h/model.L, 7.9,
                               facecolor=P_COLOR, edgecolor='none', zorder=0))
        for j, phase in enumerate(choices[i % 2]):
            ax.plot([phase_x(phase)]*2, [-.4, 7.4], c=colors[j], ls=(0, (4, 3)), lw=.9, alpha=.8, zorder=2)
        ax.text(x0+.5, -.78, rf'$N_{i+1}={slot}$, '+('down' if i % 2 == 0 else 'up'), ha='center')
        for phase, label in [(0, '$0$'), (model.h, '$L_1$')]:
            ax.text(phase_x(phase), 7.77, label, ha='center', color=GRAY)
    for y, row in enumerate(rows):
        for i, letter in enumerate(row['word']):
            ax.scatter(1.24*i+(row['phases'][i]+ETA)/model.L, y, c=colors['PQ'.index(letter)],
                       s=23, zorder=4, edgecolors='white', linewidths=.35)
        if row['word'] in PAIR:
            ax.axhspan(y-.43, y+.43, color='#888888', alpha=.15, zorder=1, linewidth=0)
    ax.set(xlim=(-.08, 3.56), ylim=(8.3, -1.16), yticks=range(8), xticks=[],
           yticklabels=['$'+''.join(f'{letter}_{slot}' for letter, slot in zip(row['word'], SLOTS))+'$'
                        for row in rows])
    ax.tick_params(axis='y', length=0, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title('(a) Crossing words in three slots')
    ax.legend(handles=[Line2D([], [], c=GRAY, ls=(0, (4, 3)), lw=.9, label='isolated phase'),
                       Line2D([], [], c=GRAY, marker='o', ls='none', ms=4, label='computed crossing'),
                       Patch(facecolor=P_COLOR, label=r'$P$ interval $(0,L_1)$'),
                       Patch(facecolor=Q_COLOR, label=r'$Q$ interval $(L_1,L)$')],
              loc='lower center', bbox_to_anchor=(.5, -.17), ncols=2, frameon=False,
              fontsize=8, handlelength=1.3, columnspacing=.8)

    # (b) Profiles of the pair.
    left, right = -2.1*model.L, 6.1*model.L
    vertices = [j*model.L+phase for j in range(-3, 8) for phase in (0., model.h)
                if left < j*model.L+phase < right]
    xs = np.unique(np.r_[np.linspace(left, right, 2200), vertices,
                        *[row['rhos'] for row in pair], corner_rhos])
    for slot in SLOTS:
        prof.axvspan(slot-ETA/model.L, slot+1-ETA/model.L, facecolor='#eef1f4', zorder=0)
    for row, color in zip(pair, colors):
        word = row['word']
        label = r'$Q_0P_2'+('P_4' if word == 'QPP' else 'Q_4')+'$'
        prof.plot(xs/model.L, model.profile(xs, row['rhos'])[:, 0], c=color, lw=1.35,
                  ls='-' if word == 'QPP' else (0, (5, 2.5)), label=label)
        prof.scatter(row['rhos']/model.L, [A]*3, c=color, s=18, zorder=5, edgecolors='white', linewidths=.3)
    prof.axhline(A, c='0.42', lw=.65, ls=':')
    prof.text(.985, A-.018, '$a$', transform=prof.get_yaxis_transform(), ha='right', va='top', color=GRAY)
    prof.set(xlim=(left/model.L, right/model.L), ylim=(-.035, 1.38),
             xlabel='$x/L$', ylabel='$u(x)$', yticks=[0, .5, 1], xticks=[0, 2, 4])
    prof.grid(axis='y', alpha=.12)
    prof.set_title(r'(b) $Q_0P_2P_4$ and $Q_0P_2Q_4$ at $a=0.56$')
    prof.legend(loc='upper left', bbox_to_anchor=(.025, .985), frameon=False,
                fontsize=9, handlelength=2.0, borderpad=.15, labelspacing=.3)

    # Inset: position of the last crossing against a, up to the corner.
    inset = prof.inset_axes([.54, .775, .425, .205])
    for row, color in zip(pair, colors):
        xx = np.array([rhos[-1]/model.L for _, rhos in paths[row['word']]])
        aa = np.array([a for a, _ in paths[row['word']]])
        inset.plot(xx, aa, color=color, lw=1.2)
        inset.scatter(xx[0], aa[0], color=color, s=17, zorder=5)
        inset.annotate('', xy=(xx[61], aa[61]), xytext=(xx[45], aa[45]),
                       arrowprops={'arrowstyle': '->', 'color': color, 'lw': 1.1})
    inset.axvline(SLOTS[-1], color='.7', ls=':', lw=.6, zorder=0)
    inset.scatter(SLOTS[-1], a_corner, color='black', s=19, zorder=6)
    inset.set(xlim=(3.95, 4.04), ylim=(.558, .5815), xticks=[3.96, 4, 4.04], yticks=[.56, .576])
    inset.set_xlabel(r'$\rho_3/L$', labelpad=1)
    inset.set_ylabel('$a$', labelpad=1)
    inset.tick_params(length=2)
    for spine in inset.spines.values():
        spine.set_color('.55')
    inset.set_axes_locator(lambda _, renderer: TransformedBbox(
        Bbox.from_bounds(.60, .765, .37, .235), prof.transAxes-fig.transFigure))

    style.finish(fig, 'global-many-crossing-profiles')
    fig.set_size_inches(style.WIDTH, 3.25)
    style.save_pdf(fig, OUT/'global-many-crossing-profiles.pdf')
    plt.close(fig)


def main():
    style.apply()
    OUT.mkdir(exist_ok=True)
    model, choices, rows, pair, paths, a_corner, corner_rhos = compute()
    plot(model, choices, rows, pair, paths, a_corner, corner_rhos)
    print(f'Figure 3: the pair Q_0P_2P_4, Q_0P_2Q_4 meets at the corner when a = {a_corner:.6f}')


if __name__ == '__main__':
    main()
