"""Figure 2: threshold fluxes, the pinning interval, and its dependence on the edge lengths.

All curves are evaluated from the explicit Floquet formulas of Section 3 of the paper.
Panels (a)-(c) show the threshold fluxes of the two equilibrium tails for
alpha = 1/4, L1 = 0.8, L2 = 1.6 and three thresholds near a_-; a decreasing front
crosses the threshold at phase s exactly where the two fluxes agree.
Panel (d) shows a_- and a_+ for equal lengths L1 = L2 = l and several alpha,
and panel (e) shows them for alpha = 1/4 against compactified lengths.

Run from the repository root:  python figure2.py
"""
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from necklace import style

OUT = Path(__file__).resolve().parent/'fig'
BLUE, ORANGE, GRAY = '#1768a0', '#c56b2d', '#767c82'
COLORS = ['#633e7f', '#1768a0', '#258776', '#202020', '#ba7935', '#ab4657', '#666b71']
ALPHAS = [.01, .1, .2, .25, .3, .4, .5]


def matrix(kappa, t):
    """Flow over length t of the linear system u' = kappa v, v' = u, where kappa = 1/p."""
    r = np.sqrt(kappa)
    c, s = np.cosh(r*t), np.sinh(r*t)
    return np.array([[c, r*s], [s/r, c]])


def endpoint_data(alpha, L1, L2):
    """Monodromy matrix A, its trace tau, d = A_22 - A_11, D = sqrt(tau^2-4), and a_-, a_+."""
    A = matrix(alpha, L2) @ matrix(1., L1)
    tau = float(np.trace(A))
    d = float(A[1, 1]-A[0, 0])
    # Equal to sqrt(tau^2-4), evaluated without cancellation.
    D = float(np.hypot(d, 2*np.sqrt(A[0, 1]*A[1, 0])))
    return A, tau, d, D, .5-d/(2*D), .5+d/(2*D)


def width(t1, t2, alpha=.25):
    """Width a_+ - a_- as a function of t1 = tanh(L1) and t2 = tanh(sqrt(alpha) L2) on [0,1]^2."""
    r = np.sqrt(alpha)
    k, c = 1/r-r, 1/r+r
    t1, t2 = np.broadcast_arrays(np.asarray(t1, float), np.asarray(t2, float))
    denominator = np.sqrt(k*k*(t1*t2)**2 + 4*t1*t1 + 4*t2*t2 + 4*c*t1*t2)
    return np.divide(k*t1*t2, denominator, out=np.zeros_like(denominator), where=denominator > 0)


class FloquetTails:
    """Threshold fluxes of the stable and unstable equilibrium tails at phase s."""

    def __init__(self, alpha, L1, L2):
        self.alpha, self.L1, self.L2 = alpha, L1, L2
        self.L = L1+L2
        self.A, self.tau, self.d, self.D, self.lower, self.upper = endpoint_data(alpha, L1, L2)
        self.lam_u = (self.tau+self.D)/2
        self.lam_s = 1/self.lam_u
        self.w = np.array([1., (self.d+self.D)/(2*self.A[0, 1])])  # unstable eigenvector
        self.v = np.array([1., (self.d-self.D)/(2*self.A[0, 1])])  # stable eigenvector

    def transport(self, s):
        if s <= self.L1:
            return matrix(1., s)
        return matrix(self.alpha, s-self.L1) @ matrix(1., self.L1)

    def flux(self, s, a):
        """Fluxes at the threshold of the stable tail of (0,0) and the unstable tail of (1,0)."""
        B = self.transport(s)
        v, w = B @ self.v, B @ self.w
        return a*v[1]/v[0], (a-1)*w[1]/w[0]

    def mismatch(self, s, a):
        S, U = self.flux(s, a)
        return U-S

    def matching_phases(self, a):
        """Crossing phases of the decreasing fronts, using monotonicity on each coefficient interval."""
        roots = [point for point in [0., self.L1] if abs(self.mismatch(point, a)) < 1e-13]
        for lo, hi in [(0., self.L1), (self.L1, self.L)]:
            flo, fhi = self.mismatch(lo, a), self.mismatch(hi, a)
            if abs(flo) > 1e-13 and abs(fhi) > 1e-13 and flo*fhi < 0:
                roots.append(brentq(lambda s: self.mismatch(s, a), lo, hi, xtol=5e-15))
        return sorted(roots)


def compute():
    model = FloquetTails(.25, .8, 1.6)
    assert np.max(abs(model.A @ model.v-model.lam_s*model.v)) < 1e-13
    assert np.max(abs(model.A @ model.w-model.lam_u*model.w)) < 1e-13
    assert abs(np.linalg.det(model.A)-1) < 1e-13
    phase_grid = np.r_[np.linspace(0, model.L1, 601), np.linspace(model.L1, model.L, 601)[1:]]
    cases = []
    for offset in [-.03, 0, .03]:
        a = model.lower+offset
        roots = model.matching_phases(a)
        ss = np.unique(np.r_[phase_grid, roots])
        cases.append({'a': a, 's': ss, 'roots': roots,
                      'fluxes': np.array([model.flux(s, a) for s in ss])})
    assert [len(case['roots']) for case in cases] == [0, 1, 2]
    ell = np.unique(np.r_[np.linspace(.01, .1, 129), np.linspace(.1, 1, 385), np.linspace(1, 10, 1537)])
    boundaries = []
    for alpha in ALPHAS:
        values = np.array([endpoint_data(alpha, length, length)[-2:] for length in ell])
        assert np.all((0 < values[:, 0]) & (values[:, 0] < .5) & (values[:, 1] < 1))
        boundaries.append(values)
    # The width formula agrees with the monodromy matrix.
    for l1, l2 in [(.01, .02), (.4, .8), (.8, 1.6), (1, 1), (2, 5), (6, 6)]:
        lower, upper = endpoint_data(.25, l1, l2)[-2:]
        assert abs(float(width(np.tanh(l1), np.tanh(l2/2)))-(upper-lower)) < 1e-13
    return model, cases, ell, boundaries


def plot(model, cases, ell, boundaries):
    flux_min = min(case['fluxes'].min() for case in cases)
    flux_max = max(case['fluxes'].max() for case in cases)
    flux_pad = .05*(flux_max-flux_min)
    fig = plt.figure(figsize=(6.30, 5.6))

    # (a)-(c) Threshold fluxes.
    top = [fig.add_axes([.11+.295*i, .69, .265, .25]) for i in range(3)]
    titles = [r'(a) $a=a_- -0.03$', r'(b) $a=a_-$', r'(c) $a=a_- +0.03$']
    for index, (ax, case) in enumerate(zip(top, cases)):
        S, U = case['fluxes'].T
        ax.plot(case['s'], S, c=ORANGE, lw=1.45)
        ax.plot(case['s'], U, c=BLUE, lw=1.45)
        ax.axvline(model.L1, c=GRAY, ls=':', lw=.75, zorder=0)
        for s in case['roots']:
            ax.scatter(s, model.flux(s, case['a'])[0], c='black', s=21, zorder=7)
        ax.set(xlim=(0, model.L), ylim=(flux_min-2*flux_pad, flux_max+flux_pad),
               xlabel='$s$', xticks=[0, model.L1, model.L],
               xticklabels=['$0$', '$L_1$', '$L$'], yticks=[-1.2, -.8, -.4])
        if index == 0:
            ax.set_ylabel('Threshold flux')
        else:
            ax.tick_params(labelleft=False)
        ax.set_title(titles[index], pad=7)
        ax.text(.95, .018, ['No match', 'One match', 'Two matches'][index], transform=ax.transAxes,
                ha='right', va='bottom', color='#40464b')
    legend_axis = fig.add_axes([.075, .565, .885, .06])
    legend_axis.set_axis_off()
    legend_axis.legend(handles=[Line2D([], [], c=ORANGE, lw=1.45, label=r'$v^{\mathrm{s}}(s,a)$'),
                                Line2D([], [], c=BLUE, lw=1.45, label=r'$v^{\mathrm{u}}(s,a)$')],
                       loc='center', ncol=2, frameon=False, handlelength=1.7, columnspacing=1.8,
                       fontsize=11)  # sets the spacing unit; style.finish sets the text to 8 pt

    # (d) Pinning interval for equal lengths.
    ax = fig.add_axes([.11, .19, .40, .31])
    for color, alpha, values in zip(COLORS, ALPHAS, boundaries):
        ax.plot(ell, values[:, 0], color=color, lw=1.25)
        ax.plot(ell, values[:, 1], color=color, lw=1.25, label=rf'$\alpha={alpha:g}$')
    ax.axhline(.5, c=GRAY, ls=':', lw=.7, zorder=0, alpha=.5)
    ax.set(xlim=(0, 10.15), ylim=(.05, .95), xlabel=r'$\ell$', ylabel='$a$',
           xticks=[0, 2, 4, 6, 8, 10], yticks=[.1, .3, .5, .7, .9])
    ax.set_title(r'(d) $a_-$ and $a_+$ for $L_1=L_2=\ell$', pad=8)
    ax.legend(loc='upper center', bbox_to_anchor=(.31, .075), bbox_transform=fig.transFigure,
              ncol=4, frameon=False, columnspacing=.8, handlelength=1.2,
              labelspacing=.4, borderpad=0, borderaxespad=0)

    # (e) Surfaces a_- and a_+ for alpha = 1/4 over compactified lengths.
    surface = fig.add_axes([.54, .095, .35, .42], projection='3d', computed_zorder=False)
    q = np.linspace(0, 1, 65)
    xx, yy = np.meshgrid(q, q)
    ww = width(xx, yy)
    for sign, color in [(-1, '#83b6d6'), (1, BLUE)]:
        zz = .5+sign*ww/2
        surface.plot_surface(xx, yy, zz, color=color, alpha=.85, linewidth=0, rcount=65, ccount=65, zorder=2)
        surface.plot(q, np.ones_like(q), zz[-1], c=BLUE, lw=.8, zorder=3)
        surface.plot(np.ones_like(q), q, zz[:, -1], c=BLUE, lw=.8, zorder=3)
    for x, y, w in [(xx[-1], yy[-1], ww[-1]), (xx[:, -1], yy[:, -1], ww[:, -1])]:
        surface.plot_surface(np.array([x, x]), np.array([y, y]), np.array([.5-w/2, .5+w/2]),
                             color=BLUE, alpha=.15, linewidth=0, zorder=1)
    surface.plot_surface(xx, yy, np.full_like(xx, .5), color=GRAY, alpha=.1, zorder=1)
    surface.plot(q, q*0, q*0+.5, c=GRAY, lw=1.5, zorder=4)
    surface.plot(q*0, q, q*0+.5, c=GRAY, lw=1.5, zorder=4)
    # The equal-length curve of panel (d), and its limit (1/3, 2/3) at the far corner.
    equal = np.linspace(0, 18, 501)
    tx, ty = np.r_[np.tanh(equal), 1], np.r_[np.tanh(equal/2), 1]
    for sign in (-1, 1):
        surface.plot(tx, ty, .5+sign*width(tx, ty)/2, c=COLORS[ALPHAS.index(.25)], lw=1.6, zorder=5)
    surface.scatter([1, 1], [1, 1], [1/3, 2/3], c='black', s=12, zorder=6)
    surface.set(xlim=(0, 1), ylim=(0, 1), zlim=(.31, .69), xlabel=r'$\tanh L_1$',
                ylabel=r'$\tanh(L_2/2)$', zlabel='$a$')
    surface.set_xticks([0, .5, 1])
    surface.set_yticks([.5, 1])
    surface.set_zticks([1/3, .5, 2/3], ['$1/3$', '$1/2$', '$2/3$'])
    surface.tick_params(axis='both', pad=1)
    surface.xaxis.labelpad = 12
    surface.yaxis.labelpad = 12
    surface.zaxis.labelpad = 0
    surface.set_proj_type('ortho')
    surface.view_init(elev=23, azim=-55)
    surface.set_box_aspect((1, 1, .75))
    surface.set_title(r'(e) $a_-$ and $a_+$ for $\alpha=1/4$', pad=8)

    style.finish(fig, 'monotone-front-summary')
    assert ax.get_legend().get_window_extent().y1 < ax.get_window_extent().y0
    style.save_pdf(fig, OUT/'monotone-front-summary.pdf')
    plt.close(fig)


def main():
    style.apply()
    plt.rcParams['path.simplify'] = False
    OUT.mkdir(exist_ok=True)
    model, cases, ell, boundaries = compute()
    plot(model, cases, ell, boundaries)
    print(f'Figure 2: a_- = {model.lower:.6f}, a_+ = {model.upper:.6f} for alpha = 1/4, L1 = 0.8, L2 = 1.6')


if __name__ == '__main__':
    main()
