"""Figures 4 and 5: period-map images of the invariant manifolds and the fronts they carry.

The coefficient has alpha = 1/4, L1 = 0.4 and L2 = 0.8, and states are sampled at
the phase s = L1/2, the midpoint of an interval where p = 1. Figure 4 uses the
threshold a = 1/2. Notation follows Section 5.4 of the paper: I is the straight
unstable segment of (1,0) from the preimage of P_0 to P_0, its first image
consists of the arc E and the spargel Sigma_0, and Sigma_j is the j-th image of
Sigma_0. The base is the segment [Q_0, P_0] of the stable line of (0,0).

Manifold arcs are images of finely and equally spaced points of I (or of the
straight segment of the unstable manifold of (0,0)) under the exact period map.
Each intersection is detected between neighbouring images and then refined by a
root finder in the seed parameter. The crossing word of a front is found by
following its trajectory to its straight equilibrium tails.

Run from the repository root:  python figures4_5.py
"""
from pathlib import Path
import numpy as np
from scipy.optimize import brentq, root
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString, Polygon as ShapelyPolygon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from necklace import style
from necklace.green import GreenFront
from necklace.switching import Map, lib, matrix

OUT = Path(__file__).resolve().parent/'fig'
H, L, S = .4, 1.2, .2  # L1, the period L, and the sampling phase s = L1/2
C = ['#1f77b4', '#17becf', '#2ca02c', '#bcbd22', '#9467bd', '#3b0f70']  # Sigma_0, ..., Sigma_5
ORANGE, PINK, GRAY = '#d55e00', '#e377c2', '#70767b'
FRONT_COLORS = ['#555555', '#222222', '#8e4999']
# Final line weights in points.
MANIFOLD, FRONT, GUIDE, LEADER, LEADER_COLOR = .65, .9, .6, .35, '#8c8c8c'

MP = Map(l1=H, l2=.8, a=.5)
# Seed parameter of the point of I whose image is Q_0, on the stable line v = -m u.
TQ = brentq(lambda t: MP.point(t, 1)[1]+MP.m*MP.point(t, 1)[0], .1, .8, xtol=1e-14)
P, Q = MP.point(0, 1), MP.point(TQ, 1)


# Computation

def manifold_images():
    """Sigma_0, ..., Sigma_5, the arc E, and the straight segment of the unstable
    manifold of (0,0) with three of its images."""
    ts = np.linspace(TQ, 1, 500001)
    ys, curves = MP.seeds(ts), {}
    for j in range(6):
        ys, _ = MP.batch(ys)
        curves[j] = ys.copy()
    outward, _ = MP.batch(MP.seeds(np.linspace(0, TQ, 8001)))
    x = np.linspace(0, .5, 500001)
    zero, zs = np.column_stack([x, MP.m*x]), []
    for j in range(4):
        if j:
            zero, _ = MP.batch(zero)
        zs.append(zero.copy())
    return ts, curves, outward, zs


def event_flow(y, T, kappa, a):
    """Python version of the flow in flow.cpp that also returns the threshold crossing positions."""
    y = np.array(y, float)
    direction = 1 if T >= 0 else -1
    y[1] *= direction
    r, left, elapsed, events = np.sqrt(kappa), abs(T), 0., []
    b = int(y[0] > a)
    if abs(y[0]-a) < 2e-12:
        if abs(y[1]) < 1e-12:
            raise ValueError('tangent threshold')
        b = int(y[1] > 0)
    for _ in range(1000):
        if left < 1e-14:
            break
        # In the variable z = exp(r t), u(t) = a is a quadratic equation A z^2 + B z + C = 0.
        d = y[0]-b
        A, B, Cq = d+r*y[1], -2*(a-b), d-r*y[1]
        zs, disc = [], B*B-4*A*Cq
        if abs(A) < 1e-15:
            if abs(B) > 1e-15:
                zs = [-Cq/B]
        elif disc >= 0:
            qq = -.5*(B+np.copysign(np.sqrt(disc), B))
            zs = [qq/A]+([Cq/qq] if qq else [])
        times = [np.log(z)/r for z in zs if np.isfinite(z) and z > 1+2e-12]
        nxt = min(times, default=np.inf)
        hit, dt = nxt <= left+2e-12, min(nxt, left)
        ch, sh = np.cosh(r*dt), np.sinh(r*dt)
        y = np.array([b+d*ch+r*y[1]*sh, d*sh/r+y[1]*ch])
        left -= dt
        elapsed += dt
        if hit:
            y[0] = a
            events.append(direction*elapsed)
            b = 1-b
    else:
        raise ValueError('event cap')
    y[1] *= direction
    return y, events


def trace_tail(y, direction, maxperiods=20):
    """Threshold crossings of the trajectory from y at phase S, followed backward (direction=-1)
    or forward (direction=1) until it lies on a straight equilibrium tail."""
    x, y, events = S, np.array(y, float), []
    if abs(y[0]-.5) < 2e-10:
        events.append(x)
    for _ in range(maxperiods):
        defect = abs(y[1]-(MP.m*(y[0]-1) if direction < 0 else -MP.m*y[0]))
        on_tail = (y[0] > .5 and y[1] < 0) if direction < 0 else (y[0] < .5 and y[1] < 0)
        if on_tail and defect < 2e-7:
            return events
        for length, kappa in [(H/2, 1), (.8, .25), (H/2, 1)]:
            y, crossings = event_flow(y, direction*length, kappa, .5)
            events.extend(x+e for e in crossings)
            x += direction*length
        if not np.isfinite(y).all() or np.linalg.norm(y) > 1e4:
            break
    raise ValueError('trajectory did not reach a straight tail')


def word_for(y, base=False):
    """Crossings and crossing word of the front through the state y at phase S.

    A point of the base already lies on the stable tail, so only the backward part is traced.
    The letter is P for a crossing in (0, L1) and Q for one in (L1, L) modulo L, and the
    subscript is the period slot, shifted so that the first slot is zero.
    """
    xs = sorted(trace_tail(y, -1)+([] if base else trace_tail(y, 1)))
    xs = [x for i, x in enumerate(xs) if i == 0 or abs(x-xs[i-1]) > 1e-8]
    slots = np.floor(np.array(xs)/L+1e-10).astype(int)
    slots -= slots[0]
    phases = np.mod(xs, L)
    assert np.all(np.minimum.reduce([phases, abs(phases-H), L-phases]) >= 1e-7)
    letters = ['P' if phase < H else 'Q' for phase in phases]
    return {'word': ''.join(f'{c}_{n}' for c, n in zip(letters, slots)),
            'latex': '$'+''.join(f'{c}_{{{n}}}' for c, n in zip(letters, slots))+'$',
            'crossings': xs}


def base_crossings(j, ts, curve):
    """Crossings of Sigma_j with the open base, each paired with the other end of its visit
    of the pocket."""
    def offset(t):
        y = MP.point(t, j+1)
        return y[1]+MP.m*y[0]
    values = curve[:, 1]+MP.m*curve[:, 0]
    roots = []
    for i in np.flatnonzero(values[:-1]*values[1:] < 0):
        t = brentq(offset, ts[i], ts[i+1], xtol=5e-16)
        y = MP.point(t, j+1)
        if abs(offset(t)) > 2e-8 or not Q[0]+1e-9 < y[0] < P[0]-1e-9:
            continue
        roots.append({'id': len(roots), 't': t, 'point': y.tolist(), **word_for(y, base=True)})
    # Consecutive crossings along the spargel bound one visit when the arc between them
    # lies below the stable line.
    for first, second in zip(roots[:-1], roots[1:]):
        between = curve[(ts > first['t']) & (ts < second['t'])]
        if len(between) and np.max(between[:, 1]+MP.m*between[:, 0]) < 1e-9:
            first['paired_with'], second['paired_with'] = second['id'], first['id']
    return roots


def line_parts(ys, bounds, max_chord=.002):
    """Pieces of a sampled arc inside bounds, broken where neighbouring samples are far apart."""
    ys = np.asarray(ys)
    valid = np.isfinite(ys).all(axis=1)
    valid &= (ys[:, 0] >= bounds[0]) & (ys[:, 0] <= bounds[1]) & (ys[:, 1] >= bounds[2]) & (ys[:, 1] <= bounds[3])
    breaks = np.r_[True, np.linalg.norm(np.diff(ys, axis=0), axis=1) > max_chord] | ~valid
    starts = np.flatnonzero(breaks)
    parts = []
    for ix, start in enumerate(starts):
        end = starts[ix+1] if ix+1 < len(starts) else len(ys)
        a = start if valid[start] else start+1
        if end-a >= 2:
            parts.append(LineString(ys[a:end]))
    return parts


def mirror_crossings(ts, curve, j=2):
    """Intersections of Sigma_j with its mirror image R(Sigma_j) under (u, v) -> (1-u, v)."""
    bounds = (.46, .54, .53, .65)
    parts = line_parts(curve, bounds)
    mirrored = [LineString(np.column_stack([1-np.asarray(p.coords)[:, 0], np.asarray(p.coords)[:, 1]]))
                for p in parts]
    inter = MultiLineString(parts).intersection(MultiLineString(mirrored))
    points = [inter] if inter.geom_type == 'Point' else [p for p in getattr(inter, 'geoms', []) if p.geom_type == 'Point']
    tree, found = cKDTree(curve), []
    for p in points:
        point = np.array(p.coords[0])
        _, i = tree.query(point)
        _, k = tree.query([1-point[0], point[1]])

        def mismatch(tt):
            u, w = MP.point(tt[0], j+1), MP.point(tt[1], j+1)
            return u-np.array([1-w[0], w[1]])
        sol = root(mismatch, [ts[i], ts[k]], tol=1e-10)
        if max(abs(mismatch(sol.x))) > 1e-8 or not np.all((sol.x >= TQ) & (sol.x <= 1)):
            continue
        y = MP.point(sol.x[0], j+1)
        if any(np.linalg.norm(y-np.array(r['point'])) < 1e-7 for r in found):
            continue
        found.append({'point': y.tolist(), **word_for(y)})
    return sorted(found, key=lambda r: (r['point'][1], r['point'][0]))


def endpoint_states():
    """P_0, P_1 and Q_0 at the endpoints a_- (phase L1) and a_+ (phase 0) of the pinning interval."""
    upper = GreenFront(H).b(0)
    records = []
    for a, phase in [(1-upper, .4), (upper, 0.)]:
        mp = Map(l1=H, l2=.8, a=a)
        stable = matrix(phase-S, 1) @ np.array([1., -mp.m])  # transported from the midpoint
        p = np.array([a, a*stable[1]/stable[0]])
        p1 = p/mp.lam
        records.append({'a': a, 'phase': phase, 'P0': p, 'P1': p1, 'Q0': p1 if phase > S else p})
    return records


def front_samples(mirror):
    """The monotone front u^P and two symmetric fronts of Figure 5(b), with period samples at S."""
    green = GreenFront(.4)
    fronts = [('P_0', np.array([green.phases(.5)[0]]))]
    fronts += [(word, np.array(next(r for r in mirror if r['word'] == word)['crossings']))
               for word in ['P_0P_3P_6', 'Q_0P_3Q_5']]
    result = []
    for word, rhos in fronts:
        # Shift by whole periods so that the first crossing lies in slot zero.
        rhos = rhos-np.floor(rhos[0]/L)*L
        sample_x = S+np.arange(-3, int(np.ceil((rhos[-1]-S)/L))+4)*L
        checks = green.verify(rhos, .5)
        assert checks['threshold_residual'] < 1e-10 and checks['max_wrong_side_at_all_segment_extrema'] < 1e-10
        result.append({'word': word, 'rhos': rhos, 'sample_x': sample_x})
    return result


# Drawing

def draw(ax, ys, color, lw=1., alpha=1., max_chord=.006, bounds=None, **kwargs):
    """Plot a sampled arc, broken where neighbouring samples are far apart, and clipped to bounds."""
    yy = np.array(ys, copy=True)
    yy[np.r_[False, np.linalg.norm(np.diff(yy, axis=0), axis=1) > max_chord]] = np.nan
    if bounds:
        x0, x1, y0, y1 = bounds
        visible = (yy[:, 0] >= x0) & (yy[:, 0] <= x1) & (yy[:, 1] >= y0) & (yy[:, 1] <= y1)
        yy[~(visible | np.r_[False, visible[:-1]] | np.r_[visible[1:], False])] = np.nan
    # Simplify each piece at a tolerance far below the printed resolution.
    valid = np.isfinite(yy).all(axis=1)
    edges = np.flatnonzero(np.diff(np.r_[False, valid, False]))
    chunks = []
    for start, end in zip(edges[::2], edges[1::2]):
        if end-start >= 2:
            chunks += [np.asarray(LineString(yy[start:end]).simplify(1e-9).coords), np.full((1, 2), np.nan)]
    if chunks:
        yy = np.vstack(chunks)
    ax.plot(yy[:, 0], yy[:, 1], color=color, lw=lw, alpha=alpha, **kwargs)


def shade(ax, points):
    ax.add_patch(Polygon(points, facecolor=ORANGE, alpha=.15, edgecolor='none', zorder=0))


def setup(ax, title, bounds, equilibria=True):
    ax.set(xlim=bounds[:2], ylim=bounds[2:], xlabel='$u$', ylabel='$v=p u_x$', title=title)
    ax.tick_params(pad=2, length=3)
    for x in [0, 1]:  # the equilibria
        if equilibria and bounds[0] < x < bounds[1] and bounds[2] < 0 < bounds[3]:
            ax.scatter(x, 0, c='black', s=10, zorder=10)


def set_line_weights(fig, fronts_zorder=None):
    """Thin lines for manifold arcs, so that the folds of a spargel stay two visible strands.

    style.finish has made lines 1.8 pt (I and the base) or 1 pt. The thick ones are kept,
    dotted guides become GUIDE, lines at fronts_zorder become FRONT and all others MANIFOLD.
    Leaders become thin and grey, so that they cannot be read as curves.
    """
    for ax in fig.axes:
        for line in ax.lines:
            if line.get_linewidth() >= 1.5:
                continue
            if line.get_linestyle() == ':':
                line.set_linewidth(GUIDE)
            elif line.get_zorder() == fronts_zorder:
                line.set_linewidth(FRONT)
            else:
                line.set_linewidth(MANIFOLD)
        for text in ax.texts:
            if getattr(text, 'arrow_patch', None):
                text.arrow_patch.set_linewidth(LEADER)
                text.arrow_patch.set_color(LEADER_COLOR)
        if ax.get_legend():
            for handle in ax.get_legend().legend_handles:
                if isinstance(handle, Line2D) and handle.get_linewidth() < 1.5:
                    handle.set_linewidth(MANIFOLD)


def dot_sizes(fig):
    for ax in fig.axes:
        for obj in ax.collections:
            if isinstance(obj, PathCollection):
                obj.set_sizes([9])
                obj.set_linewidths([0])


def figure4(ts, curves, outward, zs, crossings, mirror):
    pocket = np.vstack([outward, Q, P])
    fig = plt.figure(figsize=(6.30, 6.0))
    axs = [fig.add_axes(rect) for rect in [(.085, .57, .39, .35), (.60, .57, .39, .35),
                                           (.085, .095, .341, .35), (.745, .095, .225, .35)]]

    # (a) I, its first image E + Sigma_0, the spargels and the pocket.
    ax, bounds = axs[0], (-.03, 1.05, -.84, .78)
    shade(ax, pocket)
    ax.plot([0, P[0]], [0, P[1]], c=ORANGE, lw=.8)
    ax.plot([Q[0], P[0]], [Q[1], P[1]], c=ORANGE, lw=2)
    draw(ax, MP.seeds(np.linspace(0, 1, 10001)), C[0], lw=1.5)
    draw(ax, outward, C[0], lw=.9)
    for j in [0, 1, 2, 5]:
        draw(ax, curves[j], C[j], lw=.7, bounds=bounds, max_chord=.002)
    for pt, label, offset in [(P, '$P_0$', (5, -11)), (Q, '$Q_0$', (-22, 0)), (P/MP.lam, '$P_1$', (-22, 0))]:
        ax.scatter(*pt, c='black', s=9, zorder=8)
        ax.annotate(label, pt, xytext=offset, textcoords='offset points')
    ends = [Q/MP.lam**5, P/MP.lam**6]
    assert max(np.linalg.norm(curves[5][i]-ends[n]) for n, i in enumerate([0, -1])) < 1e-10
    for pt in ends:
        ax.scatter(*pt, c=C[5], s=5, zorder=9)
    ax.annotate(r'$\lambda^{-5}Q_0,\ P_6$', ends[0], xytext=(.02, .44), color=C[5],
                arrowprops={'arrowstyle': '-', 'lw': .4, 'color': C[5]})
    e_point = outward[np.argmin(np.linalg.norm(outward-[.39, -.70], axis=1))]
    i_point = np.array([1., 0.])+.65*(P-np.array([1., 0.]))
    for label, point, pos, color in [('$I$', i_point, (.80, -.43), C[0]), ('$E$', e_point, (.23, -.79), C[0]),
                                     ('base', .6*Q+.4*P, (.08, -.65), ORANGE),
                                     ('pocket', (.43, -.63), (.68, -.74), 'black')]:
        ax.annotate(label, point, xytext=pos, color=color, arrowprops={'arrowstyle': '-', 'lw': .45, 'color': color})
    ax.legend(handles=[Line2D([], [], c=C[j], label=rf'$\Sigma_{j}$') for j in [0, 1, 2, 5]], ncol=2,
              loc='upper right', bbox_to_anchor=(1, 1.03), fontsize=8, frameon=False, handlelength=1,
              columnspacing=.7)
    setup(ax, '(a) First image, spargels and the pocket', bounds)
    ax.set_xticks([0, .5, 1])
    ax.set_yticks([-.6, 0, .6])

    # (b) Late spargels following the unstable manifold of (0,0).
    ax, bounds = axs[1], (-.05, 1.46, -.63, .85)
    for z in zs:
        draw(ax, z, PINK, lw=1.2, bounds=bounds, max_chord=.002)
    for j in [3, 4, 5]:
        draw(ax, curves[j], C[j], lw=.65, bounds=bounds, max_chord=.002)
    setup(ax, r'(b) Spargels following $W^u(0,0)$', bounds)
    ax.legend(handles=[Line2D([], [], c=C[j], label=rf'$\Sigma_{j}$') for j in [3, 4, 5]]
              + [Line2D([], [], c=PINK, label=r'$W^u(0,0)$')],
              fontsize=8, loc='lower right', frameon=False, handlelength=1, labelspacing=.2)
    ax.set_xticks([0, .5, 1])
    ax.set_yticks([-.5, 0, .5])

    # (c) Crossings of Sigma_5 with the base, with the words of their fronts in a numbered column.
    ax, bounds = axs[2], (Q[0]-.016, P[0]+.016, P[1]-.025, Q[1]+.025)
    shade(ax, pocket)
    draw(ax, curves[5], C[5], lw=.7, bounds=bounds, max_chord=.0015)
    ax.plot([Q[0], P[0]], [Q[1], P[1]], c=ORANGE, lw=1.8)
    ordered = sorted(crossings, key=lambda r: r['point'][0])  # from Q_0 to P_0 along the base
    assert all(ordered[k]['paired_with'] == ordered[11-k]['id'] for k in range(6))  # nested visits
    levels = np.linspace(.434, .106, 12)
    word_texts = []
    for k, (r, level) in enumerate(zip(ordered, levels)):
        ax.scatter(*r['point'], c='black', s=5, zorder=9)
        ax.annotate('', r['point'], xytext=(.432, level), textcoords=fig.transFigure, annotation_clip=False,
                    arrowprops={'arrowstyle': '-', 'lw': .4, 'color': C[5]}, zorder=8)
        word_texts.append(fig.text(.442, level, r['latex'], fontsize=8, va='center'))
    # Brackets join the two crossings of each visit, nested as the visits are.
    fig.canvas.draw()
    word_right = max(t.get_window_extent().transformed(fig.transFigure.inverted()).x1 for t in word_texts)
    point = 1/(6.30*72)
    for k in range(6):
        x = word_right+6*point+(5-k)*3*point
        for xx, yy in [([x, x], [levels[k], levels[11-k]]), ([x-2*point, x], [levels[k]]*2),
                       ([x-2*point, x], [levels[11-k]]*2)]:
            fig.add_artist(Line2D(xx, yy, transform=fig.transFigure, color='#444444', lw=.6))
    for pt, label, offset in [(Q, '$Q_0$', (2, 5)), (P, '$P_0$', (-14, 4))]:
        ax.scatter(*pt, c='black', s=10, zorder=10)
        ax.annotate(label, pt, xytext=offset, textcoords='offset points')
    setup(ax, '', bounds)
    ax.set_title(r'(c) Crossings of $\Sigma_5$ with the base', loc='left')
    ax.set_xticks([.35, .40, .45])
    ax.set_yticks([-.6, -.5])

    # (d) Sigma_2 and its mirror image; two intersections lie on the threshold line u = 1/2.
    ax, bounds = axs[3], (.458, .542, .53, .585)
    draw(ax, curves[2], C[2], lw=.9, bounds=bounds, max_chord=.002)
    mirror_image = curves[2].copy()
    mirror_image[:, 0] = 1-mirror_image[:, 0]
    draw(ax, mirror_image, ORANGE, lw=.9, bounds=bounds, max_chord=.002)
    ax.axvline(.5, c=GRAY, ls=':', lw=.5)
    for r, (x, y, ha) in zip(mirror, [(.459, .563, 'left'), (.541, .548, 'right'),
                                      (.540, .568, 'right'), (.460, .578, 'left')]):
        ax.scatter(*r['point'], c='black', s=10, zorder=7)
        ax.annotate(r['latex'], r['point'], xytext=(x, y), ha=ha,
                    arrowprops={'arrowstyle': '-', 'color': GRAY, 'lw': .4})
    setup(ax, r'(d) $\Sigma_2$ and its mirror image', bounds)
    ax.set_xticks([.46, .5, .54])
    ax.set_yticks([.54, .56, .58])
    ax.legend(handles=[Line2D([], [], c=C[2], label=r'$\Sigma_2$'), Line2D([], [], c=ORANGE, label=r'$R(\Sigma_2)$')],
              loc='upper right', fontsize=8, frameon=False, handlelength=1)

    style.finish(fig, 'global-manifolds-first-return')
    dot_sizes(fig)
    for obj in axs[2].collections[:12]:
        obj.set_sizes([2.2**2])
    # Number the words 1-12, moving the column right to make room.
    for k, text in enumerate(word_texts):
        text.set_x(text.get_position()[0]+.023)
        fig.text(.450, text.get_position()[1], str(k+1), fontsize=8, ha='right', va='center')
    for obj in fig.artists:
        obj.set_xdata(np.asarray(obj.get_xdata())+.023)

    # Inset: crossings 5-8, which lie close together.
    inset = fig.add_axes([.099, .101, 1.4/6.3, .56/6], facecolor='white', zorder=20)
    bounds = (.3837, .3891, -.5353, -.5275)
    shade(inset, pocket)
    for part in line_parts(curves[5], bounds, .0015):
        xy = np.asarray(part.simplify(1e-10).coords)
        inset.plot(xy[:, 0], xy[:, 1], c=C[5], lw=1)
    inset.plot([Q[0], P[0]], [Q[1], P[1]], c=ORANGE, lw=1.8)
    for k, r in enumerate(ordered[4:8], 5):
        x, y = r['point']
        inset.scatter(x, y, s=2.2**2, c='black', linewidths=0, zorder=9)
        inset.annotate(str(k), (x, y), xytext={5: (-7, -9), 6: (8, 2), 7: (-8, 8), 8: (8, 5)}[k],
                       textcoords='offset points', fontsize=7, ha='center', zorder=10)
    inset.set(xlim=bounds[:2], ylim=bounds[2:], xticks=[], yticks=[])
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(.7)
        spine.set_color('black')
    assert not any(inset.bbox.contains(*axs[2].transData.transform(r['point'])) for r in ordered)

    set_line_weights(fig)
    style.save_pdf(fig, OUT/'global-manifolds-first-return.pdf')
    plt.close(fig)


def endpoint_panel(ax, rec, title):
    """Images of I at an endpoint of the pinning interval, sampled at the given phase."""
    a, delta = rec['a'], rec['phase']-S
    mp = Map(l1=H, l2=.8, a=a)
    e = matrix(delta, 1) @ np.array([1., mp.m])  # unstable direction transported from the midpoint
    c = np.linspace((1-a)/mp.lam, 1-a, 200001)
    seed = np.column_stack([1-c, -e[1]/e[0]*c])

    def flow(ys, t):
        z = np.array(ys, copy=True)
        for row in z:
            if lib.flow(row, t, 1., a) < 0:
                raise ValueError('flow to the sampling phase failed')
        return z
    p, p1, q = rec['P0'], rec['P1'], rec['Q0']
    draw(ax, seed, C[0], lw=1.6, label='$I$')
    zmid, first = flow(seed, -delta), None
    for n, color in enumerate(C[:3], 1):
        zmid, _ = mp.batch(zmid)
        z = flow(zmid, delta)
        if n == 1:
            first = z.copy()
        draw(ax, z, color, lw=.85, max_chord=.005, bounds=(-.3, 1.1, -1.05, .9), label=rf'$\Psi^{n}(I)$')
    ax.plot([0, p[0]], [0, p[1]], c=ORANGE, lw=.8)
    ax.plot([q[0], p[0]], [q[1], p[1]], c=ORANGE, lw=2.2)
    lower = rec['phase'] > S
    if lower:
        shade(ax, np.vstack([first, p1, p]))
    ax.scatter(*p, c='black', s=12, zorder=7)
    ax.annotate('$P_0$' if lower else '$Q_0=P_0$', p, xytext=(4, -12), textcoords='offset points')
    ax.scatter(*p1, c='black', s=12, zorder=7)
    if lower:
        ax.annotate('$Q_0=P_1$', p1, xytext=(.23, .08), ha='center',
                    arrowprops={'arrowstyle': '-', 'color': GRAY, 'lw': .45})
    else:
        ax.annotate('$P_1$', p1, xytext=(-15, -10), textcoords='offset points')
    ax.axvline(a, c=GRAY, ls=':', lw=.6)
    ax.scatter([0, 1], [0, 0], c='black', s=10)
    setup(ax, title, (-.3, 1.1, -1.05, .9), equilibria=False)
    ax.set_xticks([0, .5, 1])
    ax.set_yticks([-1, -.5, 0, .5])
    ax.set_ylabel('')
    ax.set_xlim(-.15, 1.08)
    assert max(np.linalg.norm(first[0]-p), np.linalg.norm(first[-1]-p1)) < 2e-14


def figure5(curves, mirror, endpoints, fronts):
    fig, axs = plt.subplots(1, 3, figsize=(6.30, 3.3))
    fig.subplots_adjust(left=.09, right=.99, bottom=.38, top=.92, wspace=.22)
    endpoint_panel(axs[0], endpoints[0], r'(a) $a=a_-$, $s=L_1$')
    endpoint_panel(axs[2], endpoints[1], r'(c) $a=a_+$, $s=0$')

    # (b) Fronts at a = 1/2 with their period samples, over Sigma_0, ..., Sigma_3 and their mirror images.
    ax, bounds = axs[1], (-.04, 1.04, -.8, .72)
    for j in range(4):
        draw(ax, curves[j], C[j], lw=.6, alpha=.35, bounds=bounds, max_chord=.002)
        mirror_image = curves[j].copy()
        mirror_image[:, 0] = 1-mirror_image[:, 0]
        draw(ax, mirror_image, ORANGE, lw=.6, alpha=.35, bounds=bounds, max_chord=.002)
    green = GreenFront(.4)
    for front, color in zip(fronts, FRONT_COLORS):
        rhos, sample_x, word = front['rhos'], front['sample_x'], front['word']
        # 100 points on each analytic piece between coefficient interfaces, crossings and samples.
        vertices = np.array([n*L+d for n in range(-20, 30) for d in [0, .4]])
        breaks = np.unique(np.r_[vertices, rhos, sample_x])
        xs = np.unique(np.concatenate([np.linspace(a, b, 101) for a, b in zip(breaks[:-1], breaks[1:])]))
        xy, samples = green.profile(xs, rhos), green.profile(sample_x, rhos)
        label = r'$u^P$' if word == 'P_0' else next(r['latex'] for r in mirror if r['word'] == word)
        ax.plot(xy[:, 0], xy[:, 1], c=color, lw=.9, ls='--' if word == 'P_0' else '-', label=label, zorder=4)
        ax.scatter(samples[:, 0], samples[:, 1], s=7, facecolor='white', edgecolor=color, linewidth=.5, zorder=6)
    setup(ax, r'(b) $a=1/2$, $s=L_1/2$', bounds)
    ax.set_ylabel('')
    ax.set_xticks([0, .5, 1])
    ax.set_yticks([-.5, 0, .5])
    axs[0].set_ylabel('$v=p u_x$')

    image_handles = [Line2D([], [], c=C[max(0, j-1)], lw=1.6 if j == 0 else .85,
                            label=('$I$' if j == 0 else '$\\Psi(I)$' if j == 1 else rf'$\Psi^{j}(I)$'))
                     for j in range(4)]
    front_handles, _ = ax.get_legend_handles_labels()
    background = [[Line2D([], [], c=C[j], alpha=.35, label=rf'$\Sigma_{j}$') for j in k] for k in [(0, 1), (2, 3)]]
    background.append([Line2D([], [], c=ORANGE, alpha=.35, label=r'$R(\Sigma_j)$')])
    samples = Line2D([], [], c='#555555', marker='o', markerfacecolor='white', markersize=3, ls='none',
                     label='period samples')
    style.legend_table(fig, [('(a), (c)', [image_handles[:2], image_handles[2:]]),
                             ('(b) fronts', [[h] for h in front_handles+[samples]]),
                             ('(b) background', background)])

    style.finish(fig, 'global-manifolds-endpoints')
    dot_sizes(fig)
    for obj in axs[1].collections:
        if obj.get_zorder() == 6:  # hollow period samples
            obj.set_sizes([(2.5-.4)**2])
            obj.set_linewidths([.4])
            obj.set_facecolors('none')
    for ax in fig.axes:  # black and hollow dots above the curves
        for obj in ax.collections:
            if isinstance(obj, PathCollection) and np.allclose(obj.get_facecolors()[:, :3], 0):
                obj.set_zorder(12)
    for obj in fig.artists:
        if isinstance(obj, Line2D) and obj.get_marker() == 'o':
            hollow = obj.get_markerfacecolor() == 'white'
            obj.set_markersize(2.1 if hollow else 2.2)
            obj.set_markeredgewidth(.4 if hollow else 0)
    set_line_weights(fig, fronts_zorder=4)
    # The thinner background arcs need more opacity.
    for line in axs[1].lines:
        if line.get_alpha() == .35:
            line.set_alpha(.5)
    for obj in fig.artists:
        if isinstance(obj, Line2D):
            if obj.get_alpha() == .35:
                obj.set_alpha(.5)
            if obj.get_linestyle() != 'None' and obj.get_linewidth() < 1.5:
                obj.set_linewidth(FRONT if to_hex(obj.get_color()) in FRONT_COLORS else MANIFOLD)
    style.save_pdf(fig, OUT/'global-manifolds-endpoints.pdf')
    plt.close(fig)


def main():
    style.apply()
    plt.rcParams['path.simplify'] = False
    OUT.mkdir(exist_ok=True)
    ts, curves, outward, zs = manifold_images()
    crossings = base_crossings(5, ts, curves[5])
    mirror = mirror_crossings(ts, curves[2])
    endpoints = endpoint_states()
    fronts = front_samples(mirror)
    figure4(ts, curves, outward, zs, crossings, mirror)
    figure5(curves, mirror, endpoints, fronts)
    print('Figure 4: crossing words of Sigma_5 with the base, from Q_0 to P_0:')
    for r in sorted(crossings, key=lambda r: r['point'][0]):
        print('  ', r['word'])
    print(f"Figure 5: a_- = {endpoints[0]['a']:.6f}, a_+ = {endpoints[1]['a']:.6f}")


if __name__ == '__main__':
    main()
