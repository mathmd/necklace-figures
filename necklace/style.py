"""Common typography for the figures: LaTeX text in Computer Modern at the printed size."""
import os
import matplotlib as mpl
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection
from matplotlib.font_manager import FontProperties

WIDTH = 6.30  # text width of the paper, in inches
HEIGHTS = {'monotone-front-summary': 5.6, 'global-many-crossing-profiles': 4.1,
           'global-manifolds-first-return': 6.0, 'global-manifolds-endpoints': 3.3}
PREAMBLE = (r'\usepackage{amsmath}\usepackage{amssymb}'
            r'\DeclareMathSizes{8}{8}{7}{7}\DeclareMathSizes{9}{9}{7}{7}')


def apply():
    mpl.rcParams.update({
        'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Computer Modern Roman'],
        'text.latex.preamble': PREAMBLE,
        'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'axes.linewidth': .6, 'xtick.major.width': .6, 'ytick.major.width': .6,
        'axes.spines.top': False, 'axes.spines.right': False,
        'lines.linewidth': 1.0, 'savefig.bbox': None,
        'pgf.texsystem': 'pdflatex', 'pgf.rcfonts': False, 'pgf.preamble': PREAMBLE})


def finish(fig, name):
    """Set the figure size and the sizes of all text, lines and markers.

    Titles and axis labels are 9 pt and all other text 8 pt. Lines are 1 pt,
    except that lines of at least 1.5 pt in Figures 4 and 5 become 1.8 pt.
    """
    apply()
    fig.set_size_inches(WIDTH, HEIGHTS[name])
    for text in fig.findobj(Text):
        text.set_usetex(True)
        text.set_fontfamily('serif')
        text.set_fontsize(8)
        text.set_text(text.get_text().replace(r'W^u', r'W^{\mathrm u}').replace(r'\mathdefault', r'\mathrm'))
    axes = list(fig.axes)
    for ax in fig.axes:
        axes.extend(ax.child_axes)
    thick = name in ('global-manifolds-first-return', 'global-manifolds-endpoints')
    for ax in axes:
        for title in [ax.title, ax._left_title, ax._right_title]:
            title.set_fontsize(9)
        ax.xaxis.label.set_fontsize(9)
        ax.yaxis.label.set_fontsize(9)
        axis_list = [ax.xaxis, ax.yaxis]
        if hasattr(ax, 'zaxis'):
            ax.zaxis.label.set_fontsize(9)
            axis_list.append(ax.zaxis)
        for axis in axis_list:
            if hasattr(axis, 'line'):
                axis.line.set_linewidth(.6)
        ax.tick_params(width=.6, labelsize=8)
        for spine in ax.spines.values():
            spine.set_linewidth(.6)
        for line in ax.lines:
            line.set_linewidth(1.8 if thick and line.get_linewidth() >= 1.5 else 1.0)
            if line.get_marker() not in ['None', 'none', '', None]:
                line.set_markersize(3)
        for collection in ax.collections:
            if isinstance(collection, PathCollection):
                collection.set_sizes([9])
        for text in ax.texts:
            if hasattr(text, 'arrow_patch') and text.arrow_patch:
                text.arrow_patch.set_linewidth(.6)
    for legend in fig.legends+[ax.get_legend() for ax in axes if ax.get_legend()]:
        for text in legend.get_texts():
            text.set_fontsize(8)
            text.set_usetex(True)
        legend.get_title().set_fontsize(8)
        legend.get_title().set_usetex(True)
        for handle, text in zip(legend.legend_handles, legend.get_texts()):
            if isinstance(handle, Line2D):
                handle.set_linewidth(1.8 if text.get_text() in ('$I$', 'base') else 1.0)
                handle.set_markersize(3)
    fig.canvas.draw()
    assert all(t.get_usetex() and t.get_fontsize() in (8, 9)
               for t in fig.findobj(Text) if t.get_visible() and t.get_text())
    return fig


def save_pdf(fig, path):
    """Save through the pgf backend, so that pdflatex embeds the fonts."""
    keys = ('extra_mem_top', 'extra_mem_bot')
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ.setdefault(key, '50000000')  # TeX memory for long paths
        fig.savefig(path, backend='pgf')
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def legend_table(fig, groups):
    """Draw legend groups side by side below the axes, with common baselines.

    Each group is (title, rows), and each row is a list of legend handles.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    unit_x = 1/(fig.get_figwidth()*72)
    unit_y = 1/(fig.get_figheight()*72)

    def width(text):
        return renderer.get_text_width_height_descent(text, FontProperties(size=8), ismath='TeX')[0]/fig.dpi*72

    columns = []
    for title, rows in groups:
        columns.append([max(width(row[j].get_label()) for row in rows if j < len(row))+14
                        for j in range(max(map(len, rows)))])
    widths = [max(sum(cols)+12*(len(cols)-1), width(title))
              for (title, rows), cols in zip(groups, columns)]
    gap = 24
    x = (1-(sum(widths)+gap*(len(groups)-1))*unit_x)/2
    header = .245
    for (title, rows), cols, total in zip(groups, columns, widths):
        fig.text(x, header, title, fontsize=8, va='baseline')
        for i, row in enumerate(rows):
            y = header-(i+1)*12*unit_y
            cx = x
            for j, handle in enumerate(row):
                label = handle.get_label()
                fig.add_artist(Line2D(
                    [cx, cx+10*unit_x], [y+2.5*unit_y]*2, transform=fig.transFigure,
                    color=handle.get_color(), alpha=handle.get_alpha(),
                    lw=1.8 if label == '$I$' else 1, ls=handle.get_linestyle(),
                    marker=handle.get_marker(), markevery=[0], markersize=3,
                    markerfacecolor=handle.get_markerfacecolor(),
                    markeredgecolor=handle.get_markeredgecolor()))
                fig.text(cx+14*unit_x, y, label, fontsize=8, va='baseline')
                cx += (cols[j]+12)*unit_x
        x += (total+gap)*unit_x
