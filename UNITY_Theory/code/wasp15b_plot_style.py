"""WASP-15b publication style, based on plot_figure3_r400.py."""
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator

MODEL_COLOUR = '#6F3CC3'
DATA_COLOUR = '#111111'

def apply_style():
    mpl.rcParams.update({
        'font.family': 'DejaVu Sans', 'mathtext.fontset': 'stix',
        'font.size': 8.0, 'axes.labelsize': 8.0, 'axes.titlesize': 9.0,
        'xtick.labelsize': 7.0, 'ytick.labelsize': 7.0, 'legend.fontsize': 7.5,
        'axes.linewidth': 0.7, 'axes.edgecolor': 'black',
        'text.color': 'black', 'axes.labelcolor': 'black',
        'xtick.color': 'black', 'ytick.color': 'black',
        'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
        'xtick.minor.width': 0.5, 'ytick.minor.width': 0.5,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'axes.grid': False, 'legend.frameon': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'pdf.fonttype': 42,
        'ps.fonttype': 42, 'svg.fonttype': 'none',
    })

def style_axis(ax):
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.7)
    ax.tick_params(which='both', direction='in', top=True, right=True, pad=2.0)
    ax.tick_params(which='major', length=3.0, width=0.7)
    ax.tick_params(which='minor', length=1.5, width=0.5)
    if ax.get_xscale() == 'linear':
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    if ax.get_yscale() == 'linear':
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
