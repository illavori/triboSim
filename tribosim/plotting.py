"""
plotting.py
-----------
Optional plotting style used by the examples and the paper scripts: a teal
palette from light to dark, black for measurements or references, and a
serif font.  Matplotlib is imported only when apply_style() is called, so the
package itself does not depend on it.

    from tribosim.plotting import apply_style, TEAL
    apply_style()
    ax.plot(x, q, color=TEAL[2])

Author : Iñigo Llavori Osa - Mondragon Unibertsitatea
"""

LIGHT = "#7fd0d2"
MID = "#00a0a5"
D1 = "#006d70"
DARK = "#003d3d"
GREY = "#6e6e6e"
BLACK = "#000000"
TEAL = (LIGHT, MID, D1, DARK)          # light to dark


def apply_style(font_size: float = 10) -> None:
    """Set the Matplotlib defaults: serif font, teal colour cycle, inward ticks."""
    import matplotlib as mpl
    from cycler import cycler

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 2,
        "legend.frameon": False,
        "axes.linewidth": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.prop_cycle": cycler(color=[MID, DARK, LIGHT, D1]),
        "savefig.dpi": 300,
    })
