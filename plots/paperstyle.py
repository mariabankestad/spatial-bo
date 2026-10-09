"""Figure style of the paper: Computer Modern through LaTeX, AISTATS widths,
one palette, one colormap.
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "experiments"), _os.path.join(_ROOT, "training"), _os.path.join(_ROOT, "plots")):
    _p in _sys.path or _sys.path.insert(0, _p)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEXTWIDTH, COLWIDTH = 6.75, 3.25          # inches, aistats2026.sty
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath,amssymb}",
    "font.family": "serif",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.frameon": False, "legend.handlelength": 1.4,
    "lines.linewidth": 1.4, "lines.markersize": 3.5,
    "figure.dpi": 100, "savefig.dpi": 300, "pdf.fonttype": 42,
})
# categorical palette (validated: CVD dE >= 9, normal-vision dE >= 19), fixed order
COL = {"raw": "#2a78d6", "std_bo": "#e87ba4", "g_info": "#eb6834", "g_eff": "#1baf7a", "g_unif": "#eda100"}
GREY = "#8a8a8a"
LAB = {"raw": r"$g=1$ (no weighting)", "std_bo": r"standard UCB ($\sigma$ at $\hat\ell$)",
       "g_info": r"$g_{\mathrm{geo}}$", "g_eff": r"$g_{\mathrm{eff}}$", "g_unif": r"$g_{\mathrm{uniform}}$"}
ORDER = ("raw", "g_info", "g_eff", "g_unif")          # the controlled arms: only g changes
ORDER_ALL = ("raw", "std_bo", "g_info", "g_eff", "g_unif")   # appendix versions
SEQ_CMAP = "Blues"                        # one hue, light -> dark, for magnitude
