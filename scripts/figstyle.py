"""Shared drawing primitives for the architecture figures.

``plot_arch.py`` and ``plot_probe_arch.py`` both draw a chain of tall rounded
boxes on white, in the visual language of the Phantom figure, so the pieces live
here rather than being copied into each. Nothing in this module knows what is
being drawn; callers pass the vertical centre of their own chain.
"""
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

BLUE = "#1f6fb4"          # the main data path
ACCENT = "#7b4fa8"        # whatever the figure is actually about
RED = "#c0392b"           # the supervised/geometry side of the probe
GREY = "#5a5a5a"
FROZEN = "#e8eef5"
INK = "#1a1a1a"


def vbox(ax, x, cy, label, edge=BLUE, face="white", fs=8.5, w=0.56, h=2.15,
         sub=None):
    """A tall rounded box with its label rotated, optionally with a smaller
    second line beneath the first."""
    ax.add_patch(FancyBboxPatch((x - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.015,rounding_size=0.09",
                                linewidth=1.3, edgecolor=edge, facecolor=face,
                                zorder=3))
    if sub is None:
        ax.text(x, cy, label, rotation=90, ha="center", va="center",
                fontsize=fs, color=INK, zorder=4)
    else:
        ax.text(x + 0.10, cy, label, rotation=90, ha="center", va="center",
                fontsize=fs, color=INK, zorder=4)
        ax.text(x - 0.14, cy, sub, rotation=90, ha="center", va="center",
                fontsize=fs - 1.4, color=GREY, zorder=4)
    return x + w / 2


def hbox(ax, x, y, w, h, label, edge=GREY, face="white", fs=8.5,
         weight="normal"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.2, edgecolor=edge, facecolor=face,
                                zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, color=INK,
            zorder=4, weight=weight, linespacing=1.35)
    return x + w / 2


def oplus(ax, x, y, sym="+", color=BLUE, r=0.115):
    ax.add_patch(Circle((x, y), r, fill=True, facecolor="white",
                        edgecolor=color, linewidth=1.3, zorder=5))
    ax.text(x, y - 0.005, sym, ha="center", va="center", fontsize=9.5,
            color=color, zorder=6)
    return x + r


def arrow(ax, p, q, color=BLUE, lw=1.3, style="-|>", rad=0.0, ls="-", z=2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11,
                                 linewidth=lw, color=color, zorder=z,
                                 linestyle=ls, shrinkA=0, shrinkB=0,
                                 connectionstyle=f"arc3,rad={rad}"))


def frame(ax, x0, x1, y0, y1, edge=ACCENT, lw=1.5):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                linewidth=lw, edgecolor=edge, facecolor="none",
                                zorder=1))


def canvas(plt, w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax
