"""LUCID values against observed feature values."""

import matplotlib.pyplot as plt
from matplotlib.colors import is_color_like
from matplotlib.lines import Line2D

from ._utils import (
    CMAP, COLOR, _category_colors, _colorbar, _column, _data,
    _index, _is_categorical,
)


def scatter(
    explanation, X, feature, *, x=None, color=None, cmap=CMAP,
    norm=None, categorical=False, s=16, alpha=0.8,
    colorbar=True, colorbar_kwargs=None, ax=None, **kwargs,
):
    """Plot a feature/group's LUCID against one observed feature.

    feature selects the explanation column; x selects the data column and
    defaults to feature. For groups, supply x explicitly. color can select
    another data column or be a Matplotlib color. Set categorical=True for
    nominal numeric color codes. X rows must match the explanation.
    Return the Matplotlib axes; extra keywords go to ``ax.scatter``.
    """
    values, names = _data(explanation)
    # Resolve the explanation column independently of the observed feature used for the
    # horizontal axis.
    j = _index(feature, names)
    x_values, x_name = _column(X, feature if x is None else x, names)
    if len(x_values) != len(values):
        raise ValueError("X rows must match the explanation rows in order.")
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    options = {"s": s, "alpha": alpha, "linewidths": 0}
    # Distinguish a feature name used for point colors from a literal Matplotlib color.
    color_is_column = (
        isinstance(color, str)
        and color in (X.columns if hasattr(X, "columns") else names)
    )
    handles = []
    numeric_color = False
    if color is None:
        options["color"] = COLOR
    elif not color_is_column and is_color_like(color):
        options["color"] = color
    else:
        color_values, color_name = _column(X, color, names)
        # Use discrete colors and legend entries for categories, or a continuous color scale
        # otherwise.
        if categorical or _is_categorical(color_values):
            labels, codes, palette = _category_colors(color_values, cmap)
            options["c"] = palette[codes]
            handles = [
                Line2D([], [], marker="o", linestyle="none", color=shade,
                       label=f"{color_name}: {label}")
                for label, shade in zip(labels, palette)
            ]
        else:
            options.update(c=color_values, cmap=cmap, norm=norm)
            numeric_color = True
    options.update(kwargs)
    points = ax.scatter(x_values, values[:, j], **options)
    ax.set_xlabel(str(x_name))
    ax.set_ylabel(f"LUCID value ({names[j]})")
    ax.spines[["top", "right"]].set_visible(False)

    if colorbar and numeric_color:
        bar_options = {"label": str(color_name), **(colorbar_kwargs or {})}
        _colorbar(ax, points, bar_options)
    if handles:
        ax.legend(handles=handles, frameon=False)
    return ax
