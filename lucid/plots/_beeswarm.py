"""Observation-level LUCID distributions."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from ._utils import (
    CMAP, COLOR, _category_colors, _colorbar, _data, _feature_data,
    _is_categorical, _order, _summary_axis, _swarm_offsets,
)


def beeswarm(
    explanation, X=None, *, max_display=20, order=None, cmap=CMAP,
    color=COLOR, norm=None, categorical=None, s=12, alpha=0.8,
    colorbar=True, colorbar_kwargs=None, ax=None,
):
    """Plot one LUCID point per observation and feature or group.

    X must have the same row order; DataFrame columns are matched by name.
    Numeric colors are scaled within each feature unless norm is supplied.
    List nominal feature names/indices in categorical for discrete colors.
    For group explanations, omit X to use a uniform color. Return the axes.
    """
    values, names = _data(explanation)
    indices = _order(values, names, order, max_display)
    matrix = None if X is None else _feature_data(X, names, len(values))
    categorical = set(categorical or [])
    if ax is None:
        _, ax = plt.subplots(figsize=(7, max(2, 0.5 * len(indices) + 1)))

    numeric_points = None
    for position, j in enumerate(indices):
        offsets = _swarm_offsets(values[:, j])
        options = {"s": s, "alpha": alpha, "linewidths": 0, "zorder": 3}
        if matrix is None:
            options["color"] = color
        elif names[j] in categorical or j in categorical or _is_categorical(matrix[:, j]):
            _, codes, palette = _category_colors(matrix[:, j], cmap)
            options["c"] = palette[codes]
        else:
            feature_values = np.asarray(matrix[:, j], dtype=float)
            if norm is None:
                span = np.ptp(feature_values)
                feature_values = (
                    np.full(len(values), 0.5) if span == 0
                    else (feature_values - feature_values.min()) / span
                )
            options.update(c=feature_values, cmap=cmap, norm=norm or Normalize(0, 1))

        points = ax.scatter(values[:, j], position + offsets, **options)
        if "norm" in options:
            numeric_points = points

    _summary_axis(ax, [names[j] for j in indices], "LUCID value")
    if colorbar and numeric_points is not None:
        _colorbar(ax, numeric_points, colorbar_kwargs, relative=norm is None)
    return ax
