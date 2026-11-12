"""Shared data handling and drawing utilities."""

import matplotlib.pyplot as plt
import numpy as np


COLOR = "#563276"
CMAP = "viridis"


def _data(explanation):
    values = np.asarray(explanation.values, dtype=float)
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError("Explanation values must be a nonempty 2D array.")
    if not np.isfinite(values).all():
        raise ValueError("Explanation values must be finite.")
    names = explanation.feature_names
    names = list(range(values.shape[1])) if names is None else list(names)
    if len(names) != values.shape[1]:
        raise ValueError("Labels must match the number of explanation columns.")
    return values, names


def _index(feature, names):
    return names.index(feature) if isinstance(feature, str) else int(feature)


def _order(values, names, order, max_display):
    # Default to descending mean importance, or preserve the caller's explicit feature
    # order.
    indices = (
        np.argsort(-values.mean(axis=0), kind="stable")
        if order is None
        else np.asarray([_index(item, names) for item in order], dtype=int)
    )
    if indices.size == 0 or np.any(indices < 0) or np.any(indices >= len(names)):
        raise ValueError("order must select valid explanation columns.")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("order must not contain duplicates.")
    if max_display is not None:
        if max_display < 1:
            raise ValueError("max_display must be positive or None.")
        indices = indices[:max_display]
    return indices


def _feature_data(X, names, n_rows):
    # Align DataFrame columns by feature name; array columns must already be in explanation
    # order.
    if hasattr(X, "loc"):
        matrix = X.loc[:, names].to_numpy()
    else:
        matrix = np.asarray(X)
    if matrix.shape != (n_rows, len(names)):
        raise ValueError("X must match explanation rows and columns in order.")
    return matrix


def _column(X, feature, names):
    if hasattr(X, "iloc"):
        if isinstance(feature, str):
            return X[feature].to_numpy(), feature
        return X.iloc[:, feature].to_numpy(), X.columns[feature]
    index = _index(feature, names)
    return np.asarray(X)[:, index], feature


def _is_categorical(values):
    try:
        np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return True
    return False


def _category_colors(values, cmap):
    # Map category labels to stable color indices within this feature.
    labels, codes = np.unique(values, return_inverse=True)
    palette = plt.get_cmap(cmap)(np.linspace(0.1, 0.9, len(labels)))
    return labels, codes, palette


def _swarm_offsets(values, width=0.35):
    """Stack points within equal-width bins without moving their x values."""
    span = np.ptp(values)
    bins = (
        np.zeros(len(values), dtype=int)
        if span == 0
        else np.floor(99 * (values - values.min()) / span).astype(int)
    )
    # Alternate vertical offsets around each feature row for points in the same horizontal
    # bin.
    offsets = np.zeros(len(values))
    for bin_id in np.unique(bins):
        rows = np.flatnonzero(bins == bin_id)
        levels = np.arange(len(rows))
        offsets[rows] = np.ceil(levels / 2) * (2 * (levels % 2) - 1)
    return width * offsets / max(1, np.abs(offsets).max())


def _summary_axis(ax, labels, xlabel):
    positions = np.arange(len(labels))
    ax.set_yticks(positions, labels)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_xlabel(xlabel)
    ax.axvline(0, color="0.55", linewidth=0.8, zorder=0)
    ax.grid(axis="y", color="0.9", linestyle=":", zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, labelleft=True)


def _colorbar(ax, points, options, relative=False):
    kwargs = {"label": "Feature value", "pad": 0.02}
    kwargs.update(options or {})
    colorbar = ax.figure.colorbar(points, ax=ax, **kwargs)
    # Label per-feature normalized colors as Low and High rather than original measurement
    # units.
    if relative and "ticks" not in kwargs:
        colorbar.set_ticks([0, 1], labels=["Low", "High"])
    return colorbar
