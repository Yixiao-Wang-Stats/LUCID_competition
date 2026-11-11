"""Feature or group LUCID for selected evaluation observations."""

import matplotlib.pyplot as plt
import numpy as np

from ._utils import COLOR, _data, _order, _summary_axis


def observations(
    explanation, observations=0, *, ncols=4, max_display=20,
    order=None, color=COLOR, s=20, axes=None, **kwargs,
):
    """Plot one dot per feature/group for each selected observation.

    observations is a zero-based row position or a sequence such as range(10).
    Each panel is ordered by its own LUCID values unless order is supplied.
    Return an Axes for a scalar selection, otherwise a 2D array of axes.
    Extra keywords are passed to ``ax.scatter``.
    """
    values, names = _data(explanation)
    single = isinstance(observations, (int, np.integer))
    rows = np.asarray([observations] if single else list(observations))
    if rows.ndim != 1 or rows.size == 0 or rows.dtype.kind not in "iu":
        raise ValueError("observations must select one or more integer row positions.")
    if np.any(rows < 0) or np.any(rows >= len(values)):
        raise IndexError("Observation position is out of range.")
    if ncols < 1:
        raise ValueError("ncols must be positive.")

    if axes is None:
        columns = 1 if single else ncols
        nrows = (len(rows) + columns - 1) // columns
        count = len(_order(values, names, order, max_display))
        _, axes = plt.subplots(
            nrows, columns, squeeze=False, layout="constrained",
            figsize=(3.8 * columns, max(2.2, 0.35 * count + 1.2) * nrows),
        )
    panels = np.asarray(axes, dtype=object).reshape(-1)
    if len(panels) < len(rows):
        raise ValueError("Provide at least one axes for each observation.")

    options = {"color": color, "s": s, "zorder": 3}
    options.update(kwargs)
    for row, ax in zip(rows, panels):
        scores = values[row]
        indices = _order(scores[None, :], names, order, max_display)
        ax.scatter(scores[indices], np.arange(len(indices)), **options)
        _summary_axis(
            ax, [names[j] for j in indices], f"LUCID value (observation {row})"
        )
    for ax in panels[len(rows):]:
        ax.set_axis_off()
    return panels[0] if single else np.atleast_2d(axes)
