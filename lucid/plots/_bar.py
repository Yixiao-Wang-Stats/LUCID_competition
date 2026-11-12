"""Horizontal importance bars."""

import matplotlib.pyplot as plt
import numpy as np

from ._utils import COLOR, _data, _order, _summary_axis


def bar(
    explanation, *, observation=None, max_display=20, order=None,
    color=COLOR, ax=None, **kwargs,
):
    """Plot mean feature/group LUCID, or the values for one observation.

    Return the Matplotlib axes. Extra keywords are passed to ``ax.barh``.
    """
    values, names = _data(explanation)
    scores = values.mean(axis=0) if observation is None else values[observation]
    indices = _order(scores[None, :], names, order, max_display)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, max(2, 0.45 * len(indices) + 1)))

    options = {"color": color, "height": 0.7, "edgecolor": "none"}
    options.update(kwargs)
    ax.barh(np.arange(len(indices)), scores[indices], **options)
    label = "Mean LUCID value" if observation is None else "LUCID value"
    _summary_axis(ax, [names[j] for j in indices], label)
    return ax
