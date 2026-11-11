"""Feature correlations alongside LUCID distributions."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.path import Path
from matplotlib.patches import PathPatch

from ._beeswarm import beeswarm
from ._utils import (
    CMAP, COLOR, _data, _feature_data, _is_categorical, _order,
)


def relationship(
    explanation, X, *, predictability=None, max_display=20, order=None,
    min_correlation=0, cmap=CMAP, correlation_cmap="coolwarm",
    categorical=None, s=12, alpha=0.8, colorbar=True,
    colorbar_kwargs=None, correlation_colorbar_kwargs=None, axes=None,
):
    """Combine Pearson feature correlations and a LUCID beeswarm.

    This plot requires individual-feature explanations. Optional predictability
    scores are supplied by the caller as a name-to-score mapping or an array
    in explanation-column order; no models are fitted here. Nominal features
    are excluded from Pearson links. Return a dict of Matplotlib axes with
    keys 'correlation', 'values', and, when requested, 'predictability'.
    """
    values, names = _data(explanation)
    matrix = _feature_data(X, names, len(values))
    indices = _order(values, names, order, max_display)
    if not 0 <= min_correlation <= 1:
        raise ValueError("min_correlation must be between zero and one.")
    categorical = set(categorical or [])
    scores = None
    if predictability is not None:
        scores = np.asarray(
            [predictability[name] for name in names]
            if isinstance(predictability, dict) else predictability,
            dtype=float,
        )
        if scores.shape != (len(names),):
            raise ValueError("Provide one predictability score per feature.")

    keys = ["correlation", "values"]
    widths = [1, 3]
    if scores is not None:
        keys.insert(1, "predictability")
        widths.insert(1, 1)
    if axes is None:
        _, panels = plt.subplots(
            1, len(keys), sharey=True,
            figsize=(10 if scores is None else 12, max(3, 0.5 * len(indices) + 1)),
            gridspec_kw={"width_ratios": widths}, layout="constrained",
        )
        axes = dict(zip(keys, panels))

    beeswarm(
        explanation, X, order=indices, max_display=None, cmap=cmap,
        categorical=categorical, s=s, alpha=alpha, colorbar=colorbar,
        colorbar_kwargs=colorbar_kwargs, ax=axes["values"],
    )

    network = axes["correlation"]
    norm = Normalize(-1, 1)
    palette = plt.get_cmap(correlation_cmap)
    for left, j in enumerate(indices):
        if names[j] in categorical or j in categorical or _is_categorical(matrix[:, j]):
            continue
        for right in range(left + 1, len(indices)):
            k = indices[right]
            if names[k] in categorical or k in categorical or _is_categorical(matrix[:, k]):
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                correlation = np.corrcoef(
                    np.asarray(matrix[:, j], dtype=float),
                    np.asarray(matrix[:, k], dtype=float),
                )[0, 1]
            if not np.isfinite(correlation) or abs(correlation) < min_correlation:
                continue
            bend = max(0.05, 0.8 - 0.12 * (right - left))
            path = Path(
                [(1, left), (bend, left), (bend, right), (1, right)],
                [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4],
            )
            network.add_patch(PathPatch(
                path, fill=False, color=palette(norm(correlation)),
                linewidth=0.5 + 2 * abs(correlation), alpha=0.8,
            ))
    network.scatter(np.ones(len(indices)), np.arange(len(indices)), s=10, color=COLOR)
    network.set_xlim(0, 1.1)
    network.set_ylim(len(indices) - 0.5, -0.5)
    network.set_axis_off()
    if colorbar:
        options = {
            "label": "Feature correlation", "location": "left",
            "pad": 0.03, "ticks": [-1, 0, 1],
        }
        options.update(correlation_colorbar_kwargs or {})
        network.figure.colorbar(ScalarMappable(norm=norm, cmap=palette), ax=network, **options)

    if scores is not None:
        axis = axes["predictability"]
        axis.barh(np.arange(len(indices)), scores[indices], height=0.6, color=COLOR)
        axis.set_ylim(len(indices) - 0.5, -0.5)
        axis.set_xlabel("Predictability")
        axis.tick_params(axis="y", left=False, labelleft=False)
        axis.spines[["top", "right", "left"]].set_visible(False)
    return axes
