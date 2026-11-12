# hold local values and feature names
# provide global_importance().

import numpy as np


class Explanation:
    """LUCID values for a set of evaluation observations."""

    def __init__(self, values, feature_names=None):
        self.values = np.asarray(values, dtype=float)
        self.feature_names = feature_names

    def global_importance(self):
        """Compute global LUCID by averaging over evaluation observations."""
        return self.values.mean(axis=0)