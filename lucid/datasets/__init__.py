# Expose real-data loaders and the named synthetic-dataset namespace.
from . import dgp
from ._loaders import auto_mpg, bikesharing

__all__ = ["auto_mpg", "bikesharing", "dgp"]